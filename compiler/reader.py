"""Provides the structural reading of a compiled Triton kernel into the matmuls it computes."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from triton.runtime import JITFunction

from .frontend import compile_to_ttir
from .ir import _REDUCTION_AXIS, Access, Carried, Guard, MatMul, Operation, ProgramError, TileLoop

_ASSUMED_ACCUMULATOR_ELEMENT: str = "f32"
"""Element type a dot accumulates in when its printed result type states none."""

_AXIS_VARIABLES: dict[str, tuple[str, str]] = {
    "activation": ("m", "k"),
    "weight": ("k", "n"),
    "output": ("m", "n"),
}
"""Induction variable each tensor axis of each operand carries, before any transpose."""

_PENDING_AXIS: int = -1
"""Axis a rank one iota occupies until an expand_dims operation places it in a ranked tensor."""

_ADDRESS_OPERATIONS: frozenset[str] = frozenset(
    {"arith.constant", "tt.splat", "tt.make_range", "tt.addptr", "arith.muli", "arith.addi",
     "tt.get_program_id", "arith.extsi", "arith.trunci"}
)
"""Operations forming an address or supplying a literal rather than a step of an epilogue."""

_RESHAPES: frozenset[str] = frozenset({"tt.broadcast", "tt.expand_dims"})
"""Operations carrying a value into another shape. A walk passes through these recording nothing, because the
same pair spreads a reduced row back over a tile and places an iota in an address, and only what lies behind
them tells the two apart."""

_PREDICATE_RESHAPES: frozenset[str] = frozenset(
    {"tt.broadcast", "tt.expand_dims", "arith.extsi", "arith.trunci"}
)
"""Operations that carry a predicate into its access's shape without changing which elements it selects."""

_POINTWISE_TRANSFORMS: frozenset[str] = frozenset(
    {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.extf", "arith.truncf", "arith.maxnumf",
     "arith.minnumf", "math.erf", "math.exp", "math.tanh", "math.sqrt", "math.rsqrt"}
)
"""Operations transforming a tile element by element, which change the values it holds and never where it lies. A
walk looking for the memory behind a matmul operand passes through these, leaving the addressing of an operand a
transform stands in front of readable."""

_WIDTH_CASTS: frozenset[str] = frozenset({"arith.extf", "arith.truncf"})
"""Operations changing an element's width alone, which a traced transform records none of. The machine stages one
width whatever width the kernel wrote its arithmetic in, so recording them would key one fused transform under as
many sets as there are ways to spell its precision."""

_ORDERING_PREDICATES: dict[int, tuple[bool, bool]] = {
    2: (True, False), 3: (True, True), 6: (True, False), 7: (True, True),
    4: (False, False), 5: (False, True), 8: (False, False), 9: (False, True),
}
"""Every arith.cmpi predicate that orders its operands, as its enum value to whether it reads "less" and whether it
is inclusive. Signed and unsigned forms both appear because an index is non-negative either way. Which side the axis
falls on is what makes one of these an upper bound, so it is read from the operands rather than assumed."""

_LOAD_LINE: re.Pattern[str] = re.compile(r"=\s*tt\.load\s+(?P<operands>[^:]+):")
"""A printed tt.load, up to the colon that begins its type."""

_CONSTANT_LINE: re.Pattern[str] = re.compile(
    r"^\s*(?P<name>%\S+)\s*=\s*arith\.constant\s+(?:dense<(?P<dense>[^>]*)>|(?P<scalar>[^:]+?))\s*:"
)
"""A printed arith.constant, splat or scalar."""


@dataclass(slots=True)
class _Expression:
    """Describes a resolved pointer or index expression.

    Attributes:
        base: Pointer argument the expression offsets from, or None when it carries none.
        axes: Scalar stride multiplying each tensor axis's iota, keyed by axis.
        scalar: The uniform part, as the parameter naming it or an integer.
    """

    base: str | None = None
    axes: dict[int, object] = field(default_factory=dict)
    scalar: object = None


class _Graph:
    """Indexes a compiled module once, then answers every structural question by lookup.

    Args:
        module: The compiled module.
        signature: The signature the module was built from, which names its arguments.

    Attributes:
        module: The module this graph indexes.
        _resolved: Memoized expression per resolved value.
        _zero_filled: Cached zero fill flag per load, read from the printed module on first use.
        _definitions: Defining operation per value.
        _named: Operations grouped by name, in walk order.
        _loop_of_block: Loop owning each body block.
        _body_of_loop: Body block each loop owns.
        _yield_of_loop: Terminator of each loop body.
        _carried: Value each loop body argument entered the loop holding.
        _resolvers: Expression reader per operation name.
        _arguments: Parameter name per entry function argument.
    """

    def __init__(self, module, *, signature: dict[str, str]) -> None:
        self.module = module
        self._resolved: dict[int, _Expression] = {}
        self._zero_filled: list[bool] | None = None
        operations: list = []
        module.walk(operations.append)

        self._definitions: dict[int, object] = {}
        self._blocks: dict[int, object] = {}
        self._positions: dict[int, int] = {}
        self._named: dict[str, list] = {}
        blocks: dict[int, object] = {}
        for position, operation in enumerate(operations):
            self._positions[id(operation)] = position
            self._named.setdefault(operation.get_name(), []).append(operation)
            for index in range(operation.get_num_results()):
                self._definitions[operation.get_result(index).id()] = operation
            block = operation.get_block()
            if block is not None:
                blocks.setdefault(block.get_parent().id(), block)
        self._blocks = blocks

        self._loop_of_block: dict[int, object] = {}
        self._body_of_loop: dict[int, object] = {}
        self._yield_of_loop: dict[int, object] = {}
        self._carried: dict[int, object] = {}
        for loop in self.by_name("scf.for"):
            body = blocks.get(loop.get_region(0).id())
            if body is None:
                continue
            self._loop_of_block[body.id()] = loop
            self._body_of_loop[id(loop)] = body
            for index in range(1, body.get_num_arguments()):
                # Reads argument zero as the induction variable, and argument index as loop operand two plus
                # index, which is the value it entered the loop holding.
                self._carried[body.get_argument(index).id()] = loop.get_operand(2 + index)
            self._yield_of_loop[id(loop)] = next(
                (operation for operation in self.by_name("scf.yield")
                 if operation.get_block() is not None and operation.get_block().id() == body.id()), None)

        self._resolvers = {
            "tt.broadcast": self._resolve_through, "arith.extsi": self._resolve_through,
            "arith.trunci": self._resolve_through, "tt.splat": self._resolve_splat,
            "arith.constant": self._resolve_constant, "tt.make_range": self._resolve_range,
            "tt.expand_dims": self._resolve_expand_dims, "arith.muli": self._resolve_multiply,
            "arith.addi": self._resolve_add, "tt.addptr": self._resolve_add,
        }

        names = [name for name, kind in signature.items() if kind != "constexpr"]
        function = module.get_function(module.get_entry_func_name())
        self._arguments: dict[int, str] = {
            function.args(index).id(): names[index] for index in range(function.get_num_args())
        }

    def by_name(self, name: str) -> list:
        """Returns every operation of the given name, in walk order."""
        return self._named.get(name, [])

    def enclosing_loops(self, operation) -> list:
        """Returns every loop whose body contains the given operation, outermost first."""
        chain = []
        current = operation
        while current is not None:
            block = current.get_block()
            if block is None or block.id() not in self._loop_of_block:
                break
            current = self._loop_of_block[block.id()]
            chain.append(current)
        return chain[::-1]

    def tile_loop(self, loop, *, axis: str | None = None) -> TileLoop | None:
        """Returns the trips, step and axis of a loop, or None when any of its three bounds is a runtime value."""
        lower, upper, step = (self.constant(loop.get_operand(index)) for index in range(3))
        if lower is None or upper is None or step is None or step <= 0 or upper <= lower:
            return None
        return TileLoop(trips=-(-(upper - lower) // step), step=step, bound=upper, axis=axis)

    def constant(self, value) -> int | None:
        """Returns the value of a constant, or None when it is computed at runtime."""
        defining = self._definitions.get(value.id())
        if defining is None or defining.get_name() != "arith.constant":
            return None
        try:
            return defining.get_constant_value()
        except (RuntimeError, ValueError, TypeError):
            return None

    def addresses_with(self, pointer, *, variable) -> bool:
        """Returns whether one pointer expression depends on the given induction variable.

        A loop walks the axis of whichever operand its variable addresses, which two loops sharing a step still
        answer differently.

        Args:
            pointer: The pointer value to walk back from.
            variable: The loop's induction variable.

        Returns:
            True when the variable is reached.
        """
        wanted = variable.id()
        seen: set[int] = set()
        pending = [pointer]
        while pending:
            current = pending.pop()
            key = current.id()
            if key == wanted:
                return True
            if key in seen:
                continue
            seen.add(key)
            if key in self._carried:
                pending.append(self._carried[key])
            operation = self._definitions.get(key)
            if operation is not None:
                pending.extend(operation.get_operand(index) for index in range(operation.get_num_operands()))
        return False

    def axis_of_loop(self, loop, *, pointers: dict[str, object]) -> str | None:
        """Returns the output axis one loop walks, read from which operand's address its variable reaches.

        Args:
            loop: The loop to name.
            pointers: The activation's and the weight's pointer values, keyed by role.

        Returns:
            "m" when the variable addresses the activation, "n" when it addresses the weight, or None when it
            reaches neither or both.
        """
        body = self._body_of_loop.get(id(loop))
        if body is None or not body.get_num_arguments():
            return None
        variable = body.get_argument(0)
        reached = {role for role, pointer in pointers.items()
                   if pointer is not None and self.addresses_with(pointer, variable=variable)}
        if reached == {"activation"}:
            return "m"
        if reached == {"weight"}:
            return "n"
        return None

    def axis_of_grid(self, operation, *, pointers: dict[str, object]) -> str | None:
        """Returns the output axis one program id walks, read from which operand's address it reaches.

        A grid carries its axis the way a loop carries one, the tile it lands on being whichever operand its
        identifier addresses, so this reads it the same way `axis_of_loop` reads a loop.

        Args:
            operation: The tt.get_program_id operation to name.
            pointers: The activation's and the weight's pointer values, keyed by role.

        Returns:
            "m" when the identifier addresses the activation, "n" when it addresses the weight, or None when it
            reaches neither or both.
        """
        value = operation.get_result(0)
        reached = {role for role, pointer in pointers.items()
                   if pointer is not None and self.addresses_with(pointer, variable=value)}
        if reached == {"activation"}:
            return "m"
        if reached == {"weight"}:
            return "n"
        return None

    def trace(self, value, *, among: list) -> tuple[object | None, frozenset[str]]:
        """Traces a value back to whichever of the given operations produced it.

        Args:
            value: The value to trace.
            among: The operations that terminate the trace, being the kernel's matmuls.

        Returns:
            The producing operation, or None when the value comes from memory, the arithmetic operation names
            passed through on the way, and those operations in program order.
        """
        found: set[str] = set()
        chain: dict[int, Operation] = {}
        seen: set[int] = set()
        pending = [value]
        source = None

        while pending:
            current = pending.pop()
            key = current.id()
            if key in seen:
                continue
            seen.add(key)
            operation = self._definitions.get(key)
            if operation is None:
                continue

            if operation in among:
                source = operation
                continue
            name = operation.get_name()
            if name == "scf.for":
                terminator = self._yield_of_loop.get(id(operation))
                if terminator is not None:
                    index = next(position for position in range(operation.get_num_results())
                                 if operation.get_result(position).id() == key)
                    pending.append(terminator.get_operand(index))
                continue
            if name in _RESHAPES:
                pending.extend(operation.get_operand(index) for index in range(operation.get_num_operands()))
                continue
            if name == "tt.load" or name in _ADDRESS_OPERATIONS or operation.get_num_operands() == 0:
                continue

            if name not in _WIDTH_CASTS:
                named = f"{name}.{self.reduce_kind(operation)}" if name == "tt.reduce" else name
                found.add(named)
                chain[id(operation)] = Operation(
                    name=named, position=self._positions[id(operation)], result=operation.get_result(0).id(),
                    operands=tuple(operation.get_operand(index).id()
                                   for index in range(operation.get_num_operands())))
            pending.extend(operation.get_operand(index) for index in range(operation.get_num_operands()))
        return source, frozenset(found), tuple(sorted(chain.values(), key=lambda step: step.position))

    def carried_values(self, loop, *, among: list) -> tuple[Carried, ...]:
        """Returns the values one loop hands from each trip to the next.

        Args:
            loop: The scf.for operation to read.
            among: The operations a chain terminates at, being the kernel's matmuls.

        Returns:
            One record per carried value, in the order the loop declares them.
        """
        body = self._body_of_loop.get(id(loop))
        yielded = self._yield_of_loop.get(id(loop))
        if body is None or yielded is None:
            return ()

        found: list[Carried] = []
        for index in range(1, body.get_num_arguments()):
            argument = body.get_argument(index)
            try:
                shape = tuple(argument.get_shape())
            except (RuntimeError, ValueError, TypeError):
                shape = ()
            found.append(Carried(position=index - 1, entering=loop.get_operand(2 + index).id(),
                                 inside=argument.id(), leaving=yielded.get_operand(index - 1).id(),
                                 shape=shape,
                                 chain=self.trace(yielded.get_operand(index - 1), among=among)[2]))
        return tuple(found)

    def reduce_kind(self, operation) -> str | None:
        """Returns the operation one reduce folds its axis under, read from the region it carries.

        Every reduce prints as the same operation whatever it computes, so the combiner in its region is what
        tells a row maximum from a row sum.

        Args:
            operation: The tt.reduce operation to read.

        Returns:
            The combining operation's name, or None where the region does not resolve to one.
        """
        if not operation.get_num_regions():
            return None
        block = self._blocks.get(operation.get_region(0).id())
        if block is None:
            return None
        terminator = next((found for found in self.by_name("tt.reduce.return")
                           if found.get_block() is not None and found.get_block().id() == block.id()), None)
        if terminator is None:
            return None
        combiner = self._definitions.get(terminator.get_operand(0).id())
        return None if combiner is None else combiner.get_name()

    def accesses_of(self, *, dot, store) -> dict[str, Access]:
        """Reads how each operand of one matmul is addressed.

        Notes:
            A transpose between a load and the matmul reverses which variable each loaded axis carries.

        Args:
            dot: The matmul operation.
            store: The store draining it, or None when a later matmul consumes it instead.

        Returns:
            The accesses, keyed by role.
        """
        sources = [(role, self._load_behind(dot.get_operand(index)))
                   for index, role in enumerate(("activation", "weight"))]
        if store is not None:
            sources.append(("output", (store, False)))

        accesses: dict[str, Access] = {}
        for role, (operation, transposed) in sources:
            if operation is None:
                accesses[role] = Access(role=role, base=None, strides={}, guards=())
                continue
            resolved = self.resolve(operation.get_operand(0))
            variables = _AXIS_VARIABLES[role][::-1] if transposed else _AXIS_VARIABLES[role]
            accesses[role] = Access(
                role=role,
                base=resolved.base,
                strides={variables[axis]: stride for axis, stride in resolved.axes.items()
                         if 0 <= axis < len(variables)},
                guards=self.guards_of(operation, role=role, variables=variables),
            )
        return accesses

    def guards_of(self, operation, *, role: str, variables: tuple[str, ...]) -> tuple[Guard, ...]:
        """Reads the predicate on one access as the axes it bounds, refusing one the machine cannot reproduce.

        Args:
            operation: The load or store carrying the predicate.
            role: What the operand is to the machine.
            variables: Induction variable each tensor axis of this operand carries, transpose already applied.

        Returns:
            One guard per bounded axis, empty when the access carries no predicate.

        Raises:
            ProgramError: If the predicate is not a conjunction of axes bounded from above by uniform values, or
                if a masked load reads its excluded elements as anything but zero.
        """
        index = 2 if role == "output" else 1
        if operation.get_num_operands() <= index:
            return ()

        def refuse(reason: str) -> None:
            """Raises with what was read and why padding cannot stand in for it."""
            message = (
                f"Unable to run the predicate on this kernel's {role}, which {reason}. The array has no per "
                f"element predicate, so the only mask a launch reproduces is one bounding each axis from above "
                f"by a uniform value: the host zero fills the trailing run that leaves and slices it off "
                f"coming out. Bound each axis from above, or compute the masked result in a further launch."
            )
            raise ProgramError(message)

        # Skips a store, which names no fill because the launch writes the whole padded tile. A load names one,
        # and the padding writes zeros, so a load asking for anything else computes something else.
        if role != "output" and not self.reads_zero_outside(operation):
            refuse("fills its excluded elements with something other than zero, which the padding cannot write")

        return tuple(self._guard_from(term, placements, variables=variables, refuse=refuse)
                     for term, placements in self._conjuncts(operation.get_operand(index)))

    def reads_zero_outside(self, load) -> bool:
        """Returns True when a masked load fills its excluded elements with zero.

        Notes:
            The MLIR binding returns None for a dense float constant, so every load's fill is read from the
            module's printed form and matched by position. A load with no fill selects every element and reports
            True, and a fill this reader cannot resolve reports False rather than being assumed zero.
        """
        if self._zero_filled is None:
            zeros: set[str] = set()
            flags: list[bool] = []
            for line in self.module.str_nodebug().splitlines():
                constant = _CONSTANT_LINE.match(line)
                if constant is not None:
                    text = (constant.group("dense") or constant.group("scalar") or "").strip()
                    try:
                        if float(text) == 0.0:
                            zeros.add(constant.group("name"))
                    except ValueError:
                        pass
                found = _LOAD_LINE.search(line)
                if found is not None:
                    operands = [operand.strip() for operand in found.group("operands").split(",")]
                    flags.append(len(operands) < 3 or operands[2] in zeros)
            self._zero_filled = flags
        position = self.by_name("tt.load").index(load)
        return self._zero_filled[position] if position < len(self._zero_filled) else False

    def resolve(self, value) -> _Expression:
        """Reads a value as a base pointer plus a per axis affine offset, memoized."""
        key = value.id()
        if key in self._resolved:
            return self._resolved[key]

        if key in self._arguments:
            resolved = _Expression(base=self._arguments[key], scalar=self._arguments[key])
        elif key in self._carried:
            resolved = self.resolve(self._carried[key])
        else:
            operation = self._definitions.get(key)
            handler = self._resolvers.get(operation.get_name()) if operation is not None else None
            resolved = handler(operation) if handler is not None else _Expression()

        self._resolved[key] = resolved
        return resolved

    def accumulator_elements(self) -> tuple[str, ...]:
        """Returns the element type each matmul accumulates in, in program order.

        Notes:
            The MLIR type binding exposes only fp16 and integer predicates, so the dot result types are read from
            the module's printed form.
        """
        elements = []
        for line in self.module.str_nodebug().splitlines():
            if "tt.dot" in line:
                found = re.search(r"->\s*tensor<[0-9x]+x(\w+)>", line)
                elements.append(found.group(1) if found else _ASSUMED_ACCUMULATOR_ELEMENT)
        return tuple(elements)

    def _conjuncts(self, value) -> list:
        """Returns every comparison a predicate is the conjunction of, with what placed each one in the tile.

        Notes:
            A mask is compared at rank one and only then expanded into the operand's shape, so the expand_dims
            axes crossed on the way down to it are carried out with it.

        Args:
            value: The predicate operand of a load or store.

        Returns:
            One comparison and placement pair per term, the comparison being None for a term that is not one and
            the placements running outermost first.
        """
        found, pending = [], [(value, ())]
        while pending:
            current, placements = pending.pop()
            operation = self._definitions.get(current.id())
            if operation is None:
                found.append((None, placements))
                continue
            name = operation.get_name()
            if name == "tt.expand_dims":
                placements = (*placements, operation.get_int_attr("axis"))
                pending.append((operation.get_operand(0), placements))
            elif name in _PREDICATE_RESHAPES:
                # Traces through broadcast and widen, which reshape without moving which axis an element varies along.
                pending.append((operation.get_operand(0), placements))
            elif name == "arith.andi":
                pending.extend((operation.get_operand(side), placements) for side in range(2))
            else:
                found.append((operation, placements))
        return found

    def _guard_from(self, comparison, placements: tuple[int, ...], *, variables: tuple[str, ...],
                    refuse: Callable[[str], None]) -> Guard:
        """Reads one term of a predicate as the axis it bounds and the value it bounds it to.

        Args:
            comparison: The comparison forming this term, or None when the term is not one.
            placements: The expand_dims axes crossed on the way down to the comparison, outermost first.
            variables: Induction variable each tensor axis of this operand carries.
            refuse: Raises with the reason padding cannot stand in for the predicate.

        Returns:
            The guard this term states.
        """
        if comparison is None or comparison.get_name() != "arith.cmpi":
            found = "none" if comparison is None else comparison.get_name()
            refuse(f"joins a term built by {found} rather than an integer comparison")

        ordering = _ORDERING_PREDICATES.get(comparison.get_int_attr("predicate"))
        if ordering is None:
            refuse("compares an axis for equality, which excludes elements that are not one run")
        less, inclusive = ordering

        sides = [self.resolve(comparison.get_operand(side)) for side in range(2)]
        counts = [len(side.axes) for side in sides]
        if sorted(counts) != [0, 1]:
            refuse(f"compares {counts[0]} axes against {counts[1]}, rather than one against none")
        iota_left = counts[0] == 1
        walked, uniform = sides if iota_left else sides[::-1]
        # Reads an axis left of a "less" and right of a "greater" as the same upper bound. The other two
        # orientations bound from below, which excludes a leading run the padding does not hold.
        if less != iota_left:
            refuse("bounds an axis from below, so the elements it excludes lead rather than trail")

        axis, stride = next(iter(walked.axes.items()))
        if stride != 1:
            refuse(f"walks its axis by {stride} rather than one element at a time")
        # Applies the expand_dims innermost first, as `_resolve_expand_dims` does for an address, since the
        # comparison was made at rank one.
        for inserted in reversed(placements):
            axis = (1 - inserted) if axis == _PENDING_AXIS else (axis if axis < inserted else axis + 1)
        if not 0 <= axis < len(variables):
            refuse(f"bounds tensor axis {axis}, which is not one of the {len(variables)} this operand carries")
        if uniform.scalar is None:
            refuse("compares against a value this reader cannot resolve to a parameter or a constant")

        bound = uniform.scalar
        if inclusive:
            if not isinstance(bound, int):
                refuse(f"bounds {variables[axis]} inclusively by {bound}, whose successor is not known here")
            bound += 1
        return Guard(axis=variables[axis], bound=bound)

    def _load_behind(self, value) -> tuple[object | None, bool]:
        """Returns the load producing a matmul operand, and whether a transpose sits in between.

        Args:
            value: The matmul operand to walk back from.

        Returns:
            The load, or None where the paths meet at no load or at more than one, and whether a transpose was
            crossed reaching it.
        """
        reached: dict[tuple[int, bool], tuple[object, bool]] = {}
        seen: set[tuple[int, bool]] = set()
        pending = [(value, False)]

        while pending:
            current, transposed = pending.pop()
            key = (current.id(), transposed)
            if key in seen:
                continue
            seen.add(key)
            operation = self._definitions.get(current.id())
            if operation is None:
                continue

            name = operation.get_name()
            if name == "tt.load":
                reached[(id(operation), transposed)] = (operation, transposed)
                continue
            if name == "tt.trans":
                pending.append((operation.get_operand(0), not transposed))
                continue
            if name in _POINTWISE_TRANSFORMS:
                pending.extend((operation.get_operand(index), transposed)
                               for index in range(operation.get_num_operands()))

        if len(reached) != 1:
            return None, False
        return next(iter(reached.values()))

    def _resolve_through(self, operation) -> _Expression:
        """Reads an operation that reshapes or widens without changing the expression."""
        return self.resolve(operation.get_operand(0))

    def _resolve_splat(self, operation) -> _Expression:
        """Reads a scalar broadcast, which carries the uniform part and no iota."""
        inner = self.resolve(operation.get_operand(0))
        return _Expression(base=inner.base, scalar=inner.scalar)

    def _resolve_constant(self, operation) -> _Expression:
        """Reads a literal."""
        return _Expression(scalar=self.constant(operation.get_result(0)))

    @staticmethod
    def _resolve_range(operation) -> _Expression:  # noqa: ARG004
        """Reads an iota as a unit stride waiting for an expand_dims to place it."""
        return _Expression(axes={_PENDING_AXIS: 1})

    def _resolve_expand_dims(self, operation) -> _Expression:
        """Places a pending iota on the tensor axis an expand_dims operation leaves varying."""
        inner = self.resolve(operation.get_operand(0))
        inserted = operation.get_int_attr("axis")

        def placed(axis: int) -> int:
            """Returns the tensor axis an inner axis moves to once the new one is inserted."""
            return (1 - inserted) if axis == _PENDING_AXIS else (axis if axis < inserted else axis + 1)

        return _Expression(base=inner.base, scalar=inner.scalar,
                           axes={placed(axis): stride for axis, stride in inner.axes.items()})

    def _resolve_multiply(self, operation) -> _Expression:
        """Scales the strides of the varying side of a multiply by the uniform side."""
        left = self.resolve(operation.get_operand(0))
        right = self.resolve(operation.get_operand(1))
        varying, uniform = (left, right) if left.axes else (right, left)
        if varying.axes and uniform.axes:
            return _Expression(scalar=None)
        return _Expression(
            base=varying.base or uniform.base,
            axes={axis: uniform.scalar if stride == 1 else None for axis, stride in varying.axes.items()},
        )

    def _resolve_add(self, operation) -> _Expression:
        """Merges the axes and uniform parts of both sides of an addition."""
        left = self.resolve(operation.get_operand(0))
        right = self.resolve(operation.get_operand(1))
        return _Expression(
            base=left.base or right.base,
            axes={**left.axes, **right.axes},
            scalar=left.scalar if right.scalar is None else right.scalar,
        )


def signature_of(kernel: JITFunction, *, operands: dict[str, str]) -> dict[str, str]:
    """Returns the signature a kernel is compiled against, in the order the kernel declares its parameters.

    Args:
        kernel: The function the triton.jit decorator returned.
        operands: Triton type string for each parameter a buffer is passed for, keyed by parameter name, which
            overrides the annotation that parameter carries.

    Returns:
        The signature.

    Raises:
        ProgramError: If a named operand is not a runtime parameter of the kernel.
    """
    declared = {parameter.name: parameter.is_constexpr for parameter in kernel.params}
    unknown = {name for name in operands if declared.get(name, True)}
    if unknown:
        message = (
            f"Unable to type {', '.join(sorted(unknown))} as an operand of {kernel.__name__}. Every named operand "
            f"must be one of its runtime parameters, which are "
            f"{', '.join(name for name, constexpr in declared.items() if not constexpr)}."
        )
        raise ProgramError(message)
    return {parameter.name: "constexpr" if parameter.is_constexpr else
            operands.get(parameter.name, _annotated(parameter.annotation) or "i32")
            for parameter in kernel.params}


def read_program(kernel: JITFunction, *, signature: dict[str, str],
                 constexprs: dict[str, int]) -> tuple[MatMul, ...]:
    """Reads every matmul a kernel computes, in program order.

    Args:
        kernel: The function the triton.jit decorator returned.
        signature: Parameter name to Triton type string, in declaration order.
        constexprs: Values of the constexpr parameters, which fix the tiles.

    Returns:
        The matmuls.

    Raises:
        ProgramError: If the grid carries independent products, a matmul sits outside every loop, or its loops
            leave the reduction ambiguous.
    """
    graph = _Graph(compile_to_ttir(kernel, signature=signature, constexprs=constexprs), signature=signature)
    axes = {operation.get_int_attr("axis") for operation in graph.by_name("tt.get_program_id")}
    if axes - {0, 1}:
        message = (
            f"Unable to read a grid using axes {sorted(axes)}. Two axes carry the output tile and a third carries "
            f"independent products, whose launch needs the distance the activation pointer moves between them. "
            f"Recovering that distance needs the operand's strides, which this reader does not follow."
        )
        raise ProgramError(message)

    dots = graph.by_name("tt.dot")
    stores = graph.by_name("tt.store")
    consumers = [*dots[1:], *stores]
    accumulators = graph.accumulator_elements()

    matmuls = []
    for index, dot in enumerate(dots):
        loops = graph.enclosing_loops(dot)
        if not loops:
            message = (
                f"Unable to read matmul {index} from no enclosing loop. A dot walks its reduction in a loop, "
                f"because the group that loop accumulates is what the builder's block count comes from."
            )
            raise ProgramError(message)

        rows, columns = dot.get_result(0).get_shape()
        steps = [graph.constant(loop.get_operand(2)) for loop in loops]
        if any(step is None for step in steps):
            message = f"Unable to read the step of matmul {index}'s loops. Each must be a compile time constant."
            raise ProgramError(message)

        pointers = {role: (None if operation is None else operation.get_operand(0))
                    for role, (operation, _) in (("activation", graph._load_behind(dot.get_operand(0))),
                                                 ("weight", graph._load_behind(dot.get_operand(1))))}
        source, prologue, prologue_chain = graph.trace(dot.get_operand(0), among=dots)
        at = _reduction_position(index=index, graph=graph, loops=loops, steps=steps, pointers=pointers,
                                 rows=rows, columns=columns, source=source)
        consumer = consumers[index] if index < len(consumers) else None
        epilogue, epilogue_chain = (frozenset(), ()) if consumer is None else graph.trace(
            consumer.get_operand(1 if consumer.get_name() == "tt.store" else 0), among=dots)[1:]

        def walked(among) -> tuple[TileLoop, ...]:
            """Reads each of the given loops as a tile loop, dropping any whose bounds are runtime values."""
            read = (graph.tile_loop(other, axis=graph.axis_of_loop(other, pointers=pointers)) for other in among)
            return tuple(loop for loop in read if loop is not None)

        # A dot contracting its whole depth in one operation has no reduction loop, so the contraction is the
        # extent its operands meet at and every enclosing loop walks an output axis.
        contraction = dot.get_operand(0).get_shape()[1]
        matmuls.append(MatMul(
            rows=rows, columns=columns, depth=steps[at] if at is not None else contraction,
            accumulator=accumulators[index] if index < len(accumulators) else _ASSUMED_ACCUMULATOR_ELEMENT,
            epilogue=epilogue,
            prologue=prologue,
            left_source=dots.index(source) if source is not None else None,
            reduction=graph.constant(loops[at].get_operand(1)) if at is not None else contraction,
            tile_loops=walked(loops if at is None else [*loops[:at], *loops[at + 1:]]),
            nest=_nest_of(graph=graph, loops=loops, at=at, pointers=pointers),
            accesses=graph.accesses_of(dot=dot, store=stores[index] if index < len(stores) else None),
            # A store the reduction encloses sits deeper than the reduction itself, which is the
            # accumulator the kernel keeps in memory rather than in registers.
            stores_inside_reduction=(at is not None and index < len(stores)
                                     and len(graph.enclosing_loops(stores[index])) > at),
            grid_axes=_grid_axes(graph=graph, pointers=pointers),
            carried=graph.carried_values(loops[at], among=dots) if at is not None else (),
            epilogue_chain=epilogue_chain, prologue_chain=prologue_chain,
            accumulator_chain=graph.trace(dot.get_operand(2), among=dots)[2],
        ))
    return tuple(matmuls)


def _nest_of(*, graph: _Graph, loops: list, at: int | None, pointers: dict[str, object]) -> tuple[str, ...]:
    """Returns the axis each enclosing loop walks, outermost first, with the contraction among them.

    A dot whose contraction is unlooped runs it inside every loop it sits in, so the reduction names the
    innermost position rather than one of them.

    Args:
        graph: The module being read.
        loops: The loops enclosing the matmul, outermost first.
        at: Position of the reduction loop, or None where the contraction is unlooped.
        pointers: The activation's and the weight's pointer values, keyed by role.

    Returns:
        The axes, holding the reduction axis once.
    """
    walked = tuple(_REDUCTION_AXIS if position == at else graph.axis_of_loop(loop, pointers=pointers)
                   for position, loop in enumerate(loops)
                   if position == at or graph.axis_of_loop(loop, pointers=pointers) is not None)
    return walked if at is not None else (*walked, _REDUCTION_AXIS)


def _grid_axes(*, graph: _Graph, pointers: dict[str, object]) -> frozenset[str]:
    """Returns the output axes a kernel's grid walks, being the ones its program ids address an operand along.

    An identifier reaching neither operand or both names no axis of its own, so it stands for either of them
    rather than for none: an axis the plan cannot rule out being on the grid is one it cannot read a single trip
    loop from either.

    Args:
        graph: The module being read.
        pointers: The activation's and the weight's pointer values, keyed by role.

    Returns:
        The axes the grid walks, empty for a kernel launching on one program.
    """
    axes: set[str] = set()
    for operation in graph.by_name("tt.get_program_id"):
        axis = graph.axis_of_grid(operation, pointers=pointers)
        axes.update({"m", "n"} if axis is None else {axis})
    return frozenset(axes)


def _reduction_position(*, index: int, graph: _Graph, loops: list, steps: list[int],
                        pointers: dict[str, object], rows: int, columns: int, source=None) -> int | None:
    """Returns the position of the loop walking one matmul's contraction.

    Notes:
        A tile loop's variable addresses one operand alone, so the contraction is the loop naming none, and two
        the addresses leave unnamed are told apart by their step.

        A matmul reading another's product has no addressable left operand. Its producer fills one step of the
        contraction and so runs inside it, which names the innermost loop enclosing both.

    Args:
        index: Position of the matmul in the kernel.
        graph: The indexed module the loops belong to.
        loops: The loops enclosing the matmul, outermost first.
        steps: The step each of those loops takes.
        pointers: The activation's and the weight's pointer values, keyed by role.
        rows: Token rows the matmul's tile covers.
        columns: Output columns the matmul's tile covers.
        source: The matmul producing this one's left operand, or None when it comes from memory.

    Returns:
        The position of the reduction loop, or None where the contraction runs in one dot and no loop walks it.

    Raises:
        ProgramError: If two loops walk neither operand, or if a matmul reading another's product shares no loop
            with it.
    """
    if source is not None:
        producing = graph.enclosing_loops(source)
        shared = [position for position, loop in enumerate(loops) if loop in producing]
        if shared:
            return shared[-1]
        message = (
            f"Unable to read matmul {index}'s reduction, which it shares no loop with the matmul producing its "
            f"left operand. That product is one step of this contraction, so the matmul leaving it runs inside "
            f"the loop this one contracts over. A matmul reading a whole product at once contracts in a single "
            f"step, which is the reduction loop folded away rather than a loop this reader can name."
        )
        raise ProgramError(message)

    walking = [position for position, loop in enumerate(loops)
               if graph.axis_of_loop(loop, pointers=pointers) is None]
    if len(walking) > 1:
        # Holds a tile loop advancing its operand from the body it encloses, whose own variable reaches neither
        # pointer, to the step its tile gives it.
        walking = [position for position in walking if steps[position] not in (rows, columns)]
    if len(walking) == 1:
        return walking[0]

    if not walking:
        # Every enclosing loop walks an output axis, so the dot contracts its whole depth in one operation and
        # the reduction is its operands' own extent rather than a loop this reader can name.
        return None
    message = (
        f"Unable to tell which of matmul {index}'s loops walks the reduction: {len(walking)} of them address "
        f"neither operand alone and step by neither the tile's {rows} rows nor its {columns} columns "
        f"({', '.join(str(steps[position]) for position in walking)}). One loop may walk the contraction, because "
        f"a reader that picked one of these would accumulate the wrong axis."
    )
    raise ProgramError(message)


def _annotated(annotation: str) -> str:
    """Returns the Triton type one parameter's annotation states, unquoted.

    Args:
        annotation: The annotation the kernel declared, which `from __future__ import annotations` hands over as
            the source text and so with its quotes still on.

    Returns:
        The type, or the empty string for a parameter carrying no annotation.
    """
    text = annotation.strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text
