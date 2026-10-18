"""Provides the register regions and the order one attention trip runs its operations under."""

from __future__ import annotations

from dataclasses import dataclass

from .dataflow import CARRIED, CONSTANT, STREAMED, Trip
from .ir import Operation, ProgramError

COMMIT: str = "minitpu.commit"
"""Name of the copy standing between a carried value's slot and its region, which the search places itself."""

_PRODUCT: str = "tt.dot"
"""Name the module prints a matrix product under."""

_REDUCE: str = "tt.reduce."
"""Prefix the module prints a reduce under, which carries the combiner it folds behind it."""

_SLOT_NAMES: tuple[str, ...] = ("scores", "staged", "result", "partial", "weight", "pushed", "running")
"""Every region the shared span is read under, several of which name one slot under two phases."""

_ACCUMULATOR: str = "accumulator"
"""Region the output tiles hold, which is wider than a slot and so holds whatever is computed from it."""

_STATES: int = 200_000
"""States the search walks before giving up, which bounds a trip whose operations admit no order."""


@dataclass(frozen=True, slots=True)
class Place:
    """One operation, the region its result lands in and the region each value it reads lives in.

    Attributes:
        operation: The operation placed, being a commit where the search moves a carried value itself.
        into: Region the result lands in.
        sources: Region of each value the operation reads, in the order it reads them.
    """

    operation: Operation
    into: str
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Placement:
    """One trip's operations in the order they run, each over the regions it reads and writes.

    Attributes:
        places: The operations in the order they run, the commits the search placed among them.
    """

    places: tuple[Place, ...]


def allocate(trip: Trip, *, registers) -> Placement:
    """Returns the order and the regions one trip runs its operations under.

    A reduce spends the two slots it transposes and folds through, and a product spends the slot it pops into,
    so a value living in one of those cannot cross the operation that spends it. The search therefore settles
    the order and the regions together rather than ordering first and assigning after.

    Args:
        trip: The dataflow one trip runs.
        registers: The register map the block is placed under.

    Returns:
        The placement.

    Raises:
        ProgramError: If the trip admits no order its operations all fit the shared span under.
    """
    layout = _slot_layout(registers)
    regions = _regions(trip)
    wide = frozenset({_ACCUMULATOR} if registers.output_tiles > 1 else ())
    found = _Search(trip, layout=layout, regions=regions, readers=_readers(trip), wide=wide).run()
    if found is not None:
        return Placement(places=found)

    message = (
        f"Unable to order the {len(trip.operations)} operations one trip runs inside the "
        f"{layout['slots']} slots the register file shares between its phases. A reduce spends two of them and "
        f"a product spends one, so a trip holding more values across those than the rest of the span carries is "
        f"one no order fits. Compute part of it in a further launch."
    )
    raise ProgramError(message)


def _slot_layout(registers) -> dict[str, int]:
    """Returns the slot each shared region names, read from the register map rather than restated.

    Args:
        registers: The register map the block is placed under.

    Returns:
        The slot index of each shared region, under "slots" the slots the span holds.

    Raises:
        ProgramError: If a region does not start on a slot boundary, which is a span the phases do not share
            in whole tiles.
    """
    width = registers.per_tile
    found: dict[str, int] = {"slots": registers.transient_span // width}
    for name in _SLOT_NAMES:
        offset = getattr(registers, name) - registers.transient
        if offset % width:
            message = (
                f"Unable to read {name} as a slot of the shared span, which starts {offset} registers into it "
                f"where a slot spans {width}. The search moves whole tiles between slots, so a region opening "
                f"inside one is a span it can place nothing in."
            )
            raise ProgramError(message)
        found[name] = offset // width
    return found


def _regions(trip: Trip) -> dict[int, str]:
    """Returns the region every value with one holds, being the carried values and whatever the accumulator feeds.

    Args:
        trip: The dataflow one trip runs.

    Returns:
        The region of each value the register file holds outside the shared span, keyed by identifier.
    """
    found = {identifier: value.region for identifier, value in trip.values.items()
             if value.role == CARRIED and value.region is not None}
    # A value computed from the accumulator spans the accumulator's tiles, so it is updated where it lies.
    for operation in trip.operations:
        if any(found.get(operand) == _ACCUMULATOR for operand in operation.operands):
            found[operation.result] = _ACCUMULATOR
    return found


def _readers(trip: Trip) -> dict[int, frozenset[int]]:
    """Returns the operations reading each value, keyed by identifier.

    Args:
        trip: The dataflow one trip runs.

    Returns:
        The position of every operation reading the value, keyed by the value's identifier.
    """
    found: dict[int, set[int]] = {}
    for operation in trip.operations:
        for operand in operation.operands:
            found.setdefault(operand, set()).add(operation.position)
    return {identifier: frozenset(positions) for identifier, positions in found.items()}


class _Search:
    """Walks the orders one trip admits, looking for one whose values all fit the slots they need.

    Attributes:
        operations: The operations to place, keyed by position.
        readers: Position of every operation reading each value, keyed by identifier.
        regions: Region of every value held outside the shared span, keyed by identifier.
        wide: Every region spanning more than one slot, which is one nothing stages through a slot.
        staged: Identifier of every value a load places, being a tile from memory or a stated number.
        named: Region naming each slot, keyed by slot.
        spent: Slots each operation writes over as working space, keyed by position.
        leaving: Identifier each region holds leaving one trip, keyed by region.
        slots: Slots the shared span holds.
        walked: States the search has walked.
        refused: Every state from which no order completes.
    """

    def __init__(self, trip: Trip, *, layout: dict[str, int], regions: dict[int, str],
                 readers: dict[int, frozenset[int]], wide: frozenset[str]) -> None:
        self.operations = {one.position: one for one in trip.operations}
        self.readers = readers
        self.regions = regions
        self.wide = wide
        self.staged = frozenset(identifier for identifier, value in trip.values.items()
                                if value.role in (STREAMED, CONSTANT))
        self.named = {layout[name]: name for name in ("scores", "staged", "result", "partial")}
        self.spent = {position: self._spends(one, layout=layout)
                      for position, one in self.operations.items()}
        self.leaving = {region: pair[1] for region, pair in trip.carried.items()}
        self.slots = layout["slots"]
        self.walked = 0
        self.refused: set[tuple] = set()
        self._entering = tuple(sorted((region, pair[0]) for region, pair in trip.carried.items()))

    def run(self) -> tuple[Place, ...] | None:
        """Returns the places one order runs, or None where the trip admits no order."""
        return self._step(frozenset(), (None,) * self.slots, self._entering, ())

    def _step(self, emitted: frozenset[int], occupancy: tuple, held: tuple,
              places: tuple[Place, ...]) -> tuple[Place, ...] | None:
        """Returns the places completing one order from this state, or None where none completes it.

        Args:
            emitted: Position of every operation already placed.
            occupancy: Value each slot holds, None where it holds none.
            held: Value each region outside the shared span holds, as sorted pairs.
            places: The places reaching this state.

        Returns:
            The completed order, or None.
        """
        if len(emitted) == len(self.operations) and self._settled(held):
            return places
        self.walked += 1
        state = (emitted, occupancy, held)
        if self.walked > _STATES or state in self.refused:
            return None

        for place, following, holding in self._moves(emitted, occupancy, held):
            reached = emitted if place.operation.name == COMMIT else emitted | {place.operation.position}
            found = self._step(reached, following, holding, (*places, place))
            if found is not None:
                return found
        self.refused.add(state)
        return None

    def _moves(self, emitted: frozenset[int], occupancy: tuple, held: tuple):
        """Yields every operation this state may place next, with the state placing it reaches.

        Args:
            emitted: Position of every operation already placed.
            occupancy: Value each slot holds, None where it holds none.
            held: Value each region outside the shared span holds, as sorted pairs.

        Yields:
            The place, the slots it leaves behind and the regions it leaves behind.
        """
        for region, holder in held:
            yield from self._commits(region, holder, emitted, occupancy, held)
        for position in sorted(set(self.operations) - emitted):
            yield from self._placings(position, emitted, occupancy, held)

    def _commits(self, region: str, holder: int, emitted: frozenset[int], occupancy: tuple, held: tuple):
        """Yields the copy moving one region's next value out of the slot that staged it.

        Args:
            region: The region to fill.
            holder: The value the region holds.
            emitted: Position of every operation already placed.
            occupancy: Value each slot holds, None where it holds none.
            held: Value each region outside the shared span holds, as sorted pairs.

        Yields:
            The place, the slots it leaves behind and the regions it leaves behind.
        """
        wanted = self.leaving[region]
        if holder == wanted or self._live(holder, emitted):
            return
        staged = next((index for index, one in enumerate(occupancy) if one == wanted), None)
        if staged is None:
            return
        moved = Operation(name=COMMIT, position=-1, result=wanted, operands=(wanted,))
        following = tuple(None if index == staged else one for index, one in enumerate(occupancy))
        yield (Place(operation=moved, into=region, sources=(self.named[staged],)),
               following, self._holding(held, region, wanted))

    def _placings(self, position: int, emitted: frozenset[int], occupancy: tuple, held: tuple):
        """Yields every way one operation may be placed from this state.

        Args:
            position: Position of the operation to place.
            emitted: Position of every operation already placed.
            occupancy: Value each slot holds, None where it holds none.
            held: Value each region outside the shared span holds, as sorted pairs.

        Yields:
            The place, the slots it leaves behind and the regions it leaves behind.
        """
        operation = self.operations[position]
        after = emitted | {position}
        if any(self._live(occupancy[index], after) for index in self.spent[position]):
            return
        for staging in self._stagings(operation, emitted, occupancy):
            standing = tuple(staging.get(index, one) for index, one in enumerate(occupancy))
            sources = self._sources(operation, staging, standing, held)
            if sources is None:
                continue
            for into in self._destinations(operation, staging, standing, held, after):
                cleared = tuple(None if index in self.spent[position] or
                                (one is not None and not self._live(one, after)) else one
                                for index, one in enumerate(standing))
                slot = next((index for index, name in self.named.items() if name == into), None)
                following = tuple(operation.result if index == slot else one
                                  for index, one in enumerate(cleared))
                holding = held if slot is not None else self._holding(held, into, operation.result)
                yield Place(operation=operation, into=into, sources=sources), following, holding

    def _stagings(self, operation: Operation, emitted: frozenset[int], occupancy: tuple):
        """Yields every way the operands a load places may be spread over the free slots.

        Args:
            operation: The operation being placed.
            emitted: Position of every operation already placed.
            occupancy: Value each slot holds, None where it holds none.

        Yields:
            The value each staged slot holds, keyed by slot.
        """
        wanted = [operand for operand in operation.operands if operand in self.staged]
        room = [index for index in range(self.slots)
                if not self._live(occupancy[index], emitted) and index not in self.spent[operation.position]]

        def walk(rest: list[int], taken: tuple) -> object:
            """Spreads the values still wanting a slot over the slots none of them holds yet."""
            if not rest:
                yield dict(taken)
                return
            for index in room:
                if index not in {slot for slot, _ in taken}:
                    yield from walk(rest[1:], (*taken, (index, rest[0])))

        yield from walk(wanted, ())

    def _sources(self, operation: Operation, staging: dict, occupancy: tuple, held: tuple) -> tuple | None:
        """Returns the region each operand lies in, or None where something holds one of them nowhere.

        Args:
            operation: The operation being placed.
            staging: Value each staged slot holds, keyed by slot.
            occupancy: Value each slot holds once the staging has landed.
            held: Value each region outside the shared span holds, as sorted pairs.

        Returns:
            The regions, or None.
        """
        found = []
        for operand in operation.operands:
            where = self._where(operand, occupancy, held)
            if where is None:
                return None
            found.append(where)
        return tuple(found)

    def _destinations(self, operation: Operation, staging: dict, occupancy: tuple, held: tuple,
                      after: frozenset[int]):
        """Yields every region one operation's result may land in.

        A product accumulates onto its third operand, so its result lands wherever that operand lies and
        nowhere else. Everything else lands in the region its result is held in, or in any slot or carried
        region holding nothing a later operation reads.

        Args:
            operation: The operation being placed.
            staging: Value each staged slot holds, keyed by slot.
            occupancy: Value each slot holds once the staging has landed.
            held: Value each region outside the shared span holds, as sorted pairs.
            after: Position of every operation placed once this one is.

        Yields:
            The region.
        """
        if operation.name == _PRODUCT:
            where = self._where(operation.operands[2], occupancy, held)
            if where is not None:
                yield where
            return

        forced = self.regions.get(operation.result)
        if forced is None:
            for region, holder in held:
                if not self._live(holder, after):
                    yield region
        elif not self._live(dict(held)[forced], after):
            yield forced
        if forced in self.wide:
            return
        # A region one slot wide is one a result stages through, the copy filling it placed once the value it
        # holds has been read for the last time.
        for index in range(self.slots):
            if not self._live(occupancy[index], after) and index not in self.spent[operation.position]:
                yield self.named[index]

    def _where(self, value: int, occupancy: tuple, held: tuple) -> str | None:
        """Returns the region one value lies in, or None where nothing holds it."""
        for region, holder in held:
            if holder == value:
                return region
        for index, holder in enumerate(occupancy):
            if holder == value:
                return self.named[index]
        return None

    def _live(self, value: int | None, emitted: frozenset[int]) -> bool:
        """Returns whether an operation the order has not reached yet still reads one value."""
        return value is not None and bool(self.readers.get(value, frozenset()) - emitted)

    def _settled(self, held: tuple) -> bool:
        """Returns whether every region holds the value it hands to the next trip."""
        return all(holder == self.leaving[region] for region, holder in held)

    @staticmethod
    def _holding(held: tuple, region: str, value: int) -> tuple:
        """Returns the regions one write leaves behind."""
        return tuple(sorted((name, value if name == region else holder) for name, holder in held))

    @staticmethod
    def _spends(operation: Operation, *, layout: dict[str, int]) -> frozenset[int]:
        """Returns the slots one operation writes over as working space."""
        if operation.name.startswith(_REDUCE):
            return frozenset({layout["staged"], layout["running"]})
        if operation.name == _PRODUCT:
            return frozenset({layout["result"]})
        return frozenset()
