/**
 * @file
 *
 * @brief Implements the VMEM and register allocators, which name the row of every buffer and the register of every
 * value a kernel leaves unpinned.
 */

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/EquivalenceClasses.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Matchers.h>
#include <algorithm>
#include <array>
#include <optional>
#include <vector>
#include "minitpu/minitpu/passes.h"
#include "minitpu/minitpuisa/minitpuisa_schedule.h.inc"

namespace mlir::minitpu
{
#define GEN_PASS_DEF_MINITPUALLOCATEVMEM
#define GEN_PASS_DEF_MINITPUALLOCATEREGISTERS
#include "minitpu/minitpu/passes.h.inc"
}  // namespace mlir::minitpu

using namespace mlir;
using namespace mlir::minitpu;

namespace
{
    /// Stores the vector registers in the file.
    constexpr uint32_t kVectorRegisters = 32;

    /// Stores the scalar registers in the file.
    constexpr uint32_t kScalarRegisters = 4;

    /// Stores the vector registers a weight tile occupies, which vmatload reads from consecutive registers.
    constexpr uint32_t kTileRegisters = 4;

    /// Stores the width of a scalar register's value.
    constexpr uint32_t kScalarBits = 32;

    /// Stores the highest base a weight tile may start at, which keeps its four registers inside the file.
    constexpr uint32_t kHighestTileBase = kVectorRegisters - kTileRegisters;

    /// Stores one VMEM region's rows, as a half-open range.
    struct RowRange
    {
        uint32_t first = 0;  ///< The first row.
        uint32_t end   = 0;  ///< The row past the last.
    };

    /// Provides the minitpu-allocate-vmem pass over one kernel.
    class AllocateVmemPass final : public mlir::minitpu::impl::MinitpuAllocateVmemBase<AllocateVmemPass>
    {
      public:
        /// Gives every unpinned buffer the next free rows, refusing a kernel whose buffers pass a slot's rows.
        void runOnOperation() override
        {
            llvm::SmallVector<AllocOp> allocations;
            getOperation().walk([&](AllocOp allocation) {
                allocations.push_back(allocation);
            });

            std::vector<RowRange> taken;
            for (AllocOp allocation : allocations)
            {
                if (!allocation.getBase()) continue;
                const uint32_t first = *allocation.getBase();
                taken.push_back({first, first + allocation.getBuffer().getType().getRows()});
            }

            uint32_t next = 0;
            for (AllocOp allocation : allocations)
            {
                if (allocation.getBase()) continue;
                const uint32_t rows = allocation.getBuffer().getType().getRows();
                bool moved          = true;
                while (moved)
                {
                    moved = false;
                    for (const RowRange& range : taken)
                    {
                        if (next < range.end && range.first < next + rows)
                        {
                            next  = range.end;
                            moved = true;
                        }
                    }
                }
                if (next + rows > static_cast<uint32_t>(isa::schedule::kVmemRowsPerSlot))
                {
                    InFlightDiagnostic diagnostic = allocation.emitOpError()
                                                 << "is unable to take " << rows << " rows at row " << next
                                                 << ". A kernel's buffers must fit one VMEM slot's "
                                                 << isa::schedule::kVmemRowsPerSlot << " rows.";
                    for (AllocOp other : allocations)
                    {
                        diagnostic.attachNote(other.getLoc())
                            << "buffer of " << other.getBuffer().getType().getRows() << " rows";
                    }
                    signalPassFailure();
                    return;
                }
                OpBuilder builder(allocation);
                allocation.setBaseAttr(builder.getI32IntegerAttr(static_cast<int32_t>(next)));
                taken.push_back({next, next + rows});
                next += rows;
            }
        }
    };

    /// Stores one set of values sharing a register, and the positions over which that register is held.
    struct Interval
    {
        llvm::SmallVector<Value, 4> members;  ///< The values sharing the register.
        int64_t start = 0;                    ///< The position the first member is defined at.
        int64_t end   = 0;                    ///< The position the last member is last read at.

        /// Stores each member's own span, from its definition to its last read. A loop-carried set holds its register
        /// only over these, so a value living between the carried value's last read and the next value's definition
        /// may share it.
        llvm::SmallVector<std::pair<int64_t, int64_t>, 4> spans;
        bool is_vector = false;            ///< Determines whether the register is a vector register.
        std::optional<uint32_t> pinned;    ///< The register a member pins, if any.
        std::optional<uint32_t> assigned;  ///< The register the allocator names.
        Operation* anchor = nullptr;       ///< The operation defining the first member, for diagnostics.
    };

    /// Stores one allocation unit: a single interval, or the four intervals of a weight tile in consecutive registers.
    struct Unit
    {
        llvm::SmallVector<size_t, kTileRegisters> intervals;  ///< The intervals, in register order.
        int64_t start = 0;                                    ///< The earliest start among the intervals.
        int64_t end   = 0;                                    ///< The latest end among the intervals.
    };

    /// Provides the register allocation of one kernel.
    class RegisterAllocation
    {
      public:
        /// Prepares to allocate the registers of `kernel`.
        explicit RegisterAllocation(KernelOp kernel) : _kernel(kernel)
        {}

        /// Allocates every unpinned value's register, failing with a diagnostic on the first conflict.
        LogicalResult Run()
        {
            SinkArguments();
            if (failed(IsolateLoopCarriedValues())) return failure();
            Number(_kernel.getBody().front());
            if (failed(GroupValues()) || failed(GroupTiles())) return failure();
            // Stacks the lone values from the top of the file when spreading them leaves no run of four for a weight
            // tile, which a block holding many values across a loop does.
            if (!Assign(true, false, false) && !Assign(true, true, true)) return failure();
            if (!Assign(false, false, true)) return failure();
            Record();
            return success();
        }

      private:
        /// Determines whether an i32 constant must live in a register, rather than ride an addition as its immediate.
        static bool NeedsRegister(arith::ConstantOp constant)
        {
            for (OpOperand& use : constant->getUses())
            {
                auto addition = dyn_cast<arith::AddIOp>(use.getOwner());
                if (!addition) return true;

                // Matches the lowering, which takes the right operand as the immediate whenever it is a constant.
                const bool is_left = use.getOperandNumber() == 0;
                if (is_left && matchPattern(addition.getRhs(), m_Constant())) return true;
            }
            return false;
        }

        /// Determines whether `value` lives in a register of its own.
        static bool NeedsRegister(Value value)
        {
            const Type type = value.getType();
            if (!llvm::isa<VRegType>(type) && !type.isInteger(kScalarBits)) return false;
            Operation* definition = value.getDefiningOp();
            if (definition == nullptr || llvm::isa<scf::ForOp>(definition)) return false;
            if (auto constant = dyn_cast<arith::ConstantOp>(definition)) return NeedsRegister(constant);
            return true;
        }

        /**
         * @brief Moves every argument read to just before the operation that first uses it.
         *
         * A launch argument holds one value for the whole launch, so reading it late changes nothing but the scalar
         * register it holds, which an argument first used late would otherwise hold from the kernel's start.
         */
        void SinkArguments()
        {
            Block& body = _kernel.getBody().front();
            llvm::SmallVector<ArgOp> arguments;
            for (ArgOp argument : body.getOps<ArgOp>()) arguments.push_back(argument);
            for (ArgOp argument : arguments)
            {
                Operation* first = nullptr;
                for (Operation* user : argument->getUsers())
                {
                    Operation* top = body.findAncestorOpInBlock(*user);
                    if (top != nullptr && (first == nullptr || top->isBeforeInBlock(first))) first = top;
                }
                if (first != nullptr) argument->moveBefore(first);
            }
        }

        /**
         * @brief Copies every loop's initial value that is also read elsewhere, so the loop may take its register.
         *
         * A vector value is copied with vmov and a scalar one with an addition of zero. A loop that reads its
         * carried value after computing the next one would lose it to the shared register, so it is refused.
         */
        LogicalResult IsolateLoopCarriedValues()
        {
            llvm::SmallVector<scf::ForOp> loops;
            _kernel.walk([&](scf::ForOp loop) {
                loops.push_back(loop);
            });
            for (scf::ForOp loop : loops)
            {
                auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
                for (size_t index = 0; index < loop.getInitArgs().size(); ++index)
                {
                    OpOperand& entering = loop.getInitArgsMutable()[index];
                    if (!entering.get().hasOneUse() || entering.get().getDefiningOp() == nullptr)
                    {
                        OpBuilder builder(loop);
                        Value copy;
                        if (llvm::isa<VRegType>(entering.get().getType()))
                        {
                            copy = VmovOp::create(builder, loop.getLoc(), entering.get().getType(), entering.get());
                        }
                        else
                        {
                            Value zero = arith::ConstantIntOp::create(builder, loop.getLoc(), int64_t {0}, kScalarBits);
                            copy       = arith::AddIOp::create(builder, loop.getLoc(), entering.get(), zero);
                        }
                        entering.set(copy);
                    }

                    const BlockArgument carried = loop.getRegionIterArgs()[index];
                    Operation* next             = yield.getOperand(index).getDefiningOp();
                    if (next == nullptr || next->getBlock() != loop.getBody()) continue;
                    for (Operation* reader : carried.getUsers())
                    {
                        Operation* top = loop.getBody()->findAncestorOpInBlock(*reader);
                        if (top != nullptr && next->isBeforeInBlock(top))
                        {
                            return reader->emitOpError(
                                "is unable to read a loop-carried value after the loop computes its next value. "
                                "The two share one register, so read the value before computing the next one."
                            );
                        }
                    }
                }
            }
            return success();
        }

        /// Numbers every operation in program order, a loop taking one position before its body and one after.
        // NOLINTNEXTLINE(misc-no-recursion): recurses once per nested loop, at most the AGU's four levels deep.
        void Number(Block& block)
        {
            for (Operation& operation : block)
            {
                _position[&operation] = _next_position++;
                if (auto loop = dyn_cast<scf::ForOp>(operation))
                {
                    Number(*loop.getBody());
                    _loop_end[loop] = _next_position++;
                }
            }
        }

        /// Returns the position a value is defined at.
        int64_t DefinitionOf(Value value)
        {
            if (auto argument = dyn_cast<BlockArgument>(value))
            {
                return _position[argument.getOwner()->getParentOp()];
            }
            if (auto loop = dyn_cast<scf::ForOp>(value.getDefiningOp())) return _loop_end[loop];
            return _position[value.getDefiningOp()];
        }

        /// Returns the last position a value is read at, held through every loop it is read in but defined outside.
        int64_t LastReadOf(Value value)
        {
            int64_t last       = DefinitionOf(value);
            Region* defined_in = value.getParentRegion();
            for (Operation* reader : value.getUsers())
            {
                int64_t read = _position[reader];
                for (auto loop = reader->getParentOfType<scf::ForOp>(); loop;
                     loop      = loop->getParentOfType<scf::ForOp>())
                {
                    if (!loop.getRegion().isAncestor(defined_in)) read = std::max(read, _loop_end[loop]);
                }
                last = std::max(last, read);
            }
            return last;
        }

        /// Groups every value that must share a register, being a loop's carried values, and builds their intervals.
        LogicalResult GroupValues()
        {
            llvm::EquivalenceClasses<Value> classes;
            llvm::SmallVector<Value> values;
            _kernel.walk([&](Operation* operation) {
                for (Value result : operation->getResults())
                {
                    if (NeedsRegister(result)) values.push_back(result);
                }
            });
            for (Value value : values) classes.insert(value);
            _kernel.walk([&](scf::ForOp loop) {
                auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
                for (size_t index = 0; index < loop.getInitArgs().size(); ++index)
                {
                    const Value entering = loop.getInitArgs()[index];
                    for (const Value member :
                         {Value(loop.getRegionIterArgs()[index]), yield.getOperand(index), Value(loop.getResult(index))
                         })
                    {
                        classes.unionSets(entering, member);
                    }
                }
            });

            for (auto leader = classes.begin(); leader != classes.end(); ++leader)
            {
                if (!(*leader)->isLeader()) continue;
                Interval interval;
                interval.start = INT64_MAX;
                interval.end   = INT64_MIN;
                for (auto member = classes.member_begin(**leader); member != classes.member_end(); ++member)
                {
                    Value value = *member;
                    interval.members.push_back(value);
                    interval.is_vector    = llvm::isa<VRegType>(value.getType());
                    const int64_t defined = DefinitionOf(value);
                    if (defined < interval.start)
                    {
                        interval.start  = defined;
                        interval.anchor = value.getDefiningOp() != nullptr ? value.getDefiningOp()
                                                                           : value.getParentRegion()->getParentOp();
                    }
                    interval.end = std::max(interval.end, LastReadOf(value));
                    interval.spans.emplace_back(defined, LastReadOf(value));
                    Operation* definition = value.getDefiningOp();
                    if (definition == nullptr) continue;
                    const auto pin =
                        definition->getAttrOfType<IntegerAttr>(MinitpuDialect::get_register_attribute_name());
                    if (!pin) continue;
                    const auto pinned = static_cast<uint32_t>(pin.getInt());
                    if (interval.pinned && *interval.pinned != pinned)
                    {
                        return definition->emitOpError()
                            << "is unable to live in register " << pinned << ". It shares a register with a "
                            << "loop-carried value another operation pins to register " << *interval.pinned << ".";
                    }
                    interval.pinned = pinned;
                }
                for (const Value member : interval.members) _interval_of[member] = _intervals.size();
                _intervals.push_back(std::move(interval));
            }
            return success();
        }

        /// Builds the allocation units, joining each vmatload's four operands into one tile of consecutive registers.
        LogicalResult GroupTiles()
        {
            llvm::DenseMap<size_t, std::pair<size_t, uint32_t>> tile_of;
            std::vector<Unit> tiles;
            const auto status = _kernel.walk([&](VmatloadOp load) -> WalkResult {
                const Value quarters[kTileRegisters] =
                    {load.getQuarter0(), load.getQuarter1(), load.getQuarter2(), load.getQuarter3()};
                Unit tile;
                for (uint32_t quarter = 0; quarter < kTileRegisters; ++quarter)
                {
                    const size_t interval = _interval_of.lookup(quarters[quarter]);
                    const auto existing   = tile_of.find(interval);
                    if (existing != tile_of.end() &&
                        (existing->second.first != tiles.size() || existing->second.second != quarter))
                    {
                        load.emitOpError() << "is unable to read quarter " << quarter << " from a value another "
                                           << "vmatload reads at a different position. Each value must sit at "
                                           << "one position of one weight tile.";
                        return WalkResult::interrupt();
                    }
                    tile_of[interval] = {tiles.size(), quarter};
                    tile.intervals.push_back(interval);
                }
                tiles.push_back(std::move(tile));
                return WalkResult::advance();
            });
            if (status.wasInterrupted()) return failure();

            for (Unit& tile : tiles) _units.push_back(std::move(tile));
            for (size_t interval = 0; interval < _intervals.size(); ++interval)
            {
                if (!tile_of.contains(interval)) _units.push_back({{interval}});
            }
            for (Unit& unit : _units)
            {
                unit.start = INT64_MAX;
                unit.end   = INT64_MIN;
                for (const size_t interval : unit.intervals)
                {
                    unit.start = std::min(unit.start, _intervals[interval].start);
                    unit.end   = std::max(unit.end, _intervals[interval].end);
                }
            }
            std::stable_sort(_units.begin(), _units.end(), [](const Unit& lhs, const Unit& rhs) {
                return lhs.start < rhs.start;
            });
            return success();
        }

        /// Determines whether two intervals hold their register at overlapping positions.
        static bool Overlap(const Interval& lhs, const Interval& rhs)
        {
            // Lets a value take the register of one last read by the operation defining it, which reads before
            // it writes.
            for (const auto& [left_start, left_end] : lhs.spans)
            {
                for (const auto& [right_start, right_end] : rhs.spans)
                {
                    if (left_start < right_end && right_start < left_end) return true;
                }
            }
            return false;
        }

        /// Determines whether `interval` may take `candidate` alongside every interval already assigned.
        [[nodiscard]]
        bool IsFree(const Interval& interval, const uint32_t candidate) const
        {
            for (const Interval& other : _intervals)
            {
                if (&other == &interval || other.is_vector != interval.is_vector) continue;
                const std::optional<uint32_t> held = other.assigned ? other.assigned : other.pinned;
                if (held && *held == candidate && Overlap(interval, other)) return false;
            }
            return true;
        }

        /// Returns the register `unit`'s first interval takes, or nullopt when no base leaves every interval free. A
        /// stacked lone value takes the highest free register.
        [[nodiscard]]
        std::optional<uint32_t> ChooseBase(const Unit& unit, const uint32_t registers, const bool stacked) const
        {
            const uint32_t span = unit.intervals.size();
            const uint32_t last = span == 1 ? registers - 1 : kHighestTileBase;
            for (uint32_t offset = 0; offset < span; ++offset)
            {
                const Interval& member = _intervals[unit.intervals[offset]];
                if (member.pinned)
                    return *member.pinned >= offset ? std::optional(*member.pinned - offset) : std::nullopt;
            }
            // Gives a lone value the free register idle the longest, so consecutive values land in different registers
            // and the scheduler finds no false dependence between them. A weight tile takes the lowest free run.
            std::optional<uint32_t> chosen;
            for (uint32_t base = 0; base <= last; ++base)
            {
                bool free = true;
                for (uint32_t offset = 0; offset < span && free; ++offset)
                {
                    free = IsFree(_intervals[unit.intervals[offset]], base + offset);
                }
                if (!free) continue;
                if (span > 1) return base;
                if (!chosen || stacked || _released[base] < _released[*chosen]) chosen = base;
            }
            return chosen;
        }

        /// Returns how many of the file's intervals hold a register at `position`.
        [[nodiscard]]
        uint32_t LiveAt(const int64_t position, const bool is_vector) const
        {
            uint32_t live = 0;
            for (const Interval& interval : _intervals)
            {
                if (interval.is_vector != is_vector) continue;
                live += llvm::any_of(interval.spans, [position](const std::pair<int64_t, int64_t>& span) {
                    return span.first <= position && position < span.second;
                });
            }
            return live;
        }

        /**
         * @brief Assigns a register to every unit of one file, pinned units holding their register.
         *
         * @param is_vector determines whether the vector file is assigned rather than the scalar one.
         * @param stacked determines whether a lone value takes the highest free register rather than the one idle
         * the longest.
         * @param report determines whether a unit left without a register is reported as a diagnostic.
         * @returns whether every unit got a register, every assignment of the file cleared when one did not.
         */
        bool Assign(const bool is_vector, const bool stacked, const bool report)
        {
            const uint32_t registers = is_vector ? kVectorRegisters : kScalarRegisters;
            _released.fill(INT64_MIN);
            for (const Unit& unit : _units)
            {
                if (_intervals[unit.intervals.front()].is_vector != is_vector) continue;
                const std::optional<uint32_t> base = ChooseBase(unit, registers, stacked);
                const Interval& first              = _intervals[unit.intervals.front()];
                bool fits                          = base.has_value() && *base + unit.intervals.size() <= registers;
                for (uint32_t offset = 0; fits && offset < unit.intervals.size(); ++offset)
                {
                    fits = IsFree(_intervals[unit.intervals[offset]], *base + offset);
                }
                if (!fits)
                {
                    if (report)
                    {
                        first.anchor->emitOpError()
                            << "is unable to get a " << (is_vector ? "vector" : "scalar") << " register. "
                            << LiveAt(unit.start, is_vector) << " values are live here, where the file holds "
                            << registers << (unit.intervals.size() > 1 ? ", and a weight tile needs 4 in a row." : ".");
                    }
                    for (Interval& interval : _intervals)
                    {
                        if (interval.is_vector == is_vector) interval.assigned.reset();
                    }
                    return false;
                }
                for (uint32_t offset = 0; offset < unit.intervals.size(); ++offset)
                {
                    Interval& interval        = _intervals[unit.intervals[offset]];
                    interval.assigned         = *base + offset;
                    _released[*base + offset] = std::max(_released[*base + offset], interval.end);
                }
            }
            return true;
        }

        /// Writes every assigned register onto the operations defining the values that hold it.
        void Record()
        {
            OpBuilder builder(_kernel.getContext());
            for (const Interval& interval : _intervals)
            {
                for (const Value member : interval.members)
                {
                    Operation* definition = member.getDefiningOp();
                    if (definition == nullptr || llvm::isa<scf::ForOp>(definition)) continue;
                    definition->setAttr(
                        MinitpuDialect::get_register_attribute_name(),
                        builder.getI32IntegerAttr(static_cast<int32_t>(*interval.assigned))
                    );
                }
            }
        }

        /// Stores the kernel being allocated.
        KernelOp _kernel;

        /// Stores each operation's position in program order.
        llvm::DenseMap<Operation*, int64_t> _position;

        /// Stores the position each loop ends at.
        llvm::DenseMap<Operation*, int64_t> _loop_end;

        /// Stores the next position to hand out.
        int64_t _next_position = 0;

        /// Stores every interval, one per set of values sharing a register.
        std::vector<Interval> _intervals;

        /// Stores the interval each value belongs to.
        llvm::DenseMap<Value, size_t> _interval_of;

        /// Stores the allocation units, in the order their registers are chosen.
        std::vector<Unit> _units;

        /// Stores the last position each register of the file being assigned is held to, for choosing the idlest.
        std::array<int64_t, kVectorRegisters> _released {};
    };

    /// Provides the minitpu-allocate-registers pass over one kernel.
    class AllocateRegistersPass final : public mlir::minitpu::impl::MinitpuAllocateRegistersBase<AllocateRegistersPass>
    {
      public:
        /// Names a register for every value, leaving the kernel's registers unnamed on any failure.
        void runOnOperation() override
        {
            RegisterAllocation allocation(getOperation());
            if (failed(allocation.Run())) signalPassFailure();
        }
    };
}  // namespace
