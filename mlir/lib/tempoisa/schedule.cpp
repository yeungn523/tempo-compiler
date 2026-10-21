/**
 * @file
 *
 * @brief Implements the tempoisa scheduler, which fills every bundle's delay from the latencies tempoisa/isa.td
 * publishes.
 *
 * @note The timeline matches the retired Python assembler's schedule() bundle for bundle, over the traffic traffic.h
 * models, which the recorded streams under test/Inputs/schedule hold it to.
 */

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <array>
#include <deque>
#include <optional>
#include <vector>
#include "tempo/tempoisa/passes.h"
#include "traffic.h"

namespace mlir::tempo::isa
{
#define GEN_PASS_DEF_TEMPOISASCHEDULE
#include "tempo/tempoisa/passes.h.inc"
}  // namespace mlir::tempo::isa

using namespace mlir;
using namespace mlir::tempo::isa;

namespace
{
    /// Stores the cycles after its issue a scalar write lands, which sequencer_pkg leaves to software to space for.
    constexpr int64_t kScalarWriteback = 1;

    /// Stores the longest delay one bundle's field encodes.
    constexpr uint32_t kMaxDelay = (uint32_t {1} << schedule::kDelayFieldWidthBits) - 1;

    /// Stores the passes the scheduler makes before it refuses a stream whose fillers never settle.
    constexpr uint32_t kMaxPasses = 256;

    /// Stores the inclusive range of VMEM rows one access may touch over every iteration of its loops.
    struct RowRange
    {
        int64_t first = 0;  ///< The lowest row.
        int64_t last  = 0;  ///< The highest row.
    };

    /// Stores one vst still inside its turnaround, which a later vld reading its rows waits out.
    struct RecentStore
    {
        int64_t issue = 0;  ///< The cycle the vst issued at.
        RowRange rows;      ///< The rows it may have written.
    };

    /**
     * @brief Returns the rows an access may touch, its induction term ranging over every value the loop at its level
     * takes.
     *
     * @param access the access's address fields.
     * @param loops the loops enclosing the access, outermost first.
     * @returns the range, which spans every row when the access names a loop no lbegin encloses it in.
     */
    RowRange RowsOf(const VmemAccess& access, const llvm::ArrayRef<LoopBounds> loops)
    {
        int64_t lowest_sublane  = -1;
        int64_t highest_sublane = -1;
        for (uint32_t sublane = 0; sublane <= kXMask.Mask(); ++sublane)
        {
            if (((access.mask >> sublane) & 1) == 0) continue;
            if (lowest_sublane < 0) lowest_sublane = sublane;
            highest_sublane = sublane;
        }
        RowRange rows {.first = access.row + lowest_sublane, .last = access.row + highest_sublane};
        if (!access.agu) return rows;
        if (access.level >= loops.size()) return {.first = INT64_MIN, .last = INT64_MAX};

        // Spans the induction variable from its first value to the last one below the bound.
        const LoopBounds& loop = loops[access.level];
        const int64_t step     = std::max<int64_t>(loop.step, 1);
        const int64_t last     = loop.high > loop.low ? loop.low + (loop.high - 1 - loop.low) / step * step : loop.low;
        rows.first += int64_t {loop.low} << access.shift;
        rows.last += last << access.shift;
        return rows;
    }

    /// Stores one bundle of the stream being scheduled, or a filler the scheduler inserted.
    struct StreamEntry
    {
        BundleOp bundle;        ///< The bundle, or null for an inserted filler.
        BundleTraffic traffic;  ///< What the bundle reads, writes and occupies.
        uint32_t delay = 0;     ///< The delay the bundle carries, raised as the timeline requires.
    };

    /**
     * @brief Refuses the hazards no delay repairs: two port-A readers in one bundle, a vector write beside a vmatpop, a
     * burst overflowing the MXU output FIFO, a flush of a DMA slot no descriptor uses, and a DMA slot reused unflushed.
     *
     * @param stream the program's bundles, in order.
     * @returns success when the stream carries none of the hazards, failure with a diagnostic on the first otherwise.
     */
    LogicalResult CheckUnschedulableHazards(std::vector<StreamEntry>& stream)
    {
        for (StreamEntry& entry : stream)
        {
            if (entry.traffic.port_a_readers > 1)
            {
                return entry.bundle.emitOpError(
                    "is unable to issue. Its instructions must claim VREG read port A on separate cycles, but they "
                    "claim it on the same one, so the losing reader silently reads the winner's register."
                );
            }
            if (entry.traffic.drains_results && entry.traffic.vector_writes.size() > 1)
            {
                return entry.bundle.emitOpError(
                    "is unable to issue. A vmatpop must be the only VREG writer in its bundle, because the sequencer "
                    "reserves its unslid write window alone and the wired-OR writeback drops the other write."
                );
            }
        }

        // Counts the results outstanding in each lane's MXU output FIFO, in program order.
        int64_t outstanding = 0;
        for (StreamEntry& entry : stream)
        {
            if (entry.traffic.pushes_results)
            {
                outstanding += schedule::kMxuOutputFifoEntriesPerVmatpush;
                if (outstanding > schedule::kMxuOutputFifoDepth)
                {
                    return entry.bundle.emitOpError()
                        << "is unable to push. The MXU output FIFO must hold every outstanding result, but this "
                           "vmatpush leaves "
                        << outstanding << " where it holds " << schedule::kMxuOutputFifoDepth
                        << ", and overflow drops results silently.";
                }
            }
            else if (entry.traffic.drains_results)
            {
                outstanding = std::max<int64_t>(0, outstanding - schedule::kMxuOutputFifoEntriesPerVmatpop);
            }
        }

        uint32_t issued_slots = 0;
        for (StreamEntry& entry : stream)
        {
            if (entry.traffic.dma_slot) issued_slots |= uint32_t {1} << *entry.traffic.dma_slot;
        }
        std::array<std::optional<size_t>, schedule::kDmaSlotsCount> occupied {};
        for (size_t index = 0; index < stream.size(); ++index)
        {
            const BundleTraffic& traffic = stream[index].traffic;
            if ((traffic.flush_mask & ~issued_slots) != 0)
            {
                return stream[index].bundle.emitOpError(
                    "is unable to flush. A flush must wait only on DMA slots a descriptor issues on, because only a "
                    "descriptor asserts the done bit and the device otherwise stays wedged."
                );
            }
            for (size_t slot = 0; slot < occupied.size(); ++slot)
            {
                if (((traffic.flush_mask >> slot) & 1) != 0) occupied[slot].reset();
            }
            if (!traffic.dma_slot) continue;
            if (occupied[*traffic.dma_slot])
            {
                return stream[index].bundle.emitOpError()
                    << "is unable to issue on DMA slot " << *traffic.dma_slot
                    << ". A slot must be flushed before it carries another descriptor, but the descriptor "
                    << *occupied[*traffic.dma_slot] << " bundles earlier still holds it.";
            }
            occupied[*traffic.dma_slot] = index;
        }
        return success();
    }

    /// Returns the fillers needed to bridge a gap of `gap` cycles, each spacing one delay and one issue cycle.
    uint32_t CountFillers(const int64_t gap)
    {
        uint32_t fillers = 0;
        while (int64_t {fillers + 1} * kMaxDelay + fillers < gap) ++fillers;
        return fillers;
    }

    /**
     * @brief Raises every bundle's delay until the stream meets each latency, inserting fillers where a gap exceeds
     * what one delay field spans.
     *
     * @param stream the program's bundles, in order, whose delays and fillers are updated in place.
     * @param program the program the bundles belong to, which a failure's diagnostic is attached to.
     * @returns success once the delays settle, failure when fillers keep being inserted past the pass limit.
     */
    LogicalResult ComputeDelays(std::vector<StreamEntry>& stream, ProgramOp program)
    {
        for (uint32_t pass = 0; pass < kMaxPasses; ++pass)
        {
            std::array<int64_t, kVectorRegisterCount> vector_ready {};
            std::array<int64_t, kScalarRegisterCount> scalar_ready {};
            std::array<int64_t, kVectorRegisterCount> streamed_until {};
            std::vector<RecentStore> recent_stores;
            llvm::SmallVector<LoopBounds, 4> loops;
            std::optional<int64_t> last_stage;
            std::optional<int64_t> last_drain;
            llvm::DenseSet<int64_t> write_port;
            std::deque<int64_t> results_ready;
            int64_t matrix_free = 0;
            int64_t port_a_free = 0;
            int64_t vmem_free   = 0;
            int64_t previous    = 0;
            std::optional<std::pair<size_t, uint32_t>> insertion;

            for (size_t index = 0; index < stream.size(); ++index)
            {
                const BundleTraffic& traffic = stream[index].traffic;
                int64_t issue                = index == 0 ? 0 : previous + 1 + stream[index - 1].delay;

                // Prices scalar RAW for every bundle, descriptors included, since one reads its base and stride from
                // scalar registers. WAW and WAR come free from the in-order advance of the issue cycle.
                for (const uint32_t scalar : traffic.scalar_reads) issue = std::max(issue, scalar_ready[scalar]);
                if (!traffic.is_dma)
                {
                    for (const uint32_t vector : traffic.vector_reads) issue = std::max(issue, vector_ready[vector]);
                    if (traffic.matrix != nullptr) issue = std::max(issue, matrix_free);
                    if (traffic.drains_results && !results_ready.empty())
                    {
                        issue = std::max(issue, results_ready.front());
                    }
                    if (traffic.claims_port_a) issue = std::max(issue, port_a_free);
                    if (traffic.accesses_vmem) issue = std::max(issue, vmem_free);
                    // Holds a write until a matrix command streaming the register has read it, since the command
                    // reads its operands over its whole occupancy and nothing interlocks a write landing first.
                    for (const WriteReservation& write : traffic.vector_writes)
                    {
                        issue = std::max(issue, streamed_until[write.vector_register] - write.first + 1);
                    }
                    const auto port_taken = [&](const int64_t at) {
                        for (const WriteReservation& write : traffic.vector_writes)
                        {
                            for (int64_t cycle = write.first; cycle <= write.last; ++cycle)
                            {
                                if (write_port.contains(at + cycle)) return true;
                            }
                        }
                        return false;
                    };
                    while (port_taken(issue)) ++issue;
                }

                // Holds what the register model does not reach: VMEM rows a vst wrote, and the transpose staging array.
                if (traffic.vmem_read)
                {
                    const RowRange read = RowsOf(*traffic.vmem_read, loops);
                    for (const RecentStore& store : recent_stores)
                    {
                        if (store.rows.first <= read.last && read.first <= store.rows.last)
                        {
                            issue = std::max(issue, store.issue + schedule::kVmemTurnaroundCycles);
                        }
                    }
                }
                if (traffic.stages_transpose && last_stage)
                {
                    issue = std::max(issue, *last_stage + schedule::kTransposeInIntervalCycles);
                }
                if (traffic.drains_transpose)
                {
                    if (last_stage) issue = std::max(issue, *last_stage + schedule::kTransposeInToOutCycles);
                    if (last_drain) issue = std::max(issue, *last_drain + schedule::kTransposeOutIntervalCycles);
                }

                if (index > 0 && issue - previous - 1 > kMaxDelay)
                {
                    insertion = {index, CountFillers(issue - previous - 1)};
                    break;
                }

                if (!traffic.is_dma)
                {
                    for (const WriteReservation& write : traffic.vector_writes)
                    {
                        for (int64_t cycle = write.first; cycle <= write.last; ++cycle)
                            write_port.insert(issue + cycle);
                        vector_ready[write.vector_register] = issue + write.last + 1;
                    }
                    if (traffic.accesses_vmem) vmem_free = issue + 1;
                    if (traffic.matrix != nullptr)
                    {
                        matrix_free = issue + traffic.matrix->occupancy_cycles + 1;
                        if (traffic.matrix->holds_read_port_a)
                        {
                            port_a_free = matrix_free;
                            for (const uint32_t vector : traffic.vector_reads)
                            {
                                streamed_until[vector] = issue + traffic.matrix->occupancy_cycles;
                            }
                        }
                        if (traffic.pushes_results)
                        {
                            results_ready.push_back(issue + traffic.matrix->result_latency_cycles);
                        }
                        else if (traffic.drains_results && !results_ready.empty())
                        {
                            results_ready.pop_front();
                        }
                    }
                }
                for (const uint32_t scalar : traffic.scalar_writes) scalar_ready[scalar] = issue + kScalarWriteback + 1;
                if (traffic.vmem_written) recent_stores.push_back({issue, RowsOf(*traffic.vmem_written, loops)});
                if (traffic.opens_loop) loops.push_back(*traffic.opens_loop);
                if (traffic.closes_loop && !loops.empty()) loops.pop_back();
                if (traffic.stages_transpose) last_stage = issue;
                if (traffic.drains_transpose) last_drain = issue;
                llvm::erase_if(recent_stores, [&](const RecentStore& store) {
                    return store.issue + schedule::kVmemTurnaroundCycles <= issue;
                });
                if (index > 0)
                {
                    uint32_t& delay = stream[index - 1].delay;
                    delay           = std::max<uint32_t>(delay, static_cast<uint32_t>(issue - previous - 1));
                }
                previous = issue;
            }

            if (!insertion) return success();
            const auto [position, fillers] = *insertion;
            stream.insert(
                stream.begin() + static_cast<std::ptrdiff_t>(position),
                fillers,
                StreamEntry {.bundle = nullptr, .traffic = {}, .delay = kMaxDelay}
            );
        }
        return program.emitOpError() << "is unable to be scheduled. Its fillers must settle within " << kMaxPasses
                                     << " passes, but the program still needed one.";
    }

    /// Writes the computed delays onto the bundles and materializes every inserted filler as an empty bundle.
    void ApplyDelays(std::vector<StreamEntry>& stream)
    {
        BundleOp previous = nullptr;
        for (StreamEntry& entry : stream)
        {
            if (!entry.bundle)
            {
                OpBuilder builder(previous);
                builder.setInsertionPointAfter(previous);
                entry.bundle = BundleOp::create(builder, previous.getLoc(), entry.delay);
                entry.bundle.getBody().emplaceBlock();
            }
            entry.bundle.setDelay(entry.delay);
            previous = entry.bundle;
        }
    }

    /// Provides the tempoisa-schedule pass over one program.
    class SchedulePass final : public mlir::tempo::isa::impl::TempoisaScheduleBase<SchedulePass>
    {
      public:
        /// Schedules the program, leaving it untouched when any hazard is refused.
        void runOnOperation() override
        {
            std::vector<StreamEntry> stream;
            for (Operation& operation : getOperation().getBody().front())
            {
                auto bundle                            = cast<BundleOp>(operation);
                const FailureOr<BundleTraffic> traffic = CollectTraffic(bundle);
                if (failed(traffic))
                {
                    signalPassFailure();
                    return;
                }
                stream.push_back({.bundle = bundle, .traffic = *traffic, .delay = bundle.getDelay()});
            }

            if (failed(CheckUnschedulableHazards(stream)) || failed(ComputeDelays(stream, getOperation())))
            {
                signalPassFailure();
                return;
            }
            ApplyDelays(stream);
        }
    };
}  // namespace
