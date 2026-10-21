/**
 * @file
 *
 * @brief Implements the minitpuisa packer, which merges independent instructions between barriers into shared bundles.
 *
 * @note The region schedule matches the retired Python assembler's pack() bundle for bundle, over the traffic traffic.h
 * models, which the recorded streams under test/Inputs/schedule hold it to.
 */

#include <llvm/ADT/DenseSet.h>
#include <map>
#include <vector>
#include "minitpu/minitpuisa/passes.h"
#include "traffic.h"

namespace mlir::minitpu::isa
{
#define GEN_PASS_DEF_MINITPUISAPACK
#include "minitpu/minitpuisa/passes.h.inc"
}  // namespace mlir::minitpu::isa

using namespace mlir;
using namespace mlir::minitpu::isa;

namespace
{
    /// Stores the cycles after its issue a scalar write lands, which sequencer_pkg leaves to software to space for.
    constexpr int64_t kScalarWriteback = 1;

    /// Stores the slots that pin a bundle to its position, being the ones whose consumers sit outside any region.
    constexpr uint32_t kBarrierSlots = (uint32_t {1} << static_cast<uint32_t>(kSlot::kM)) |
                                       (uint32_t {1} << static_cast<uint32_t>(kSlot::kS)) |
                                       (uint32_t {1} << static_cast<uint32_t>(kSlot::kF));

    /// Stores one bundle of a region together with its traffic.
    struct RegionEntry
    {
        BundleOp bundle;        ///< The bundle.
        BundleTraffic traffic;  ///< What the bundle reads, writes and occupies.
    };

    /// Stores one bundle of the packed program: a barrier kept in place, a merge of region bundles, or an empty bundle.
    struct PackedBundle
    {
        llvm::SmallVector<BundleOp, 2> members;  ///< The bundles merged into this one, empty for an empty bundle.
    };

    /// Determines whether a bundle must keep its position: a DMA bundle, or one carrying a matrix, scalar or control
    /// instruction.
    bool IsBarrier(const BundleTraffic& traffic)
    {
        return traffic.is_dma || (traffic.slots & kBarrierSlots) != 0;
    }

    /// Determines whether a bundle encodes to the all-zero word, carrying no instruction and no delay.
    bool IsEmpty(BundleOp bundle)
    {
        return bundle.getBody().front().empty() && bundle.getDelay() == 0;
    }

    /// Determines whether `registers` holds `target`.
    bool Holds(const llvm::ArrayRef<uint32_t> registers, const uint32_t target)
    {
        return llvm::is_contained(registers, target);
    }

    /**
     * @brief Returns the cycles `later` must issue after `earlier`, being 0 when the two are independent.
     *
     * @param earlier the traffic of the bundle placed first in program order.
     * @param later the traffic of the bundle being placed.
     * @returns the minimum issue distance between the two.
     */
    int64_t DependenceGap(const BundleTraffic& earlier, const BundleTraffic& later)
    {
        const int64_t occupancy = earlier.matrix != nullptr ? earlier.matrix->occupancy_cycles : 0;
        int64_t gap             = 0;
        llvm::SmallVector<uint32_t, 2> later_writes;
        for (const WriteReservation& write : later.vector_writes) later_writes.push_back(write.vector_register);

        for (const WriteReservation& write : earlier.vector_writes)
        {
            if (Holds(later.vector_reads, write.vector_register)) gap = std::max(gap, write.last + 1);
            if (Holds(later_writes, write.vector_register)) gap = std::max<int64_t>(gap, 1);
        }
        for (const uint32_t scalar : earlier.scalar_writes)
        {
            if (Holds(later.scalar_reads, scalar)) gap = std::max(gap, kScalarWriteback + 1);
            if (Holds(later.scalar_writes, scalar)) gap = std::max<int64_t>(gap, 1);
        }

        // Holds an operand a streaming command still reads for that command's whole occupancy.
        if (llvm::any_of(later_writes, [&](const uint32_t vector) {
                return Holds(earlier.vector_reads, vector);
            }))
        {
            gap = std::max(gap, occupancy != 0 ? occupancy : 1);
        }
        if (llvm::any_of(later.scalar_writes, [&](const uint32_t scalar) {
                return Holds(earlier.scalar_reads, scalar);
            }))
        {
            gap = std::max<int64_t>(gap, 1);
        }
        if (earlier.matrix != nullptr && later.matrix != nullptr) gap = std::max(gap, occupancy);

        // Keeps read port A for a vmatload's whole occupancy, the one held command that produces no results.
        const bool earlier_loads_weights =
            earlier.matrix != nullptr && earlier.matrix->holds_read_port_a && !earlier.pushes_results;
        if (earlier_loads_weights && later.claims_port_a) gap = std::max(gap, occupancy);
        if (earlier.accesses_vmem && later.accesses_vmem && (earlier.stores_vmem || later.stores_vmem))
        {
            gap = std::max<int64_t>(gap, 1);
        }

        // Keeps every vtxin and vtxout in program order, since they share the transpose staging array, which no
        // register names. The scheduler then spaces them by the array's own intervals.
        const bool earlier_transposes = earlier.stages_transpose || earlier.drains_transpose;
        if (earlier_transposes && (later.stages_transpose || later.drains_transpose)) gap = std::max<int64_t>(gap, 1);
        return gap;
    }

    /**
     * @brief List-schedules one region, merging bundles that land on one cycle.
     *
     * @param region the region's non-empty bundles, in program order.
     * @param packed the packed program, which receives the region's merged bundles in cycle order.
     */
    void PackRegion(const std::vector<RegionEntry>& region, std::vector<PackedBundle>& packed)
    {
        if (region.size() < 2)
        {
            for (const RegionEntry& entry : region) packed.push_back({{entry.bundle}});
            return;
        }

        // Holds each occupied cycle's merged bundles, the slots they take, and their port-A readers.
        struct Cycle
        {
            PackedBundle merged;          ///< The bundles merged on this cycle.
            uint32_t slots          = 0;  ///< The slots they occupy, one bit per kSlot.
            uint32_t port_a_readers = 0;  ///< The instructions claiming read port A on this cycle.
        };

        std::map<int64_t, Cycle> cycles;
        std::vector<int64_t> cycle_of(region.size(), 0);
        std::vector<size_t> pushes;
        size_t pops = 0;

        for (size_t index = 0; index < region.size(); ++index)
        {
            const BundleTraffic& traffic = region[index].traffic;
            int64_t ready                = 0;
            for (size_t earlier = 0; earlier < index; ++earlier)
            {
                const int64_t gap = DependenceGap(region[earlier].traffic, traffic);
                if (gap != 0) ready = std::max(ready, cycle_of[earlier] + gap);
            }

            // Drains the i-th push with the i-th pop, which waits out the array's result latency.
            if (traffic.drains_results && pops < pushes.size())
            {
                const RegionEntry& push = region[pushes[pops]];
                ready = std::max(ready, cycle_of[pushes[pops]] + push.traffic.matrix->result_latency_cycles);
                ++pops;
            }
            if (traffic.pushes_results) pushes.push_back(index);

            while (true)
            {
                const auto held = cycles.find(ready);
                if (held == cycles.end()) break;
                if ((held->second.slots & traffic.slots) == 0 &&
                    held->second.port_a_readers + traffic.port_a_readers <= 1)
                {
                    break;
                }
                ++ready;
            }
            Cycle& cycle = cycles[ready];
            cycle.merged.members.push_back(region[index].bundle);
            cycle.slots |= traffic.slots;
            cycle.port_a_readers += traffic.port_a_readers;
            cycle_of[index] = ready;
        }
        for (auto& [cycle, held] : cycles) packed.push_back(std::move(held.merged));
    }

    /// Returns the program's bundles split at its barriers, each region packed, in program order.
    FailureOr<std::vector<PackedBundle>> PackProgram(ProgramOp program)
    {
        std::vector<PackedBundle> packed;
        std::vector<RegionEntry> region;
        uint32_t pending_empties = 0;
        for (Operation& operation : program.getBody().front())
        {
            auto bundle                            = cast<BundleOp>(operation);
            const FailureOr<BundleTraffic> traffic = CollectTraffic(bundle);
            if (failed(traffic)) return failure();

            if (IsBarrier(*traffic))
            {
                PackRegion(region, packed);
                region.clear();
                packed.insert(packed.end(), pending_empties, PackedBundle {});
                pending_empties = 0;
                packed.push_back({{bundle}});
            }
            else if (!IsEmpty(bundle))
            {
                region.push_back({bundle, *traffic});
            }
            else if (!region.empty())
            {
                pending_empties = 0;
            }
            else
            {
                ++pending_empties;
            }
        }
        PackRegion(region, packed);
        packed.insert(packed.end(), pending_empties, PackedBundle {});
        return packed;
    }

    /**
     * @brief Rebuilds the program body as the packed bundles, in order.
     *
     * A lone member is moved as it stands. Several members merge into one new bundle carrying their instructions and
     * the bitwise OR of their delays, which is what merging their words produces.
     *
     * @param program the program to rebuild.
     * @param packed the packed bundles, in order.
     */
    void RebuildProgram(ProgramOp program, const std::vector<PackedBundle>& packed)
    {
        Block& body = program.getBody().front();
        std::vector<Operation*> originals;
        for (Operation& operation : body) originals.push_back(&operation);

        llvm::DenseSet<Operation*> kept;
        OpBuilder builder(program.getContext());
        for (const PackedBundle& bundle : packed)
        {
            if (bundle.members.size() == 1)
            {
                BundleOp lone     = bundle.members.front();
                Operation* member = lone.getOperation();
                member->moveBefore(&body, body.end());
                kept.insert(member);
                continue;
            }

            builder.setInsertionPointToEnd(&body);
            BundleOp first          = bundle.members.empty() ? BundleOp() : bundle.members.front();
            const Location location = first ? first.getLoc() : program.getLoc();
            uint32_t delay          = 0;
            for (BundleOp member : bundle.members) delay |= member.getDelay();
            auto merged  = BundleOp::create(builder, location, delay);
            Block& slots = merged.getBody().emplaceBlock();
            for (BundleOp member : bundle.members)
            {
                for (Operation& instruction : llvm::make_early_inc_range(member.getBody().front()))
                {
                    instruction.moveBefore(&slots, slots.end());
                }
            }
        }
        for (Operation* original : originals)
        {
            if (!kept.contains(original)) original->erase();
        }
    }

    /// Provides the minitpuisa-pack pass over one program.
    class PackPass final : public mlir::minitpu::isa::impl::MinitpuisaPackBase<PackPass>
    {
      public:
        /// Packs the program, leaving it untouched when an instruction is one the records do not name.
        void runOnOperation() override
        {
            const FailureOr<std::vector<PackedBundle>> packed = PackProgram(getOperation());
            if (failed(packed))
            {
                signalPassFailure();
                return;
            }
            RebuildProgram(getOperation(), *packed);
        }
    };
}  // namespace
