/**
 * @file
 *
 * @brief Implements the traffic analysis the tempoisa passes share.
 */

#include "traffic.h"

using namespace mlir;
using namespace mlir::tempo::isa;

namespace
{
    /// Stores the base a register role's offset range is written in, as in reg_idx+0..3.
    constexpr uint32_t kRoleRadix = 10;

    /// Returns the address fields of a vld or vst.
    template <typename AccessOp>
    VmemAccess AccessOf(AccessOp access)
    {
        return {
            .row   = access.getRow(),
            .agu   = access.getAgu(),
            .level = access.getLevel(),
            .shift = access.getShift(),
            .mask  = access.getMask(),
        };
    }

    /// Returns the value of the operand `role` names, or 0 where the instruction carries no such operand.
    uint32_t ReadOperand(Operation& operation, const llvm::StringRef role)
    {
        // Reads the field's zero bits where an instruction carries no operand of that role, as the hardware decoder
        // does.
        const std::optional<Attribute> attribute = operation.getInherentAttr(role);
        if (!attribute) return 0;
        return static_cast<uint32_t>(cast<IntegerAttr>(*attribute).getInt());
    }

    /**
     * @brief Appends the registers one register role reads, expanding a range such as reg_idx+0..3.
     *
     * @param operation the instruction carrying the role.
     * @param role the role, as tempoisa/isa.td's register_reads names it.
     * @param register_count the size of the register file, which a range wraps around.
     * @param reads the registers read, which receives the role's registers.
     */
    void AppendReads(
        Operation& operation,
        const std::string_view role,
        const size_t register_count,
        llvm::SmallVectorImpl<uint32_t>& reads
    )
    {
        const auto [name, range] = llvm::StringRef(role.data(), role.size()).split('+');
        const uint32_t base      = ReadOperand(operation, name);
        if (range.empty())
        {
            reads.push_back(base);
            return;
        }

        const auto [low_text, high_text] = range.split("..");
        uint32_t low                     = 0;
        uint32_t high                    = 0;
        low_text.getAsInteger(kRoleRadix, low);
        high_text.getAsInteger(kRoleRadix, high);
        for (uint32_t offset = low; offset <= high; ++offset)
        {
            reads.push_back(static_cast<uint32_t>((base + offset) % register_count));
        }
    }

}  // namespace

FailureOr<BundleTraffic> mlir::tempo::isa::CollectTraffic(BundleOp bundle)
{
    BundleTraffic traffic;
    for (Operation& operation : bundle.getBody().front())
    {
        const schedule::OperationFacts* facts = schedule::FindOperation(operation.getName().stripDialect());
        if (facts == nullptr)
        {
            return operation.emitOpError(
                "is unable to be scheduled. Its mnemonic must be one tempoisa/isa.td publishes facts for."
            );
        }

        traffic.slots |= uint32_t {1} << static_cast<uint32_t>(facts->slot);
        switch (facts->slot)
        {
            case kSlot::kE:
                // Prices every E instruction as reading its ra field and claiming port A, the over-claim traffic.h
                // names.
                traffic.vector_reads.push_back(ReadOperand(operation, "ra"));
                for (const std::string_view role : facts->register_reads)
                {
                    if (!role.empty() && role != "ra")
                    {
                        AppendReads(operation, role, kVectorRegisterCount, traffic.vector_reads);
                    }
                }
                traffic.claims_port_a = true;
                ++traffic.port_a_readers;
                traffic.stages_transpose = llvm::isa<VtxinOp>(operation);
                traffic.drains_transpose = llvm::isa<VtxoutOp>(operation);
                if (facts->writeback_first >= 0)
                {
                    traffic.vector_writes.push_back(
                        {ReadOperand(operation, facts->register_write), facts->writeback_first, facts->writeback_last}
                    );
                }
                break;
            case kSlot::kX:
            case kSlot::kM:
                for (const std::string_view role : facts->register_reads)
                {
                    if (!role.empty()) AppendReads(operation, role, kVectorRegisterCount, traffic.vector_reads);
                }
                if (!facts->register_write.empty() && facts->writeback_first >= 0)
                {
                    traffic.vector_writes.push_back(
                        {ReadOperand(operation, facts->register_write), facts->writeback_first, facts->writeback_last}
                    );
                }
                traffic.claims_port_a = traffic.claims_port_a || facts->claims_read_port_a;
                if (facts->slot == kSlot::kX)
                {
                    traffic.accesses_vmem = true;
                    traffic.stores_vmem   = facts->register_write.empty();
                    if (auto load = dyn_cast<VldOp>(operation)) traffic.vmem_read = AccessOf(load);
                    if (auto store = dyn_cast<VstOp>(operation)) traffic.vmem_written = AccessOf(store);
                    if (facts->claims_read_port_a) ++traffic.port_a_readers;
                    break;
                }
                traffic.matrix         = schedule::FindMatrixCommand(facts->mnemonic);
                traffic.pushes_results = traffic.matrix != nullptr && traffic.matrix->result_latency_cycles >= 0;
                traffic.drains_results = !facts->register_write.empty();
                break;
            case kSlot::kS:
                for (const std::string_view role : facts->register_reads)
                {
                    if (!role.empty()) AppendReads(operation, role, kScalarRegisterCount, traffic.scalar_reads);
                }
                traffic.scalar_writes.push_back(ReadOperand(operation, facts->register_write));
                break;
            case kSlot::kF:
                if (auto flush = dyn_cast<FlushSlotOp>(operation)) traffic.flush_mask = flush.getMask();
                if (auto loop = dyn_cast<LbeginOp>(operation))
                {
                    traffic.opens_loop = LoopBounds {.low = loop.getLo(), .high = loop.getHi(), .step = loop.getStep()};
                }
                traffic.closes_loop = traffic.closes_loop || llvm::isa<LoopEndOp>(operation);
                break;
            case kSlot::kD:
                traffic.is_dma   = true;
                traffic.dma_slot = ReadOperand(operation, "dma_slot");
                for (const std::string_view role : facts->register_reads)
                {
                    if (!role.empty()) AppendReads(operation, role, kScalarRegisterCount, traffic.scalar_reads);
                }
                break;
        }
    }
    return traffic;
}
