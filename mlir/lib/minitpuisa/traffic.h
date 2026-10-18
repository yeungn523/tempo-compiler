/**
 * @file
 *
 * @brief Provides the traffic analysis the minitpuisa passes share: what each bundle reads, writes and occupies.
 *
 * @note The model matches codegen/asm.py's decoder, including its over-claims: every E instruction reads its ra field
 * and claims read port A, and every scalar operation reads its rs field. Matching it keeps the passes' images identical
 * to asm.py's while the Python assembler remains the reference.
 */

#ifndef MINITPU_LIB_MINITPUISA_TRAFFIC_H
#define MINITPU_LIB_MINITPUISA_TRAFFIC_H

#include <llvm/ADT/SmallVector.h>
#include <optional>
#include "minitpu/minitpuisa/minitpuisa.h"
#include "minitpu/minitpuisa/minitpuisa_schedule.h.inc"

namespace mlir::minitpu::isa
{
    /// Stores the number of vector registers, which a vector register field addresses in full.
    static constexpr size_t kVectorRegisterCount = kMRegister.Mask() + 1;

    /// Stores the number of scalar registers, which a scalar register field addresses in full.
    static constexpr size_t kScalarRegisterCount = kSRd.Mask() + 1;

    /// Stores one vector register write and the cycles, relative to issue, it holds the write port for.
    struct WriteReservation
    {
        uint32_t vector_register = 0;  ///< The register written.
        int64_t first            = 0;  ///< The first cycle the write port is held.
        int64_t last             = 0;  ///< The last cycle the write port is held.
    };

    /// Stores what one bundle reads, writes and occupies, which is everything the timeline prices it by.
    struct BundleTraffic
    {
        bool is_dma = false;                                   ///< Determines whether the bundle carries a descriptor.
        llvm::SmallVector<uint32_t, 4> vector_reads;           ///< The vector registers read.
        llvm::SmallVector<WriteReservation, 2> vector_writes;  ///< The vector registers written.
        llvm::SmallVector<uint32_t, 2> scalar_reads;           ///< The scalar registers read.
        llvm::SmallVector<uint32_t, 1> scalar_writes;          ///< The scalar registers written.
        bool claims_port_a      = false;                       ///< Determines whether the bundle waits for read port A.
        uint32_t port_a_readers = 0;      ///< The instructions claiming port A on the bundle's own issue cycle.
        bool accesses_vmem      = false;  ///< Determines whether the bundle holds the VMEM compute port.
        const schedule::MatrixCommandFacts* matrix = nullptr;  ///< The matrix command issued, or nullptr.
        bool pushes_results = false;       ///< Determines whether the matrix command produces results.
        bool drains_results = false;       ///< Determines whether the matrix command drains results.
        std::optional<uint32_t> dma_slot;  ///< The DMA slot the descriptor issues on.
        uint32_t flush_mask = 0;           ///< The DMA slots the bundle waits on, one bit each.
        bool stores_vmem    = false;       ///< Determines whether the bundle's VMEM access writes VMEM.
        uint32_t slots      = 0;           ///< The slots the bundle occupies, one bit per kSlot.
    };

    /// Returns what one bundle reads, writes and occupies, or failure for an instruction the records do not name.
    FailureOr<BundleTraffic> CollectTraffic(BundleOp bundle);
}  // namespace mlir::minitpu::isa

#endif  // MINITPU_LIB_MINITPUISA_TRAFFIC_H
