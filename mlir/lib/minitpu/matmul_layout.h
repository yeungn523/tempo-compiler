/**
 * @file
 *
 * @brief Provides the layout a minitpu.matmul is lowered over, shared by its verifier and its lowering.
 */

#ifndef MINITPU_LIB_MINITPU_MATMUL_LAYOUT_H
#define MINITPU_LIB_MINITPU_MATMUL_LAYOUT_H

#include <cstdint>

namespace mlir::minitpu::matmul
{
    /// Stores the elements one VMEM row and one DRAM row hold, being the array's width.
    static constexpr uint32_t kTile = 16;

    /// Stores the bytes one row moves between DRAM and VMEM.
    static constexpr uint32_t kRowBytes = kTile * 2;

    /// Stores the rows one vector register holds, one per sublane.
    static constexpr uint32_t kSublanes = 4;

    /// Stores the vector registers one 16x16 weight tile occupies.
    static constexpr uint32_t kWeightRegisters = kTile / kSublanes;

    /// Stores the tokens one launch covers, whose pushes fill the MXU output FIFO's 32 entries exactly.
    static constexpr uint32_t kTokens = 32;

    /// Stores the vector registers one block's tokens occupy, each holding four tokens.
    static constexpr uint32_t kTokenGroups = kTokens / kSublanes;

    /// Stores the rows one DMA descriptor moves at most, which is the weight chunk the lowering streams.
    static constexpr uint32_t kChunkRows = 128;

    /// Stores the k-tiles one weight chunk covers.
    static constexpr uint32_t kChunkTiles = kChunkRows / kTile;

    /// Stores the rows of the zeroed tile heading the drain buffer, which the accumulators start from.
    static constexpr uint32_t kZeroRows = kTile;

    /// Stores the VMEM rows one bank holds, which every buffer of a launch shares.
    static constexpr uint32_t kBankRows = 8192;

    /// Returns the VMEM rows a launch of the given depth stages: the zero tile, the activations, two weight chunks and
    /// one column tile of output.
    constexpr uint32_t StagedRows(const uint32_t depth)
    {
        return kZeroRows + kTokens * (depth / kTile) + 2 * kChunkRows + kTokens;
    }
}  // namespace mlir::minitpu::matmul

#endif  // MINITPU_LIB_MINITPU_MATMUL_LAYOUT_H
