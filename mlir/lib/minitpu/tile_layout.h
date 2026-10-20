/**
 * @file
 *
 * @brief Provides the layout the minitpu tile ops are lowered over, shared by their verifiers and their lowering.
 */

#ifndef MINITPU_LIB_MINITPU_TILE_LAYOUT_H
#define MINITPU_LIB_MINITPU_TILE_LAYOUT_H

#include <cstdint>

namespace mlir::minitpu::tile
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

    /// Stores the rows one DMA descriptor moves at most, which is the chunk every streamed operand moves in.
    static constexpr uint32_t kChunkRows = 128;

    /// Stores the feature tiles one chunk of a block's activations covers, each tile being one row per token.
    static constexpr uint32_t kChunkFeatureTiles = kChunkRows / kTokens;

    /// Stores the rows a broadcast row takes per feature tile, the row repeated across every sublane.
    static constexpr uint32_t kBroadcastRows = kSublanes;

    /// Stores the k-tiles one weight chunk covers.
    static constexpr uint32_t kChunkTiles = kChunkRows / kTile;

    /// Stores the rows of the zeroed tile heading the drain buffer, which the accumulators start from.
    static constexpr uint32_t kZeroRows = kTile;

    /// Stores the VMEM rows one bank holds, which every buffer of a launch shares.
    static constexpr uint32_t kBankRows = 8192;

    /// Stores the rows each scalar a LayerNorm's constants block states takes, the scalar repeated across one register.
    static constexpr uint32_t kScalarRows = kSublanes;

    /// Stores the scalars a LayerNorm's constants block states after gamma and beta: the reciprocal width and epsilon.
    static constexpr uint32_t kNormScalars = 2;

    /// Returns the rows a LayerNorm's constants block takes: gamma, beta, and one register per scalar.
    constexpr uint32_t NormConstantRows(const uint32_t features)
    {
        return 2 * (features / kTile) + kNormScalars * kScalarRows;
    }

    /// Returns the VMEM rows a LayerNorm of the given width stages: its constants, and two token rows each of input and
    /// output.
    constexpr uint32_t LayernormStagedRows(const uint32_t features)
    {
        return NormConstantRows(features) + 4 * (features / kTile);
    }

    /// Returns the VMEM rows a product of the given shape stages: the zero tile, or the bias row when it adds one, the
    /// activations, two weight chunks and one column tile of output.
    constexpr uint32_t MatmulStagedRows(const uint32_t depth, const uint32_t columns, const bool bias)
    {
        const uint32_t start = bias ? kBroadcastRows * (columns / kTile) : kZeroRows;
        return start + kTokens * (depth / kTile) + 2 * kChunkRows + kTokens;
    }

    /// Returns the VMEM rows a sum of the given width stages: two chunks of each streamed operand, one chunk of output,
    /// and the whole broadcast row when it broadcasts.
    constexpr uint32_t AddStagedRows(const uint32_t features, const bool broadcast)
    {
        const uint32_t streamed = broadcast ? 2 : 4;
        return streamed * kChunkRows + kChunkRows + (broadcast ? kBroadcastRows * (features / kTile) : 0);
    }
}  // namespace mlir::minitpu::tile

#endif  // MINITPU_LIB_MINITPU_TILE_LAYOUT_H
