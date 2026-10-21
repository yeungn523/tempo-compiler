/**
 * @file
 *
 * @brief Implements the minitpu dialect: its types, its descriptor verifier, and the operations TableGen generates
 * from minitpu_ops.td.
 *
 * @note The verify and initialize names below are the ones MLIR's generated code calls, so they keep MLIR's spelling.
 */

#include "minitpu/minitpu/minitpu.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Matchers.h>
#include "tile_layout.h"

using namespace mlir;
using namespace mlir::minitpu;

#include "minitpu/minitpu/minitpu_dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "minitpu/minitpu/minitpu_types.cpp.inc"

/// Registers every type and operation the dialect defines.
void MinitpuDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "minitpu/minitpu/minitpu_types.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "minitpu/minitpu/minitpu_ops.cpp.inc"
        >();
}

namespace
{
    /**
     * @brief Verifies that a descriptor moving `rows` rows from a constant `row` stays inside its buffer.
     *
     * A row the program computes is held to the buffer by the loop lowering, which resolves it.
     *
     * @param operation the descriptor.
     * @param buffer the VMEM buffer it moves rows into or out of.
     * @param row the first row it moves.
     * @param rows the rows it moves.
     * @returns success when the rows fit or the row is not a constant, failure with a diagnostic otherwise.
     */
    LogicalResult VerifyDescriptorBounds(Operation* operation, const VmemType buffer, Value row, const uint32_t rows)
    {
        APInt first;
        if (!matchPattern(row, m_ConstantInt(&first))) return success();
        const uint64_t last = first.getZExtValue() + rows;
        if (last <= buffer.getRows()) return success();
        return operation->emitOpError() << "is unable to move rows " << first.getZExtValue() << " through " << last - 1
                                        << ". A descriptor must stay inside its buffer, which holds "
                                        << buffer.getRows() << " rows.";
    }
}  // namespace

/// Verifies that a minitpu.reg attribute names a register of the file its operation's result lives in.
LogicalResult MinitpuDialect::verifyOperationAttribute(Operation* operation, const NamedAttribute attribute)
{
    if (attribute.getName() != get_register_attribute_name())
    {
        return operation->emitOpError() << "carries " << attribute.getName()
                                        << ", which the minitpu dialect does not define.";
    }
    const auto index = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!index || operation->getNumResults() != 1)
    {
        return operation->emitOpError() << "is unable to carry " << get_register_attribute_name()
                                        << ". It must be an integer on an operation defining one value.";
    }

    // Holds a vector value to the 32 vector registers and a scalar value to the 4 scalar registers.
    const Type type         = operation->getResult(0).getType();
    const bool is_vector    = isa<VRegType>(type);
    const bool is_scalar    = type.isInteger(32);
    const int64_t registers = is_vector ? 32 : 4;
    if (!is_vector && !is_scalar)
    {
        return operation->emitOpError() << "is unable to carry " << get_register_attribute_name()
                                        << ". Only a vector or i32 scalar value lives in a register, but it defines "
                                        << type << ".";
    }
    if (index.getInt() < 0 || index.getInt() >= registers)
    {
        return operation->emitOpError() << "is unable to live in " << (is_vector ? 'v' : 's') << index.getInt()
                                        << ". It must name one of the " << registers << " "
                                        << (is_vector ? "vector" : "scalar") << " registers.";
    }
    return success();
}

/// Verifies that the product's shape is one the lowering covers and that its fixed buffers leave room for activations.
LogicalResult MatmulOp::verify()
{
    if (getTokens() != tile::kTokens)
    {
        return emitOpError() << "is unable to cover " << getTokens() << " tokens. One launch covers " << tile::kTokens
                             << ", whose pushes fill the MXU output FIFO exactly.";
    }
    if (getDepth() % tile::kTile != 0 || getColumns() % tile::kTile != 0)
    {
        return emitOpError() << "is unable to cover a depth of " << getDepth() << " by " << getColumns()
                             << " columns. Both must be a whole number of " << tile::kTile
                             << "-wide tiles, the host padding a narrower operand with zeros.";
    }

    // Splits a contraction deeper than one bank holds into segments, so only the fixed buffers bound the width.
    if (tile::MatmulSegmentLimit(getColumns(), getBias() != nullptr) == 0)
    {
        return emitOpError() << "is unable to stage " << getColumns()
                             << " columns. Its bias row and weight chunks take "
                             << tile::MatmulFixedRows(getColumns(), getBias() != nullptr)
                             << " VMEM rows, leaving one bank of " << tile::kBankRows
                             << " no room for a chunk of activations.";
    }
    return success();
}

/// Verifies that the sum's shape is one the lowering covers and that its staged rows fit one VMEM bank.
LogicalResult AddOp::verify()
{
    if (getTokens() != tile::kTokens)
    {
        return emitOpError() << "is unable to cover " << getTokens() << " tokens. One launch covers " << tile::kTokens
                             << ", which fill one feature tile's rows.";
    }

    // Holds the width to an even number of chunks, since the steady loop alternates two buffers between them.
    const uint32_t pair = 2 * tile::kChunkFeatureTiles * tile::kTile;
    if (getFeatures() % pair != 0)
    {
        return emitOpError() << "is unable to cover " << getFeatures() << " features. The width must be a whole number "
                             << "of " << pair << ", an even number of " << tile::kChunkRows << "-row chunks.";
    }
    if (tile::AddStagedRows(getFeatures(), getBroadcast()) > tile::kBankRows)
    {
        return emitOpError() << "is unable to stage " << getFeatures() << " features. Its buffers take "
                             << tile::AddStagedRows(getFeatures(), getBroadcast())
                             << " VMEM rows, where one bank holds " << tile::kBankRows << ".";
    }
    return success();
}

/// Verifies that the LayerNorm's shape is one the lowering covers and that its staged rows fit one VMEM bank.
LogicalResult LayernormOp::verify()
{
    if (getTokens() != tile::kTokens)
    {
        return emitOpError() << "is unable to cover " << getTokens() << " tokens. One launch covers " << tile::kTokens
                             << ", which fill one feature tile's rows.";
    }

    // Holds a token to whole registers and to one descriptor, which gathers its rows from every feature tile.
    const uint32_t register_features = tile::kSublanes * tile::kTile;
    if (getFeatures() % register_features != 0 || getFeatures() / tile::kTile > tile::kChunkRows)
    {
        return emitOpError() << "is unable to cover " << getFeatures() << " features. The width must be a whole number "
                             << "of " << register_features << ", and at most " << tile::kChunkRows * tile::kTile
                             << " so one descriptor gathers a token.";
    }
    if (tile::LayernormStagedRows(getFeatures()) > tile::kBankRows)
    {
        return emitOpError() << "is unable to stage " << getFeatures() << " features. Its buffers take "
                             << tile::LayernormStagedRows(getFeatures()) << " VMEM rows, where one bank holds "
                             << tile::kBankRows << ".";
    }
    return success();
}

/// Verifies that the attention's shape is one the lowering covers and that its staged rows fit one VMEM bank.
LogicalResult AttentionOp::verify()
{
    const bool causal = getCausal();
    if (causal == (getCache() != nullptr))
    {
        return emitOpError() << "is unable to tell where its keys lie. A causal attention reads them from the queries "
                             << "tensor and any other reads them from a cache, so exactly one of causal and a cache "
                             << "must be given.";
    }
    if (getHead() % tile::kTile != 0 || getKeys() % tile::kTokens != 0)
    {
        return emitOpError() << "is unable to cover a head of " << getHead() << " over " << getKeys() << " keys. The "
                             << "head must be a whole number of " << tile::kTile << "-wide tiles, at most "
                             << tile::kMaxHeadTiles * tile::kTile << ", and the keys a whole number of "
                             << tile::kTokens << "-token blocks.";
    }
    const uint32_t staged = tile::AttentionStagedRows(getHead(), getKeys(), causal);
    if (staged > tile::kBankRows)
    {
        return emitOpError() << "is unable to stage " << getKeys() << " keys. Its buffers take " << staged
                             << " VMEM rows, where one bank holds " << tile::kBankRows << ".";
    }
    return success();
}

/// Verifies that the copy's stride is a whole number of rows, which the descriptors it issues step by.
LogicalResult CopyOp::verify()
{
    if (getStride() % tile::kRowBytes == 0) return success();
    return emitOpError() << "is unable to step " << getStride() << " bytes between rows. The stride must be a whole "
                         << "number of " << tile::kRowBytes << "-byte rows.";
}

/// Verifies that the load's rows stay inside its buffer.
LogicalResult VmemldOp::verify()
{
    return VerifyDescriptorBounds(*this, getBuffer().getType(), getRow(), getRows());
}

/// Verifies that the store's rows stay inside its buffer.
LogicalResult VmemstOp::verify()
{
    return VerifyDescriptorBounds(*this, getBuffer().getType(), getRow(), getRows());
}

#define GET_OP_CLASSES
#include "minitpu/minitpu/minitpu_ops.cpp.inc"
