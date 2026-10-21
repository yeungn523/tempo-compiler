/**
 * @file
 *
 * @brief Implements the expansion of minitpu tile ops into machine operations over virtual registers and buffers.
 */

#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <tuple>
#include "minitpu/minitpu/passes.h"
#include "tile_layout.h"

namespace mlir::minitpu
{
#define GEN_PASS_DEF_MINITPUTILETOVECTOR
#include "minitpu/minitpu/passes.h.inc"
}  // namespace mlir::minitpu

using namespace mlir;
using namespace mlir::minitpu;
using namespace mlir::minitpu::tile;

namespace
{
    /// Stores the DMA slot mask naming both slots.
    constexpr uint32_t kBothSlots = 3;

    /// Stores the sublane mask naming all four sublanes, so an access reads or writes a whole register.
    constexpr uint32_t kEverySublane = 15;

    /// Stores the width of a scalar register's value.
    constexpr uint32_t kScalarBits = 32;

    /// Stores the most bytes one scalar addition advances a pointer by, the reach of saddi's signed immediate.
    constexpr int64_t kAddReach = INT16_MAX;

    /// Provides the operations one expansion builds, each at the builder's insertion point.
    class Expansion
    {
      public:
        /// Prepares to build at `builder`'s insertion point with `location` on every operation.
        Expansion(OpBuilder& builder, const Location location) : _builder(builder), _location(location)
        {}

        /// Returns an index constant.
        Value Index(const int64_t value)
        {
            return arith::ConstantIndexOp::create(_builder, _location, value);
        }

        /// Returns an i32 constant, which lives in a scalar register when an operation other than an addition
        /// reads it.
        Value Scalar(const int64_t value)
        {
            return arith::ConstantIntOp::create(_builder, _location, value, kScalarBits);
        }

        /// Returns a scalar register advanced by `bytes`, in as many additions as saddi's immediate takes.
        Value Advance(Value pointer, int64_t bytes)
        {
            while (bytes != 0)
            {
                const int64_t step = std::clamp<int64_t>(bytes, -kAddReach, kAddReach);
                pointer            = arith::AddIOp::create(_builder, _location, pointer, Scalar(step));
                bytes -= step;
            }
            return pointer;
        }

        /// Returns `value` read afresh when it is a launch argument, which smovarg reads at any point, so a value used
        /// again after a long loop holds no scalar register across it; returns any other value as it is.
        Value Reread(Value value)
        {
            auto argument = value.getDefiningOp<ArgOp>();
            if (!argument) return value;
            return ArgOp::create(_builder, _location, value.getType(), argument.getIndex()).getResult();
        }

        /// Returns a VMEM buffer of `rows` rows.
        Value Buffer(const uint32_t rows)
        {
            return AllocOp::create(_builder, _location, VmemType::get(_builder.getContext(), rows), nullptr)
                .getBuffer();
        }

        /// Issues one descriptor loading `rows` rows from `pointer` into `buffer` at `row`, on `slot`.
        void
        Load(Value buffer, const int64_t row, Value pointer, Value stride, const uint32_t rows, const uint32_t slot)
        {
            VmemldOp::create(_builder, _location, buffer, Index(row), pointer, stride, rows, slot);
        }

        /// Issues one descriptor draining `rows` rows of `buffer` from `row` to `pointer`, on `slot`.
        void
        Drain(Value buffer, const int64_t row, Value pointer, Value stride, const uint32_t rows, const uint32_t slot)
        {
            VmemstOp::create(_builder, _location, buffer, Index(row), pointer, stride, rows, slot);
        }

        /// Blocks until every DMA slot `mask` names reports done.
        void Flush(const uint32_t mask)
        {
            FlushSlotOp::create(_builder, _location, mask);
        }

        /// Loads `rows` consecutive host rows from `pointer` into `buffer`, one descriptor per chunk on alternating
        /// slots, then waits for every slot it used.
        void Stage(Value buffer, Value pointer, Value stride, const uint32_t rows)
        {
            uint32_t issued = 0;
            for (uint32_t row = 0, descriptor = 0; row < rows; row += kChunkRows, ++descriptor)
            {
                const uint32_t slot = descriptor % 2;
                if (descriptor >= 2) Flush(1U << slot);
                Load(buffer, row, pointer, stride, std::min(kChunkRows, rows - row), slot);
                if (row + kChunkRows < rows) pointer = Advance(pointer, int64_t {kChunkRows} * kRowBytes);
                issued |= 1U << slot;
            }
            Flush(issued);
        }

        /// Returns the vector register loaded from `row` of `buffer`.
        Value LoadRegister(Value buffer, Value row)
        {
            return VldOp::create(_builder, _location, VRegType::get(_builder.getContext()), buffer, row, kEverySublane)
                .getResult();
        }

        /// Returns `induction` times `scale` plus `offset`, as a row the AGU adds.
        Value Row(Value induction, const int64_t scale, const int64_t offset)
        {
            Value scaled = arith::MulIOp::create(_builder, _location, induction, Index(scale));
            return arith::AddIOp::create(_builder, _location, scaled, Index(offset));
        }

        /// Returns the builder every operation is created through.
        OpBuilder& get_builder()
        {
            return _builder;
        }

        /// Returns the location every operation carries.
        [[nodiscard]]
        Location get_location() const
        {
            return _location;
        }

      private:
        /// Stores the builder every operation is created through.
        OpBuilder& _builder;

        /// Stores the location every operation carries.
        Location _location;
    };

    /// Stores the buffers one product stages.
    struct MatmulBuffers
    {
        Value start;                   ///< The zero tile, or the bias row, the accumulators start from.
        Value activations;             ///< One segment of the block's activations, resident while it is walked.
        std::array<Value, 2> weights;  ///< The two weight chunks, alternating so one lands while the other is pushed.
        Value output;                  ///< One column tile of the product, staged for its drain.
    };

    /**
     * @brief Builds one chunk's k-tile loop: a weight tile loaded, eight activation registers pushed and eight results
     * accumulated per k-tile.
     *
     * @param expansion the expansion building the loop.
     * @param buffers the product's buffers.
     * @param chunk the chunk's index within its segment, which offsets every activation row.
     * @param tiles the k-tiles the chunk covers, fewer than a whole chunk only for a segment's last.
     * @param accumulators the accumulators entering the chunk.
     * @returns the accumulators leaving the chunk.
     */
    llvm::SmallVector<Value> BuildChunk(
        Expansion& expansion,
        const MatmulBuffers& buffers,
        const uint32_t chunk,
        const uint32_t tiles,
        llvm::ArrayRef<Value> accumulators
    )
    {
        OpBuilder& builder      = expansion.get_builder();
        const Location location = expansion.get_location();
        auto loop               = scf::ForOp::create(
            builder,
            location,
            expansion.Index(0),      // lower bound
            expansion.Index(tiles),  // upper bound
            expansion.Index(1),      // step
            ValueRange(accumulators)
        );
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(loop.getBody());
        Value tile          = loop.getInductionVar();
        Value weight_buffer = buffers.weights[chunk % 2];

        // Loads the k-tile's 16x16 weight tile, four rows per register, into four registers the allocator keeps in a
        // row.
        std::array<Value, kWeightRegisters> quarters {};
        for (uint32_t quarter = 0; quarter < kWeightRegisters; ++quarter)
        {
            quarters[quarter] =
                expansion.LoadRegister(weight_buffer, expansion.Row(tile, kTile, int64_t {quarter} * kSublanes));
        }
        VmatloadOp::create(builder, location, quarters[0], quarters[1], quarters[2], quarters[3]);

        // Pushes all eight token groups before popping, which fills the output FIFO's 32 entries and no more.
        const int64_t chunk_row = int64_t {chunk} * kChunkTiles * kTokens;
        for (uint32_t group = 0; group < kTokenGroups; ++group)
        {
            Value row = expansion.Row(tile, kTokens, chunk_row + int64_t {group} * kSublanes);
            VmatpushOp::create(builder, location, expansion.LoadRegister(buffers.activations, row));
        }
        llvm::SmallVector<Value> next;
        for (uint32_t group = 0; group < kTokenGroups; ++group)
        {
            Value product = VmatpopOp::create(builder, location, VRegType::get(builder.getContext()));
            Value carried = loop.getRegionIterArgs()[group];
            next.push_back(VaddOp::create(builder, location, carried.getType(), carried, product));
        }
        scf::YieldOp::create(builder, location, next);
        return llvm::SmallVector<Value>(loop.getResults());
    }

    /**
     * @brief Passes the resident activations through the GELU in place, once for every column tile that reads them.
     *
     * @param expansion the expansion building the loop.
     * @param activations the activation buffer.
     * @param rows the rows the buffer holds.
     */
    void ActivateInPlace(Expansion& expansion, Value activations, const uint32_t rows)
    {
        OpBuilder& builder      = expansion.get_builder();
        const Location location = expansion.get_location();
        auto loop               = scf::ForOp::create(
            builder,
            location,
            expansion.Index(0),                 // lower bound
            expansion.Index(rows / kSublanes),  // upper bound
            expansion.Index(1)                  // step
        );
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(loop.getBody());
        Value row       = expansion.Row(loop.getInductionVar(), kSublanes, 0);
        Value staged    = expansion.LoadRegister(activations, row);
        Value activated = VgeluOp::create(builder, location, staged.getType(), staged);
        VstOp::create(builder, location, activated, activations, row, kEverySublane);
    }

    /**
     * @brief Builds one segment of a product's contraction: its activations staged, then every column tile's
     * accumulators carried through the segment's chunks and drained.
     *
     * @details The first segment starts from the zero tile or the bias row, a later one from the partial product the
     * segment before drained.
     *
     * @param expansion the expansion building the segment.
     * @param product the product being expanded.
     * @param buffers the product's buffers.
     * @param first the segment's first k-tile.
     * @param tiles the k-tiles the segment covers.
     * @param stride the scalar holding the bytes between consecutive host rows.
     */
    void BuildSegment(
        Expansion& expansion,
        MatmulOp product,
        const MatmulBuffers& buffers,
        const uint32_t first,
        const uint32_t tiles,
        Value stride
    )
    {
        OpBuilder& builder             = expansion.get_builder();
        const Location location        = expansion.get_location();
        const uint32_t depth_tiles     = product.getDepth() / kTile;
        const uint32_t chunks          = (tiles + kChunkTiles - 1) / kChunkTiles;
        const uint32_t activation_rows = kTokens * tiles;
        const bool resumes             = first != 0;

        // Reads each pointer afresh in a resuming segment, since the scalar file holds no argument across a segment.
        const auto pointer = [&](Value value) {
            return resumes ? expansion.Reread(value) : value;
        };
        Value activations = expansion.Advance(pointer(product.getActivations()), int64_t {first} * kTokens * kRowBytes);
        expansion.Stage(buffers.activations, activations, stride, activation_rows);
        if (product.getActivate()) ActivateInPlace(expansion, buffers.activations, activation_rows);
        Value weights = expansion.Advance(pointer(product.getWeights()), int64_t {first} * kTile * kRowBytes);
        Value drain   = expansion.Advance(pointer(product.getDrain()), int64_t {kZeroRows} * kRowBytes);

        auto columns = scf::ForOp::create(
            builder,
            location,
            expansion.Index(0),                             // lower bound
            expansion.Index(product.getColumns() / kTile),  // upper bound
            expansion.Index(1),                             // step
            ValueRange {weights, drain}
        );
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(columns.getBody());
        Value column_weights = columns.getRegionIterArgs()[0];
        Value output         = columns.getRegionIterArgs()[1];

        // Starts the accumulators from the zero tile or the bias row, or in a resuming segment from the partial product
        // read back beside the first chunk.
        const auto load_first = [&] {
            expansion.Load(buffers.weights[0], 0, column_weights, stride, std::min(tiles, kChunkTiles) * kTile, 0);
            column_weights = expansion.Advance(column_weights, int64_t {kChunkRows} * kRowBytes);
        };
        llvm::SmallVector<Value> accumulators;
        if (resumes)
        {
            load_first();
            expansion.Load(buffers.output, 0, output, stride, kTokens, 1);
            expansion.Flush(1U << 1);
            for (uint32_t group = 0; group < kTokenGroups; ++group)
            {
                Value row = expansion.Index(int64_t {group} * kSublanes);
                accumulators.push_back(expansion.LoadRegister(buffers.output, row));
            }
        }
        else
        {
            Value start =
                product.getBias() ? expansion.Row(columns.getInductionVar(), kBroadcastRows, 0) : expansion.Index(0);
            for (uint32_t group = 0; group < kTokenGroups; ++group)
            {
                accumulators.push_back(expansion.LoadRegister(buffers.start, start));
            }
            load_first();
        }

        // Prefetches each next chunk into the other buffer before pushing the current.
        for (uint32_t chunk = 0; chunk < chunks; ++chunk)
        {
            const uint32_t covered = std::min(kChunkTiles, tiles - chunk * kChunkTiles);
            expansion.Flush(1U << (chunk % 2));
            if (chunk + 1 < chunks)
            {
                const uint32_t next = (chunk + 1) % 2;
                const uint32_t rows = std::min(kChunkTiles, tiles - (chunk + 1) * kChunkTiles) * kTile;
                expansion.Load(buffers.weights[next], 0, column_weights, stride, rows, next);
                column_weights = expansion.Advance(column_weights, int64_t {kChunkRows} * kRowBytes);
            }
            accumulators = BuildChunk(expansion, buffers, chunk, covered, accumulators);
        }

        // Stores the column tile and drains it on the slot the last chunk freed, waiting before the next reuses it.
        for (uint32_t group = 0; group < kTokenGroups; ++group)
        {
            VstOp::create(
                builder,
                location,
                accumulators[group],
                buffers.output,
                expansion.Index(int64_t {group} * kSublanes),
                kEverySublane
            );
        }
        const uint32_t drain_slot = chunks % 2;
        expansion.Drain(buffers.output, 0, output, stride, kTokens, drain_slot);
        expansion.Flush(1U << drain_slot);
        output = expansion.Advance(output, int64_t {kTokens} * kRowBytes);

        // Steps the weights from where the segment's chunks leave them to the segment's start in the next column tile.
        const int64_t walked = int64_t {chunks} * kChunkRows;
        column_weights       = expansion.Advance(column_weights, (int64_t {depth_tiles} * kTile - walked) * kRowBytes);
        scf::YieldOp::create(builder, location, ValueRange {column_weights, output});
    }

    /// Expands one minitpu.matmul into the machine operations that compute it, then erases it.
    void ExpandMatmul(MatmulOp product)
    {
        OpBuilder builder(product);
        Expansion expansion(builder, product.getLoc());
        const uint32_t depth_tiles = product.getDepth() / kTile;
        const bool bias            = product.getBias() != nullptr;
        const uint32_t segment     = MatmulSegmentTiles(product.getDepth(), product.getColumns(), bias);
        Value stride               = expansion.Scalar(kRowBytes);

        // Starts the accumulators from the bias row when the product adds one, its tiles one register each, and
        // otherwise from the zero tile heading the drain buffer.
        const uint32_t bias_rows = kBroadcastRows * (product.getColumns() / kTile);
        MatmulBuffers buffers {
            .start       = expansion.Buffer(bias ? bias_rows : kZeroRows),
            .activations = expansion.Buffer(kTokens * segment),
            .weights     = {expansion.Buffer(kChunkRows), expansion.Buffer(kChunkRows)},
            .output      = expansion.Buffer(kTokens),
        };
        if (bias)
        {
            expansion.Stage(buffers.start, product.getBias(), stride, bias_rows);
        }
        else
        {
            expansion.Stage(buffers.start, product.getDrain(), stride, kZeroRows);
        }
        for (uint32_t first = 0; first < depth_tiles; first += segment)
        {
            BuildSegment(expansion, product, buffers, first, std::min(segment, depth_tiles - first), stride);
        }
        product.erase();
    }

    /// Stores the buffers one sum stages and the stride its descriptors share.
    struct AddBuffers
    {
        std::array<Value, 2> left;   ///< The two left chunks, alternating so the next lands while the current is added.
        std::array<Value, 2> right;  ///< The two right chunks, or null values when the right operand broadcasts.
        Value bias;                  ///< The whole broadcast row, or a null value when the right operand streams.
        Value output;                ///< One chunk of the sum, staged for its drain.
        Value stride;                ///< The scalar holding the bytes between consecutive host rows.
    };

    /// Stores the host pointers a sum advances by one chunk at a time.
    struct AddPointers
    {
        Value left;    ///< The next left chunk to load.
        Value right;   ///< The next right chunk to load, unused once a broadcast row is staged.
        Value output;  ///< The next output chunk to drain.
    };

    /**
     * @brief Builds one chunk of a sum: the wait for its operands, the next chunk's prefetch, the additions and the
     * drain.
     *
     * @details A broadcast sum loads on slot 0 and drains on slot 1. A streamed sum loads on both slots and drains on
     * slot 0 once the prefetch on it lands.
     *
     * @param expansion the expansion building the chunk.
     * @param buffers the sum's buffers.
     * @param pointers the host pointers, each advanced past what the chunk moves.
     * @param parity the operand buffer the chunk reads, the prefetch filling the other.
     * @param bias_row the broadcast row's register row for each of the chunk's feature tiles.
     * @param first determines whether no drain precedes the chunk, whose slot then carries nothing to wait on.
     * @param prefetch determines whether a next chunk follows, whose operands the chunk loads.
     */
    void BuildAddChunk(
        Expansion& expansion,
        const AddBuffers& buffers,
        AddPointers& pointers,
        const uint32_t parity,
        const llvm::function_ref<Value(uint32_t)> bias_row,
        const bool first,
        const bool prefetch
    )
    {
        OpBuilder& builder      = expansion.get_builder();
        const Location location = expansion.get_location();
        const bool broadcast    = buffers.bias != nullptr;
        const int64_t chunk     = int64_t {kChunkRows} * kRowBytes;
        expansion.Flush(broadcast && first ? 1 : kBothSlots);
        if (prefetch)
        {
            expansion.Load(buffers.left[1 - parity], 0, pointers.left, buffers.stride, kChunkRows, 0);
            pointers.left = expansion.Advance(pointers.left, chunk);
            if (!broadcast)
            {
                expansion.Load(buffers.right[1 - parity], 0, pointers.right, buffers.stride, kChunkRows, 1);
                pointers.right = expansion.Advance(pointers.right, chunk);
            }
        }

        // Adds one feature tile at a time, its loads placed before the previous tile's stores. The packer keeps every
        // VMEM access behind an earlier store, so this order lets the one VMEM port run the loads and stores back to
        // back while the additions wait out the load latency beside them. The broadcast row is read once per tile.
        llvm::SmallVector<Value> pending;
        auto store_pending = [&](const uint32_t tile) {
            for (uint32_t group = 0; group < pending.size(); ++group)
            {
                Value row = expansion.Index(int64_t {tile} * kTokens + int64_t {group} * kSublanes);
                VstOp::create(builder, location, pending[group], buffers.output, row, kEverySublane);
            }
            pending.clear();
        };
        for (uint32_t tile = 0; tile < kChunkFeatureTiles; ++tile)
        {
            Value shared = broadcast ? expansion.LoadRegister(buffers.bias, bias_row(tile)) : Value {};
            llvm::SmallVector<std::pair<Value, Value>, kTokenGroups> operands;
            for (uint32_t group = 0; group < kTokenGroups; ++group)
            {
                Value row   = expansion.Index(int64_t {tile} * kTokens + int64_t {group} * kSublanes);
                Value left  = expansion.LoadRegister(buffers.left[parity], row);
                Value right = broadcast ? shared : expansion.LoadRegister(buffers.right[parity], row);
                operands.emplace_back(left, right);
            }
            if (tile > 0) store_pending(tile - 1);
            for (const auto& [left, right] : operands)
            {
                pending.push_back(VaddOp::create(builder, location, left.getType(), left, right));
            }
        }
        store_pending(kChunkFeatureTiles - 1);

        const uint32_t drain_slot = broadcast ? 1 : 0;
        if (!broadcast && prefetch) expansion.Flush(1);
        expansion.Drain(buffers.output, 0, pointers.output, buffers.stride, kChunkRows, drain_slot);
        pointers.output = expansion.Advance(pointers.output, chunk);
    }

    /// Expands one minitpu.add into the machine operations that compute it, then erases it.
    void ExpandAdd(AddOp sum)
    {
        OpBuilder builder(sum);
        Expansion expansion(builder, sum.getLoc());
        const Location location = sum.getLoc();
        const bool broadcast    = sum.getBroadcast();
        const uint32_t tiles    = sum.getFeatures() / kTile;
        const uint32_t chunks   = tiles / kChunkFeatureTiles;
        const int64_t chunk     = int64_t {kChunkRows} * kRowBytes;

        AddBuffers buffers {
            .left   = {expansion.Buffer(kChunkRows), expansion.Buffer(kChunkRows)},
            .right  = {},
            .bias   = nullptr,
            .output = expansion.Buffer(kChunkRows),
            .stride = expansion.Scalar(kRowBytes),
        };
        AddPointers pointers {.left = sum.getLeft(), .right = sum.getRight(), .output = sum.getOutput()};

        // Stages the whole broadcast row on alternating slots, or the right operand's first chunk beside the left's.
        if (broadcast)
        {
            const uint32_t rows = kBroadcastRows * tiles;
            buffers.bias        = expansion.Buffer(rows);
            expansion.Stage(buffers.bias, pointers.right, buffers.stride, rows);
        }
        else
        {
            buffers.right = {expansion.Buffer(kChunkRows), expansion.Buffer(kChunkRows)};
            expansion.Load(buffers.right[0], 0, pointers.right, buffers.stride, kChunkRows, 1);
            pointers.right = expansion.Advance(pointers.right, chunk);
        }
        expansion.Load(buffers.left[0], 0, pointers.left, buffers.stride, kChunkRows, 0);
        pointers.left = expansion.Advance(pointers.left, chunk);

        // Returns the broadcast row of a feature tile in a chunk the expansion knows.
        auto fixed_row = [&](const uint32_t index) {
            return [&expansion, index](const uint32_t tile) {
                return expansion.Index(int64_t {index * kChunkFeatureTiles + tile} * kBroadcastRows);
            };
        };

        // Peels the first chunk, whose drain slot carries nothing to wait on, and the last, which prefetches nothing.
        BuildAddChunk(expansion, buffers, pointers, 0, fixed_row(0), /*first=*/true, /*prefetch=*/true);
        if (chunks > 2)
        {
            llvm::SmallVector<Value> carried {pointers.left, pointers.output};
            if (!broadcast) carried.push_back(pointers.right);
            auto pairs = scf::ForOp::create(
                builder,
                location,
                expansion.Index(0),                 // lower bound
                expansion.Index((chunks - 2) / 2),  // upper bound
                expansion.Index(1),                 // step
                ValueRange(carried)
            );
            {
                OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToStart(pairs.getBody());
                Value pair            = pairs.getInductionVar();
                AddPointers advancing = {
                    .left   = pairs.getRegionIterArgs()[0],
                    .right  = broadcast ? pointers.right : pairs.getRegionIterArgs()[2],
                    .output = pairs.getRegionIterArgs()[1],
                };

                // Covers chunks 2 * pair + 1 and 2 * pair + 2, which read the second buffer and then the first.
                for (uint32_t step = 0; step < 2; ++step)
                {
                    auto walked_row = [&expansion, pair, step](const uint32_t tile) {
                        const int64_t offset = int64_t {(1 + step) * kChunkFeatureTiles + tile} * kBroadcastRows;
                        return expansion.Row(pair, 2 * kChunkFeatureTiles * kBroadcastRows, offset);
                    };
                    BuildAddChunk(
                        expansion,
                        buffers,
                        advancing,
                        1 - step,
                        walked_row,
                        /*first=*/false,
                        /*prefetch=*/true
                    );
                }
                llvm::SmallVector<Value> next {advancing.left, advancing.output};
                if (!broadcast) next.push_back(advancing.right);
                scf::YieldOp::create(builder, location, next);
            }
            pointers.left   = pairs.getResult(0);
            pointers.output = pairs.getResult(1);
            if (!broadcast) pointers.right = pairs.getResult(2);
        }
        BuildAddChunk(expansion, buffers, pointers, 1, fixed_row(chunks - 1), /*first=*/false, /*prefetch=*/false);
        expansion.Flush(broadcast ? 2 : 1);
        sum.erase();
    }

    /// Stores the constants block's row holding the tile of the entering maximum.
    constexpr int64_t kInitialRow = 0;

    /// Stores the constants block's row holding the tile of zeros.
    constexpr int64_t kZeroRow = kTile;

    /// Stores the constants block's row holding the first mask tile.
    constexpr int64_t kMaskRow = int64_t {kAttentionStatedTiles} * kTile;

    /// Stores the sublane mask naming sublane zero alone.
    constexpr uint32_t kFirstSublane = 1;

    /// Stores the tiles one batch of value pushes covers, which together fill the MXU output FIFO's 32 entries.
    constexpr uint32_t kPushTiles = kTokenGroups / kSublanes;

    /// Stores the bytes one half of a token block's tile takes, being one query block's rows.
    constexpr int64_t kHalfTileBytes = int64_t {kQueryBlock} * kRowBytes;

    /// Returns the result of a vector operation of two operands.
    template <typename BinaryOp>
    Value Binary(Expansion& expansion, Value left, Value right)
    {
        return BinaryOp::create(expansion.get_builder(), expansion.get_location(), left.getType(), left, right);
    }

    /// Returns the result of a vector operation of one operand.
    template <typename UnaryOp>
    Value Unary(Expansion& expansion, Value operand)
    {
        return UnaryOp::create(expansion.get_builder(), expansion.get_location(), operand.getType(), operand);
    }

    /// Returns the tile held in four registers, register r carrying rows 4r through 4r + 3, transposed.
    std::array<Value, kSublanes> Transpose(Expansion& expansion, const std::array<Value, kSublanes>& quarters)
    {
        OpBuilder& builder      = expansion.get_builder();
        const Location location = expansion.get_location();
        for (uint32_t quarter = 0; quarter < kSublanes; ++quarter)
        {
            VtxinOp::create(builder, location, quarters[quarter], quarter);
        }
        std::array<Value, kSublanes> columns {};
        for (uint32_t quarter = 0; quarter < kSublanes; ++quarter)
        {
            columns[quarter] = VtxoutOp::create(builder, location, VRegType::get(builder.getContext()), quarter);
        }
        return columns;
    }

    /// Issues descriptors on alternating slots, waiting on a slot before it carries another and on both at the end.
    class DescriptorQueue
    {
      public:
        /// Prepares to issue through `expansion`.
        explicit DescriptorQueue(Expansion& expansion) : _expansion(expansion)
        {}

        /**
         * @brief Moves `count` tiles of `rows` rows each, `byte_step` bytes apart in host memory and `row_step` rows
         * apart in `buffer`, advancing `pointer` in place onto the last tile.
         *
         * @param drain determines whether the tiles move from the buffer to host memory rather than into it.
         * @param buffer the VMEM buffer.
         * @param first_row the buffer row the first tile moves to or from.
         * @param row_step the buffer rows between consecutive tiles.
         * @param pointer the host address of the first tile.
         * @param byte_step the host bytes between consecutive tiles.
         * @param count the tiles moved.
         * @param rows the rows each tile spans.
         * @param stride the scalar holding the bytes between consecutive rows.
         * @returns the host address of the last tile.
         */
        Value Move(
            const bool drain,
            Value buffer,
            const int64_t first_row,
            const int64_t row_step,
            Value pointer,
            const int64_t byte_step,
            const uint32_t count,
            const uint32_t rows,
            Value stride
        )
        {
            for (uint32_t tile = 0; tile < count; ++tile)
            {
                const uint32_t slot = _issued_count % 2;
                if (_issued_count >= 2) _expansion.Flush(1U << slot);
                const int64_t row = first_row + int64_t {tile} * row_step;
                if (drain) _expansion.Drain(buffer, row, pointer, stride, rows, slot);
                else _expansion.Load(buffer, row, pointer, stride, rows, slot);
                _used |= 1U << slot;
                ++_issued_count;
                if (tile + 1 < count) pointer = _expansion.Advance(pointer, byte_step);
            }
            return pointer;
        }

        /// Waits for every slot a descriptor was issued on.
        void Finish()
        {
            if (_used != 0) _expansion.Flush(_used);
        }

      private:
        /// Stores the expansion the descriptors are built through.
        Expansion& _expansion;

        /// Stores the descriptors issued so far.
        uint32_t _issued_count = 0;

        /// Stores the slots a descriptor was issued on, one bit each.
        uint32_t _used = 0;
    };

    /// Stores the host layout one attention walks, in bytes.
    struct AttentionLayout
    {
        uint32_t tiles;             ///< The tiles one head spans.
        uint32_t heads;             ///< The heads the tensors carry.
        uint32_t keys;              ///< The keys each query attends over at most.
        bool causal;                ///< Whether the keys are the queries tensor's own, each query reading up to itself.
        int64_t query_block_bytes;  ///< The bytes one token block of the queries tensor spans.
        int64_t key_block_bytes;    ///< The bytes one token block of the tensor holding the keys spans.
        int64_t model_bytes;        ///< The bytes from one head's keys to its values, and from its queries to its keys.
        int64_t head_bytes;         ///< The bytes from one head's tiles to the next head's.
    };

    /// Stores the VMEM buffers and the stride one attention shares across its heads and query blocks.
    struct AttentionBuffers
    {
        Value constants;   ///< The entering maximum, zeros and the mask tiles.
        Value queries;     ///< One query block's queries, a tile per 16 head features.
        Value transposed;  ///< The same queries transposed, each tile the stationary operand of a score product.
        Value operands;    ///< One key token block's keys and then its values, a tile per 16 head features.
        Value scratch;     ///< The rows the per-query folds turn through.
        Value output;      ///< One query block's output, staged for its drain.
        Value stride;      ///< The scalar holding the bytes between consecutive rows.
    };

    /// Stores what one query block carries across its keys.
    struct AttentionState
    {
        llvm::SmallVector<Value>
            output;     ///< The output's registers, transposed: four head features each, tile after tile.
        Value maximum;  ///< The running maximum, one query per lane, every sublane the same.
        Value total;    ///< The running sum, laid out as the maximum.

        /// Returns every carried value, in the order a loop carries them.
        [[nodiscard]]
        llvm::SmallVector<Value> Values() const
        {
            llvm::SmallVector<Value> values(output);
            values.push_back(maximum);
            values.push_back(total);
            return values;
        }

        /// Takes every carried value from `values`, in the order Values() lists them.
        void Assign(const ValueRange values)
        {
            for (size_t index = 0; index < output.size(); ++index) output[index] = values[index];
            maximum = values[output.size()];
            total   = values[output.size() + 1];
        }
    };

    /// Stores the scratch rows one fold turns through: two partials, each read back past itself, and the spread.
    constexpr int64_t kFoldRows = 24;

    /**
     * @brief Returns the four registers' elements combined across registers and sublanes into every sublane, one
     * query per lane, turning the partials through VMEM, whose consecutive rows are the only path between sublanes.
     *
     * @param expansion the expansion building the fold.
     * @param scratch the buffer the fold turns through.
     * @param first_row the first of the `kFoldRows` scratch rows the fold owns.
     * @param quarters the registers combined.
     * @returns the combined register, every sublane the same.
     */
    template <typename CombineOp>
    Value FoldSublanes(
        Expansion& expansion,
        Value scratch,
        const int64_t first_row,
        const std::array<Value, kSublanes>& quarters
    )
    {
        OpBuilder& builder      = expansion.get_builder();
        const Location location = expansion.get_location();
        Value value             = Binary<CombineOp>(
            expansion,
            Binary<CombineOp>(expansion, quarters[0], quarters[1]),
            Binary<CombineOp>(expansion, quarters[2], quarters[3])
        );

        // Halves the sublanes twice, reading each partial back two and then one rows on. The rows read past each
        // partial are stale, so only sublane zero carries the whole fold.
        constexpr std::array<std::pair<int64_t, int64_t>, 2> kFolds {
            {{0, 2}, {8, 1}}
        };
        for (const auto& [row, distance] : kFolds)
        {
            VstOp::create(builder, location, value, scratch, expansion.Index(first_row + row), kEverySublane);
            Value shifted = expansion.LoadRegister(scratch, expansion.Index(first_row + row + distance));
            value         = Binary<CombineOp>(expansion, value, shifted);
        }

        // Spreads sublane zero over four consecutive rows, which one load reads back as agreeing sublanes.
        constexpr int64_t kSpread = 16;
        for (uint32_t sublane = 0; sublane < kSublanes; ++sublane)
        {
            VstOp::create(
                builder,
                location,
                value,
                scratch,
                expansion.Index(first_row + kSpread + sublane),
                kFirstSublane
            );
        }
        return expansion.LoadRegister(scratch, expansion.Index(first_row + kSpread));
    }

    /**
     * @brief Builds one block of 16 keys attended by one block of 16 queries under the online softmax.
     *
     * @details The block is held transposed, one query per lane, so each query's maximum and sum combine across
     * registers and sublanes rather than lanes. The probabilities stay stationary for every value tile.
     *
     * @param expansion the expansion building the block.
     * @param buffers the attention's buffers.
     * @param tiles the tiles one head spans.
     * @param state the query block's carried values, updated past the block.
     * @param half the half of the staged key token block the block reads.
     * @param counter the key loop's counter the mask tile is walked by, or a null value for a fixed mask tile.
     * @param mask_row the constants row of the block's mask tile, the counter's walk aside, or a negative row for none.
     */
    void BuildKeyBlock(
        Expansion& expansion,
        const AttentionBuffers& buffers,
        const uint32_t tiles,
        AttentionState& state,
        const uint32_t half,
        Value counter,
        const int64_t mask_row
    )
    {
        OpBuilder& builder      = expansion.get_builder();
        const Location location = expansion.get_location();
        const Type vector       = VRegType::get(builder.getContext());
        const int64_t key_row   = int64_t {half} * kQueryBlock;
        auto tile_at            = [&](Value buffer, const int64_t first_row) {
            std::array<Value, kSublanes> quarters {};
            for (uint32_t quarter = 0; quarter < kSublanes; ++quarter)
            {
                quarters[quarter] = expansion.LoadRegister(buffer, expansion.Index(first_row + quarter * kSublanes));
            }
            return quarters;
        };
        auto stationary = [&](const std::array<Value, kSublanes>& quarters) {
            VmatloadOp::create(builder, location, quarters[0], quarters[1], quarters[2], quarters[3]);
        };

        std::array<Value, kSublanes> scores {};
        for (uint32_t tile = 0; tile < tiles; ++tile)
        {
            stationary(tile_at(buffers.transposed, int64_t {tile} * kQueryBlock));
            for (uint32_t group = 0; group < kSublanes; ++group)
            {
                const int64_t row = int64_t {tile} * kTokens + key_row + group * kSublanes;
                VmatpushOp::create(builder, location, expansion.LoadRegister(buffers.operands, expansion.Index(row)));
            }
            for (uint32_t group = 0; group < kSublanes; ++group)
            {
                Value product = VmatpopOp::create(builder, location, vector);
                scores[group] = tile == 0 ? product : Binary<VaddOp>(expansion, scores[group], product);
            }
        }
        if (mask_row >= 0)
        {
            for (uint32_t group = 0; group < kSublanes; ++group)
            {
                const int64_t row = mask_row + group * kSublanes;
                Value at          = counter ? expansion.Row(counter, 2 * kTile, row) : expansion.Index(row);
                scores[group] = Binary<VaddOp>(expansion, scores[group], expansion.LoadRegister(buffers.constants, at));
            }
        }

        // Takes the running maximum past the block's, and the rescale factor, the probabilities and their sum against
        // it.
        Value block   = FoldSublanes<VmaxOp>(expansion, buffers.scratch, 0, scores);
        Value maximum = Binary<VmaxOp>(expansion, state.maximum, block);
        Value rescale = Unary<VexpOp>(expansion, Binary<VsubOp>(expansion, state.maximum, maximum));
        for (Value& held : state.output) held = Binary<VmulOp>(expansion, held, rescale);
        std::array<Value, kSublanes> probabilities {};
        for (uint32_t group = 0; group < kSublanes; ++group)
        {
            probabilities[group] = Unary<VexpOp>(expansion, Binary<VsubOp>(expansion, scores[group], maximum));
        }
        Value sum     = FoldSublanes<VaddOp>(expansion, buffers.scratch, kFoldRows, probabilities);
        state.total   = Binary<VaddOp>(expansion, Binary<VmulOp>(expansion, state.total, rescale), sum);
        state.maximum = maximum;

        // Holds the probabilities stationary and pushes every value tile against them, transposed, in batches the
        // output FIFO holds without a pop between.
        stationary(probabilities);
        for (uint32_t first = 0; first < tiles; first += kPushTiles)
        {
            const uint32_t batch = std::min(kPushTiles, tiles - first);
            for (uint32_t tile = first; tile < first + batch; ++tile)
            {
                const std::array<Value, kSublanes> values =
                    Transpose(expansion, tile_at(buffers.operands, int64_t {tiles + tile} * kTokens + key_row));
                for (uint32_t group = 0; group < kSublanes; ++group)
                {
                    VmatpushOp::create(builder, location, values[group]);
                }
            }
            for (uint32_t tile = first; tile < first + batch; ++tile)
            {
                for (uint32_t group = 0; group < kSublanes; ++group)
                {
                    Value& held = state.output[tile * kSublanes + group];
                    held        = Binary<VaddOp>(expansion, held, VmatpopOp::create(builder, location, vector));
                }
            }
        }
    }

    /**
     * @brief Stages one key token block's keys and then its values.
     *
     * @param expansion the expansion building the descriptors.
     * @param buffers the attention's buffers.
     * @param layout the host layout.
     * @param pointer the host address of the token block's first key tile of the head.
     * @param advance determines whether the pointer moves on to the next token block's, which another stage reads.
     * @returns the pointer, moved on when `advance` is set.
     */
    Value StageKeys(
        Expansion& expansion,
        const AttentionBuffers& buffers,
        const AttentionLayout& layout,
        Value pointer,
        const bool advance
    )
    {
        const int64_t walked = int64_t {layout.tiles - 1} * kTokenTileBytes;
        DescriptorQueue queue(expansion);
        pointer = queue.Move(
            false,
            buffers.operands,
            0,
            kTokens,
            pointer,
            kTokenTileBytes,
            layout.tiles,
            kTokens,
            buffers.stride
        );
        pointer = expansion.Advance(pointer, layout.model_bytes - walked);
        pointer = queue.Move(
            false,
            buffers.operands,
            int64_t {layout.tiles} * kTokens,
            kTokens,
            pointer,
            kTokenTileBytes,
            layout.tiles,
            kTokens,
            buffers.stride
        );
        queue.Finish();
        return advance ? expansion.Advance(pointer, layout.key_block_bytes - layout.model_bytes - walked) : pointer;
    }

    /**
     * @brief Builds a loop over `count` key token blocks, each staging its keys and values and attending both halves.
     *
     * @param expansion the expansion building the loop.
     * @param buffers the attention's buffers.
     * @param layout the host layout.
     * @param state the query block's carried values, updated past the loop.
     * @param pointer the host address of the first token block's first key tile of the head, moved past the last.
     * @param count the token blocks the loop walks.
     * @param masked determines whether each key block adds the mask tile the constants block holds for it.
     */
    void BuildKeyLoop(
        Expansion& expansion,
        const AttentionBuffers& buffers,
        const AttentionLayout& layout,
        AttentionState& state,
        Value& pointer,
        const uint32_t count,
        const bool masked
    )
    {
        OpBuilder& builder      = expansion.get_builder();
        const Location location = expansion.get_location();
        llvm::SmallVector<Value> carried {pointer};
        llvm::append_range(carried, state.Values());
        auto loop = scf::ForOp::create(
            builder,
            location,
            expansion.Index(0),      // lower bound
            expansion.Index(count),  // upper bound
            expansion.Index(1),      // step
            ValueRange(carried)
        );
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(loop.getBody());
            AttentionState walked = state;
            walked.Assign(loop.getRegionIterArgs().drop_front());
            Value next = StageKeys(expansion, buffers, layout, loop.getRegionIterArgs()[0], true);
            for (uint32_t half = 0; half < 2; ++half)
            {
                const int64_t mask_row = masked ? kMaskRow + int64_t {half} * kTile : -1;
                BuildKeyBlock(expansion, buffers, layout.tiles, walked, half, loop.getInductionVar(), mask_row);
            }
            llvm::SmallVector<Value> yielded {next};
            llvm::append_range(yielded, walked.Values());
            scf::YieldOp::create(builder, location, yielded);
        }
        pointer = loop.getResult(0);
        state.Assign(loop.getResults().drop_front());
    }

    /**
     * @brief Builds one query block: its queries staged and transposed, its keys attended, and its output normalised
     * and drained over its queries.
     *
     * @param expansion the expansion building the block.
     * @param buffers the attention's buffers.
     * @param layout the host layout.
     * @param queries the host address of the head's first query tile.
     * @param keys the host address of the head's first cached key tile, unused by a causal attention.
     * @param block the query block's index, two per token block.
     */
    void BuildQueryBlock(
        Expansion& expansion,
        const AttentionBuffers& buffers,
        const AttentionLayout& layout,
        Value queries,
        Value keys,
        const uint32_t block
    )
    {
        OpBuilder& builder      = expansion.get_builder();
        const Location location = expansion.get_location();
        const uint32_t token    = block / 2;
        const uint32_t half     = block % 2;
        const int64_t offset    = int64_t {token} * layout.query_block_bytes + int64_t {half} * kHalfTileBytes;

        DescriptorQueue loads(expansion);
        loads.Move(
            false,
            buffers.queries,
            0,
            kQueryBlock,
            expansion.Advance(queries, offset),
            kTokenTileBytes,
            layout.tiles,
            kQueryBlock,
            buffers.stride
        );
        loads.Finish();
        for (uint32_t tile = 0; tile < layout.tiles; ++tile)
        {
            std::array<Value, kSublanes> quarters {};
            for (uint32_t group = 0; group < kSublanes; ++group)
            {
                const int64_t row = int64_t {tile} * kQueryBlock + group * kSublanes;
                quarters[group]   = expansion.LoadRegister(buffers.queries, expansion.Index(row));
            }
            const std::array<Value, kSublanes> transposed = Transpose(expansion, quarters);
            for (uint32_t group = 0; group < kSublanes; ++group)
            {
                const int64_t row = int64_t {tile} * kQueryBlock + group * kSublanes;
                VstOp::create(
                    builder,
                    location,
                    transposed[group],
                    buffers.transposed,
                    expansion.Index(row),
                    kEverySublane
                );
            }
        }

        AttentionState state;
        state.maximum = expansion.LoadRegister(buffers.constants, expansion.Index(kInitialRow));
        state.total   = expansion.LoadRegister(buffers.constants, expansion.Index(kZeroRow));
        for (uint32_t index = 0; index < layout.tiles * kSublanes; ++index)
        {
            state.output.push_back(expansion.LoadRegister(buffers.constants, expansion.Index(kZeroRow)));
        }

        // Attends a causal block's full token blocks, then the token block holding its own queries, whose keys past
        // each query are masked by the diagonal tile and whose second half the first query half never reads.
        if (layout.causal)
        {
            Value pointer = expansion.Advance(queries, layout.model_bytes);
            if (token > 0) BuildKeyLoop(expansion, buffers, layout, state, pointer, token, false);
            StageKeys(expansion, buffers, layout, pointer, false);
            if (half == 1) BuildKeyBlock(expansion, buffers, layout.tiles, state, 0, Value {}, -1);
            BuildKeyBlock(expansion, buffers, layout.tiles, state, half, Value {}, kMaskRow);
        }
        else
        {
            Value pointer = keys;
            BuildKeyLoop(expansion, buffers, layout, state, pointer, layout.keys / kTokens, true);
        }

        // Normalises each query's output by its sum, transposes it back to one query per row, and drains it over the
        // queries.
        Value reciprocal = Unary<VrecipOp>(expansion, state.total);
        for (uint32_t tile = 0; tile < layout.tiles; ++tile)
        {
            std::array<Value, kSublanes> normalised {};
            for (uint32_t group = 0; group < kSublanes; ++group)
            {
                normalised[group] = Binary<VmulOp>(expansion, state.output[tile * kSublanes + group], reciprocal);
            }
            const std::array<Value, kSublanes> rows = Transpose(expansion, normalised);
            for (uint32_t group = 0; group < kSublanes; ++group)
            {
                const int64_t row = int64_t {tile} * kQueryBlock + group * kSublanes;
                VstOp::create(builder, location, rows[group], buffers.output, expansion.Index(row), kEverySublane);
            }
        }
        DescriptorQueue drains(expansion);
        drains.Move(
            true,
            buffers.output,
            0,
            kQueryBlock,
            expansion.Advance(queries, offset),
            kTokenTileBytes,
            layout.tiles,
            kQueryBlock,
            buffers.stride
        );
        drains.Finish();
    }

    /// Expands one minitpu.attention into the machine operations that compute it, then erases it.
    void ExpandAttention(AttentionOp attention)
    {
        OpBuilder builder(attention);
        Expansion expansion(builder, attention.getLoc());
        const Location location = attention.getLoc();
        const uint32_t tiles    = attention.getHead() / kTile;
        const uint32_t heads    = attention.getHeads();
        const bool causal       = attention.getCausal();
        const int64_t model     = int64_t {heads} * tiles * kTokenTileBytes;
        const AttentionLayout layout {
            .tiles             = tiles,
            .heads             = heads,
            .keys              = attention.getKeys(),
            .causal            = causal,
            .query_block_bytes = kZeroTileBytes + 3 * model,
            .key_block_bytes   = kZeroTileBytes + (causal ? 3 : 2) * model,
            .model_bytes       = model,
            .head_bytes        = int64_t {tiles} * kTokenTileBytes,
        };
        const uint32_t constant_rows = AttentionConstantRows(layout.keys, causal);
        AttentionBuffers buffers {
            .constants  = expansion.Buffer(constant_rows),
            .queries    = expansion.Buffer(tiles * kQueryBlock),
            .transposed = expansion.Buffer(tiles * kQueryBlock),
            .operands   = expansion.Buffer(2 * tiles * kTokens),
            .scratch    = expansion.Buffer(kAttentionScratchRows),
            .output     = expansion.Buffer(tiles * kQueryBlock),
            .stride     = expansion.Scalar(kRowBytes),
        };
        expansion.Stage(buffers.constants, attention.getConstants(), buffers.stride, constant_rows);

        // Walks the heads, each one head's tiles past the last in every tensor.
        llvm::SmallVector<Value> carried {attention.getQueries()};
        if (!causal) carried.push_back(attention.getCache());
        auto walk = scf::ForOp::create(
            builder,
            location,
            expansion.Index(0),      // lower bound
            expansion.Index(heads),  // upper bound
            expansion.Index(1),      // step
            ValueRange(carried)
        );
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(walk.getBody());
            Value queries         = walk.getRegionIterArgs()[0];
            Value keys            = causal ? Value {} : walk.getRegionIterArgs()[1];
            const uint32_t blocks = causal ? layout.keys / kQueryBlock : 1;
            for (uint32_t block = 0; block < blocks; ++block)
            {
                BuildQueryBlock(expansion, buffers, layout, queries, keys, block);
            }
            llvm::SmallVector<Value> next {expansion.Advance(queries, layout.head_bytes)};
            if (!causal) next.push_back(expansion.Advance(keys, layout.head_bytes));
            scf::YieldOp::create(builder, location, next);
        }
        attention.erase();
    }

    /// Stores the two buffers a copy alternates between and the stride its descriptors share.
    struct CopyBuffers
    {
        std::array<Value, 2> chunks;  ///< The two chunk buffers, one loading while the other drains.
        Value stride;                 ///< The scalar holding the bytes between consecutive rows at both ends.
    };

    /// Stores the host pointers a copy advances by one chunk at a time.
    struct CopyPointers
    {
        Value source;       ///< The next chunk to load.
        Value destination;  ///< The next chunk to drain.
    };

    /// Stores what one step of a copy moves.
    struct CopyStep
    {
        uint32_t parity;           ///< The buffer the step loads into, the drain reading the other.
        uint32_t load_rows;        ///< The rows the step loads, zero once every chunk is loaded.
        uint32_t drain_rows;       ///< The rows of the previous chunk the step drains.
        bool first;                ///< Whether no drain precedes the step, whose slot then carries nothing to wait on.
        bool advance_source;       ///< Whether another load follows, which reads past this one.
        bool advance_destination;  ///< Whether another drain follows, which writes past this one.
    };

    /**
     * @brief Builds one step of a copy: the wait, the next chunk's load on slot 0 and the previous chunk's drain on
     * slot 1, so the drain runs beside the load.
     *
     * @param expansion the expansion building the step.
     * @param buffers the copy's buffers.
     * @param pointers the host pointers, each advanced past what the step moves when another step follows.
     * @param step what the step moves.
     * @param advance the bytes one chunk spans at both ends.
     */
    void BuildCopyStep(
        Expansion& expansion,
        const CopyBuffers& buffers,
        CopyPointers& pointers,
        const CopyStep& step,
        const int64_t advance
    )
    {
        expansion.Flush(step.first ? 1 : kBothSlots);
        if (step.load_rows > 0)
        {
            expansion.Load(buffers.chunks[step.parity], 0, pointers.source, buffers.stride, step.load_rows, 0);
            if (step.advance_source) pointers.source = expansion.Advance(pointers.source, advance);
        }
        expansion.Drain(buffers.chunks[1 - step.parity], 0, pointers.destination, buffers.stride, step.drain_rows, 1);
        if (step.advance_destination) pointers.destination = expansion.Advance(pointers.destination, advance);
    }

    /// Expands one minitpu.copy into the descriptors that move it, then erases it.
    void ExpandCopy(CopyOp copy)
    {
        OpBuilder builder(copy);
        Expansion expansion(builder, copy.getLoc());
        const Location location = copy.getLoc();
        const uint32_t rows     = copy.getRows();
        const uint32_t chunks   = (rows + kChunkRows - 1) / kChunkRows;
        const int64_t advance   = int64_t {kChunkRows} * copy.getStride();
        const uint32_t span     = std::min(rows, kChunkRows);
        auto rows_of            = [&](const uint32_t chunk) {
            return std::min(kChunkRows, rows - chunk * kChunkRows);
        };

        CopyBuffers buffers {
            .chunks = {expansion.Buffer(span), chunks > 1 ? expansion.Buffer(span) : Value {}},
            .stride = expansion.Scalar(copy.getStride()),
        };
        CopyPointers pointers {.source = copy.getSource(), .destination = copy.getDestination()};

        // Step c loads chunk c, when one remains, and drains chunk c - 1. Every step is the same but its buffer
        // parity, its load size and whether its pointers advance, so the steps past the first two whose loads are
        // whole chunks that another load follows run as a loop over pairs.
        auto step_of = [&](const uint32_t index) {
            return CopyStep {
                .parity              = index % 2,
                .load_rows           = index < chunks ? rows_of(index) : 0,
                .drain_rows          = rows_of(index - 1),
                .first               = index == 1,
                .advance_source      = index + 1 < chunks,
                .advance_destination = index < chunks,
            };
        };
        expansion.Load(buffers.chunks[0], 0, pointers.source, buffers.stride, rows_of(0), 0);
        if (chunks > 1) pointers.source = expansion.Advance(pointers.source, advance);
        uint32_t index = 1;
        if (chunks > 1) BuildCopyStep(expansion, buffers, pointers, step_of(index++), advance);
        const uint32_t pairs = chunks > 3 ? (chunks - 3) / 2 : 0;
        if (pairs > 0)
        {
            auto loop = scf::ForOp::create(
                builder,
                location,
                expansion.Index(0),      // lower bound
                expansion.Index(pairs),  // upper bound
                expansion.Index(1),      // step
                ValueRange {pointers.source, pointers.destination}
            );
            {
                OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToStart(loop.getBody());
                CopyPointers advancing {
                    .source      = loop.getRegionIterArgs()[0],
                    .destination = loop.getRegionIterArgs()[1],
                };
                BuildCopyStep(expansion, buffers, advancing, step_of(index), advance);
                BuildCopyStep(expansion, buffers, advancing, step_of(index + 1), advance);
                scf::YieldOp::create(builder, location, ValueRange {advancing.source, advancing.destination});
            }
            pointers.source      = loop.getResult(0);
            pointers.destination = loop.getResult(1);
            index += 2 * pairs;
        }
        for (; index <= chunks; ++index) BuildCopyStep(expansion, buffers, pointers, step_of(index), advance);
        expansion.Flush(2);
        copy.erase();
    }

    /// Stores the most registers one LayerNorm token holds resident, beyond which it streams them from its buffer.
    constexpr uint32_t kResidentRegisters = 12;

    /// Stores the registers a streamed LayerNorm token sums at a time.
    constexpr uint32_t kStreamGroup = 8;

    /// Stores the registers one LayerNorm token normalises in a batch, its gamma and beta loads placed before the
    /// previous batch's stores.
    constexpr uint32_t kNormBatch = 4;

    /// Stores the buffers and registers a LayerNorm shares across its tokens.
    struct NormBuffers
    {
        std::array<Value, 2> rows;    ///< The two token buffers, alternating so the next token lands while one is read.
        std::array<Value, 2> output;  ///< The two output buffers, alternating so one drains while the next is written.
        Value constants;              ///< Gamma, beta, the reciprocal width and epsilon.
        Value inverse;                ///< The register holding the reciprocal width.
        Value epsilon;                ///< The register holding epsilon.
        Value stride;                 ///< The scalar holding the bytes between one token's rows in host memory.
        uint32_t tiles = 0;           ///< The rows one token takes, one per 16 features.
    };

    /// Stores the host pointers a LayerNorm advances by one token at a time.
    struct NormPointers
    {
        Value rows;    ///< The next token to gather.
        Value output;  ///< The next token to drain.
    };

    /// Returns the elementwise sum of `values`, added as a balanced tree so its depth grows with their logarithm.
    Value SumTree(OpBuilder& builder, const Location location, llvm::SmallVector<Value> values)
    {
        while (values.size() > 1)
        {
            llvm::SmallVector<Value> next;
            for (size_t index = 0; index + 1 < values.size(); index += 2)
            {
                next.push_back(
                    VaddOp::create(builder, location, values[index].getType(), values[index], values[index + 1])
                );
            }
            if (values.size() % 2 != 0) next.push_back(values.back());
            values = std::move(next);
        }
        return values.front();
    }

    /**
     * @brief Builds one token of a LayerNorm: the wait for its rows, the next token's prefetch, the normalisation and
     * the drain.
     *
     * @details Each register holds 64 of the token's features, so the mean and the variance each sum the registers as
     * a tree before one vredsum. The variance sums the centred squares, as the host reference does.
     *
     * @param expansion the expansion building the token.
     * @param buffers the LayerNorm's buffers and shared registers.
     * @param pointers the host pointers, each advanced past what the token moves.
     * @param parity the token and output buffers the token uses, the prefetch filling the other token buffer.
     * @param first determines whether no drain precedes the token, whose slot then carries nothing to wait on.
     * @param prefetch determines whether a next token follows, whose rows the token gathers.
     */
    void BuildNormToken(
        Expansion& expansion,
        const NormBuffers& buffers,
        NormPointers& pointers,
        const uint32_t parity,
        const bool first,
        const bool prefetch
    )
    {
        OpBuilder& builder      = expansion.get_builder();
        const Location location = expansion.get_location();
        const Type vector       = VRegType::get(builder.getContext());
        const uint32_t count    = buffers.tiles / kSublanes;
        expansion.Flush(1);
        if (prefetch)
        {
            expansion.Load(buffers.rows[1 - parity], 0, pointers.rows, buffers.stride, buffers.tiles, 0);
            pointers.rows = expansion.Advance(pointers.rows, kRowBytes);
        }

        // Holds a narrow token's registers and centred values for the whole token, and streams a wider one's from its
        // buffer in groups, recomputing each centred value where it is used, so the register file bounds no width.
        const bool resident = count <= kResidentRegisters;
        auto load           = [&](const uint32_t index) {
            return expansion.LoadRegister(buffers.rows[parity], expansion.Index(int64_t {index} * kSublanes));
        };
        auto reduce = [&](const llvm::function_ref<Value(uint32_t)> term) {
            llvm::SmallVector<Value> partials;
            const uint32_t group = resident ? count : kStreamGroup;
            for (uint32_t first = 0; first < count; first += group)
            {
                llvm::SmallVector<Value> terms;
                for (uint32_t index = first; index < std::min(first + group, count); ++index)
                {
                    terms.push_back(term(index));
                }
                partials.push_back(SumTree(builder, location, terms));
            }
            return VredsumOp::create(builder, location, vector, SumTree(builder, location, partials)).getResult();
        };

        llvm::SmallVector<Value> values;
        if (resident)
        {
            for (uint32_t index = 0; index < count; ++index) values.push_back(load(index));
        }
        Value total = reduce([&](const uint32_t index) {
            return resident ? values[index] : load(index);
        });
        Value mean  = VmulOp::create(builder, location, vector, total, buffers.inverse);

        llvm::SmallVector<Value> centred;
        auto centre = [&](Value value) {
            return VsubOp::create(builder, location, vector, value, mean).getResult();
        };
        Value spread   = reduce([&](const uint32_t index) {
            if (resident) centred.push_back(centre(values[index]));
            Value deviation = resident ? centred.back() : centre(load(index));
            return VmulOp::create(builder, location, vector, deviation, deviation).getResult();
        });
        Value variance = VmulOp::create(builder, location, vector, spread, buffers.inverse);
        Value floored  = VaddOp::create(builder, location, vector, variance, buffers.epsilon);
        Value scale    = VrsqrtOp::create(builder, location, vector, floored);

        // Scales and shifts a batch at a time, its gamma and beta loads placed before the previous batch's stores, as
        // the sum orders its tiles, so the stores never hold the loads back.
        llvm::SmallVector<std::pair<uint32_t, Value>> pending;
        auto store_pending = [&]() {
            for (const auto& [index, result] : pending)
            {
                Value row = expansion.Index(int64_t {index} * kSublanes);
                VstOp::create(builder, location, result, buffers.output[parity], row, kEverySublane);
            }
            pending.clear();
        };
        for (uint32_t start = 0; start < count; start += kNormBatch)
        {
            const uint32_t end = std::min(start + kNormBatch, count);
            llvm::SmallVector<std::tuple<Value, Value, Value>, kNormBatch> affine;
            for (uint32_t index = start; index < end; ++index)
            {
                const int64_t row = int64_t {index} * kSublanes;
                affine.emplace_back(
                    expansion.LoadRegister(buffers.constants, expansion.Index(row)),
                    expansion.LoadRegister(buffers.constants, expansion.Index(buffers.tiles + row)),
                    resident ? Value {} : load(index)
                );
            }
            store_pending();
            for (uint32_t index = start; index < end; ++index)
            {
                const auto& [gamma, beta, value] = affine[index - start];
                Value deviation                  = resident ? centred[index] : centre(value);
                Value normalised                 = VmulOp::create(builder, location, vector, deviation, scale);
                Value scaled                     = VmulOp::create(builder, location, vector, normalised, gamma);
                pending.emplace_back(index, VaddOp::create(builder, location, vector, scaled, beta));
            }
        }
        store_pending();

        if (!first) expansion.Flush(2);
        expansion.Drain(buffers.output[parity], 0, pointers.output, buffers.stride, buffers.tiles, 1);
        pointers.output = expansion.Advance(pointers.output, kRowBytes);
    }

    /// Expands one minitpu.layernorm into the machine operations that compute it, then erases it.
    void ExpandLayernorm(LayernormOp norm)
    {
        OpBuilder builder(norm);
        Expansion expansion(builder, norm.getLoc());
        const Location location      = norm.getLoc();
        const uint32_t tiles         = norm.getFeatures() / kTile;
        const uint32_t constant_rows = NormConstantRows(norm.getFeatures());

        NormBuffers buffers {
            .rows      = {expansion.Buffer(tiles), expansion.Buffer(tiles)},
            .output    = {expansion.Buffer(tiles), expansion.Buffer(tiles)},
            .constants = expansion.Buffer(constant_rows),
            .inverse   = nullptr,
            .epsilon   = nullptr,
            .stride    = nullptr,
            .tiles     = tiles,
        };

        // Stages the constants block on alternating slots, then keeps its two scalars in registers for every token.
        expansion.Stage(buffers.constants, norm.getConstants(), expansion.Scalar(kRowBytes), constant_rows);
        buffers.inverse = expansion.LoadRegister(buffers.constants, expansion.Index(2 * tiles));
        buffers.epsilon = expansion.LoadRegister(buffers.constants, expansion.Index(2 * tiles + kScalarRows));

        // Gathers each token's rows from every feature tile, which lie one block of tokens apart in host memory.
        buffers.stride = expansion.Scalar(int64_t {kTokens} * kRowBytes);
        NormPointers pointers {.rows = norm.getRows(), .output = norm.getOutput()};
        expansion.Load(buffers.rows[0], 0, pointers.rows, buffers.stride, tiles, 0);
        pointers.rows = expansion.Advance(pointers.rows, kRowBytes);

        // Peels the first token, whose drain slot carries nothing to wait on, and the last, which prefetches nothing.
        BuildNormToken(expansion, buffers, pointers, 0, /*first=*/true, /*prefetch=*/true);
        auto pairs = scf::ForOp::create(
            builder,
            location,
            expansion.Index(0),                  // lower bound
            expansion.Index((kTokens - 2) / 2),  // upper bound
            expansion.Index(1),                  // step
            ValueRange {pointers.rows, pointers.output}
        );
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(pairs.getBody());
            NormPointers advancing {.rows = pairs.getRegionIterArgs()[0], .output = pairs.getRegionIterArgs()[1]};
            BuildNormToken(expansion, buffers, advancing, 1, /*first=*/false, /*prefetch=*/true);
            BuildNormToken(expansion, buffers, advancing, 0, /*first=*/false, /*prefetch=*/true);
            scf::YieldOp::create(builder, location, ValueRange {advancing.rows, advancing.output});
        }
        pointers = {.rows = pairs.getResult(0), .output = pairs.getResult(1)};
        BuildNormToken(expansion, buffers, pointers, 1, /*first=*/false, /*prefetch=*/false);
        expansion.Flush(2);
        norm.erase();
    }

    /// Provides the minitpu-tile-to-vector pass over one kernel.
    class TileToVectorPass final : public mlir::minitpu::impl::MinitpuTileToVectorBase<TileToVectorPass>
    {
      public:
        /// Expands every tile op in the kernel.
        void runOnOperation() override
        {
            llvm::SmallVector<MatmulOp> products;
            getOperation().walk([&](MatmulOp product) {
                products.push_back(product);
            });
            for (MatmulOp product : products) ExpandMatmul(product);
            llvm::SmallVector<AddOp> sums;
            getOperation().walk([&](AddOp sum) {
                sums.push_back(sum);
            });
            for (AddOp sum : sums) ExpandAdd(sum);
            llvm::SmallVector<LayernormOp> norms;
            getOperation().walk([&](LayernormOp norm) {
                norms.push_back(norm);
            });
            for (LayernormOp norm : norms) ExpandLayernorm(norm);
            llvm::SmallVector<AttentionOp> attentions;
            getOperation().walk([&](AttentionOp attention) {
                attentions.push_back(attention);
            });
            for (AttentionOp attention : attentions) ExpandAttention(attention);
            llvm::SmallVector<CopyOp> copies;
            getOperation().walk([&](CopyOp copy) {
                copies.push_back(copy);
            });
            for (CopyOp copy : copies) ExpandCopy(copy);
        }
    };
}  // namespace
