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

        /// Returns a scalar register advanced by `bytes`.
        Value Advance(Value pointer, const int64_t bytes)
        {
            return arith::AddIOp::create(_builder, _location, pointer, Scalar(bytes));
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
        Value activations;             ///< The block's activations, resident for the whole launch.
        std::array<Value, 2> weights;  ///< The two weight chunks, alternating so one lands while the other is pushed.
        Value output;                  ///< One column tile of the product, staged for its drain.
    };

    /**
     * @brief Builds one chunk's k-tile loop: a weight tile loaded, eight activation registers pushed and eight results
     * accumulated per k-tile.
     *
     * @param expansion the expansion building the loop.
     * @param buffers the product's buffers.
     * @param chunk the chunk's index within the contraction, which offsets every activation row.
     * @param accumulators the accumulators entering the chunk.
     * @returns the accumulators leaving the chunk.
     */
    llvm::SmallVector<Value> BuildChunk(
        Expansion& expansion,
        const MatmulBuffers& buffers,
        const uint32_t chunk,
        llvm::ArrayRef<Value> accumulators
    )
    {
        OpBuilder& builder      = expansion.get_builder();
        const Location location = expansion.get_location();
        auto loop               = scf::ForOp::create(
            builder,
            location,
            expansion.Index(0),            // lower bound
            expansion.Index(kChunkTiles),  // upper bound
            expansion.Index(1),            // step
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

    /// Expands one minitpu.matmul into the machine operations that compute it, then erases it.
    void ExpandMatmul(MatmulOp product)
    {
        OpBuilder builder(product);
        Expansion expansion(builder, product.getLoc());
        const Location location        = product.getLoc();
        const uint32_t depth           = product.getDepth();
        const uint32_t chunks          = depth / kChunkRows;
        const uint32_t activation_rows = kTokens * (depth / kTile);
        Value stride                   = expansion.Scalar(kRowBytes);

        // Starts the accumulators from the bias row when the product adds one, its tiles one register each, and
        // otherwise from the zero tile heading the drain buffer.
        Value bias               = product.getBias();
        const uint32_t bias_rows = kBroadcastRows * (product.getColumns() / kTile);
        MatmulBuffers buffers {
            .start       = expansion.Buffer(bias ? bias_rows : kZeroRows),
            .activations = expansion.Buffer(activation_rows),
            .weights     = {expansion.Buffer(kChunkRows), expansion.Buffer(kChunkRows)},
            .output      = expansion.Buffer(kTokens),
        };
        if (bias)
        {
            expansion.Stage(buffers.start, bias, stride, bias_rows);
        }
        else
        {
            expansion.Stage(buffers.start, product.getDrain(), stride, kZeroRows);
        }
        expansion.Stage(buffers.activations, product.getActivations(), stride, activation_rows);
        if (product.getActivate()) ActivateInPlace(expansion, buffers.activations, activation_rows);
        Value drain = expansion.Advance(product.getDrain(), int64_t {kZeroRows} * kRowBytes);

        auto columns = scf::ForOp::create(
            builder,
            location,
            expansion.Index(0),                             // lower bound
            expansion.Index(product.getColumns() / kTile),  // upper bound
            expansion.Index(1),                             // step
            ValueRange {product.getWeights(), drain}
        );
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(columns.getBody());
            Value weights = columns.getRegionIterArgs()[0];
            Value output  = columns.getRegionIterArgs()[1];

            llvm::SmallVector<Value> accumulators;
            Value start = bias ? expansion.Row(columns.getInductionVar(), kBroadcastRows, 0) : expansion.Index(0);
            for (uint32_t group = 0; group < kTokenGroups; ++group)
            {
                accumulators.push_back(expansion.LoadRegister(buffers.start, start));
            }

            // Loads the first chunk, then prefetches each next chunk into the other buffer before pushing the current.
            expansion.Load(buffers.weights[0], 0, weights, stride, kChunkRows, 0);
            weights = expansion.Advance(weights, int64_t {kChunkRows} * kRowBytes);
            for (uint32_t chunk = 0; chunk < chunks; ++chunk)
            {
                expansion.Flush(1U << (chunk % 2));
                if (chunk + 1 < chunks)
                {
                    const uint32_t next = (chunk + 1) % 2;
                    expansion.Load(buffers.weights[next], 0, weights, stride, kChunkRows, next);
                    weights = expansion.Advance(weights, int64_t {kChunkRows} * kRowBytes);
                }
                accumulators = BuildChunk(expansion, buffers, chunk, accumulators);
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
            scf::YieldOp::create(builder, location, ValueRange {weights, output});
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
     * @details A broadcast sum loads on slot 0 and drains on slot 1, so the drain runs beside the next prefetch. A
     * streamed sum loads its two operands on both slots and drains on slot 0 once the prefetch on it lands.
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
     * @details The token's rows land one per 16 features, so each register holds 64 of its features and vredsum, which
     * sums a whole register, reduces over features alone. The mean and the variance each sum the token's registers as a
     * tree before one vredsum, and the variance sums the centred values' squares, as the host reference does. Gamma and
     * beta line up with the token's registers row for row.
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
        }
    };
}  // namespace
