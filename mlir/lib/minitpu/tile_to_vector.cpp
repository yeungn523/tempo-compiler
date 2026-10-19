/**
 * @file
 *
 * @brief Implements the expansion of minitpu tile ops into machine operations over virtual registers and buffers.
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <array>
#include "matmul_layout.h"
#include "minitpu/minitpu/passes.h"

namespace mlir::minitpu
{
#define GEN_PASS_DEF_MINITPUTILETOVECTOR
#include "minitpu/minitpu/passes.h.inc"
}  // namespace mlir::minitpu

using namespace mlir;
using namespace mlir::minitpu;
using namespace mlir::minitpu::matmul;

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
        Value zero;                    ///< The zero tile the accumulators start from.
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

        MatmulBuffers buffers {
            .zero        = expansion.Buffer(kZeroRows),
            .activations = expansion.Buffer(activation_rows),
            .weights     = {expansion.Buffer(kChunkRows), expansion.Buffer(kChunkRows)},
            .output      = expansion.Buffer(kTokens),
        };

        // Stages the zero tile heading the drain buffer, then the block's activations on alternating slots.
        expansion.Load(buffers.zero, 0, product.getDrain(), stride, kZeroRows, 0);
        expansion.Flush(1);
        Value activations          = product.getActivations();
        const uint32_t descriptors = activation_rows / kChunkRows;
        for (uint32_t descriptor = 0; descriptor < descriptors; ++descriptor)
        {
            const uint32_t slot = descriptor % 2;
            if (descriptor >= 2) expansion.Flush(1U << slot);
            expansion
                .Load(buffers.activations, int64_t {descriptor} * kChunkRows, activations, stride, kChunkRows, slot);
            activations = expansion.Advance(activations, int64_t {kChunkRows} * kRowBytes);
        }
        expansion.Flush(descriptors > 1 ? kBothSlots : 1);
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
            for (uint32_t group = 0; group < kTokenGroups; ++group)
            {
                accumulators.push_back(expansion.LoadRegister(buffers.zero, expansion.Index(0)));
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
        }
    };
}  // namespace
