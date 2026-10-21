/**
 * @file
 *
 * @brief Implements the conversion of Triton kernels into minitpu kernels, each kernel becoming the one tile op that
 * computes it.
 *
 * A kernel is read as the host launches its image: once per block of tokens, so the loop over token blocks is the
 * host's and the tile op covers one block. Each loop is identified by what it walks, the rows or the columns of an
 * operand, since the canonicalizer folds away a loop the kernel's shape gives one iteration. Every address the kernel
 * computes is resolved into an affine form over the tensor's indices and the enclosing loops' counters, and is held to
 * the dense row-major layout the host packs each operand from. A product's masks state its extents, past which the host
 * pads its operands with zeros to the whole tiles the tile op covers. A kernel the conversion does not recognise, or
 * one that addresses an operand any other way, is refused with a diagnostic naming what differs.
 */

#include <llvm/ADT/StringExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Matchers.h>
#include <triton/Dialect/Triton/IR/Dialect.h>
#include <cmath>
#include <optional>
#include <utility>
#include "minitpu/minitpu/passes.h"
#include "tile_layout.h"

namespace mlir::minitpu
{
#define GEN_PASS_DEF_CONVERTTRITONTOMINITPU
#include "minitpu/minitpu/passes.h.inc"
}  // namespace mlir::minitpu

using namespace mlir;
using namespace mlir::minitpu;
namespace tt = mlir::triton;

namespace
{
    /// Stores the tolerance a recognised float constant is matched to, relative to its value.
    constexpr double kConstantTolerance = 1e-6;

    /// Stores the factor the exact GELU scales its argument by outside the error function.
    constexpr double kGeluHalf = 0.5;

    /// Stores the divisor the exact GELU scales its argument by.
    constexpr double kRootTwo = 1.4142135623730951;

    /// Stores the highest constant read as masking a score, which no real score reaches.
    constexpr double kMaskBound = -1.0e9;

    /// Stores the dimension a broadcast feature row is repeated along, being the tokens.
    constexpr int32_t kTokenAxis = 0;

    /// Stores the dimension a broadcast per-token value is repeated along, being the features.
    constexpr int32_t kFeatureAxis = 1;

    /// Stores a loop counter and the elements an address advances per unit of it.
    using Counter = std::pair<Value, int64_t>;

    /// Stores an address or an offset as an affine form, in elements, over its tensor's indices and the loop counters
    /// it depends on.
    struct Affine
    {
        Value base;                     ///< The pointer argument the address is based on, or null for an offset.
        int64_t constant = 0;           ///< The constant term.
        SmallVector<int64_t> strides;   ///< The elements one index along each tensor dimension advances.
        SmallVector<Counter> counters;  ///< The elements one unit of each loop counter advances.

        /// Returns the elements one unit of `counter` advances, zero when the form does not depend on it.
        [[nodiscard]]
        int64_t Step(Value counter) const
        {
            for (const auto& [known, step] : counters)
            {
                if (known == counter) return step;
            }
            return 0;
        }

        /// Adds `step` elements per unit of `counter`.
        void AddStep(Value counter, const int64_t step)
        {
            for (auto* it = counters.begin(); it != counters.end(); ++it)
            {
                if (it->first != counter) continue;
                it->second += step;
                if (it->second == 0) counters.erase(it);
                return;
            }
            if (step != 0) counters.emplace_back(counter, step);
        }
    };

    /// Stores a loop's constant bounds, which the conversion holds to start at zero and to step evenly to its end.
    struct Loop
    {
        scf::ForOp op;      ///< The loop.
        Value counter;      ///< The loop's counter.
        int64_t upper = 0;  ///< The bound its counter stops before.
        int64_t step  = 0;  ///< The amount its counter advances per iteration.

        /// Returns the extent its iterations cover, past its bound when the last iteration overruns it.
        [[nodiscard]]
        int64_t Covered() const
        {
            return (upper + step - 1) / step * step;
        }
    };

    /// Returns `value`'s tensor rank, zero for a scalar.
    int64_t Rank(Value value)
    {
        const auto tensor = dyn_cast<RankedTensorType>(value.getType());
        return tensor ? tensor.getRank() : 0;
    }

    /// Returns `value`'s tensor shape, empty for a scalar.
    ArrayRef<int64_t> Shape(Value value)
    {
        const auto tensor = dyn_cast<RankedTensorType>(value.getType());
        return tensor ? tensor.getShape() : ArrayRef<int64_t> {};
    }

    /// Returns the element type a pointer or a tensor of pointers points to.
    Type Pointee(Value pointer)
    {
        Type type          = getElementTypeOrSelf(pointer.getType());
        const auto pointee = dyn_cast<tt::PointerType>(type);
        return pointee ? pointee.getPointeeType() : Type {};
    }

    /**
     * @brief Resolves `operation`'s bounds, holding them to constants that start at zero and step evenly to the end.
     *
     * @param operation the loop.
     * @param ragged determines whether the last iteration may overrun the bound, which a kernel masking the overrun
     * reads as the padding the host lays out.
     * @returns the bounds, or failure with a diagnostic on the loop.
     */
    FailureOr<Loop> Bounds(scf::ForOp operation, const bool ragged = false)
    {
        const std::optional<int64_t> lower = getConstantIntValue(operation.getLowerBound());
        const std::optional<int64_t> upper = getConstantIntValue(operation.getUpperBound());
        const std::optional<int64_t> step  = getConstantIntValue(operation.getStep());
        if (!lower || !upper || !step || *lower != 0 || *step <= 0 || (!ragged && *upper % *step != 0))
        {
            return operation.emitError()
                << "is a loop the conversion is unable to read. Each loop must run from 0 to a "
                << "constant bound in a constant step dividing it, which fixes the shape the "
                << "image covers.";
        }
        return Loop {.op = operation, .counter = operation.getInductionVar(), .upper = *upper, .step = *step};
    }

    /// Returns the sum of two forms of one rank, based on whichever pointer either is based on.
    Affine Added(Affine left, const Affine& right)
    {
        left.base = left.base ? left.base : right.base;
        left.constant += right.constant;
        for (const auto [dimension, stride] : llvm::enumerate(right.strides)) left.strides[dimension] += stride;
        for (const auto& [counter, step] : right.counters) left.AddStep(counter, step);
        return left;
    }

    /// Returns `form` negated, which must be based on no pointer.
    Affine Negated(Affine form)
    {
        form.constant = -form.constant;
        for (int64_t& stride : form.strides) stride = -stride;
        for (auto& [counter, step] : form.counters) step = -step;
        return form;
    }

    /// Resolves the addresses and offsets a kernel computes into affine forms.
    class AddressAnalysis
    {
      public:
        /**
         * @brief Resolves `value` into an affine form.
         *
         * @param value the address or offset.
         * @returns the form, or failure with a diagnostic on the step the resolution is unable to follow.
         */
        // NOLINTNEXTLINE(misc-no-recursion): recurses once per operation computing the address, a short chain.
        FailureOr<Affine> Of(Value value)
        {
            if (const auto known = _known.find(value); known != _known.end()) return known->second;
            FailureOr<Affine> form = Compute(value);
            if (failed(form)) return failure();

            // Drops the stride of every dimension one index wide, whose only index is zero.
            for (const auto [dimension, extent] : llvm::enumerate(Shape(value)))
            {
                if (extent == 1) form->strides[dimension] = 0;
            }
            _known.try_emplace(value, *form);
            return form;
        }

      private:
        /// Stores every form resolved so far.
        llvm::DenseMap<Value, Affine> _known;

        /// Returns the sum of two forms, at most one of which is based on a pointer.
        static FailureOr<Affine> Sum(Operation* operation, const Affine& left, const Affine& right)
        {
            if (left.base && right.base)
            {
                return operation->emitError() << "adds two pointers, which no address the conversion reads does.";
            }
            return Added(left, right);
        }

        /// Returns whether `form` is a constant alone.
        static bool IsConstant(const Affine& form)
        {
            return !form.base && form.counters.empty() && llvm::all_of(form.strides, [](int64_t s) {
                return s == 0;
            });
        }

        /// Resolves a value carried around `loop`, which must advance by a constant each iteration.
        // NOLINTNEXTLINE(misc-no-recursion): recurses once per operation computing the address, a short chain.
        FailureOr<Affine> Carried(scf::ForOp loop, BlockArgument argument)
        {
            const unsigned index = argument.getArgNumber() - loop.getNumInductionVars();
            auto advance         = loop.getYieldedValues()[index].getDefiningOp<tt::AddPtrOp>();
            if (!advance || advance.getPtr() != argument)
            {
                return loop.emitError() << "carries an address that advances by something other than a constant "
                                        << "each iteration, which the conversion is unable to follow.";
            }
            const FailureOr<Affine> delta = Of(advance.getOffset());
            const FailureOr<Loop> bounds  = Bounds(loop, true);
            if (failed(delta) || failed(bounds)) return failure();
            if (!IsConstant(*delta) || delta->constant % bounds->step != 0)
            {
                return advance.emitError() << "advances a carried address by something other than a whole multiple "
                                           << "of the loop's step, which the conversion is unable to follow.";
            }
            FailureOr<Affine> form = Of(loop.getInitArgs()[index]);
            if (failed(form)) return failure();
            form->AddStep(loop.getInductionVar(), delta->constant / bounds->step);
            return form;
        }

        /// Resolves `value` from its definition.
        // NOLINTNEXTLINE(misc-no-recursion): recurses once per operation computing the address, a short chain.
        FailureOr<Affine> Compute(Value value)
        {
            Affine form;
            form.strides.assign(Rank(value), 0);

            if (const auto argument = dyn_cast<BlockArgument>(value))
            {
                Operation* owner = argument.getOwner()->getParentOp();
                if (llvm::isa<tt::FuncOp>(owner) && llvm::isa<tt::PointerType>(value.getType()))
                {
                    form.base = value;
                    return form;
                }
                if (auto loop = dyn_cast<scf::ForOp>(owner))
                {
                    if (value != loop.getInductionVar()) return Carried(loop, argument);
                    form.counters.emplace_back(value, 1);
                    return form;
                }
                return owner->emitError() << "passes an address through a block argument the conversion is unable "
                                          << "to follow.";
            }

            APInt integer;
            if (matchPattern(value, m_ConstantInt(&integer)))
            {
                form.constant = integer.getSExtValue();
                return form;
            }

            Operation* operation = value.getDefiningOp();
            if (auto range = dyn_cast<tt::MakeRangeOp>(operation))
            {
                form.constant   = range.getStart();
                form.strides[0] = 1;
                return form;
            }
            if (auto splat = dyn_cast<tt::SplatOp>(operation))
            {
                FailureOr<Affine> scalar = Of(splat.getSrc());
                if (failed(scalar)) return failure();
                scalar->strides.assign(Rank(value), 0);
                return scalar;
            }
            if (auto expand = dyn_cast<tt::ExpandDimsOp>(operation))
            {
                FailureOr<Affine> source = Of(expand.getSrc());
                if (failed(source)) return failure();
                source->strides.insert(source->strides.begin() + expand.getAxis(), 0);
                return source;
            }
            if (auto broadcast = dyn_cast<tt::BroadcastOp>(operation)) return Of(broadcast.getSrc());
            if (auto advance = dyn_cast<tt::AddPtrOp>(operation))
            {
                const FailureOr<Affine> pointer = Of(advance.getPtr());
                const FailureOr<Affine> offset  = Of(advance.getOffset());
                if (failed(pointer) || failed(offset)) return failure();
                return Sum(operation, *pointer, *offset);
            }
            if (auto addition = dyn_cast<arith::AddIOp>(operation))
            {
                const FailureOr<Affine> left  = Of(addition.getLhs());
                const FailureOr<Affine> right = Of(addition.getRhs());
                if (failed(left) || failed(right)) return failure();
                return Sum(operation, *left, *right);
            }
            if (auto difference = dyn_cast<arith::SubIOp>(operation))
            {
                const FailureOr<Affine> left  = Of(difference.getLhs());
                const FailureOr<Affine> right = Of(difference.getRhs());
                if (failed(left) || failed(right)) return failure();
                if (right->base)
                {
                    return operation->emitError() << "subtracts a pointer, which no address the conversion reads does.";
                }
                return Sum(operation, *left, Negated(*right));
            }
            if (auto product = dyn_cast<arith::MulIOp>(operation))
            {
                FailureOr<Affine> left  = Of(product.getLhs());
                FailureOr<Affine> right = Of(product.getRhs());
                if (failed(left) || failed(right)) return failure();
                if (!IsConstant(*right)) std::swap(left, right);
                if (!IsConstant(*right) || left->base)
                {
                    return operation->emitError() << "multiplies an address by something other than a constant, "
                                                  << "which leaves it outside the affine forms the conversion reads.";
                }
                const int64_t scale = right->constant;
                left->constant *= scale;
                for (int64_t& stride : left->strides) stride *= scale;
                for (auto& [counter, step] : left->counters) step *= scale;
                return left;
            }
            return operation->emitError() << "computes an address in a way the conversion is unable to follow. An "
                                          << "address must be a pointer argument plus sums and constant multiples of "
                                          << "ranges, loop counters and constants.";
        }
    };

    /// Returns the constant value `value` holds or splats, if it is a float constant.
    std::optional<double> FloatConstant(Value value)
    {
        APFloat bound(0.0);
        if (!matchPattern(value, m_ConstantFloat(&bound))) return std::nullopt;
        bool lost = false;
        bound.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &lost);
        return bound.convertToDouble();
    }

    /// Returns whether `value` is a float constant within the matching tolerance of `expected`.
    bool IsFloat(Value value, const double expected)
    {
        const std::optional<double> constant = FloatConstant(value);
        return constant && std::abs(*constant - expected) <= kConstantTolerance * std::abs(expected);
    }

    /// Returns the operands of `value`'s definition when it is a `BinaryOp`, in both orders, or an empty list.
    template <typename BinaryOp>
    SmallVector<std::pair<Value, Value>, 2> Orders(Value value)
    {
        auto operation = value ? value.getDefiningOp<BinaryOp>() : BinaryOp {};
        if (!operation) return {};
        return {
            {operation.getLhs(), operation.getRhs()},
            {operation.getRhs(), operation.getLhs()}
        };
    }

    /// Returns the value `value` broadcasts along a new dimension `axis`, or null when it is no such broadcast.
    Value Broadcasted(Value value, const int32_t axis)
    {
        if (!value) return {};
        auto broadcast = value.getDefiningOp<tt::BroadcastOp>();
        auto expand    = (broadcast ? broadcast.getSrc() : value).getDefiningOp<tt::ExpandDimsOp>();
        return expand && static_cast<int32_t>(expand.getAxis()) == axis ? expand.getSrc() : Value {};
    }

    /// Returns the value `value` widens from, or `value` itself when it is no widening.
    Value Narrow(Value value)
    {
        auto widened = value ? value.getDefiningOp<arith::ExtFOp>() : arith::ExtFOp {};
        return widened ? widened.getIn() : value;
    }

    /**
     * @brief Returns the load defining `value`, or a null op.
     *
     * @param value the loaded value.
     * @param masked determines whether a masked load filling what it masks with zero is accepted, the zeros being the
     * padding the host lays out past the operand's extent.
     */
    tt::LoadOp Load(Value value, const bool masked = false)
    {
        auto load = value ? value.getDefiningOp<tt::LoadOp>() : tt::LoadOp {};
        if (!load || !load.getMask()) return load;
        return masked && load.getOther() && IsFloat(load.getOther(), 0.0) ? load : tt::LoadOp {};
    }

    /// Returns the load the exact GELU in `value` activates, or a null op when `value` is no GELU of a widened load,
    /// accepting a masked load when `masked` is set.
    tt::LoadOp Gelu(Value value, const bool masked = false)
    {
        auto narrowed = value.getDefiningOp<arith::TruncFOp>();
        if (!narrowed) return {};
        for (const auto& [half, error] : Orders<arith::MulFOp>(narrowed.getIn()))
        {
            for (const auto& [argument, factor] : Orders<arith::MulFOp>(half))
            {
                if (!IsFloat(factor, kGeluHalf)) continue;
                for (const auto& [function, one] : Orders<arith::AddFOp>(error))
                {
                    auto erf = function.getDefiningOp<math::ErfOp>();
                    if (!erf || !IsFloat(one, 1.0)) continue;
                    auto quotient = erf.getOperand().getDefiningOp<arith::DivFOp>();
                    if (!quotient || quotient.getLhs() != argument || !IsFloat(quotient.getRhs(), kRootTwo)) continue;
                    if (auto load = Load(Narrow(argument), masked); load && Narrow(argument) != argument) return load;
                }
            }
        }
        return {};
    }

    /// Returns the term `value`'s loop adds into a zeroed accumulator each iteration, or null, and the loop.
    Value Accumulated(Value value, scf::ForOp& loop)
    {
        const auto result = dyn_cast_if_present<OpResult>(value);
        loop              = result ? dyn_cast<scf::ForOp>(result.getOwner()) : scf::ForOp {};
        if (!loop) return {};
        const unsigned index = result.getResultNumber();
        if (!IsFloat(loop.getInitArgs()[index], 0.0)) return {};
        Value carried = loop.getRegionIterArgs()[index];
        for (const auto& [accumulator, term] : Orders<arith::AddFOp>(loop.getYieldedValues()[index]))
        {
            if (accumulator == carried) return term;
        }
        return {};
    }

    /// Returns the tensor `value` folds along its features under `CombineOp`, or null when it is no such fold.
    template <typename CombineOp>
    Value FeatureFold(Value value)
    {
        auto reduce = value ? value.getDefiningOp<tt::ReduceOp>() : tt::ReduceOp {};
        if (!reduce || reduce.getSrcs().size() != 1 || reduce.getAxis() != kFeatureAxis) return {};
        return isa_and_nonnull<CombineOp>(reduce.getSingleCombiner()) ? reduce.getSrcs()[0] : Value {};
    }

    /// Returns the tensor `value` sums along its features, or null when it is no such sum.
    Value FeatureSum(Value value)
    {
        return FeatureFold<arith::AddFOp>(value);
    }

    /// Returns whether `value` is a float constant below every real score: negative infinity or a mask value.
    bool IsMasked(Value value)
    {
        const std::optional<double> constant = FloatConstant(value);
        return constant && *constant <= kMaskBound;
    }

    /// Stores the loops walking one operand's rows and columns, each absent when the kernel's shape gives it one
    /// iteration and the canonicalizer folds it away.
    struct Walk
    {
        std::optional<Loop> rows;     ///< The loop advancing the operand one row per unit of its counter.
        std::optional<Loop> columns;  ///< The loop advancing the operand one column per unit of its counter.
    };

    /// Returns `extent` rounded up to a whole number of tiles, the extent the host pads an operand to.
    int64_t RoundUp(const int64_t extent)
    {
        return (extent + tile::kTile - 1) / tile::kTile * tile::kTile;
    }

    /// Returns whether two optional loops are the same loop or both absent.
    bool Same(const std::optional<Loop>& left, const std::optional<Loop>& right)
    {
        return left.has_value() == right.has_value() && (!left || left->op == right->op);
    }

    /// Appends `loop`'s counter advancing `step` elements to `counters`, when the loop is present.
    void AddCounter(SmallVector<Counter>& counters, const std::optional<Loop>& loop, const int64_t step)
    {
        if (loop) counters.emplace_back(loop->counter, step);
    }

    /// Returns the term a zeroed accumulator `value` sums, through the loop accumulating it in `scope` when there is
    /// one, or null when `value` is no such sum.
    Value Summand(Value value, Operation* scope)
    {
        scf::ForOp loop;
        Value term = Accumulated(value, loop);
        if (loop) return term && loop->getParentOp() == scope ? term : Value {};

        // Reads a loop of one iteration, which the canonicalizer folds into the zeroed accumulator plus its one term.
        for (const auto& [zero, single] : Orders<arith::AddFOp>(value))
        {
            if (IsFloat(zero, 0.0)) return single;
        }
        return value;
    }

    /// Converts one Triton kernel into the minitpu kernel computing one of its token blocks.
    class KernelConversion
    {
      public:
        /// Prepares to convert `function`, an attention kernel among them walking `heads` heads in its image.
        KernelConversion(tt::FuncOp function, const uint32_t heads) : _function(function), _heads(heads)
        {}

        /**
         * @brief Converts the kernel, whose one store `store` writes its result, replacing it with a minitpu kernel.
         *
         * @param store the kernel's only store.
         * @returns success once the kernel is replaced, failure with a diagnostic otherwise.
         */
        LogicalResult Convert(tt::StoreOp store)
        {
            SmallVector<tt::DotOp> products;
            bool reduces = false;
            _function.walk([&](Operation* operation) {
                if (auto dot = dyn_cast<tt::DotOp>(operation)) products.push_back(dot);
                if (llvm::isa<tt::ReduceOp>(operation)) reduces = true;
            });
            if (products.size() == 2) return ConvertAttention(products[0], products[1], store);
            if (products.size() > 2)
            {
                return _function.emitError() << "takes " << products.size() << " dot products. A product converts "
                                             << "into minitpu.matmul when one dot product accumulates its result, "
                                             << "and attention into minitpu.attention when two do.";
            }
            if (!products.empty()) return ConvertMatmul(products.front(), store);
            if (reduces) return ConvertLayernorm(store);
            return ConvertAdd(store);
        }

      private:
        /// Stores the kernel being converted.
        tt::FuncOp _function;

        /// Stores the heads an attention image walks.
        uint32_t _heads;

        /// Stores the addresses resolved so far.
        AddressAnalysis _addresses;

        /// Returns the loops enclosing `operation`, innermost first, below `scope` or else the kernel, accepting loops
        /// whose last iteration overruns their bound when `ragged` is set.
        FailureOr<SmallVector<Loop>>
        Enclosing(Operation* operation, Operation* scope = nullptr, const bool ragged = false)
        {
            SmallVector<Loop> loops;
            for (Operation* parent = operation->getParentOp(); parent && parent != scope && parent != _function;
                 parent            = parent->getParentOp())
            {
                auto loop = dyn_cast<scf::ForOp>(parent);
                if (!loop)
                {
                    return parent->emitError() << "is control flow the conversion is unable to read. A kernel's "
                                               << "tiles may sit only in counted scf.for loops.";
                }
                const FailureOr<Loop> bounds = Bounds(loop, ragged);
                if (failed(bounds)) return failure();
                loops.push_back(*bounds);
            }
            return loops;
        }

        /**
         * @brief Finds the loops among `loops` walking the rows and the columns of the two-dimensional operand
         * `pointer` addresses.
         *
         * @param access the load or store reading it, which carries the diagnostic.
         * @param role the operand's name, as the diagnostic names it.
         * @param pointer the address.
         * @param loops the loops that may walk it.
         * @returns the loops found, or failure with a diagnostic when a loop walks it any other way.
         */
        FailureOr<Walk> Classify(Operation* access, StringRef role, Value pointer, ArrayRef<Loop> loops)
        {
            const FailureOr<Affine> form = _addresses.Of(pointer);
            if (failed(form)) return failure();
            if (form->strides.size() != 2)
            {
                return access->emitError() << "addresses the " << role << " as a tensor of " << form->strides.size()
                                           << " dimensions. A tile op's operands are tiles of rows and columns.";
            }
            Walk walk;
            for (const Loop& loop : loops)
            {
                const int64_t step = form->Step(loop.counter);
                if (step == 0) continue;
                std::optional<Loop>& slot = step == form->strides[0] ? walk.rows : walk.columns;
                if ((step != form->strides[0] && step != form->strides[1]) || slot)
                {
                    return loop.op->emitError() << "walks the " << role << " " << step << " elements per unit of its "
                                                << "counter, which is neither one row nor one column of it.";
                }
                slot = loop;
            }
            return walk;
        }

        /**
         * @brief Holds the address `pointer` resolves to to a dense row-major operand.
         *
         * @param access the load or store reading it, which carries the diagnostic.
         * @param role the operand's name, as the diagnostic names it.
         * @param pointer the address.
         * @param strides the elements each tensor dimension must advance.
         * @param counters the elements each loop counter must advance.
         * @returns the pointer argument it is based on, or failure with a diagnostic.
         */
        FailureOr<Value>
        Expect(Operation* access, StringRef role, Value pointer, ArrayRef<int64_t> strides, ArrayRef<Counter> counters)
        {
            const FailureOr<Affine> form = _addresses.Of(pointer);
            if (failed(form)) return failure();
            bool matches = form->base && form->constant == 0 && ArrayRef<int64_t>(form->strides) == strides &&
                           form->counters.size() == counters.size();
            for (const auto& [counter, step] : counters) matches = matches && form->Step(counter) == step;
            if (matches) return form->base;

            InFlightDiagnostic error = access->emitError()
                                    << "addresses the " << role << " in a layout the host does not pack it in. "
                                    << "Each operand must be a dense row-major tensor read from its pointer argument, "
                                    << "one launch per block of tokens.";
            error.attachNote() << "expected element strides [" << strides << "] and found [" << form->strides
                               << "] with " << form->counters.size() << " loop step(s) and offset " << form->constant;
            return failure();
        }

        /**
         * @brief Holds the operands' pointer arguments to the kernel's arguments, in order and with nothing else, the
         * order the image's launch arguments follow.
         *
         * @param bases the pointer argument each operand is based on, in launch order.
         * @param roles the operands' names, as the diagnostic names them.
         * @returns success when they match, failure with a diagnostic otherwise.
         */
        LogicalResult Arguments(ArrayRef<Value> bases, ArrayRef<StringRef> roles)
        {
            bool ordered = _function.getNumArguments() == bases.size();
            for (const auto [position, base] : llvm::enumerate(bases))
            {
                ordered = ordered && cast<BlockArgument>(base).getArgNumber() == position;
            }
            if (ordered) return success();
            return _function.emitError() << "must take exactly its " << llvm::join(roles, ", ")
                                         << " pointers, in that order, which is the order its launch passes them.";
        }

        /// Opens the minitpu kernel that replaces the Triton kernel and returns a builder at its start.
        OpBuilder Open()
        {
            OpBuilder builder(_function);
            auto kernel = KernelOp::create(builder, _function.getLoc(), _function.getSymName());
            builder.setInsertionPointToStart(&kernel.getBody().emplaceBlock());
            return builder;
        }

        /// Returns the launch argument `index`.
        Value Argument(OpBuilder& builder, const uint32_t index)
        {
            return ArgOp::create(builder, _function.getLoc(), builder.getI32Type(), index).getResult();
        }

        /// Stores the bound each dimension of an access's tensor is masked under, absent where its mask bounds none.
        using Extents = SmallVector<std::optional<int64_t>, 2>;

        /**
         * @brief Reads the bound one comparison in a mask holds one dimension of the operand under.
         *
         * @param comparison the comparison.
         * @param role the operand's name, as the diagnostic names it.
         * @param address the operand's address.
         * @param dimensions the dimension of the operand each dimension of the comparison's tensor stands for.
         * @returns the operand's dimension and the bound its index, plus the counters walking it, stays below.
         */
        FailureOr<std::pair<int64_t, int64_t>>
        Comparison(arith::CmpIOp comparison, StringRef role, const Affine& address, ArrayRef<int64_t> dimensions)
        {
            using arith::CmpIPredicate;
            const CmpIPredicate predicate = comparison.getPredicate();
            const bool flipped            = predicate == CmpIPredicate::sgt || predicate == CmpIPredicate::ugt ||
                                 predicate == CmpIPredicate::sge || predicate == CmpIPredicate::uge;
            const bool inclusive = predicate == CmpIPredicate::sle || predicate == CmpIPredicate::ule ||
                                   predicate == CmpIPredicate::sge || predicate == CmpIPredicate::uge;
            const bool ordered          = predicate != CmpIPredicate::eq && predicate != CmpIPredicate::ne;
            const FailureOr<Affine> lhs = _addresses.Of(comparison.getLhs());
            const FailureOr<Affine> rhs = _addresses.Of(comparison.getRhs());
            if (failed(lhs) || failed(rhs)) return failure();

            // Reads `index < bound` as `index - bound < 0`, the difference walking exactly one dimension by one.
            const Affine difference = flipped ? Added(*rhs, Negated(*lhs)) : Added(*lhs, Negated(*rhs));
            int64_t walked          = -1;
            for (const auto [dimension, stride] : llvm::enumerate(difference.strides))
            {
                if (stride == 0) continue;
                walked = stride == 1 && walked < 0 ? static_cast<int64_t>(dimension) : -2;
            }
            const int64_t operand_dimension = walked >= 0 ? dimensions[walked] : -1;

            // Holds the mask to the counters walking the operand along that dimension, so it bounds the whole operand.
            bool aligned = ordered && !lhs->base && !rhs->base && operand_dimension >= 0;
            for (const Affine* form : {&difference, &address})
            {
                for (const auto& [counter, step] : form->counters)
                {
                    if (!aligned) break;
                    const int64_t stride = address.strides[operand_dimension];
                    const bool walks     = stride != 0 && address.Step(counter) == stride;
                    aligned              = difference.Step(counter) == (walks ? 1 : 0);
                }
            }
            if (!aligned)
            {
                return comparison.emitError()
                    << "masks the " << role << " in a way the conversion is unable to read. Each comparison must hold "
                    << "one index, plus the loop counters walking the operand along it, below a constant, which is the "
                    << "operand's extent the host pads past with zeros.";
            }
            return std::pair {operand_dimension, -difference.constant + (inclusive ? 1 : 0)};
        }

        /**
         * @brief Reads the bound `mask` holds each dimension of the operand `pointer` addresses under.
         *
         * @param access the load or store the mask guards.
         * @param role the operand's name, as the diagnostic names it.
         * @param pointer the address.
         * @param mask the mask, or null for an unmasked access.
         * @returns the bound of each dimension, absent where the mask bounds none, or failure with a diagnostic.
         */
        FailureOr<Extents> Masked(Operation* access, StringRef role, Value pointer, Value mask)
        {
            Extents extents(Rank(pointer));
            if (!mask) return extents;
            const FailureOr<Affine> address = _addresses.Of(pointer);
            if (failed(address)) return failure();

            // Walks the conjunction, tracking which of the operand's dimensions each term's dimensions stand for.
            auto identity = llvm::to_vector(llvm::seq<int64_t>(0, Rank(pointer)));
            SmallVector<std::pair<Value, SmallVector<int64_t>>> pending {
                {mask, identity}
            };
            while (!pending.empty())
            {
                auto [term, dimensions] = pending.pop_back_val();
                if (auto conjunction = term.getDefiningOp<arith::AndIOp>())
                {
                    pending.emplace_back(conjunction.getLhs(), dimensions);
                    pending.emplace_back(conjunction.getRhs(), dimensions);
                    continue;
                }
                if (auto broadcast = term.getDefiningOp<tt::BroadcastOp>())
                {
                    pending.emplace_back(broadcast.getSrc(), dimensions);
                    continue;
                }
                if (auto expand = term.getDefiningOp<tt::ExpandDimsOp>())
                {
                    dimensions.erase(dimensions.begin() + expand.getAxis());
                    pending.emplace_back(expand.getSrc(), dimensions);
                    continue;
                }
                if (matchPattern(term, m_One())) continue;
                auto comparison = term.getDefiningOp<arith::CmpIOp>();
                if (!comparison)
                {
                    return access->emitError() << "masks the " << role << " with something other than comparisons "
                                               << "joined by and, which the conversion is unable to read.";
                }
                const FailureOr<std::pair<int64_t, int64_t>> bound = Comparison(comparison, role, *address, dimensions);
                if (failed(bound)) return failure();
                std::optional<int64_t>& extent = extents[bound->first];
                extent                         = extent ? std::min(*extent, bound->second) : bound->second;
            }
            return extents;
        }

        /**
         * @brief Resolves the extent a product covers along one axis from its loop or tile and the masks bounding it.
         *
         * @param product the dot product, which carries the diagnostic.
         * @param axis the axis's name, as the diagnostic names it.
         * @param loop the loop walking the axis, or nullopt when the tile covers it in one step.
         * @param tile the tile's extent along the axis.
         * @param masks the bound each access's mask holds the axis under, absent where it holds none.
         * @returns the extent, or failure with a diagnostic when the masks disagree or a loop overruns its bound
         * unmasked.
         */
        static FailureOr<int64_t> Extent(
            tt::DotOp product,
            StringRef axis,
            const std::optional<Loop>& loop,
            const int64_t tile,
            ArrayRef<std::optional<int64_t>> masks
        )
        {
            const int64_t stated  = loop ? loop->upper : tile;
            const int64_t covered = loop ? loop->Covered() : tile;
            std::optional<int64_t> masked;
            for (const std::optional<int64_t>& bound : masks)
            {
                if (!bound) continue;
                if ((masked && *masked != *bound) || *bound <= 0)
                {
                    return product.emitError() << "is unable to be read as a matrix product: the masks bounding its "
                                               << axis << " must agree on one positive extent.";
                }
                masked = bound;
            }
            if (!masked && stated != covered)
            {
                return product.emitError() << "is unable to be read as a matrix product: a loop overrunning its bound "
                                           << "along the " << axis << " must mask the overrun.";
            }
            return masked ? std::min(*masked, stated) : stated;
        }

        /// Refuses the kernel with `message`, naming the family it was read as.
        static LogicalResult Refuse(Operation* operation, StringRef family, StringRef message)
        {
            return operation->emitError() << "is unable to be read as a " << family << ": " << message;
        }

        /// Refuses a tile whose rows differ from the block loop's step, which the host launches the image once per.
        static LogicalResult
        CheckBlock(Operation* operation, StringRef family, const std::optional<Loop>& block, const int64_t tokens)
        {
            if (!block || block->step == tokens) return success();
            return Refuse(operation, family, "the loop over token blocks must step by the tile's rows.");
        }

        /// Converts a kernel whose one dot product accumulates its result into minitpu.matmul.
        LogicalResult ConvertMatmul(tt::DotOp product, tt::StoreOp store)
        {
            constexpr StringRef kFamily = "matrix product";
            const bool activate         = static_cast<bool>(Gelu(product.getA(), true));
            tt::LoadOp activation       = activate ? Gelu(product.getA(), true) : Load(product.getA(), true);
            tt::LoadOp weight           = Load(product.getB(), true);
            if (!activation || !weight)
            {
                return Refuse(
                    product,
                    kFamily,
                    "both operands must be loads, masked ones filling with zero, the left one optionally through "
                    "the exact GELU."
                );
            }

            // Finds each loop by what it walks: the token blocks walk the activations' rows, the contraction their
            // columns and the weights' rows, and the column loop the weights' columns.
            const FailureOr<SmallVector<Loop>> loops = Enclosing(product, nullptr, true);
            if (failed(loops)) return failure();
            const FailureOr<Walk> left  = Classify(activation, "activations", activation.getPtr(), *loops);
            const FailureOr<Walk> right = Classify(weight, "weights", weight.getPtr(), *loops);
            if (failed(left) || failed(right)) return failure();
            const std::optional<Loop>& block       = left->rows;
            const std::optional<Loop>& contraction = left->columns;
            const std::optional<Loop>& columns     = right->columns;
            const size_t walked                    = (block ? 1 : 0) + (contraction ? 1 : 0) + (columns ? 1 : 0);
            if (!Same(contraction, right->rows) || walked != loops->size())
            {
                return Refuse(
                    product,
                    kFamily,
                    "each loop must walk the token blocks, the contraction or the output columns, the "
                    "contraction stepping the activations' columns and the weights' rows together."
                );
            }

            // Holds the accumulator to one carried around the contraction, or to the dot product's own when the
            // contraction takes one step, starting from zero or from a bias row.
            Value start  = product.getC();
            Value result = product.getResult();
            if (contraction)
            {
                scf::ForOp loop      = contraction->op;
                const auto carried   = dyn_cast<BlockArgument>(product.getC());
                const unsigned first = loop.getNumInductionVars();
                if (!carried || carried.getOwner()->getParentOp() != loop || carried.getArgNumber() < first ||
                    loop.getYieldedValues()[carried.getArgNumber() - first] != product.getResult())
                {
                    return Refuse(product, kFamily, "its accumulator must be carried around the contraction loop.");
                }
                start  = loop.getInitArgs()[carried.getArgNumber() - first];
                result = loop.getResult(carried.getArgNumber() - first);
            }
            tt::LoadOp bias;
            if (!IsFloat(start, 0.0)) bias = Load(Narrow(Broadcasted(start, kTokenAxis)), true);
            if (!IsFloat(start, 0.0) && !bias)
            {
                return Refuse(product, kFamily, "its accumulator must start from zero or from a broadcast bias row.");
            }

            // Reads a bias the kernel adds once the contraction ends the same as one the accumulator starts from.
            Value stored = store.getValue();
            if (!bias)
            {
                for (const auto& [total, row] : Orders<arith::AddFOp>(stored))
                {
                    tt::LoadOp added = Load(Narrow(Broadcasted(row, kTokenAxis)), true);
                    if (total != result || !added) continue;
                    bias   = added;
                    stored = total;
                }
            }
            if (stored != result) return Refuse(store, kFamily, "the kernel must store the accumulated product.");

            // Reads each axis's extent from its loop or tile and the masks bounding it, the host padding the operands
            // past it with zeros to the whole tiles the op covers.
            const FailureOr<Extents> left_mask =
                Masked(activation, "activations", activation.getPtr(), activation.getMask());
            const FailureOr<Extents> right_mask = Masked(weight, "weights", weight.getPtr(), weight.getMask());
            const FailureOr<Extents> drain_mask = Masked(store, "result", store.getPtr(), store.getMask());
            const FailureOr<Extents> bias_mask =
                bias ? Masked(bias, "bias", bias.getPtr(), bias.getMask()) : FailureOr<Extents>(Extents(1));
            if (failed(left_mask) || failed(right_mask) || failed(drain_mask) || failed(bias_mask)) return failure();
            const ArrayRef<int64_t> tile = Shape(product.getA());
            const int64_t tokens         = tile[0];
            const int64_t width          = Shape(product.getB())[1];
            const FailureOr<int64_t> rows =
                Extent(product, "tokens", block, tokens, {(*left_mask)[0], (*drain_mask)[0]});
            const FailureOr<int64_t> k =
                Extent(product, "contraction", contraction, tile[1], {(*left_mask)[1], (*right_mask)[0]});
            const FailureOr<int64_t> n =
                Extent(product, "columns", columns, width, {(*right_mask)[1], (*drain_mask)[1], (*bias_mask)[0]});
            if (failed(rows) || failed(k) || failed(n)) return failure();
            if ((contraction && contraction->step != tile[1]) || (columns && columns->step != width))
            {
                return Refuse(
                    product,
                    kFamily,
                    "the contraction and column loops must step by the tile's depth and width."
                );
            }
            if (failed(CheckBlock(product, kFamily, block, tokens))) return failure();
            Type bf16 = BFloat16Type::get(product.getContext());
            if (Pointee(activation.getPtr()) != bf16 || Pointee(weight.getPtr()) != bf16 ||
                Pointee(store.getPtr()) != Float32Type::get(product.getContext()) ||
                (bias && Pointee(bias.getPtr()) != bf16))
            {
                return Refuse(
                    product,
                    kFamily,
                    "the operands and the bias must be BF16 and the result FP32, the types the array reads "
                    "and drains."
                );
            }

            SmallVector<Counter> reads;
            AddCounter(reads, block, *k);
            AddCounter(reads, contraction, 1);
            SmallVector<Counter> weighs;
            AddCounter(weighs, columns, 1);
            AddCounter(weighs, contraction, *n);
            SmallVector<Counter> drains;
            AddCounter(drains, block, *n);
            AddCounter(drains, columns, 1);
            SmallVector<Counter> biases;
            AddCounter(biases, columns, 1);
            const FailureOr<Value> activations = Expect(activation, "activations", activation.getPtr(), {*k, 1}, reads);
            const FailureOr<Value> weights     = Expect(weight, "weights", weight.getPtr(), {*n, 1}, weighs);
            const FailureOr<Value> drain       = Expect(store, "result", store.getPtr(), {*n, 1}, drains);
            const FailureOr<Value> offsets =
                bias ? Expect(bias, "bias", bias.getPtr(), {1}, biases) : FailureOr<Value>(Value {});
            if (failed(activations) || failed(weights) || failed(drain) || failed(offsets)) return failure();
            if (failed(
                    bias ? Arguments(
                               {*activations, *weights, *drain, *offsets},
                               {"activations", "weights", "result", "bias"}
                           )
                         : Arguments({*activations, *weights, *drain}, {"activations", "weights", "result"})
                ))
            {
                return failure();
            }

            OpBuilder builder = Open();
            Value first       = Argument(builder, 0);
            Value second      = Argument(builder, 1);
            Value third       = Argument(builder, 2);
            Value fourth      = bias ? Argument(builder, 3) : Value {};
            MatmulOp::create(
                builder,
                _function.getLoc(),
                first,
                second,
                third,
                fourth,
                static_cast<uint32_t>(tokens),
                static_cast<uint32_t>(RoundUp(*k)),
                static_cast<uint32_t>(RoundUp(*n)),
                activate
            );
            return success();
        }

        /**
         * @brief Holds an integer index tensor to the indices one loop walks along one dimension.
         *
         * @param value the index tensor, two-dimensional.
         * @param loop the loop whose counter advances it by one per unit, or nullopt when the loop is folded away.
         * @param dimension the dimension the indices run along, the other being broadcast.
         * @returns whether the tensor is exactly the loop's counter plus a range along `dimension`.
         */
        bool IsIndex(Value value, const std::optional<Loop>& loop, const int64_t dimension)
        {
            const FailureOr<Affine> form = _addresses.Of(value);
            if (failed(form) || form->base || form->constant != 0 || form->strides.size() != 2) return false;
            SmallVector<int64_t> strides(2, 0);
            strides[dimension]    = 1;
            const size_t counters = loop ? 1 : 0;
            return ArrayRef<int64_t>(form->strides) == ArrayRef<int64_t>(strides) &&
                   form->counters.size() == counters && (!loop || form->Step(loop->counter) == 1);
        }

        /**
         * @brief Converts a kernel attending one head under the online softmax into minitpu.attention.
         *
         * @details The result is the image the board launches: every head of the fused QKV tensor at once, each
         * head's output written over its queries.
         *
         * @param scores the dot product of the queries and the transposed keys.
         * @param attended the dot product of the probabilities and the values, accumulating the output.
         * @param store the kernel's only store, writing the normalised output.
         * @returns success once the kernel is replaced, failure with a diagnostic otherwise.
         */
        LogicalResult ConvertAttention(tt::DotOp scores, tt::DotOp attended, tt::StoreOp store)
        {
            constexpr StringRef kFamily = "attention";
            auto keys_loop              = dyn_cast<scf::ForOp>(attended->getParentOp());
            if (!keys_loop || scores->getParentOp() != keys_loop)
            {
                return Refuse(attended, kFamily, "both dot products must sit in one loop over the keys.");
            }

            // Reads the carried running maximum, running sum and output off the output's rescale.
            const unsigned first = keys_loop.getNumInductionVars();
            Value maximum_in;
            Value total_in;
            Value output_in;
            Value rescale;
            for (const auto& [carried, factor] : Orders<arith::MulFOp>(attended.getC()))
            {
                if (output_in || !llvm::is_contained(keys_loop.getRegionIterArgs(), carried)) continue;
                output_in = carried;
                rescale   = Broadcasted(factor, kFeatureAxis);
            }
            auto exponent    = rescale ? rescale.getDefiningOp<math::ExpOp>() : math::ExpOp {};
            auto difference  = exponent ? exponent.getOperand().getDefiningOp<arith::SubFOp>() : arith::SubFOp {};
            maximum_in       = difference ? difference.getLhs() : Value {};
            Value maximum    = difference ? difference.getRhs() : Value {};
            const auto index = [&](Value carried) {
                return cast<BlockArgument>(carried).getArgNumber() - first;
            };
            if (!output_in || !maximum_in || !llvm::is_contained(keys_loop.getRegionIterArgs(), maximum_in) ||
                keys_loop.getYieldedValues()[index(output_in)] != attended.getResult() ||
                keys_loop.getYieldedValues()[index(maximum_in)] != maximum)
            {
                return Refuse(
                    attended,
                    kFamily,
                    "the loop must carry the output, rescaling it by exp(old maximum - new maximum) before "
                    "accumulating the probabilities pushed against the values, and carry the new maximum."
                );
            }

            // Reads the new maximum off the masked scores, and the probabilities and their sum against it.
            Value masked;
            for (const auto& [carried, block] : Orders<arith::MaxNumFOp>(maximum))
            {
                if (carried == maximum_in && FeatureFold<arith::MaxNumFOp>(block))
                    masked = FeatureFold<arith::MaxNumFOp>(block);
            }
            auto narrowed      = attended.getA().getDefiningOp<arith::TruncFOp>();
            auto probabilities = narrowed ? narrowed.getIn().getDefiningOp<math::ExpOp>() : math::ExpOp {};
            auto centring =
                probabilities ? probabilities.getOperand().getDefiningOp<arith::SubFOp>() : arith::SubFOp {};
            if (!masked || !centring || centring.getLhs() != masked ||
                Broadcasted(centring.getRhs(), kFeatureAxis) != maximum)
            {
                return Refuse(
                    attended,
                    kFamily,
                    "the probabilities must be exp(masked scores - new maximum), the new maximum being the "
                    "old one past the masked scores' maximum over the keys."
                );
            }
            for (Value carried : keys_loop.getRegionIterArgs())
            {
                if (carried == output_in || carried == maximum_in) continue;
                total_in = carried;
            }
            bool sums = false;
            for (const auto& [kept, added] : Orders<arith::AddFOp>(keys_loop.getYieldedValues()[index(total_in)]))
            {
                for (const auto& [carried, factor] : Orders<arith::MulFOp>(kept))
                {
                    sums = sums || (carried == total_in && Broadcasted(factor, kFeatureAxis) == Value {} &&
                                    factor == rescale && FeatureSum(added) == probabilities.getResult());
                }
            }
            const bool cleared = IsFloat(keys_loop.getInitArgs()[index(total_in)], 0.0) &&
                                 IsFloat(keys_loop.getInitArgs()[index(output_in)], 0.0) &&
                                 IsMasked(keys_loop.getInitArgs()[index(maximum_in)]);
            if (!sums || !cleared || keys_loop.getNumRegionIterArgs() != 3)
            {
                return Refuse(
                    attended,
                    kFamily,
                    "the loop must carry exactly the maximum, entering below every score, the sum and the "
                    "output, both entering at zero, the sum rescaled and added the probabilities' sum."
                );
            }

            // Reads the mask: a causal one keeping each key up to its query, or one keeping the keys below a length.
            Value product = masked;
            std::optional<int64_t> length;
            bool causal = false;
            Value keep;
            if (auto choice = masked.getDefiningOp<arith::SelectOp>())
            {
                if (!IsMasked(choice.getFalseValue()))
                    return Refuse(choice, kFamily, "a masked score must be a mask value.");
                product = choice.getTrueValue();
                keep    = choice.getCondition();
            }
            auto score_product = product.getDefiningOp<tt::DotOp>();
            auto transposed    = score_product ? score_product.getB().getDefiningOp<tt::TransOp>() : tt::TransOp {};
            tt::LoadOp query   = score_product ? Load(score_product.getA()) : tt::LoadOp {};
            tt::LoadOp key     = transposed ? Load(transposed.getSrc()) : tt::LoadOp {};
            tt::LoadOp value   = Load(attended.getB());
            if (score_product != scores || !query || !key || !value || !IsFloat(scores.getC(), 0.0) || store.getMask())
            {
                return Refuse(
                    scores,
                    kFamily,
                    "the scores must be the unmasked queries' load against the transposed keys' load, from "
                    "zero, and the output must multiply the values' load and store unmasked."
                );
            }

            // Finds the query and key loops by what they walk, each absent when the shape gives it one iteration.
            const FailureOr<SmallVector<Loop>> loops = Enclosing(scores);
            if (failed(loops)) return failure();
            const FailureOr<Walk> queries = Classify(query, "queries", query.getPtr(), *loops);
            const FailureOr<Walk> keys    = Classify(key, "keys", key.getPtr(), *loops);
            if (failed(queries) || failed(keys)) return failure();
            const std::optional<Loop>& query_loop = queries->rows;
            const std::optional<Loop>& key_loop   = keys->rows;
            if (!key_loop || key_loop->op != keys_loop || queries->columns || keys->columns ||
                (query_loop ? 2U : 1U) != loops->size())
            {
                return Refuse(scores, kFamily, "one loop must walk the keys, inside at most one walking the queries.");
            }
            const ArrayRef<int64_t> query_tile = Shape(query.getResult());
            const ArrayRef<int64_t> key_tile   = Shape(key.getResult());
            const int64_t head                 = query_tile[1];
            const int64_t tokens               = query_loop ? query_loop->upper : query_tile[0];
            const int64_t key_count            = key_loop->upper;
            if (key_tile[1] != head || key_loop->step != key_tile[0] ||
                (query_loop && query_loop->step != query_tile[0]))
            {
                return Refuse(scores, kFamily, "the loops must step by the query and key tiles' rows over one head.");
            }

            if (keep)
            {
                auto widened = keep.getDefiningOp<tt::BroadcastOp>();
                auto compare = (widened ? widened.getSrc() : keep).getDefiningOp<arith::CmpIOp>();
                if (compare && compare.getPredicate() == arith::CmpIPredicate::sge)
                {
                    causal = IsIndex(compare.getLhs(), query_loop, 0) && IsIndex(compare.getRhs(), key_loop, 1);
                }
                if (compare && compare.getPredicate() == arith::CmpIPredicate::sle)
                {
                    causal = IsIndex(compare.getLhs(), key_loop, 1) && IsIndex(compare.getRhs(), query_loop, 0);
                }
                APInt bound;
                if (compare && compare.getPredicate() == arith::CmpIPredicate::slt &&
                    IsIndex(compare.getLhs(), key_loop, 1) && matchPattern(compare.getRhs(), m_ConstantInt(&bound)))
                {
                    length = bound.getSExtValue();
                }
                if (!causal && (!length || *length < 1 || *length > key_count))
                {
                    return Refuse(
                        compare ? compare.getOperation() : scores.getOperation(),
                        kFamily,
                        "the mask must keep each key up to its own query, or the keys below a length."
                    );
                }
            }
            if (causal != static_cast<bool>(query_loop) || (causal && tokens != key_count) ||
                (!causal && query_tile[0] != tile::kQueryBlock))
            {
                return Refuse(
                    scores,
                    kFamily,
                    "a causal kernel must walk its queries over the same tokens as its keys, and any other "
                    "must attend one block of 16 queries."
                );
            }

            // Holds the output to the queries' sum-normalised accumulation, and every operand to a dense row-major
            // head.
            auto out_narrowed = store.getValue().getDefiningOp<arith::TruncFOp>();
            auto quotient     = out_narrowed ? out_narrowed.getIn().getDefiningOp<arith::DivFOp>() : arith::DivFOp {};
            if (!quotient || quotient.getLhs() != keys_loop.getResult(index(output_in)) ||
                Broadcasted(quotient.getRhs(), kFeatureAxis) != keys_loop.getResult(index(total_in)))
            {
                return Refuse(store, kFamily, "the kernel must store the output divided by its sum, narrowed to BF16.");
            }
            Type bf16 = BFloat16Type::get(store.getContext());
            if (Pointee(query.getPtr()) != bf16 || Pointee(key.getPtr()) != bf16 || Pointee(value.getPtr()) != bf16 ||
                Pointee(store.getPtr()) != bf16)
            {
                return Refuse(store, kFamily, "the queries, keys, values and output must be BF16.");
            }
            SmallVector<Counter> per_query;
            AddCounter(per_query, query_loop, head);
            SmallVector<Counter> per_key;
            AddCounter(per_key, key_loop, head);
            const FailureOr<Value> query_base = Expect(query, "queries", query.getPtr(), {head, 1}, per_query);
            const FailureOr<Value> key_base   = Expect(key, "keys", key.getPtr(), {head, 1}, per_key);
            const FailureOr<Value> value_base = Expect(value, "values", value.getPtr(), {head, 1}, per_key);
            const FailureOr<Value> output     = Expect(store, "output", store.getPtr(), {head, 1}, per_query);
            if (failed(query_base) || failed(key_base) || failed(value_base) || failed(output)) return failure();
            if (failed(
                    Arguments({*query_base, *key_base, *value_base, *output}, {"queries", "keys", "values", "output"})
                ))
            {
                return failure();
            }

            OpBuilder builder = Open();
            Value constants   = Argument(builder, 0);
            Value cache       = causal ? Value {} : Argument(builder, 1);
            Value fused       = Argument(builder, 2);
            AttentionOp::create(
                builder,
                _function.getLoc(),
                constants,
                fused,
                cache,
                _heads,
                static_cast<uint32_t>(head),
                static_cast<uint32_t>(key_count),
                causal
            );
            return success();
        }

        /// Converts a kernel storing the elementwise sum of two operands, one optionally a broadcast feature row, into
        /// minitpu.add.
        LogicalResult ConvertAdd(tt::StoreOp store)
        {
            constexpr StringRef kFamily = "sum";
            tt::LoadOp left;
            tt::LoadOp right;
            bool broadcast = false;
            for (const auto& [streamed, other] : Orders<arith::AddFOp>(store.getValue()))
            {
                if (!Load(streamed)) continue;
                left      = Load(streamed);
                right     = Load(other);
                broadcast = !right;
                if (broadcast) right = Load(Broadcasted(other, kTokenAxis));
                if (right) break;
            }
            if (!left || !right || store.getMask())
            {
                return Refuse(
                    store,
                    kFamily,
                    "the kernel must store the sum of two unmasked loads, the right one optionally a feature "
                    "row broadcast over the tokens, through an unmasked store. The conversion recognises no "
                    "other kernel without a dot product or a reduction."
                );
            }

            const FailureOr<SmallVector<Loop>> loops = Enclosing(store);
            if (failed(loops)) return failure();
            const FailureOr<Walk> walk = Classify(store, "result", store.getPtr(), *loops);
            if (failed(walk)) return failure();
            const size_t walked = (walk->rows ? 1 : 0) + (walk->columns ? 1 : 0);
            if (walked != loops->size())
            {
                return Refuse(store, kFamily, "each loop must walk the token blocks or the features.");
            }
            const ArrayRef<int64_t> tile = Shape(store.getValue());
            const int64_t features       = walk->columns ? walk->columns->upper : tile[1];
            if (walk->columns && walk->columns->step != tile[1])
            {
                return Refuse(store, kFamily, "the feature loop must step by the tile's width.");
            }
            if (failed(CheckBlock(store, kFamily, walk->rows, tile[0]))) return failure();
            Type bf16 = BFloat16Type::get(store.getContext());
            if (Pointee(left.getPtr()) != bf16 || Pointee(right.getPtr()) != bf16 || Pointee(store.getPtr()) != bf16)
            {
                return Refuse(store, kFamily, "the operands and the result must be BF16.");
            }

            SmallVector<Counter> streamed;
            AddCounter(streamed, walk->rows, features);
            AddCounter(streamed, walk->columns, 1);
            SmallVector<Counter> row;
            AddCounter(row, walk->columns, 1);
            const FailureOr<Value> first  = Expect(left, "left operand", left.getPtr(), {features, 1}, streamed);
            const FailureOr<Value> second = broadcast
                                              ? Expect(right, "bias row", right.getPtr(), {1}, row)
                                              : Expect(right, "right operand", right.getPtr(), {features, 1}, streamed);
            const FailureOr<Value> output = Expect(store, "result", store.getPtr(), {features, 1}, streamed);
            if (failed(first) || failed(second) || failed(output)) return failure();
            if (failed(Arguments({*first, *second, *output}, {"left", "right", "result"}))) return failure();

            OpBuilder builder = Open();
            Value a           = Argument(builder, 0);
            Value b           = Argument(builder, 1);
            Value c           = Argument(builder, 2);
            AddOp::create(
                builder,
                _function.getLoc(),
                a,
                b,
                c,
                static_cast<uint32_t>(tile[0]),
                static_cast<uint32_t>(features),
                broadcast
            );
            return success();
        }

        /// Converts a kernel normalising each token over its features, then scaling and shifting it by gamma and beta,
        /// into minitpu.layernorm.
        LogicalResult ConvertLayernorm(tt::StoreOp store)
        {
            constexpr StringRef kFamily              = "LayerNorm";
            const FailureOr<SmallVector<Loop>> loops = Enclosing(store);
            if (failed(loops)) return failure();
            const FailureOr<Walk> walk = Classify(store, "result", store.getPtr(), *loops);
            if (failed(walk)) return failure();
            if ((walk->rows ? 1U : 0U) + (walk->columns ? 1U : 0U) != loops->size())
            {
                return Refuse(store, kFamily, "each loop must walk the token blocks or the features.");
            }
            const std::optional<Loop>& block = walk->rows;
            const ArrayRef<int64_t> tile     = Shape(store.getValue());
            const int64_t features           = walk->columns ? walk->columns->upper : tile[1];
            scf::ForOp blocks                = block ? block->op : scf::ForOp {};
            Operation* scope                 = blocks ? blocks.getOperation() : _function.getOperation();

            // Holds a load of the normalised rows, in whichever feature loop it sits, to the row-major input.
            Value input;
            auto expect_rows = [&](tt::LoadOp load) -> bool {
                if (!load) return false;
                const FailureOr<SmallVector<Loop>> around = Enclosing(load);
                if (failed(around)) return false;
                const FailureOr<Walk> reads = Classify(load, "rows", load.getPtr(), *around);
                if (failed(reads) || !Same(reads->rows, block)) return false;
                const int64_t width = reads->columns ? reads->columns->upper : Shape(load)[1];
                if (width != features || (reads->columns && reads->columns->step != Shape(load)[1])) return false;
                SmallVector<Counter> counters;
                AddCounter(counters, block, features);
                AddCounter(counters, reads->columns, 1);
                const FailureOr<Value> base = Expect(load, "rows", load.getPtr(), {features, 1}, counters);
                if (failed(base) || (input && *base != input)) return false;
                input = *base;
                return true;
            };
            // Holds a sum along the features to one over the whole width, divided by it.
            auto mean_of = [&](Value value) -> Value {
                auto quotient = value ? value.getDefiningOp<arith::DivFOp>() : arith::DivFOp {};
                if (!quotient || !IsFloat(quotient.getRhs(), static_cast<double>(features))) return {};
                Value sum = FeatureSum(quotient.getLhs());
                return sum ? Summand(sum, scope) : Value {};
            };

            // Reads the store as truncf(((x - mean) * scale) * gamma + beta), each operation in either order.
            auto narrowed = store.getValue().getDefiningOp<arith::TruncFOp>();
            Value centred;
            Value scale;
            tt::LoadOp gamma;
            tt::LoadOp beta;
            for (const auto& [scaled, shift] : Orders<arith::AddFOp>(narrowed ? narrowed.getIn() : Value {}))
            {
                tt::LoadOp shifts = Load(Narrow(Broadcasted(shift, kTokenAxis)));
                for (const auto& [normalised, row] : Orders<arith::MulFOp>(shifts ? scaled : Value {}))
                {
                    tt::LoadOp scales = Load(Narrow(Broadcasted(row, kTokenAxis)));
                    for (const auto& [difference, factor] : Orders<arith::MulFOp>(scales ? normalised : Value {}))
                    {
                        if (centred || !Broadcasted(factor, kFeatureAxis)) continue;
                        centred = difference;
                        scale   = Broadcasted(factor, kFeatureAxis);
                        gamma   = scales;
                        beta    = shifts;
                    }
                }
            }
            if (!centred || store.getMask())
            {
                return Refuse(
                    store,
                    kFamily,
                    "the kernel must store (x - mean) * rsqrt(variance + epsilon) * gamma + beta, narrowed "
                    "to BF16, through an unmasked store."
                );
            }

            // Holds the centring to the mean of the same rows, and reads the mean from its own feature loop.
            auto centring = centred.getDefiningOp<arith::SubFOp>();
            Value mean    = centring ? Broadcasted(centring.getRhs(), kFeatureAxis) : Value {};
            if (!mean || !expect_rows(Load(Narrow(centring.getLhs()))) || !expect_rows(Load(Narrow(mean_of(mean)))))
            {
                return Refuse(store, kFamily, "each token must be centred by the mean of its own features.");
            }

            // Reads the scale as rsqrt(sum((x - mean)^2) / width + epsilon), the squares centred by the same mean.
            auto root = scale.getDefiningOp<math::RsqrtOp>();
            std::optional<double> epsilon;
            Value variance;
            for (const auto& [quotient, floor] : Orders<arith::AddFOp>(root ? root.getOperand() : Value {}))
            {
                if (variance || !FloatConstant(floor) || !(*FloatConstant(floor) > 0.0)) continue;
                epsilon  = FloatConstant(floor);
                variance = mean_of(quotient);
            }
            auto square    = variance ? variance.getDefiningOp<arith::MulFOp>() : arith::MulFOp {};
            auto deviation = square && square.getLhs() == square.getRhs()
                               ? square.getLhs().getDefiningOp<arith::SubFOp>()
                               : arith::SubFOp {};
            if (!deviation || Broadcasted(deviation.getRhs(), kFeatureAxis) != mean ||
                !expect_rows(Load(Narrow(deviation.getLhs()))))
            {
                return Refuse(
                    store,
                    kFamily,
                    "each token must be scaled by rsqrt of its centred squares' mean plus a positive "
                    "epsilon."
                );
            }

            if (walk->columns && walk->columns->step != tile[1])
            {
                return Refuse(store, kFamily, "the feature loop must step by the tile's width.");
            }
            if (failed(CheckBlock(store, kFamily, block, tile[0]))) return failure();
            Type bf16 = BFloat16Type::get(store.getContext());
            if (Pointee(input) != bf16 || Pointee(gamma.getPtr()) != bf16 || Pointee(beta.getPtr()) != bf16 ||
                Pointee(store.getPtr()) != bf16)
            {
                return Refuse(store, kFamily, "the rows, gamma, beta and the result must be BF16.");
            }

            SmallVector<Counter> row;
            AddCounter(row, walk->columns, 1);
            SmallVector<Counter> streamed;
            AddCounter(streamed, block, features);
            AddCounter(streamed, walk->columns, 1);
            const FailureOr<Value> scales = Expect(gamma, "gamma row", gamma.getPtr(), {1}, row);
            const FailureOr<Value> shifts = Expect(beta, "beta row", beta.getPtr(), {1}, row);
            const FailureOr<Value> output = Expect(store, "result", store.getPtr(), {features, 1}, streamed);
            if (failed(scales) || failed(shifts) || failed(output)) return failure();
            if (failed(Arguments({input, *scales, *shifts, *output}, {"rows", "gamma", "beta", "result"})))
            {
                return failure();
            }

            OpBuilder builder = Open();
            Value normalised  = Argument(builder, 0);
            Value constants   = Argument(builder, 1);
            Value result      = Argument(builder, 2);
            LayernormOp::create(
                builder,
                _function.getLoc(),
                normalised,
                constants,
                result,
                static_cast<uint32_t>(tile[0]),
                static_cast<uint32_t>(features),
                APFloat(static_cast<float>(*epsilon))
            );
            return success();
        }
    };

    /**
     * @brief Converts one Triton kernel into a minitpu kernel and erases it.
     *
     * @param function the kernel.
     * @returns success once it is replaced, failure with a diagnostic otherwise.
     */
    LogicalResult ConvertFunction(tt::FuncOp function, const uint32_t heads)
    {
        SmallVector<tt::StoreOp> stores;
        function.walk([&](tt::StoreOp store) {
            stores.push_back(store);
        });
        if (stores.size() != 1)
        {
            return function.emitError() << "stores " << stores.size() << " times. A kernel converts into one tile op, "
                                        << "which writes one result.";
        }

        KernelConversion conversion(function, heads);
        if (failed(conversion.Convert(stores.front()))) return failure();
        function.erase();
        return success();
    }

    /// Provides the pass converting every Triton kernel in a module into a minitpu kernel.
    class ConvertTritonPass final : public mlir::minitpu::impl::ConvertTritonToMinitpuBase<ConvertTritonPass>
    {
      public:
        using ConvertTritonToMinitpuBase::ConvertTritonToMinitpuBase;

        void runOnOperation() override
        {
            SmallVector<tt::FuncOp> functions;
            getOperation().walk([&](tt::FuncOp function) {
                functions.push_back(function);
            });
            for (tt::FuncOp function : functions)
            {
                if (failed(ConvertFunction(function, heads))) signalPassFailure();
            }
        }
    };
}  // namespace
