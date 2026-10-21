/**
 * @file
 *
 * @brief Implements the lowering of tempo kernels into tempoisa programs, one bundle per machine operation.
 */

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Matchers.h>
#include "tempo/tempo/passes.h"
#include "tempo/tempoisa/encoding.h"

namespace mlir::tempo
{
#define GEN_PASS_DEF_CONVERTTEMPOTOISA
#include "tempo/tempo/passes.h.inc"
}  // namespace mlir::tempo

using namespace mlir;
using namespace mlir::tempo;

namespace
{
    /// Stores the deepest loop nest the AGU's level field addresses.
    constexpr size_t kMaxLoopDepth = isa::kXLevel.Mask() + 1;

    /// Stores the vector registers a weight tile occupies, which vmatload reads from consecutive registers.
    constexpr uint32_t kTileRegisters = 4;

    /// Stores the vector registers in the file, which consecutive weight registers wrap around.
    constexpr uint32_t kVectorRegisters = isa::kMRegister.Mask() + 1;

    /// Stores a row resolved to the instruction's static row and its induction term.
    struct ResolvedRow
    {
        int64_t constant = 0;      ///< The static part of the row.
        bool walks       = false;  ///< Determines whether an induction variable is added to the row.
        uint32_t level   = 0;      ///< The loop whose induction variable is added, 0 being the outermost.
        uint32_t shift   = 0;      ///< The left shift applied to that induction variable.
    };

    /// Provides the lowering of one kernel's operations into the bundles of one program.
    class KernelLowering
    {
      public:
        /// Prepares to append the kernel's bundles to `program`.
        explicit KernelLowering(isa::ProgramOp program) : _builder(program.getContext()), _program(program)
        {}

        /// Lowers every operation of `block` in order, failing on the first one it is unable to lower.
        // NOLINTNEXTLINE(misc-no-recursion): recurses once per nested loop or row term, which the AGU bounds.
        LogicalResult LowerBlock(Block& block)
        {
            for (Operation& operation : block)
            {
                if (failed(LowerOperation(operation))) return failure();
            }
            return success();
        }

        /// Appends the halt every program ends in.
        void Finish(const Location location)
        {
            Emit<isa::HaltOp>(location);
        }

      private:
        /// Appends one bundle carrying one instruction built from `arguments`.
        template <typename IsaOp, typename... Arguments>
        void Emit(const Location location, Arguments... arguments)
        {
            OpBuilder::InsertionGuard guard(_builder);
            _builder.setInsertionPointToEnd(&_program.getBody().front());
            auto bundle = isa::BundleOp::create(_builder, location, uint32_t {0});
            _builder.setInsertionPointToEnd(&bundle.getBody().emplaceBlock());
            IsaOp::create(_builder, location, arguments...);
        }

        /// Returns the register `value` lives in, read through any loop that carries it.
        // NOLINTNEXTLINE(misc-no-recursion): recurses once per nested loop or row term, which the AGU bounds.
        FailureOr<uint32_t> RegisterOf(Value value, Operation* user)
        {
            if (auto argument = dyn_cast<BlockArgument>(value))
            {
                auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
                if (loop && argument.getArgNumber() > 0)
                {
                    return RegisterOf(loop.getInitArgs()[argument.getArgNumber() - 1], user);
                }
                return user->emitOpError("is unable to read a loop's induction variable as a register value.");
            }
            Operation* definition = value.getDefiningOp();
            if (auto loop = dyn_cast<scf::ForOp>(definition))
            {
                return RegisterOf(loop.getInitArgs()[cast<OpResult>(value).getResultNumber()], user);
            }
            const auto index = definition->getAttrOfType<IntegerAttr>(TempoDialect::get_register_attribute_name());
            if (!index)
            {
                return user->emitOpError()
                    << "reads a value with no register. Every value must carry "
                    << TempoDialect::get_register_attribute_name() << ", which -tempo-allocate-registers writes, but "
                    << definition->getName() << " carries none.";
            }
            return static_cast<uint32_t>(index.getInt());
        }

        /// Returns the base row of the buffer `value` names.
        static FailureOr<uint32_t> BaseOf(Value buffer, Operation* user)
        {
            auto allocation = buffer.getDefiningOp<AllocOp>();
            if (!allocation || !allocation.getBase())
            {
                return user->emitOpError(
                    "reads a buffer with no base row. Every buffer must carry base, which -tempo-allocate-vmem "
                    "writes."
                );
            }
            return *allocation.getBase();
        }

        /**
         * @brief Resolves a row to a static part plus at most one induction term, as the AGU adds it.
         *
         * @param row the row value.
         * @param user the operation reading the row, which a diagnostic is attached to.
         * @returns the resolved row, or failure with a diagnostic when the row is not of a form the AGU adds.
         */
        // NOLINTNEXTLINE(misc-no-recursion): recurses once per nested loop or row term, which the AGU bounds.
        FailureOr<ResolvedRow> ResolveRow(Value row, Operation* user)
        {
            APInt constant;
            if (matchPattern(row, m_ConstantInt(&constant))) return ResolvedRow {.constant = constant.getSExtValue()};

            if (auto argument = dyn_cast<BlockArgument>(row))
            {
                for (size_t level = 0; level < _loops.size(); ++level)
                {
                    if (_loops[level].getInductionVar() == argument)
                    {
                        return ResolvedRow {.walks = true, .level = static_cast<uint32_t>(level)};
                    }
                }
            }

            Operation* definition = row.getDefiningOp();
            if (auto addition = dyn_cast_if_present<arith::AddIOp>(definition))
            {
                const FailureOr<ResolvedRow> lhs = ResolveRow(addition.getLhs(), user);
                const FailureOr<ResolvedRow> rhs = ResolveRow(addition.getRhs(), user);
                if (failed(lhs) || failed(rhs)) return failure();
                if (lhs->walks && rhs->walks)
                {
                    return user->emitOpError(
                        "is unable to address a row two induction variables walk. The AGU adds one."
                    );
                }
                ResolvedRow sum = lhs->walks ? *lhs : *rhs;
                sum.constant    = lhs->constant + rhs->constant;
                return sum;
            }

            std::optional<uint32_t> shift;
            Value scaled;
            if (auto product = dyn_cast_if_present<arith::MulIOp>(definition))
            {
                for (const auto [factor, other] : {
                         std::pair {product.getRhs(), product.getLhs()},
                         std::pair {product.getLhs(), product.getRhs()}
                })
                {
                    if (matchPattern(factor, m_ConstantInt(&constant)) && constant.isPowerOf2())
                    {
                        shift  = constant.logBase2();
                        scaled = other;
                        break;
                    }
                }
            }
            else if (auto shifted = dyn_cast_if_present<arith::ShLIOp>(definition))
            {
                if (matchPattern(shifted.getRhs(), m_ConstantInt(&constant)))
                {
                    shift  = static_cast<uint32_t>(constant.getZExtValue());
                    scaled = shifted.getLhs();
                }
            }
            if (shift)
            {
                FailureOr<ResolvedRow> resolved = ResolveRow(scaled, user);
                if (failed(resolved)) return failure();
                resolved->constant <<= *shift;
                resolved->shift += *shift;
                return resolved;
            }
            return user->emitOpError(
                "is unable to address its row. A row must be a constant, or a constant plus an enclosing loop's "
                "induction variable shifted left by a power of two."
            );
        }

        /// Returns the X-slot fields of a vld or vst reading `row` of `buffer`.
        FailureOr<ResolvedRow> AddressRow(Value buffer, Value row, Operation* user)
        {
            const FailureOr<uint32_t> base  = BaseOf(buffer, user);
            FailureOr<ResolvedRow> resolved = ResolveRow(row, user);
            if (failed(base) || failed(resolved)) return failure();
            resolved->constant += *base;
            if (resolved->constant < 0 || resolved->constant > static_cast<int64_t>(isa::kXRow.Mask()) ||
                resolved->shift > isa::kXShift.Mask())
            {
                return user->emitOpError() << "is unable to address row " << resolved->constant << " shifted by "
                                           << resolved->shift << ". The row field reaches " << isa::kXRow.Mask()
                                           << " and the shift field " << isa::kXShift.Mask() << ".";
            }
            return resolved;
        }

        /// Lowers a vector operation of two operands into the instruction of the same mnemonic.
        template <typename IsaOp, typename SourceOp>
        LogicalResult LowerBinary(SourceOp operation)
        {
            const FailureOr<uint32_t> rd = RegisterOf(operation.getResult(), operation);
            const FailureOr<uint32_t> ra = RegisterOf(operation.getLhs(), operation);
            const FailureOr<uint32_t> rb = RegisterOf(operation.getRhs(), operation);
            if (failed(rd) || failed(ra) || failed(rb)) return failure();
            Emit<IsaOp>(operation.getLoc(), *rd, *ra, *rb);
            return success();
        }

        /// Lowers a vector operation of one operand into the instruction of the same mnemonic.
        template <typename IsaOp, typename SourceOp>
        LogicalResult LowerUnary(SourceOp operation)
        {
            const FailureOr<uint32_t> rd = RegisterOf(operation.getResult(), operation);
            const FailureOr<uint32_t> ra = RegisterOf(operation.getOperand(), operation);
            if (failed(rd) || failed(ra)) return failure();
            Emit<IsaOp>(operation.getLoc(), *rd, *ra);
            return success();
        }

        /// Lowers a DMA descriptor, whose VMEM address is an immediate and so must be a constant row.
        template <typename IsaOp, typename SourceOp>
        LogicalResult LowerDescriptor(SourceOp operation)
        {
            const FailureOr<uint32_t> base    = BaseOf(operation.getBuffer(), operation);
            const FailureOr<ResolvedRow> row  = ResolveRow(operation.getRow(), operation);
            const FailureOr<uint32_t> pointer = RegisterOf(operation.getBase(), operation);
            const FailureOr<uint32_t> stride  = RegisterOf(operation.getStride(), operation);
            if (failed(base) || failed(row) || failed(pointer) || failed(stride)) return failure();
            const int64_t address = *base + row->constant;
            if (row->walks || address < 0 || address > static_cast<int64_t>(isa::kDVmem.Mask()))
            {
                return operation.emitOpError()
                    << "is unable to address VMEM row " << address << ". A descriptor's VMEM address is an "
                    << "immediate, so its row must be a constant reaching at most " << isa::kDVmem.Mask() << ".";
            }
            Emit<IsaOp>(
                operation.getLoc(),
                operation.getDmaSlot(),
                static_cast<uint32_t>(address),
                operation.getRows(),
                *pointer,
                *stride
            );
            return success();
        }

        /// Lowers an scf.for into a hardware loop around its body.
        // NOLINTNEXTLINE(misc-no-recursion): recurses once per nested loop or row term, which the AGU bounds.
        LogicalResult LowerLoop(scf::ForOp loop)
        {
            APInt low;
            APInt high;
            APInt step;
            if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&low)) ||
                !matchPattern(loop.getUpperBound(), m_ConstantInt(&high)) ||
                !matchPattern(loop.getStep(), m_ConstantInt(&step)))
            {
                return loop.emitOpError("is unable to run as a hardware loop. Its bounds and step must be constants.");
            }
            if (_loops.size() >= kMaxLoopDepth || low.getZExtValue() > isa::kLoopLow.Mask() ||
                step.getZExtValue() > isa::kLoopStep.Mask() || high.getZExtValue() > isa::kLoopHigh.Mask())
            {
                return loop.emitOpError()
                    << "is unable to run as a hardware loop. The machine nests " << kMaxLoopDepth
                    << " loops, starts one at most at " << isa::kLoopLow.Mask() << ", steps it at most by "
                    << isa::kLoopStep.Mask() << " and bounds it at most at " << isa::kLoopHigh.Mask() << ".";
            }

            // Holds every loop-carried value to one register, since no instruction moves it between iterations.
            auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
            for (const auto [initial, yielded] : llvm::zip(loop.getInitArgs(), yield.getOperands()))
            {
                const FailureOr<uint32_t> entering = RegisterOf(initial, loop);
                const FailureOr<uint32_t> leaving  = RegisterOf(yielded, loop);
                if (failed(entering) || failed(leaving)) return failure();
                if (*entering != *leaving)
                {
                    return loop.emitOpError()
                        << "is unable to carry a value from register " << *leaving << " into register " << *entering
                        << ". A loop-carried value must stay in one register.";
                }
            }

            Emit<isa::LbeginOp>(
                loop.getLoc(),
                static_cast<uint32_t>(high.getZExtValue()),  // hi
                static_cast<uint32_t>(low.getZExtValue()),   // lo
                static_cast<uint32_t>(step.getZExtValue())   // step
            );
            _loops.push_back(loop);
            const LogicalResult body = LowerBlock(*loop.getBody());
            _loops.pop_back();
            if (failed(body)) return failure();
            Emit<isa::LoopEndOp>(loop.getLoc());
            return success();
        }

        /// Lowers an arith operation: an i32 constant holding a register, or an i32 addition of a constant.
        LogicalResult LowerArithmetic(Operation& operation)
        {
            // Leaves index arithmetic to the row resolution that reads it, since it never reaches a register.
            if (llvm::all_of(operation.getResultTypes(), [](const Type type) {
                    return type.isIndex();
                }))
            {
                return success();
            }

            if (auto constant = dyn_cast<arith::ConstantIntOp>(operation))
            {
                // Treats a constant with no register as an immediate its users fold in.
                if (!constant->hasAttr(TempoDialect::get_register_attribute_name())) return success();
                const int64_t value = constant.value();
                if (value < 0 || value > static_cast<int64_t>(isa::kSImmediate.Mask()))
                {
                    return constant.emitOpError() << "is unable to load " << value << " into a register. smovi "
                                                  << "carries 0 through " << isa::kSImmediate.Mask() << ".";
                }
                const FailureOr<uint32_t> rd = RegisterOf(constant.getResult(), constant);
                if (failed(rd)) return failure();
                Emit<isa::SmoviOp>(constant.getLoc(), *rd, static_cast<uint32_t>(value));
                return success();
            }

            if (auto addition = dyn_cast<arith::AddIOp>(operation))
            {
                APInt immediate;
                Value source = addition.getLhs();
                if (!matchPattern(addition.getRhs(), m_ConstantInt(&immediate)))
                {
                    source = addition.getRhs();
                    if (!matchPattern(addition.getLhs(), m_ConstantInt(&immediate)))
                    {
                        return addition.emitOpError(
                            "is unable to add two registers. A scalar addition must add a constant, which saddi "
                            "carries as its immediate."
                        );
                    }
                }
                const int64_t value = immediate.getSExtValue();
                if (value < INT16_MIN || value > INT16_MAX)
                {
                    return addition.emitOpError() << "is unable to add " << value << ". saddi carries a signed "
                                                  << "16-bit immediate.";
                }
                const FailureOr<uint32_t> rd = RegisterOf(addition.getResult(), addition);
                const FailureOr<uint32_t> rs = RegisterOf(source, addition);
                if (failed(rd) || failed(rs)) return failure();
                Emit<isa::SaddiOp>(addition.getLoc(), *rd, *rs, static_cast<uint32_t>(value));
                return success();
            }
            return operation.emitOpError("has no lowering into tempoisa.");
        }

        /// Lowers one operation into the bundles it runs as.
        // NOLINTNEXTLINE(misc-no-recursion): recurses once per nested loop or row term, which the AGU bounds.
        LogicalResult LowerOperation(Operation& operation)
        {
            if (operation.getDialect() != nullptr && llvm::isa<arith::ArithDialect>(operation.getDialect()))
            {
                return LowerArithmetic(operation);
            }
            return llvm::TypeSwitch<Operation*, LogicalResult>(&operation)
                .Case<AllocOp, scf::YieldOp>([](auto) {
                    return success();
                })
                // NOLINTNEXTLINE(misc-no-recursion): lowers a loop body, at most the AGU's four levels deep.
                .Case<scf::ForOp>([&](scf::ForOp loop) {
                    return LowerLoop(loop);
                })
                .Case<VaddOp>([&](VaddOp source) {
                    return LowerBinary<isa::VaddOp>(source);
                })
                .Case<VsubOp>([&](VsubOp source) {
                    return LowerBinary<isa::VsubOp>(source);
                })
                .Case<VmulOp>([&](VmulOp source) {
                    return LowerBinary<isa::VmulOp>(source);
                })
                .Case<VmaxOp>([&](VmaxOp source) {
                    return LowerBinary<isa::VmaxOp>(source);
                })
                .Case<VminOp>([&](VminOp source) {
                    return LowerBinary<isa::VminOp>(source);
                })
                .Case<VmovOp>([&](VmovOp source) {
                    return LowerUnary<isa::VmovOp>(source);
                })
                .Case<VgeluOp>([&](VgeluOp source) {
                    return LowerUnary<isa::VgeluOp>(source);
                })
                .Case<VexpOp>([&](VexpOp source) {
                    return LowerUnary<isa::VexpOp>(source);
                })
                .Case<VrecipOp>([&](VrecipOp source) {
                    return LowerUnary<isa::VrecipOp>(source);
                })
                .Case<VrsqrtOp>([&](VrsqrtOp source) {
                    return LowerUnary<isa::VrsqrtOp>(source);
                })
                .Case<VredsumOp>([&](VredsumOp source) {
                    return LowerUnary<isa::VredsumOp>(source);
                })
                .Case<VredmaxOp>([&](VredmaxOp source) {
                    return LowerUnary<isa::VredmaxOp>(source);
                })
                .Case<VtxinOp>([&](VtxinOp source) -> LogicalResult {
                    const FailureOr<uint32_t> ra = RegisterOf(source.getOperand(), source);
                    if (failed(ra)) return failure();
                    Emit<isa::VtxinOp>(source.getLoc(), *ra, source.getQuarter());
                    return success();
                })
                .Case<VtxoutOp>([&](VtxoutOp source) -> LogicalResult {
                    const FailureOr<uint32_t> rd = RegisterOf(source.getResult(), source);
                    if (failed(rd)) return failure();
                    Emit<isa::VtxoutOp>(source.getLoc(), *rd, source.getQuarter());
                    return success();
                })
                .Case<VldOp>([&](VldOp source) -> LogicalResult {
                    const FailureOr<uint32_t> vreg   = RegisterOf(source.getResult(), source);
                    const FailureOr<ResolvedRow> row = AddressRow(source.getBuffer(), source.getRow(), source);
                    if (failed(vreg) || failed(row)) return failure();
                    Emit<isa::VldOp>(
                        source.getLoc(),
                        *vreg,
                        static_cast<uint32_t>(row->constant),
                        row->walks,
                        row->level,
                        row->shift,
                        source.getMask()
                    );
                    return success();
                })
                .Case<VstOp>([&](VstOp source) -> LogicalResult {
                    const FailureOr<uint32_t> vreg   = RegisterOf(source.getValue(), source);
                    const FailureOr<ResolvedRow> row = AddressRow(source.getBuffer(), source.getRow(), source);
                    if (failed(vreg) || failed(row)) return failure();
                    Emit<isa::VstOp>(
                        source.getLoc(),
                        *vreg,
                        static_cast<uint32_t>(row->constant),
                        row->walks,
                        row->level,
                        row->shift,
                        source.getMask()
                    );
                    return success();
                })
                .Case<VmatloadOp>([&](VmatloadOp source) -> LogicalResult {
                    // Holds the four weight registers to consecutive registers, which vmatload reads from one base.
                    const Value quarters[kTileRegisters] =
                        {source.getQuarter0(), source.getQuarter1(), source.getQuarter2(), source.getQuarter3()};
                    const FailureOr<uint32_t> first = RegisterOf(quarters[0], source);
                    if (failed(first)) return failure();
                    for (uint32_t quarter = 1; quarter < kTileRegisters; ++quarter)
                    {
                        const FailureOr<uint32_t> next = RegisterOf(quarters[quarter], source);
                        if (failed(next)) return failure();
                        if (*next != (*first + quarter) % kVectorRegisters)
                        {
                            return source.emitOpError() << "is unable to read its weight quarter " << quarter
                                                        << " from v" << *next << ". The four quarters must sit "
                                                        << "in consecutive registers from v" << *first << ".";
                        }
                    }
                    Emit<isa::VmatloadOp>(source.getLoc(), *first);
                    return success();
                })
                .Case<VmatpushOp>([&](VmatpushOp source) -> LogicalResult {
                    const FailureOr<uint32_t> vreg = RegisterOf(source.getOperand(), source);
                    if (failed(vreg)) return failure();
                    Emit<isa::VmatpushOp>(source.getLoc(), *vreg);
                    return success();
                })
                .Case<VmatpopOp>([&](VmatpopOp source) -> LogicalResult {
                    const FailureOr<uint32_t> vreg = RegisterOf(source.getResult(), source);
                    if (failed(vreg)) return failure();
                    Emit<isa::VmatpopOp>(source.getLoc(), *vreg);
                    return success();
                })
                .Case<ArgOp>([&](ArgOp source) -> LogicalResult {
                    const FailureOr<uint32_t> rd = RegisterOf(source.getResult(), source);
                    if (failed(rd)) return failure();
                    Emit<isa::SmovArgOp>(source.getLoc(), *rd, source.getIndex());
                    return success();
                })
                .Case<VmemldOp>([&](VmemldOp source) {
                    return LowerDescriptor<isa::VmemldOp>(source);
                })
                .Case<VmemstOp>([&](VmemstOp source) {
                    return LowerDescriptor<isa::VmemstOp>(source);
                })
                .Case<FlushSlotOp>([&](FlushSlotOp source) {
                    Emit<isa::FlushSlotOp>(source.getLoc(), source.getMask());
                    return success();
                })
                .Case<VmemswapOp>([&](VmemswapOp source) {
                    Emit<isa::VmemswapOp>(source.getLoc());
                    return success();
                })
                .Default([](Operation* unknown) {
                    return unknown->emitOpError("has no lowering into tempoisa.");
                });
        }

        /// Stores the builder every bundle is created through.
        OpBuilder _builder;

        /// Stores the program the bundles are appended to.
        isa::ProgramOp _program;

        /// Stores the loops enclosing the operation being lowered, outermost first, whose depth is the AGU level.
        llvm::SmallVector<scf::ForOp, kMaxLoopDepth> _loops;
    };

    /// Provides the convert-tempo-to-isa pass over one module.
    class ConvertToIsaPass final : public mlir::tempo::impl::ConvertTempoToIsaBase<ConvertToIsaPass>
    {
      public:
        /// Replaces every kernel with the program it lowers to, leaving the module untouched on any failure.
        void runOnOperation() override
        {
            llvm::SmallVector<KernelOp> kernels(getOperation().getOps<KernelOp>());
            for (KernelOp kernel : kernels)
            {
                OpBuilder builder(kernel);
                auto program = isa::ProgramOp::create(builder, kernel.getLoc(), kernel.getSymName());
                program.getBody().emplaceBlock();
                KernelLowering lowering(program);
                if (failed(lowering.LowerBlock(kernel.getBody().front())))
                {
                    program.erase();
                    signalPassFailure();
                    return;
                }
                lowering.Finish(kernel.getLoc());
                kernel.erase();
            }
        }
    };
}  // namespace
