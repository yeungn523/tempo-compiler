/**
 * @file
 *
 * @brief Implements the minitpuisa dialect: its register syntax, the bundle and program verifiers, and the
 * instructions TableGen generates from minitpuisa_ops.td.
 *
 * @note The parse, print, verify and initialize names below are the ones MLIR's generated code calls, so they keep
 * MLIR's spelling.
 */

#include "minitpu/minitpuisa/minitpuisa.h"
#include <array>

using namespace mlir;
using namespace mlir::minitpu::isa;

namespace
{
    /// Stores the base a register index is written in.
    constexpr uint32_t kRegisterIndexRadix = 10;

    /**
     * @brief Parses a register written as its prefix and index, such as v3 or s1.
     *
     * @param parser the parser positioned at the register.
     * @param attribute the attribute that receives the register index.
     * @param prefix the letter naming the register file.
     * @returns success once the register is parsed, failure with a diagnostic otherwise.
     */
    ParseResult ParseRegister(OpAsmParser& parser, IntegerAttr& attribute, const char prefix)
    {
        const SMLoc location = parser.getCurrentLocation();
        StringRef keyword;
        if (parser.parseKeyword(&keyword)) return failure();

        uint32_t index = 0;
        if (keyword.size() < 2 || keyword.front() != prefix ||
            keyword.drop_front().getAsInteger(kRegisterIndexRadix, index))
        {
            return parser.emitError(location) << "Unable to parse the register. It must be written as '" << prefix
                                              << "<index>', but got '" << keyword << "'.";
        }
        attribute = parser.getBuilder().getI32IntegerAttr(static_cast<int32_t>(index));
        return success();
    }

    /// Returns the letter naming a slot in diagnostics.
    char SlotLetter(const kSlot slot)
    {
        switch (slot)
        {
            case kSlot::kE: return 'E';
            case kSlot::kM: return 'M';
            case kSlot::kX: return 'X';
            case kSlot::kS: return 'S';
            case kSlot::kF: return 'F';
            case kSlot::kD: return 'D';
        }
        return '?';
    }

    /// Determines whether a bundle carries the halt instruction.
    bool CarriesHalt(Operation& bundle)
    {
        return llvm::any_of(cast<BundleOp>(bundle).getBody().front(), [](Operation& operation) {
            return llvm::isa<HaltOp>(operation);
        });
    }
}  // namespace

/// Parses a vector register written as v<index>, which the custom<VectorRegister> directive names.
static ParseResult parseVectorRegister(OpAsmParser& parser, IntegerAttr& attribute)
{
    return ParseRegister(parser, attribute, 'v');
}

/// Parses a scalar register written as s<index>, which the custom<ScalarRegister> directive names.
static ParseResult parseScalarRegister(OpAsmParser& parser, IntegerAttr& attribute)
{
    return ParseRegister(parser, attribute, 's');
}

/// Prints a vector register as v<index>.
static void printVectorRegister(OpAsmPrinter& printer, Operation*, const IntegerAttr attribute)
{
    printer << 'v' << attribute.getInt();
}

/// Prints a scalar register as s<index>.
static void printScalarRegister(OpAsmPrinter& printer, Operation*, const IntegerAttr attribute)
{
    printer << 's' << attribute.getInt();
}

#include "minitpu/minitpuisa/minitpuisa_dialect.cpp.inc"
#include "minitpu/minitpuisa/minitpuisa_interfaces.cpp.inc"

/// Registers every instruction and container the dialect defines.
void ISADialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "minitpu/minitpuisa/minitpuisa_ops.cpp.inc"
        >();
}

/// Verifies that the bundle holds machine instructions alone, one per slot, and a descriptor only by itself.
LogicalResult BundleOp::verify()
{
    std::array<bool, kSlotCount> taken {};
    Operation* descriptor = nullptr;
    uint32_t instructions = 0;
    for (Operation& operation : getBody().front())
    {
        auto instruction = dyn_cast<SlotOpInterface>(operation);
        if (!instruction)
        {
            return operation.emitOpError(
                "is unable to issue in a bundle. A bundle must hold machine instructions alone."
            );
        }

        const kSlot slot    = instruction.get_slot();
        bool& slot_is_taken = taken[static_cast<size_t>(slot)];
        if (slot_is_taken)
        {
            return operation.emitOpError() << "is unable to claim the " << SlotLetter(slot)
                                           << " slot. A bundle must carry one instruction per slot, but an earlier "
                                              "instruction in this bundle already holds it.";
        }
        slot_is_taken = true;
        if (slot == kSlot::kD) descriptor = &operation;
        ++instructions;
    }

    // Rejects a descriptor sharing its bundle, because the descriptor overlays every compute slot and whatever shares
    // the bundle with it would be lost.
    if (descriptor != nullptr && instructions > 1)
    {
        return descriptor->emitOpError() << "is unable to share its bundle. A DMA descriptor must issue alone, but "
                                            "the bundle carries "
                                         << instructions << " instructions.";
    }
    return success();
}

/// Verifies that the program holds bundles alone and ends in one carrying halt.
LogicalResult ProgramOp::verify()
{
    Block& body = getBody().front();
    for (Operation& operation : body)
    {
        if (!llvm::isa<BundleOp>(operation))
        {
            return operation.emitOpError("is unable to stand in a program. A program must hold bundles alone.");
        }
    }

    // Requires the halt, because a launch completes only once the machine halts.
    if (body.empty() || !CarriesHalt(body.back()))
    {
        return emitOpError("is unable to complete a launch. A program must end in a bundle carrying halt.");
    }
    return success();
}

#define GET_OP_CLASSES
#include "minitpu/minitpuisa/minitpuisa_ops.cpp.inc"
