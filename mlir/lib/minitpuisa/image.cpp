/**
 * @file
 *
 * @brief Implements the translation between a minitpuisa program and image bytes.
 */

#include "minitpu/minitpuisa/image.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Format.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Verifier.h>
#include <array>

using namespace mlir;
using namespace mlir::minitpu::isa;

namespace
{
    /// Stores the compute slots a non-DMA bundle may carry, in the order they are decoded.
    constexpr std::array<kSlot, 5> kComputeSlots {kSlot::kE, kSlot::kM, kSlot::kX, kSlot::kS, kSlot::kF};

    /// Stores the bits in each half of a bundle word as diagnostics print it.
    constexpr uint32_t kHalfBits = kBundleBits / 2;

    /// Stores the hexadecimal digits in each half of a bundle word as diagnostics print it.
    constexpr uint32_t kHalfDigits = kHalfBits / 4;

    /// Stores the mask selecting one byte of a word.
    constexpr uint32_t kByteMask = 0xFF;

    /// Stores the decoder of every instruction, keyed by its slot and the code its payload carries.
    using DecoderTable = llvm::DenseMap<std::pair<uint32_t, uint32_t>, const SlotOpInterface::Concept*>;

    /// Returns the decoder of every slot instruction the context has registered.
    DecoderTable CollectDecoders(MLIRContext* context)
    {
        DecoderTable decoders;
        for (const RegisteredOperationName name : context->getRegisteredOperations())
        {
            const auto* instruction = name.getInterface<SlotOpInterface>();
            if (instruction == nullptr) continue;
            decoders[{static_cast<uint32_t>(instruction->get_slot()), instruction->get_code()}] = instruction;
        }
        return decoders;
    }

    /// Returns a bundle word as the 32 hexadecimal digits diagnostics print.
    std::string FormatWord(const Word word)
    {
        std::string text;
        llvm::raw_string_ostream stream(text);
        stream << "0x" << llvm::format_hex_no_prefix(static_cast<uint64_t>(word >> kHalfBits), kHalfDigits)
               << llvm::format_hex_no_prefix(static_cast<uint64_t>(word), kHalfDigits);
        return text;
    }

    /// Returns the bundle word stored little-endian at the start of `bytes`.
    Word ReadWord(const llvm::StringRef bytes)
    {
        Word word = 0;
        for (uint32_t index = kBundleBytes; index > 0; --index)
        {
            word = (word << kBitsPerByte) | static_cast<uint8_t>(bytes[index - 1]);
        }
        return word;
    }

    /**
     * @brief Builds the instruction one slot payload carries, at the builder's insertion point.
     *
     * @param builder the builder positioned inside the bundle.
     * @param location the location the instruction carries.
     * @param decoders the decoder of every registered instruction.
     * @param slot the slot the payload came from.
     * @param payload the slot's payload.
     * @returns success once the instruction is built, failure with a diagnostic when no instruction carries the code.
     */
    LogicalResult DecodeSlot(
        OpBuilder& builder,
        const Location location,
        const DecoderTable& decoders,
        const kSlot slot,
        const uint64_t payload
    )
    {
        const uint32_t code = CodeField(slot).Extract(payload);
        const auto decoder  = decoders.find({static_cast<uint32_t>(slot), code});
        if (decoder == decoders.end())
        {
            return emitError(location) << "Unable to decode the bundle. Its slot " << static_cast<uint32_t>(slot)
                                       << " payload must carry a code an instruction defines, but got " << code << ".";
        }
        decoder->second->Decode(builder, location, payload);
        return success();
    }
}  // namespace

Word mlir::minitpu::isa::EncodeBundle(BundleOp bundle)
{
    Word word        = 0;
    bool carries_dma = false;
    for (Operation& operation : bundle.getBody().front())
    {
        auto instruction = cast<SlotOpInterface>(operation);
        const kSlot slot = instruction.get_slot();
        carries_dma      = carries_dma || slot == kSlot::kD;
        word |= SlotField(slot).PlaceIn(instruction.Encode());
    }
    if (carries_dma) return word | (Word {1} << kDmaBit) | kDmaDelay.PlaceIn(bundle.getDelay());
    return word | kDelay.PlaceIn(bundle.getDelay());
}

LogicalResult mlir::minitpu::isa::ExportImage(ProgramOp program, llvm::raw_ostream& output)
{
    for (Operation& operation : program.getBody().front())
    {
        const Word word = EncodeBundle(cast<BundleOp>(operation));
        for (uint32_t index = 0; index < kBundleBytes; ++index)
        {
            output << static_cast<char>(static_cast<uint8_t>(word >> (index * kBitsPerByte)) & kByteMask);
        }
    }
    return success();
}

OwningOpRef<ModuleOp>
mlir::minitpu::isa::ImportImage(const llvm::StringRef bytes, const llvm::StringRef name, MLIRContext* context)
{
    context->getOrLoadDialect<ISADialect>();
    const Location image_location = NameLoc::get(StringAttr::get(context, name));
    if (bytes.empty() || bytes.size() % kBundleBytes != 0)
    {
        emitError(image_location) << "Unable to read the image. Its size must be a positive multiple of "
                                  << kBundleBytes << " bytes, one bundle each, but got " << bytes.size() << ".";
        return nullptr;
    }

    OpBuilder builder(context);
    OwningOpRef<ModuleOp> module = ModuleOp::create(image_location);
    builder.setInsertionPointToEnd(module->getBody());
    auto program = ProgramOp::create(builder, image_location, name);
    program.getBody().emplaceBlock();

    const DecoderTable decoders = CollectDecoders(context);
    const uint32_t bundle_count = bytes.size() / kBundleBytes;
    for (uint32_t index = 0; index < bundle_count; ++index)
    {
        // Names each bundle by its position in the image, since an image has no source lines to point a diagnostic at.
        const Location location =
            NameLoc::get(StringAttr::get(context, llvm::Twine(name) + " bundle " + llvm::Twine(index)));
        const Word word   = ReadWord(bytes.substr(index * kBundleBytes, kBundleBytes));
        const bool is_dma = ((word >> kDmaBit) & 1) != 0;

        builder.setInsertionPointToEnd(&program.getBody().front());
        auto bundle = BundleOp::create(
            builder,
            location,
            static_cast<uint32_t>(is_dma ? kDmaDelay.ExtractFrom(word) : kDelay.ExtractFrom(word))
        );
        builder.setInsertionPointToEnd(&bundle.getBody().emplaceBlock());

        if (is_dma)
        {
            if (failed(DecodeSlot(builder, location, decoders, kSlot::kD, kDSlot.ExtractFrom(word)))) return nullptr;
        }
        else
        {
            for (const kSlot slot : kComputeSlots)
            {
                const uint64_t payload = SlotField(slot).ExtractFrom(word);
                if (payload == 0) continue;
                if (failed(DecodeSlot(builder, location, decoders, slot, payload))) return nullptr;
            }
        }

        // Holds the decoded bundle to the word it came from, which catches every bit no instruction field models.
        const Word encoded = EncodeBundle(bundle);
        if (encoded != word)
        {
            emitError(location) << "Unable to decode the bundle exactly. Its instructions must encode back to "
                                << FormatWord(word) << ", but encode to " << FormatWord(encoded) << ".";
            return nullptr;
        }
    }

    if (failed(mlir::verify(*module))) return nullptr;
    return module;
}
