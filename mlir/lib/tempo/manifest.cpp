/**
 * @file
 *
 * @brief Implements the export of the manifest entry and the constants block the board runtime launches an image with.
 */

#include "tempo/tempo/manifest.h"
#include <llvm/ADT/Twine.h>
#include <llvm/ADT/bit.h>
#include <llvm/Support/JSON.h>
#include <array>
#include <cstdint>
#include <string>
#include "tile_layout.h"

using namespace mlir;
using namespace mlir::tempo;
using namespace mlir::tempo::tile;

namespace
{
    /// Stores the indentation the JSON is printed with, which keeps an entry legible when the tool is run by hand.
    constexpr unsigned kJsonIndent = 2;

    /// Stores the per-feature rows a LayerNorm's constants block opens with, in order, as the runtime's packer names
    /// the arrays it writes them from.
    constexpr std::array<llvm::StringLiteral, 2> kNormRows {"gamma_pointer", "beta_pointer"};

    /// Stores the low half of a float32's bits, which rounding to BF16 drops.
    constexpr uint32_t kHalfBits = 16;

    /// Stores the bias that rounds a float32's dropped half to nearest, ties going to even.
    constexpr uint32_t kRoundingBias = 0x7FFF;

    /// Stores the bits in one byte of the block.
    constexpr uint32_t kByteBits = 8;

    /// Stores the mask selecting one byte of a word.
    constexpr uint32_t kByteMask = 0xFF;

    /**
     * @brief Returns the one tile op `kernel` holds.
     *
     * @param kernel the kernel.
     * @returns the tile op, a null pointer for a kernel holding only its arguments, or failure with a diagnostic when
     * it holds several.
     */
    FailureOr<Operation*> TileOp(KernelOp kernel)
    {
        Operation* found = nullptr;
        for (Operation& operation : kernel.getBody().front())
        {
            if (llvm::isa<ArgOp>(operation)) continue;
            if (found != nullptr)
            {
                // Reports at the kernel's location alone, since the op itself would print its body beside the error.
                return emitError(kernel.getLoc())
                    << "Unable to describe the " << kernel.getSymName() << " kernel by one manifest entry. A kernel "
                    << "is one image running one tile op, but it holds " << found->getName() << " and "
                    << operation.getName() << ".";
            }
            found = &operation;
        }
        return found;
    }

    /// Returns the name of the image file a kernel's tile op compiles to, or of the halt image a kernel holding none
    /// compiles to.
    std::string ImageName(Operation* operation)
    {
        if (operation == nullptr) return "halt";
        if (auto product = llvm::dyn_cast<MatmulOp>(operation))
        {
            return (llvm::Twine("gemm_") + llvm::Twine(product.getDepth()) + "x" + llvm::Twine(product.getColumns()) +
                    (product.getActivate() ? "_gelu" : "") + (product.getBias() ? "_bias" : ""))
                .str();
        }
        if (auto sum = llvm::dyn_cast<AddOp>(operation))
        {
            return (llvm::Twine("add_") + (sum.getBroadcast() ? "bias_" : "") + llvm::Twine(sum.getFeatures())).str();
        }
        if (auto norm = llvm::dyn_cast<LayernormOp>(operation))
        {
            return (llvm::Twine("layernorm_") + llvm::Twine(norm.getFeatures())).str();
        }
        if (auto attention = llvm::dyn_cast<AttentionOp>(operation))
        {
            return (llvm::Twine(attention.getCausal() ? "attention_" : "decode_attention_") +
                    llvm::Twine(attention.getKeys()))
                .str();
        }
        auto copy = llvm::cast<CopyOp>(operation);
        return (llvm::Twine("copy_") + llvm::Twine(copy.getRows()) + "x" + llvm::Twine(copy.getStride())).str();
    }

    /// Returns the name of the file a causal attention's constants block is written to beside its image.
    std::string ConstantsName(Operation* operation)
    {
        return ImageName(operation) + "_constants.bin";
    }

    /// Writes the fields of one image's manifest entry into the object `stream` has open.
    void WriteEntry(llvm::json::OStream& stream, Operation* operation)
    {
        if (operation == nullptr)
        {
            stream.attribute("kind", "halt");
            return;
        }
        if (auto product = llvm::dyn_cast<MatmulOp>(operation))
        {
            stream.attribute("kind", product.getBias() ? "gemm_bias" : "gemm");
            stream.attribute("depth", product.getDepth());
            stream.attribute("columns", product.getColumns());
            stream.attribute("activate", product.getActivate());

            // States the words the drain buffer takes, which the runtime allocates: the zeroed tile, then one row of
            // each column tile per token.
            stream.attribute("drain_words", int64_t {kZeroRows} * kTile + int64_t {product.getColumns()} * kTokens);
            return;
        }
        if (auto sum = llvm::dyn_cast<AddOp>(operation))
        {
            stream.attribute("kind", sum.getBroadcast() ? "add_bias" : "add");
            stream.attribute("features", sum.getFeatures());
            return;
        }
        if (auto norm = llvm::dyn_cast<LayernormOp>(operation))
        {
            // States what the constants block holds after gamma and beta: the reciprocal width, then epsilon.
            stream.attribute("kind", "layernorm");
            stream.attribute("features", norm.getFeatures());
            stream.attributeArray("rows", [&] {
                for (const llvm::StringLiteral row : kNormRows) stream.value(row);
            });
            stream.attributeArray("scalars", [&] {
                stream.value(1.0 / norm.getFeatures());
                stream.value(norm.getEpsilon().convertToDouble());
            });
            return;
        }
        if (auto attention = llvm::dyn_cast<AttentionOp>(operation))
        {
            if (attention.getCausal())
            {
                stream.attribute("kind", "attention");
                stream.attribute("tokens", attention.getKeys());
                stream.attribute("heads", attention.getHeads());
                stream.attribute("constants", ConstantsName(operation));
                return;
            }

            // States what the runtime packs each step's constants block from, its masks laid out keys by queries.
            stream.attribute("kind", "decode_attention");
            stream.attribute("keys", attention.getKeys());
            stream.attribute("heads", attention.getHeads());
            stream.attribute("entering", kEnteringMaximum);
            stream.attributeArray("stated", [] {
            });
            stream.attribute("transposed_masks", true);
            return;
        }
        auto copy = llvm::cast<CopyOp>(operation);
        stream.attribute("kind", "copy");
        stream.attribute("rows", copy.getRows());
        stream.attribute("stride", copy.getStride());
    }

    /// Returns a value narrowed to float32 in the BF16 container, rounding to nearest even as the host's packers do.
    uint16_t Bf16(const double value)
    {
        const auto bits = llvm::bit_cast<uint32_t>(static_cast<float>(value));
        return static_cast<uint16_t>((bits + kRoundingBias + ((bits >> kHalfBits) & 1U)) >> kHalfBits);
    }

    /// Writes one BF16 word little-endian.
    void WriteWord(llvm::raw_ostream& output, const uint16_t word)
    {
        output << static_cast<char>(word & kByteMask) << static_cast<char>(word >> kByteBits);
    }
}  // namespace

LogicalResult mlir::tempo::ExportManifest(KernelOp kernel, llvm::raw_ostream& output)
{
    const FailureOr<Operation*> operation = TileOp(kernel);
    if (failed(operation)) return failure();
    llvm::json::OStream stream(output, kJsonIndent);
    stream.object([&] {
        stream.attributeObject(ImageName(*operation) + ".bin", [&] {
            WriteEntry(stream, *operation);
        });
    });
    output << "\n";
    return success();
}

LogicalResult mlir::tempo::ExportConstants(KernelOp kernel, llvm::raw_ostream& output)
{
    const FailureOr<Operation*> operation = TileOp(kernel);
    if (failed(operation)) return failure();
    auto attention = llvm::dyn_cast_if_present<AttentionOp>(*operation);
    if (!attention || !attention.getCausal())
    {
        return emitError(kernel.getLoc())
            << "Unable to export a constants block from the " << kernel.getSymName() << " kernel. Only a causal "
            << "attention's image reads one no launch varies, a decode attention's masks following each step's length.";
    }

    // Writes the stated tiles, then the diagonal mask tile laid out keys by queries, masking each key past its query.
    for (uint32_t row = 0; row < kTile; ++row)
    {
        for (uint32_t column = 0; column < kTile; ++column) WriteWord(output, Bf16(kEnteringMaximum));
    }
    for (uint32_t row = 0; row < kTile; ++row)
    {
        for (uint32_t column = 0; column < kTile; ++column) WriteWord(output, Bf16(0.0));
    }
    for (uint32_t key = 0; key < kTile; ++key)
    {
        for (uint32_t query = 0; query < kTile; ++query) WriteWord(output, Bf16(key > query ? kMaskedScore : 0.0));
    }
    return success();
}
