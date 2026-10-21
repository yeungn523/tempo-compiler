/**
 * @file
 *
 * @brief Provides the export of what the board runtime launches an image with: its manifest entry, and the constants
 * block it uploads beside the image unchanged.
 *
 * Both are read from the kernel's tile op, the one place the image's shape and its host layout are stated, so the
 * runtime launches every image with exactly what its lowering assumed.
 */

#ifndef TEMPO_TEMPO_MANIFEST_H
#define TEMPO_TEMPO_MANIFEST_H

#include <llvm/Support/raw_ostream.h>
#include "tempo/tempo/tempo.h"

namespace mlir::tempo
{
    /**
     * @brief Writes a kernel's manifest entry as a JSON object holding one member: the entry, under the name of the
     * image file.
     *
     * @param kernel the kernel, holding at most one tile op.
     * @param output the stream that receives the JSON.
     * @returns success once the entry is written, failure with a diagnostic when the kernel holds several tile ops.
     */
    LogicalResult ExportManifest(KernelOp kernel, llvm::raw_ostream& output);

    /**
     * @brief Writes the constants block a kernel's image reads unchanged on every launch, as little-endian BF16 words.
     *
     * @param kernel the kernel, holding a causal attention, the one tile op whose constants block no launch varies.
     * @param output the stream that receives the block.
     * @returns success once the block is written, failure with a diagnostic when the kernel's image has no such block.
     */
    LogicalResult ExportConstants(KernelOp kernel, llvm::raw_ostream& output);
}  // namespace mlir::tempo

#endif  // TEMPO_TEMPO_MANIFEST_H
