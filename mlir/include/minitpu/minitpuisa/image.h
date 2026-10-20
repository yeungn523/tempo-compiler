/**
 * @file
 *
 * @brief Provides the translation between a minitpuisa program and the image bytes the board's IRAM loader takes.
 *
 * An image is the program's bundles in fetch order, each 16 little-endian bytes.
 */

#ifndef MINITPU_MINITPUISA_IMAGE_H
#define MINITPU_MINITPUISA_IMAGE_H

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include "minitpu/minitpuisa/minitpuisa.h"

namespace mlir::minitpu::isa
{
    /**
     * @brief Returns the 128-bit word one bundle encodes to.
     *
     * @param bundle the bundle to encode, which the verifier has already held to one instruction per slot.
     * @returns the bundle word.
     */
    Word EncodeBundle(BundleOp bundle);

    /**
     * @brief Writes a program as image bytes.
     *
     * @param program the program to write.
     * @param output the stream that receives the image.
     * @returns success once every bundle is written.
     */
    LogicalResult ExportImage(ProgramOp program, llvm::raw_ostream& output);

    /**
     * @brief Reads image bytes into a module holding one program.
     *
     * Every bundle is re-encoded once decoded and held to the word it came from, so a bit no instruction field models
     * fails the import with the bundle's index rather than vanishing.
     *
     * @param bytes the image.
     * @param name the program's symbol name, which also names every bundle's location with the bundle's index.
     * @param context the context the minitpuisa dialect is loaded into.
     * @returns the module, or a null reference once a diagnostic has been emitted.
     */
    OwningOpRef<ModuleOp> ImportImage(llvm::StringRef bytes, llvm::StringRef name, MLIRContext* context);
}  // namespace mlir::minitpu::isa

#endif  // MINITPU_MINITPUISA_IMAGE_H
