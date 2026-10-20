/**
 * @file
 *
 * @brief Provides minitpu-translate, which turns a minitpuisa program into image bytes and image bytes back into a
 * program.
 */

#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>
#include "minitpu/minitpuisa/image.h"

namespace
{
    /// Loads the minitpuisa dialect, the one dialect either translation reads or writes.
    void RegisterDialects(mlir::DialectRegistry& registry)
    {
        registry.insert<mlir::minitpu::isa::ISADialect>();
    }

    /// Registers the image import, which names the program after the stem of the file it reads.
    const mlir::TranslateToMLIRRegistration kImport(
        "import-minitpu-image",
        "Reads a MiniTPU image into the minitpuisa dialect",
        [](llvm::SourceMgr& source, mlir::MLIRContext* context) -> mlir::OwningOpRef<mlir::Operation*> {
            const llvm::MemoryBuffer* buffer = source.getMemoryBuffer(source.getMainFileID());
            const llvm::StringRef name       = llvm::sys::path::stem(buffer->getBufferIdentifier());
            return mlir::minitpu::isa::ImportImage(buffer->getBuffer(), name, context);
        },
        RegisterDialects
    );

    /// Registers the image export, which writes the module's one program.
    const mlir::TranslateFromMLIRRegistration kExport(
        "export-minitpu-image",
        "Writes a minitpuisa program as a MiniTPU image",
        [](mlir::Operation* operation, llvm::raw_ostream& output) -> mlir::LogicalResult {
            mlir::minitpu::isa::ProgramOp program;
            operation->walk([&](mlir::minitpu::isa::ProgramOp found) {
                program = found;
            });
            if (!program)
            {
                return operation->emitError("Unable to write an image. The module must hold a minitpuisa.program.");
            }
            return mlir::minitpu::isa::ExportImage(program, output);
        },
        RegisterDialects
    );
}  // namespace

/// Hands the command line to MLIR's translation driver.
int main(int argc, char** argv)
{
    return mlir::failed(mlir::mlirTranslateMain(argc, argv, "MiniTPU translation driver\n"));
}
