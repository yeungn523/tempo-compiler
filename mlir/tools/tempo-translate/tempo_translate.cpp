/**
 * @file
 *
 * @brief Provides tempo-translate, which turns a tempoisa program into image bytes and image bytes back into a
 * program, and writes what the board runtime launches a tempo kernel's image with.
 */

#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>
#include "tempo/tempo/manifest.h"
#include "tempo/tempoisa/image.h"

namespace
{
    /// Loads the tempoisa dialect, the one dialect either image translation reads or writes.
    void RegisterDialects(mlir::DialectRegistry& registry)
    {
        registry.insert<mlir::tempo::isa::ISADialect>();
    }

    /// Loads the tempo dialect, whose tile ops the launch translations read.
    void RegisterKernelDialects(mlir::DialectRegistry& registry)
    {
        registry.insert<mlir::tempo::TempoDialect>();
    }

    /// Returns the module's one tempo kernel, or a null op with a diagnostic naming `purpose` when it holds none.
    mlir::tempo::KernelOp FindKernel(mlir::Operation* operation, const llvm::StringRef purpose)
    {
        mlir::tempo::KernelOp kernel;
        operation->walk([&](mlir::tempo::KernelOp found) {
            kernel = found;
        });
        if (!kernel)
            operation->emitError() << "Unable to write " << purpose << ". The module must hold a tempo.kernel.";
        return kernel;
    }

    /// Registers the image import, which names the program after the stem of the file it reads.
    const mlir::TranslateToMLIRRegistration kImport(
        "import-tempo-image",
        "Reads a MiniTPU image into the tempoisa dialect",
        [](llvm::SourceMgr& source, mlir::MLIRContext* context) -> mlir::OwningOpRef<mlir::Operation*> {
            const llvm::MemoryBuffer* buffer = source.getMemoryBuffer(source.getMainFileID());
            const llvm::StringRef name       = llvm::sys::path::stem(buffer->getBufferIdentifier());
            return mlir::tempo::isa::ImportImage(buffer->getBuffer(), name, context);
        },
        RegisterDialects
    );

    /// Registers the image export, which writes the module's one program.
    const mlir::TranslateFromMLIRRegistration kExport(
        "export-tempo-image",
        "Writes a tempoisa program as a MiniTPU image",
        [](mlir::Operation* operation, llvm::raw_ostream& output) -> mlir::LogicalResult {
            mlir::tempo::isa::ProgramOp program;
            operation->walk([&](mlir::tempo::isa::ProgramOp found) {
                program = found;
            });
            if (!program)
            {
                return operation->emitError("Unable to write an image. The module must hold a tempoisa.program.");
            }
            return mlir::tempo::isa::ExportImage(program, output);
        },
        RegisterDialects
    );

    /// Registers the manifest export, which writes the entry the runtime launches the kernel's image with.
    const mlir::TranslateFromMLIRRegistration kManifest(
        "export-tempo-manifest",
        "Writes the manifest entry the board runtime launches a tempo kernel's image with",
        [](mlir::Operation* operation, llvm::raw_ostream& output) -> mlir::LogicalResult {
            mlir::tempo::KernelOp kernel = FindKernel(operation, "a manifest entry");
            return kernel ? mlir::tempo::ExportManifest(kernel, output) : mlir::failure();
        },
        RegisterKernelDialects
    );

    /// Registers the constants export, which writes the block the runtime uploads beside the kernel's image.
    const mlir::TranslateFromMLIRRegistration kConstants(
        "export-tempo-constants",
        "Writes the constants block a tempo kernel's image reads unchanged on every launch",
        [](mlir::Operation* operation, llvm::raw_ostream& output) -> mlir::LogicalResult {
            mlir::tempo::KernelOp kernel = FindKernel(operation, "a constants block");
            return kernel ? mlir::tempo::ExportConstants(kernel, output) : mlir::failure();
        },
        RegisterKernelDialects
    );
}  // namespace

/// Hands the command line to MLIR's translation driver.
int main(int argc, char** argv)
{
    return mlir::failed(mlir::mlirTranslateMain(argc, argv, "Tempo translation driver\n"));
}
