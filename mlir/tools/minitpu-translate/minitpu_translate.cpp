/**
 * @file
 *
 * @brief Provides minitpu-translate, which turns a minitpuisa program into image bytes and image bytes back into a
 * program, and writes what the board runtime launches a minitpu kernel's image with.
 */

#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>
#include "minitpu/minitpu/manifest.h"
#include "minitpu/minitpuisa/image.h"

namespace
{
    /// Loads the minitpuisa dialect, the one dialect either image translation reads or writes.
    void RegisterDialects(mlir::DialectRegistry& registry)
    {
        registry.insert<mlir::minitpu::isa::ISADialect>();
    }

    /// Loads the minitpu dialect, whose tile ops the launch translations read.
    void RegisterKernelDialects(mlir::DialectRegistry& registry)
    {
        registry.insert<mlir::minitpu::MinitpuDialect>();
    }

    /// Returns the module's one minitpu kernel, or a null op with a diagnostic naming `purpose` when it holds none.
    mlir::minitpu::KernelOp FindKernel(mlir::Operation* operation, const llvm::StringRef purpose)
    {
        mlir::minitpu::KernelOp kernel;
        operation->walk([&](mlir::minitpu::KernelOp found) {
            kernel = found;
        });
        if (!kernel)
            operation->emitError() << "Unable to write " << purpose << ". The module must hold a minitpu.kernel.";
        return kernel;
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

    /// Registers the manifest export, which writes the entry the runtime launches the kernel's image with.
    const mlir::TranslateFromMLIRRegistration kManifest(
        "export-minitpu-manifest",
        "Writes the manifest entry the board runtime launches a minitpu kernel's image with",
        [](mlir::Operation* operation, llvm::raw_ostream& output) -> mlir::LogicalResult {
            mlir::minitpu::KernelOp kernel = FindKernel(operation, "a manifest entry");
            return kernel ? mlir::minitpu::ExportManifest(kernel, output) : mlir::failure();
        },
        RegisterKernelDialects
    );

    /// Registers the constants export, which writes the block the runtime uploads beside the kernel's image.
    const mlir::TranslateFromMLIRRegistration kConstants(
        "export-minitpu-constants",
        "Writes the constants block a minitpu kernel's image reads unchanged on every launch",
        [](mlir::Operation* operation, llvm::raw_ostream& output) -> mlir::LogicalResult {
            mlir::minitpu::KernelOp kernel = FindKernel(operation, "a constants block");
            return kernel ? mlir::minitpu::ExportConstants(kernel, output) : mlir::failure();
        },
        RegisterKernelDialects
    );
}  // namespace

/// Hands the command line to MLIR's translation driver.
int main(int argc, char** argv)
{
    return mlir::failed(mlir::mlirTranslateMain(argc, argv, "MiniTPU translation driver\n"));
}
