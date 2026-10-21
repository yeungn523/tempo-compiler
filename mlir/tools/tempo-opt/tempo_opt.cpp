/**
 * @file
 *
 * @brief Provides tempo-opt, the mlir-opt driver that parses Triton's tt dialect as the Triton 3.8.0 wheel prints it
 * and runs the MiniTPU passes over it.
 */

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>
#include <triton/Dialect/Triton/IR/Dialect.h>
#include "tempo/tempo/passes.h"
#include "tempo/tempo/tempo.h"
#include "tempo/tempoisa/passes.h"
#include "tempo/tempoisa/tempoisa.h"

/// Registers every dialect and pass the pipeline reads and runs, then hands the command line to MLIR's driver.
int main(int argc, char** argv)
{
    mlir::registerTransformsPasses();
    mlir::tempo::isa::registerTempoISAPasses();
    mlir::tempo::registerTempoPasses();

    mlir::DialectRegistry registry;
    registry.insert<
        mlir::triton::TritonDialect,
        mlir::arith::ArithDialect,
        mlir::math::MathDialect,
        mlir::scf::SCFDialect,
        mlir::cf::ControlFlowDialect,
        mlir::ub::UBDialect,
        mlir::tempo::TempoDialect,
        mlir::tempo::isa::ISADialect>();

    return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Tempo optimizer driver\n", registry));
}
