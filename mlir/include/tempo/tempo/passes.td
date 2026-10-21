/**
 * @file passes.td
 * @brief Provides the passes that run over tempo kernels.
 */

#ifndef TEMPO_TEMPO_PASSES_TD
#define TEMPO_TEMPO_PASSES_TD

include "mlir/Pass/PassBase.td"

/**
 * @brief Provides the conversion of every Triton kernel into a tempo kernel holding the one tile op that computes it.
 */
def ConvertTritonToTempo : Pass<"convert-tt-to-tempo", "::mlir::ModuleOp">
{
    let summary = "Convert every Triton kernel into a tempo kernel of one tile op";
    let dependentDialects = ["::mlir::tempo::TempoDialect"];
    let options = [
        Option<"heads", "attention-heads", "uint32_t", "1",
               "Heads an attention image walks, the Triton kernel attending one of them">
    ];
}

/**
 * @brief Provides the lowering of every tempo kernel into a tempoisa program, one bundle per machine operation.
 */
def ConvertTempoToIsa : Pass<"convert-tempo-to-isa", "::mlir::ModuleOp">
{
    let summary = "Lower every tempo kernel into a tempoisa program";
    let dependentDialects = ["::mlir::tempo::isa::ISADialect"];
}

/**
 * @brief Provides the expansion of every tile op into the machine operations that compute it.
 */
def TempoTileToVector : Pass<"tempo-tile-to-vector", "::mlir::tempo::KernelOp">
{
    let summary = "Expand every tile op into machine operations over virtual registers and buffers";
    let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect"];
}

/**
 * @brief Provides the VMEM allocator, which gives every buffer without a pinned base the next free rows.
 */
def TempoAllocateVmem : Pass<"tempo-allocate-vmem", "::mlir::tempo::KernelOp">
{
    let summary = "Give every unpinned buffer the next free VMEM rows";
}

/**
 * @brief Provides the register allocator, which names a vector or scalar register for every value without one.
 */
def TempoAllocateRegisters : Pass<"tempo-allocate-registers", "::mlir::tempo::KernelOp">
{
    let summary = "Name a register for every vector and scalar value";
}

#endif  // TEMPO_TEMPO_PASSES_TD
