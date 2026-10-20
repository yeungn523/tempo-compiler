/**
 * @file passes.td
 * @brief Provides the passes that run over minitpu kernels.
 */

#ifndef MINITPU_MINITPU_PASSES_TD
#define MINITPU_MINITPU_PASSES_TD

include "mlir/Pass/PassBase.td"

/**
 * @brief Provides the conversion of every Triton kernel into a minitpu kernel holding the one tile op that computes it.
 */
def ConvertTritonToMinitpu : Pass<"convert-tt-to-minitpu", "::mlir::ModuleOp">
{
    let summary = "Convert every Triton kernel into a minitpu kernel of one tile op";
    let dependentDialects = ["::mlir::minitpu::MinitpuDialect"];
    let options = [
        Option<"heads", "attention-heads", "uint32_t", "1",
               "Heads an attention image walks, the Triton kernel attending one of them">
    ];
}

/**
 * @brief Provides the lowering of every minitpu kernel into a minitpuisa program, one bundle per machine operation.
 */
def ConvertMinitpuToIsa : Pass<"convert-minitpu-to-isa", "::mlir::ModuleOp">
{
    let summary = "Lower every minitpu kernel into a minitpuisa program";
    let dependentDialects = ["::mlir::minitpu::isa::ISADialect"];
}

/**
 * @brief Provides the expansion of every tile op into the machine operations that compute it.
 */
def MinitpuTileToVector : Pass<"minitpu-tile-to-vector", "::mlir::minitpu::KernelOp">
{
    let summary = "Expand every tile op into machine operations over virtual registers and buffers";
    let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::scf::SCFDialect"];
}

/**
 * @brief Provides the VMEM allocator, which gives every buffer without a pinned base the next free rows.
 */
def MinitpuAllocateVmem : Pass<"minitpu-allocate-vmem", "::mlir::minitpu::KernelOp">
{
    let summary = "Give every unpinned buffer the next free VMEM rows";
}

/**
 * @brief Provides the register allocator, which names a vector or scalar register for every value without one.
 */
def MinitpuAllocateRegisters : Pass<"minitpu-allocate-registers", "::mlir::minitpu::KernelOp">
{
    let summary = "Name a register for every vector and scalar value";
}

#endif  // MINITPU_MINITPU_PASSES_TD
