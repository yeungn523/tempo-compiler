/**
 * @file passes.td
 * @brief Provides the passes that run over minitpu kernels.
 */

#ifndef MINITPU_MINITPU_PASSES_TD
#define MINITPU_MINITPU_PASSES_TD

include "mlir/Pass/PassBase.td"

/**
 * @brief Provides the lowering of every minitpu kernel into a minitpuisa program, one bundle per machine operation.
 *
 * @details Every vector and scalar value must name its register through minitpu.reg and every buffer its row through
 * base, as the allocation passes write them. A vld or vst row resolves to the instruction's static row plus an
 * induction term, being a constant or a constant plus an enclosing loop's induction variable shifted left by a power of
 * two. Each scf.for becomes lbegin and loop_end, and a loop-carried value must stay in one register across its initial
 * value, its iterations and its result. An i32 constant carrying a register becomes smovi, an i32 addition of a
 * constant becomes saddi, and the program ends in halt. Every bundle leaves with no delay for the scheduler to fill.
 */
def ConvertMinitpuToIsa : Pass<"convert-minitpu-to-isa", "::mlir::ModuleOp">
{
    let summary = "Lower every minitpu kernel into a minitpuisa program";
    let dependentDialects = ["::mlir::minitpu::isa::ISADialect"];
}

/**
 * @brief Provides the VMEM allocator, which gives every buffer without a pinned base the next free rows.
 *
 * @details Buffers are laid out in program order from row zero, each starting past the last and stepping over any
 * pinned buffer it would overlap. No rows are reused, so a buffer a DMA descriptor still fills or drains is never
 * handed to another, a hazard the machine interlocks nothing against. A kernel whose buffers pass a VMEM slot's rows
 * is refused, naming each buffer.
 */
def MinitpuAllocateVmem : Pass<"minitpu-allocate-vmem", "::mlir::minitpu::KernelOp">
{
    let summary = "Give every unpinned buffer the next free VMEM rows";
}

/**
 * @brief Provides the register allocator, which names a vector or scalar register for every value without one.
 *
 * @details A value is live from its definition to its last use, and through the whole of any loop it is defined outside
 * of and read inside. A loop-carried value shares one register across its initial value, its iterations and its result,
 * and an initial value read elsewhere is copied first so sharing it is sound. The four operands of a vmatload take four
 * consecutive registers. Pinned values keep their register, and every other value takes the lowest register free over
 * its whole lifetime. A point where more values are live than the file holds is refused, naming how many are live.
 */
def MinitpuAllocateRegisters : Pass<"minitpu-allocate-registers", "::mlir::minitpu::KernelOp">
{
    let summary = "Name a register for every vector and scalar value";
}

#endif  // MINITPU_MINITPU_PASSES_TD
