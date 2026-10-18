/**
 * @file passes.td
 * @brief Provides the passes that run over minitpuisa programs.
 */

#ifndef MINITPU_MINITPUISA_PASSES_TD
#define MINITPU_MINITPUISA_PASSES_TD

include "mlir/Pass/PassBase.td"

/**
 * @brief Provides the scheduler, which fills every bundle's delay so the program is legal on a machine that interlocks
 * nothing.
 *
 * @details Each bundle issues at the earliest cycle its vector and scalar operands, the matrix controller, read port
 * A, the VMEM compute port, and the single vector write port allow, and the gap before it becomes the previous
 * bundle's delay. A gap past the delay field's reach is bridged with empty bundles. The program keeps its order, so
 * scalar WAW and WAR hazards and VMEM ownership stay with whatever placed the bundles. Five hazards no delay repairs
 * are refused before any delay is placed. Every latency and capacity is read from codegen/minitpuisa.td.
 */
def MinitpuisaSchedule : Pass<"minitpuisa-schedule", "::mlir::minitpu::isa::ProgramOp">
{
    let summary = "Fill each bundle's delay so the program meets every latency the machine leaves to software";
}

/**
 * @brief Provides the packer, which merges independent instructions into shared bundles ahead of the scheduler.
 *
 * @details The program splits at its barriers, being any bundle carrying a descriptor or an M, S or F instruction, and
 * each barrier keeps its position. Between them, every bundle is list-scheduled to the earliest cycle its operands, its
 * slots and read port A allow, and bundles landing on one cycle merge. Empty bundles inside a region are dropped,
 * because the scheduler re-derives every gap, while those directly before a barrier are kept as the scalar writeback
 * gap a descriptor reads across.
 */
def MinitpuisaPack : Pass<"minitpuisa-pack", "::mlir::minitpu::isa::ProgramOp">
{
    let summary = "Merge independent instructions between barriers into shared bundles";
}

#endif  // MINITPU_MINITPUISA_PASSES_TD
