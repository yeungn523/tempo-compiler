/**
 * @file
 *
 * @brief Provides the passes that run over minitpuisa programs, and the function that registers them.
 */

#ifndef MINITPU_MINITPUISA_PASSES_H
#define MINITPU_MINITPUISA_PASSES_H

#include <mlir/Pass/Pass.h>
#include "minitpu/minitpuisa/minitpuisa.h"

namespace mlir::minitpu::isa
{
#define GEN_PASS_DECL
#include "minitpu/minitpuisa/passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "minitpu/minitpuisa/passes.h.inc"
}  // namespace mlir::minitpu::isa

#endif  // MINITPU_MINITPUISA_PASSES_H
