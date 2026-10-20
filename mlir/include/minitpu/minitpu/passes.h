/**
 * @file
 *
 * @brief Provides the passes that run over minitpu kernels, and the function that registers them.
 */

#ifndef MINITPU_MINITPU_PASSES_H
#define MINITPU_MINITPU_PASSES_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include "minitpu/minitpu/minitpu.h"
#include "minitpu/minitpuisa/minitpuisa.h"

namespace mlir::minitpu
{
#define GEN_PASS_DECL
#include "minitpu/minitpu/passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "minitpu/minitpu/passes.h.inc"
}  // namespace mlir::minitpu

#endif  // MINITPU_MINITPU_PASSES_H
