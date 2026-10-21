/**
 * @file
 *
 * @brief Provides the passes that run over tempo kernels, and the function that registers them.
 */

#ifndef TEMPO_TEMPO_PASSES_H
#define TEMPO_TEMPO_PASSES_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include "tempo/tempo/tempo.h"
#include "tempo/tempoisa/tempoisa.h"

namespace mlir::tempo
{
#define GEN_PASS_DECL
#include "tempo/tempo/passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "tempo/tempo/passes.h.inc"
}  // namespace mlir::tempo

#endif  // TEMPO_TEMPO_PASSES_H
