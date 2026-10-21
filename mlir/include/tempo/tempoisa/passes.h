/**
 * @file
 *
 * @brief Provides the passes that run over tempoisa programs, and the function that registers them.
 */

#ifndef TEMPO_TEMPOISA_PASSES_H
#define TEMPO_TEMPOISA_PASSES_H

#include <mlir/Pass/Pass.h>
#include "tempo/tempoisa/tempoisa.h"

namespace mlir::tempo::isa
{
#define GEN_PASS_DECL
#include "tempo/tempoisa/passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "tempo/tempoisa/passes.h.inc"
}  // namespace mlir::tempo::isa

#endif  // TEMPO_TEMPOISA_PASSES_H
