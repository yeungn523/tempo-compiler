/**
 * @file
 *
 * @brief Provides the tempoisa dialect, whose operations are the MiniTPU's machine instructions grouped into
 * bundles.
 */

#ifndef TEMPO_TEMPOISA_TEMPOISA_H
#define TEMPO_TEMPOISA_TEMPOISA_H

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/RegionKindInterface.h>
#include <mlir/IR/SymbolTable.h>
#include "tempo/tempoisa/encoding.h"
#include "tempo/tempoisa/tempoisa_dialect.h.inc"
#include "tempo/tempoisa/tempoisa_interfaces.h.inc"

#define GET_OP_CLASSES
#include "tempo/tempoisa/tempoisa_ops.h.inc"

#endif  // TEMPO_TEMPOISA_TEMPOISA_H
