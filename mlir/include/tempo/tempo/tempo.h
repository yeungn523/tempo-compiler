/**
 * @file
 *
 * @brief Provides the tempo dialect, which states a kernel as machine operations over virtual registers and VMEM
 * buffers.
 */

#ifndef TEMPO_TEMPO_TEMPO_H
#define TEMPO_TEMPO_TEMPO_H

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/RegionKindInterface.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include "tempo/tempo/tempo_dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "tempo/tempo/tempo_types.h.inc"

#define GET_OP_CLASSES
#include "tempo/tempo/tempo_ops.h.inc"

#endif  // TEMPO_TEMPO_TEMPO_H
