/**
 * @file
 *
 * @brief Provides the minitpu dialect, which states a kernel as machine operations over virtual registers and VMEM
 * buffers.
 */

#ifndef MINITPU_MINITPU_MINITPU_H
#define MINITPU_MINITPU_MINITPU_H

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/RegionKindInterface.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include "minitpu/minitpu/minitpu_dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "minitpu/minitpu/minitpu_types.h.inc"

#define GET_OP_CLASSES
#include "minitpu/minitpu/minitpu_ops.h.inc"

#endif  // MINITPU_MINITPU_MINITPU_H
