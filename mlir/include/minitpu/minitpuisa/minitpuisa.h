/**
 * @file
 *
 * @brief Provides the minitpuisa dialect, whose operations are the MiniTPU's machine instructions grouped into
 * bundles.
 */

#ifndef MINITPU_MINITPUISA_MINITPUISA_H
#define MINITPU_MINITPUISA_MINITPUISA_H

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/RegionKindInterface.h>
#include <mlir/IR/SymbolTable.h>
#include "minitpu/minitpuisa/encoding.h"
#include "minitpu/minitpuisa/minitpuisa_dialect.h.inc"
#include "minitpu/minitpuisa/minitpuisa_interfaces.h.inc"

#define GET_OP_CLASSES
#include "minitpu/minitpuisa/minitpuisa_ops.h.inc"

#endif  // MINITPU_MINITPUISA_MINITPUISA_H
