# Reads every mnemonic's group from codegen/minitpuisa.td, which the include path below reaches.
set(LLVM_TARGET_DEFINITIONS minitpu_ops.td)
set(_MINITPU_INCLUDES "-I${MINITPU_REPOSITORY_DIR}")
mlir_tablegen(minitpu_dialect.h.inc -gen-dialect-decls -dialect=minitpu ${_MINITPU_INCLUDES})
mlir_tablegen(minitpu_dialect.cpp.inc -gen-dialect-defs -dialect=minitpu ${_MINITPU_INCLUDES})
mlir_tablegen(minitpu_types.h.inc -gen-typedef-decls -typedefs-dialect=minitpu ${_MINITPU_INCLUDES})
mlir_tablegen(minitpu_types.cpp.inc -gen-typedef-defs -typedefs-dialect=minitpu ${_MINITPU_INCLUDES})
mlir_tablegen(minitpu_ops.h.inc -gen-op-decls ${_MINITPU_INCLUDES})
mlir_tablegen(minitpu_ops.cpp.inc -gen-op-defs ${_MINITPU_INCLUDES})
add_public_tablegen_target(MinitpuIncGen)

set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls -name MiniTPU)
add_public_tablegen_target(MinitpuPassIncGen)
