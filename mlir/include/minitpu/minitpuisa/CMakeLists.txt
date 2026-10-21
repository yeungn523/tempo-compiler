# Reads every slot and opcode from isa.td, the ISA contract beside this file.
set(LLVM_TARGET_DEFINITIONS minitpuisa_ops.td)
set(_ISA_INCLUDES "-I${MINITPU_REPOSITORY_DIR}/mlir/include")
mlir_tablegen(minitpuisa_dialect.h.inc -gen-dialect-decls -dialect=minitpuisa ${_ISA_INCLUDES})
mlir_tablegen(minitpuisa_dialect.cpp.inc -gen-dialect-defs -dialect=minitpuisa ${_ISA_INCLUDES})
mlir_tablegen(minitpuisa_interfaces.h.inc -gen-op-interface-decls ${_ISA_INCLUDES})
mlir_tablegen(minitpuisa_interfaces.cpp.inc -gen-op-interface-defs ${_ISA_INCLUDES})
mlir_tablegen(minitpuisa_ops.h.inc -gen-op-decls ${_ISA_INCLUDES})
mlir_tablegen(minitpuisa_ops.cpp.inc -gen-op-defs ${_ISA_INCLUDES})

set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls -name MiniTPUISA)

set(LLVM_TARGET_DEFINITIONS isa.td)
tablegen(MINITPU minitpuisa_schedule.h.inc -gen-minitpu-sched)
add_public_tablegen_target(MinitpuIsaIncGen)
