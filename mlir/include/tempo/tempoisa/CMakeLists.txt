# Reads every slot and opcode from isa.td, the ISA contract beside this file.
set(LLVM_TARGET_DEFINITIONS tempoisa_ops.td)
set(_ISA_INCLUDES "-I${TEMPO_REPOSITORY_DIR}/mlir/include")
mlir_tablegen(tempoisa_dialect.h.inc -gen-dialect-decls -dialect=tempoisa ${_ISA_INCLUDES})
mlir_tablegen(tempoisa_dialect.cpp.inc -gen-dialect-defs -dialect=tempoisa ${_ISA_INCLUDES})
mlir_tablegen(tempoisa_interfaces.h.inc -gen-op-interface-decls ${_ISA_INCLUDES})
mlir_tablegen(tempoisa_interfaces.cpp.inc -gen-op-interface-defs ${_ISA_INCLUDES})
mlir_tablegen(tempoisa_ops.h.inc -gen-op-decls ${_ISA_INCLUDES})
mlir_tablegen(tempoisa_ops.cpp.inc -gen-op-defs ${_ISA_INCLUDES})

set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls -name TempoISA)

set(LLVM_TARGET_DEFINITIONS isa.td)
tablegen(TEMPO tempoisa_schedule.h.inc -gen-tempo-sched)
add_public_tablegen_target(TempoIsaIncGen)
