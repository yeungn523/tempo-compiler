# Reads every mnemonic's group from tempoisa/isa.td, the ISA contract, which the include path below reaches.
set(LLVM_TARGET_DEFINITIONS tempo_ops.td)
set(_TEMPO_INCLUDES "-I${TEMPO_REPOSITORY_DIR}/mlir/include")
mlir_tablegen(tempo_dialect.h.inc -gen-dialect-decls -dialect=tempo ${_TEMPO_INCLUDES})
mlir_tablegen(tempo_dialect.cpp.inc -gen-dialect-defs -dialect=tempo ${_TEMPO_INCLUDES})
mlir_tablegen(tempo_types.h.inc -gen-typedef-decls -typedefs-dialect=tempo ${_TEMPO_INCLUDES})
mlir_tablegen(tempo_types.cpp.inc -gen-typedef-defs -typedefs-dialect=tempo ${_TEMPO_INCLUDES})
mlir_tablegen(tempo_ops.h.inc -gen-op-decls ${_TEMPO_INCLUDES})
mlir_tablegen(tempo_ops.cpp.inc -gen-op-defs ${_TEMPO_INCLUDES})
add_public_tablegen_target(TempoIncGen)

set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls -name Tempo)
add_public_tablegen_target(TempoPassIncGen)
