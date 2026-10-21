add_llvm_executable(tempo-opt tempo_opt.cpp)
llvm_update_compile_flags(tempo-opt)
target_link_libraries(tempo-opt PRIVATE
  TempoDialect
  TempoIsa
  TempoTritonIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRUBDialect
  MLIROptLib
  MLIRTransforms
)
mlir_check_all_link_libraries(tempo-opt)
