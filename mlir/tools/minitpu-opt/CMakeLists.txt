add_llvm_executable(minitpu-opt minitpu_opt.cpp)
llvm_update_compile_flags(minitpu-opt)
target_link_libraries(minitpu-opt PRIVATE
  MinitpuIsa
  MinitpuTritonIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRUBDialect
  MLIROptLib
  MLIRTransforms
)
mlir_check_all_link_libraries(minitpu-opt)
