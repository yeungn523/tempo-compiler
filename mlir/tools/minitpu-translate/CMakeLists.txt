add_llvm_executable(minitpu-translate minitpu_translate.cpp)
llvm_update_compile_flags(minitpu-translate)
target_link_libraries(minitpu-translate PRIVATE
  MinitpuDialect
  MinitpuIsa
  MLIRTranslateLib
)
mlir_check_all_link_libraries(minitpu-translate)
