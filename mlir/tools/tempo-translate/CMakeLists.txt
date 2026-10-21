add_llvm_executable(tempo-translate tempo_translate.cpp)
llvm_update_compile_flags(tempo-translate)
target_link_libraries(tempo-translate PRIVATE
  TempoDialect
  TempoIsa
  MLIRTranslateLib
)
mlir_check_all_link_libraries(tempo-translate)
