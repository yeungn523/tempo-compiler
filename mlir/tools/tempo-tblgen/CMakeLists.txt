# Builds the TableGen driver carrying the MiniTPU's own backends, and points LLVM's tablegen() at it for the TEMPO
# project, so a directory below generates from tempoisa/isa.td as it would with mlir-tblgen.
set(LLVM_LINK_COMPONENTS Support TableGen)
add_llvm_executable(tempo-tblgen tempo_tblgen.cpp)
llvm_update_compile_flags(tempo-tblgen)
target_link_libraries(tempo-tblgen PRIVATE LLVMTableGen LLVMSupport)

set(TEMPO_TABLEGEN_EXE "$<TARGET_FILE:tempo-tblgen>" CACHE INTERNAL "")
set(TEMPO_TABLEGEN_TARGET tempo-tblgen CACHE INTERNAL "")
