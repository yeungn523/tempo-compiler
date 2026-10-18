# Builds the TableGen driver carrying the MiniTPU's own backends, and points LLVM's tablegen() at it for the MINITPU
# project, so a directory below generates from codegen/minitpuisa.td as it would with mlir-tblgen.
set(LLVM_LINK_COMPONENTS Support TableGen)
add_llvm_executable(minitpu-tblgen minitpu_tblgen.cpp)
llvm_update_compile_flags(minitpu-tblgen)
target_link_libraries(minitpu-tblgen PRIVATE LLVMTableGen LLVMSupport)

set(MINITPU_TABLEGEN_EXE "$<TARGET_FILE:minitpu-tblgen>" CACHE INTERNAL "")
set(MINITPU_TABLEGEN_TARGET minitpu-tblgen CACHE INTERNAL "")
