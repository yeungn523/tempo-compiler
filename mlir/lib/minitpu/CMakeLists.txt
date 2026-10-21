add_mlir_library(MinitpuDialect
  allocate.cpp
  manifest.cpp
  minitpu.cpp
  tile_to_vector.cpp
  to_isa.cpp
  tt_to_minitpu.cpp

  DEPENDS
  MinitpuIncGen
  MinitpuPassIncGen

  LINK_LIBS PUBLIC
  MinitpuIsa
  MLIRIR
  MinitpuTritonIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRBytecodeOpInterface
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
)
