add_mlir_library(MinitpuDialect
  allocate.cpp
  minitpu.cpp
  tile_to_vector.cpp
  to_isa.cpp

  DEPENDS
  MinitpuIncGen
  MinitpuPassIncGen

  LINK_LIBS PUBLIC
  MinitpuIsa
  MLIRIR
  MLIRArithDialect
  MLIRBytecodeOpInterface
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
)
