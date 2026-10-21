add_mlir_library(TempoDialect
  allocate.cpp
  manifest.cpp
  tempo.cpp
  tile_to_vector.cpp
  to_isa.cpp
  tt_to_tempo.cpp

  DEPENDS
  TempoIncGen
  TempoPassIncGen

  LINK_LIBS PUBLIC
  TempoIsa
  MLIRIR
  TempoTritonIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRBytecodeOpInterface
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
)
