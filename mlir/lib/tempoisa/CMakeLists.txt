add_mlir_library(TempoIsa
  image.cpp
  tempoisa.cpp
  pack.cpp
  schedule.cpp
  traffic.cpp

  DEPENDS
  TempoIsaIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBytecodeOpInterface
  MLIRPass
)
