add_mlir_library(MinitpuIsa
  image.cpp
  minitpuisa.cpp
  pack.cpp
  schedule.cpp
  traffic.cpp

  DEPENDS
  MinitpuIsaIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBytecodeOpInterface
  MLIRPass
)
