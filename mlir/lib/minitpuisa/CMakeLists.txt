add_mlir_library(MinitpuIsa
  image.cpp
  minitpuisa.cpp

  DEPENDS
  MinitpuIsaIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRBytecodeOpInterface
)
