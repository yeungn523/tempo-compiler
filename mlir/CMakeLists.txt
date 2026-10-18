# The MLIR pipeline: Triton's tt dialect in, MiniTPU images out.
#
# Builds against the LLVM Triton 3.8.0 pins (cmake/llvm-info.json in Triton's source, unpacked by Triton's own
# build helper into ~/.triton/llvm), and compiles Triton's tt dialect from that release's source, so minitpu-opt
# parses exactly what the Triton wheel prints.
cmake_minimum_required(VERSION 3.20)
project(minitpu-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_filename_component(MINITPU_REPOSITORY_DIR "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)

set(TRITON_SOURCE_DIR "$ENV{HOME}/toolchains/triton-3.8.0" CACHE PATH "Triton v3.8.0 source checkout")
if(NOT EXISTS "${TRITON_SOURCE_DIR}/include/triton/Dialect/Triton/IR/TritonOps.td")
  message(FATAL_ERROR "TRITON_SOURCE_DIR (${TRITON_SOURCE_DIR}) holds no Triton source. Clone tag v3.8.0 there.")
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIR from ${MLIR_DIR} (LLVM ${LLVM_PACKAGE_VERSION})")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(
  "${TRITON_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/third_party/triton/include"
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include")
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(third_party/triton)
add_subdirectory(include/minitpu/minitpuisa)
add_subdirectory(lib/minitpuisa)
add_subdirectory(tools)
add_subdirectory(test)
