# The MLIR pipeline: Triton's tt dialect in, MiniTPU images out.
#
# Builds against the LLVM Triton pins and compiles Triton's tt dialect from that release's source, so tempo-opt parses
# exactly what the Triton wheel prints. pyproject.toml's [tool.tempo.toolchain] pins both, tools/toolchain/setup_toolchain.py
# fetches them, and configuring refuses an LLVM built from any other commit.
cmake_minimum_required(VERSION 3.20)
project(tempo-mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_filename_component(TEMPO_REPOSITORY_DIR "${CMAKE_SOURCE_DIR}/.." ABSOLUTE)

# Reads one string pin from pyproject.toml's [tool.tempo.toolchain] table into `variable`.
function(tempo_read_pin key variable)
  file(STRINGS "${TEMPO_REPOSITORY_DIR}/pyproject.toml" lines REGEX "^${key} *= *\"")
  if(NOT lines)
    message(FATAL_ERROR "pyproject.toml pins no '${key}' under [tool.tempo.toolchain].")
  endif()
  string(REGEX REPLACE "^${key} *= *\"([^\"]*)\".*" "\\1" value "${lines}")
  set(${variable} "${value}" PARENT_SCOPE)
endfunction()

tempo_read_pin(triton-tag TEMPO_TRITON_TAG)
tempo_read_pin(llvm-commit TEMPO_LLVM_COMMIT)
string(REGEX REPLACE "^v" "" _triton_version "${TEMPO_TRITON_TAG}")

set(TRITON_SOURCE_DIR "$ENV{HOME}/toolchains/triton-${_triton_version}" CACHE PATH "Triton source at the pinned tag")
if(NOT EXISTS "${TRITON_SOURCE_DIR}/include/triton/Dialect/Triton/IR/TritonOps.td")
  message(FATAL_ERROR "TRITON_SOURCE_DIR (${TRITON_SOURCE_DIR}) holds no Triton source. Run tools/toolchain/setup_toolchain.py.")
endif()

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIR from ${MLIR_DIR} (LLVM ${LLVM_PACKAGE_VERSION})")

# Holds the LLVM found to the pinned commit, which the package records in its own VCSRevision.h.
file(STRINGS "${LLVM_INCLUDE_DIRS}/llvm/Support/VCSRevision.h" _llvm_revision REGEX "LLVM_REVISION")
if(NOT _llvm_revision MATCHES "${TEMPO_LLVM_COMMIT}")
  message(FATAL_ERROR "The LLVM at ${LLVM_DIR} records '${_llvm_revision}', but pyproject.toml pins "
                      "${TEMPO_LLVM_COMMIT}. Run tools/toolchain/setup_toolchain.py and configure with the LLVM it names.")
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(MLIR_BINARY_DIR "${CMAKE_BINARY_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
  "${TRITON_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/third_party/triton/include")
include_directories(
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include")
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(third_party/triton)
add_subdirectory(tools/tempo-tblgen)
add_subdirectory(include/tempo/tempoisa)
add_subdirectory(include/tempo/tempo)
add_subdirectory(lib/tempoisa)
add_subdirectory(lib/tempo)
add_subdirectory(tools)
add_subdirectory(test)
