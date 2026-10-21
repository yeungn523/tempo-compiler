# Triton's tt dialect, compiled from TRITON_SOURCE_DIR. Mirrors the two CMakeLists.txt Triton carries for it
# (include/triton/Dialect/Triton/IR and lib/Dialect/Triton/IR) without the rest of Triton's build.
set(_TT_INCLUDE "${TRITON_SOURCE_DIR}/include/triton/Dialect/Triton/IR")
set(_TT_LIB "${TRITON_SOURCE_DIR}/lib/Dialect/Triton/IR")
set(_TT_OUT "include/triton/Dialect/Triton/IR")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${_TT_OUT}")

# tablegen() writes each output relative to the current binary directory, so the generated headers land in a tree
# whose layout matches Triton's own includes, which the top-level include path points at.
function(_tt_tablegen td)
  set(LLVM_TARGET_DEFINITIONS "${_TT_INCLUDE}/${td}")
  foreach(pair IN LISTS ARGN)
    string(REPLACE ":" ";" pair "${pair}")
    list(GET pair 0 file)
    list(GET pair 1 flag)
    tablegen(MLIR "${_TT_OUT}/${file}" ${flag} "-I${TRITON_SOURCE_DIR}/include")
  endforeach()
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} PARENT_SCOPE)
endfunction()

_tt_tablegen(TritonOps.td Ops.h.inc:-gen-op-decls Ops.cpp.inc:-gen-op-defs
             OpsEnums.h.inc:-gen-enum-decls OpsEnums.cpp.inc:-gen-enum-defs)
_tt_tablegen(TritonDialect.td Dialect.h.inc:-gen-dialect-decls Dialect.cpp.inc:-gen-dialect-defs)
_tt_tablegen(TritonTypes.td Types.h.inc:-gen-typedef-decls Types.cpp.inc:-gen-typedef-defs)
_tt_tablegen(TritonInterfaces.td AttrInterfaces.h.inc:-gen-attr-interface-decls
             AttrInterfaces.cpp.inc:-gen-attr-interface-defs)
_tt_tablegen(TritonOpInterfaces.td OpInterfaces.h.inc:-gen-op-interface-decls
             OpInterfaces.cpp.inc:-gen-op-interface-defs)
_tt_tablegen(TritonTypeInterfaces.td TypeInterfaces.h.inc:-gen-type-interface-decls
             TypeInterfaces.cpp.inc:-gen-type-interface-defs)

set(LLVM_TARGET_DEFINITIONS "${_TT_LIB}/Canonicalize.td")
tablegen(MLIR TritonCanonicalize.inc -gen-rewriters "-I${TRITON_SOURCE_DIR}/include")

add_public_tablegen_target(TempoTritonTableGen)

add_mlir_library(TempoTritonIR
  "${_TT_LIB}/Dialect.cpp"
  "${_TT_LIB}/DiscardableAttributes.cpp"
  "${_TT_LIB}/Ops.cpp"
  "${_TT_LIB}/Traits.cpp"
  "${_TT_LIB}/Types.cpp"
  "${_TT_LIB}/OpInterfaces.cpp"
  "${_TT_LIB}/Utility.cpp"

  DEPENDS
  TempoTritonTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRFunctionInterfaces
  MLIRCallInterfaces
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
target_include_directories(TempoTritonIR PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
