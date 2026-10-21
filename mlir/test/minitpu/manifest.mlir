// Each kernel's manifest entry is read from its tile op, under the image file name the op fixes, and a causal
// attention's constants block is written beside it: the entering maximum, zeros, then the diagonal mask tile.

// RUN: minitpu-translate --split-input-file --export-minitpu-manifest %s | FileCheck %s
// RUN: minitpu-translate --export-minitpu-constants %S/../Inputs/causal_attention.mlir | od -An -tx2 -v | FileCheck %s --check-prefix=CONSTANTS
// RUN: not minitpu-translate --split-input-file --export-minitpu-constants %s 2>&1 | FileCheck %s --check-prefix=NOCONSTANTS

// CHECK:      "gemm_3072x768_gelu_bias.bin": {
// CHECK-NEXT:   "kind": "gemm_bias",
// CHECK-NEXT:   "depth": 3072,
// CHECK-NEXT:   "columns": 768,
// CHECK-NEXT:   "activate": true,
// CHECK-NEXT:   "drain_words": 24832
minitpu.kernel @gemm {
  %a = minitpu.arg 0
  %w = minitpu.arg 1
  %d = minitpu.arg 2
  %b = minitpu.arg 3
  minitpu.matmul %a, %w -> %d bias %b tokens 32 depth 3072 columns 768 activate
}

// -----

// CHECK:      "add_bias_768.bin": {
// CHECK-NEXT:   "kind": "add_bias",
// CHECK-NEXT:   "features": 768
minitpu.kernel @add {
  %l = minitpu.arg 0
  %r = minitpu.arg 1
  %o = minitpu.arg 2
  minitpu.add %l, %r -> %o tokens 32 features 768 broadcast
}

// -----

// CHECK:      "layernorm_768.bin": {
// CHECK-NEXT:   "kind": "layernorm",
// CHECK-NEXT:   "features": 768,
// CHECK-NEXT:   "rows": [
// CHECK-NEXT:     "gamma_pointer",
// CHECK-NEXT:     "beta_pointer"
// CHECK-NEXT:   ],
// CHECK-NEXT:   "scalars": [
// CHECK-NEXT:     0.0013020833333333333,
// CHECK-NEXT:     9.9999997473787516e-06
minitpu.kernel @norm {
  %x = minitpu.arg 0
  %c = minitpu.arg 1
  %o = minitpu.arg 2
  minitpu.layernorm %x, %c -> %o tokens 32 features 768 epsilon 1.0e-05
}

// -----

// CHECK:      "attention_64.bin": {
// CHECK-NEXT:   "kind": "attention",
// CHECK-NEXT:   "tokens": 64,
// CHECK-NEXT:   "heads": 12,
// CHECK-NEXT:   "constants": "attention_64_constants.bin"
minitpu.kernel @causal {
  %c = minitpu.arg 0
  %q = minitpu.arg 2
  minitpu.attention %c, %q heads 12 head 64 keys 64 causal
}

// -----

// CHECK:      "decode_attention_256.bin": {
// CHECK-NEXT:   "kind": "decode_attention",
// CHECK-NEXT:   "keys": 256,
// CHECK-NEXT:   "heads": 12,
// CHECK-NEXT:   "entering": -1e+30,
// CHECK-NEXT:   "stated": [],
// CHECK-NEXT:   "transposed_masks": true
minitpu.kernel @decode {
  %c = minitpu.arg 0
  %k = minitpu.arg 1
  %q = minitpu.arg 2
  minitpu.attention %c, %q cache %k heads 12 head 64 keys 256
}

// -----

// CHECK:      "copy_96x1024.bin": {
// CHECK-NEXT:   "kind": "copy",
// CHECK-NEXT:   "rows": 96,
// CHECK-NEXT:   "stride": 1024
minitpu.kernel @copy {
  %s = minitpu.arg 0
  %d = minitpu.arg 2
  minitpu.copy %s -> %d rows 96 stride 1024
}

// -----

// CHECK:      "halt.bin": {
// CHECK-NEXT:   "kind": "halt"
minitpu.kernel @halt {
}

// The entering maximum's tile is -1e30 in BF16 (0xf14a) and the zero tile is zeros, 256 words or 32 lines each. Each
// row of the diagonal tile is one key, masking every query before it: key 0 masks none, key 1 masks query 0, and key 15
// masks all but the last.
// CONSTANTS-COUNT-32: f14a f14a f14a f14a f14a f14a f14a f14a
// CONSTANTS-COUNT-32: 0000 0000 0000 0000 0000 0000 0000 0000
// CONSTANTS-NEXT:     0000 0000 0000 0000 0000 0000 0000 0000
// CONSTANTS-NEXT:     0000 0000 0000 0000 0000 0000 0000 0000
// CONSTANTS-NEXT:     f14a 0000 0000 0000 0000 0000 0000 0000
// CONSTANTS:          f14a f14a f14a f14a f14a f14a f14a 0000
// NOCONSTANTS: Unable to export a constants block from the gemm kernel. Only a causal attention's image reads one
