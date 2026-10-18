// Every kernel a GPT-2 small layer launches parses, verifies, and prints back unchanged: minitpu-opt reads what the
// Triton 3.8.0 wheel prints. Regenerate the inputs with tools/dump_gpt2_ttir.py.

// RUN: minitpu-opt %S/../Inputs/gemm_768x2304.ttir -o %t.gemm_768x2304 && diff -I '^$' %S/../Inputs/gemm_768x2304.ttir %t.gemm_768x2304
// RUN: minitpu-opt %S/../Inputs/gemm_3072x768_gelu.ttir -o %t.gemm_3072x768_gelu && diff -I '^$' %S/../Inputs/gemm_3072x768_gelu.ttir %t.gemm_3072x768_gelu
// RUN: minitpu-opt %S/../Inputs/add_768.ttir -o %t.add_768 && diff -I '^$' %S/../Inputs/add_768.ttir %t.add_768
// RUN: minitpu-opt %S/../Inputs/add_bias_768.ttir -o %t.add_bias_768 && diff -I '^$' %S/../Inputs/add_bias_768.ttir %t.add_bias_768
// RUN: minitpu-opt %S/../Inputs/layernorm_768.ttir -o %t.layernorm_768 && diff -I '^$' %S/../Inputs/layernorm_768.ttir %t.layernorm_768
// RUN: minitpu-opt %S/../Inputs/attention_64.ttir -o %t.attention_64 && diff -I '^$' %S/../Inputs/attention_64.ttir %t.attention_64
// RUN: minitpu-opt %S/../Inputs/decode_attention_256.ttir -o %t.decode_attention_256 && diff -I '^$' %S/../Inputs/decode_attention_256.ttir %t.decode_attention_256
// RUN: minitpu-opt %S/../Inputs/gemm_768x2304.ttir | FileCheck %s

// CHECK-LABEL: tt.func public @gpt2_matmul
// CHECK:         scf.for
// CHECK:           tt.dot
// CHECK:         tt.store
