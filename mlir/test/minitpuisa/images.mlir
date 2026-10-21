// Every image the board package launches reads into minitpuisa and writes back byte for byte, which holds the
// dialect's encoder to the one the images were assembled with.

// RUN: minitpu-translate --import-minitpu-image %images/add_768.bin -o %t.add_768.mlir && minitpu-translate --export-minitpu-image %t.add_768.mlir -o %t.add_768.bin && cmp %images/add_768.bin %t.add_768.bin
// RUN: minitpu-translate --import-minitpu-image %images/add_bias_768.bin -o %t.add_bias_768.mlir && minitpu-translate --export-minitpu-image %t.add_bias_768.mlir -o %t.add_bias_768.bin && cmp %images/add_bias_768.bin %t.add_bias_768.bin
// RUN: minitpu-translate --import-minitpu-image %images/attention_64.bin -o %t.attention_64.mlir && minitpu-translate --export-minitpu-image %t.attention_64.mlir -o %t.attention_64.bin && cmp %images/attention_64.bin %t.attention_64.bin
// RUN: minitpu-translate --import-minitpu-image %images/copy_3072x32.bin -o %t.copy_3072x32.mlir && minitpu-translate --export-minitpu-image %t.copy_3072x32.mlir -o %t.copy_3072x32.bin && cmp %images/copy_3072x32.bin %t.copy_3072x32.bin
// RUN: minitpu-translate --import-minitpu-image %images/copy_96x1024.bin -o %t.copy_96x1024.mlir && minitpu-translate --export-minitpu-image %t.copy_96x1024.mlir -o %t.copy_96x1024.bin && cmp %images/copy_96x1024.bin %t.copy_96x1024.bin
// RUN: minitpu-translate --import-minitpu-image %images/decode_attention_1024.bin -o %t.decode_attention_1024.mlir && minitpu-translate --export-minitpu-image %t.decode_attention_1024.mlir -o %t.decode_attention_1024.bin && cmp %images/decode_attention_1024.bin %t.decode_attention_1024.bin
// RUN: minitpu-translate --import-minitpu-image %images/decode_attention_256.bin -o %t.decode_attention_256.mlir && minitpu-translate --export-minitpu-image %t.decode_attention_256.mlir -o %t.decode_attention_256.bin && cmp %images/decode_attention_256.bin %t.decode_attention_256.bin
// RUN: minitpu-translate --import-minitpu-image %images/decode_attention_64.bin -o %t.decode_attention_64.mlir && minitpu-translate --export-minitpu-image %t.decode_attention_64.mlir -o %t.decode_attention_64.bin && cmp %images/decode_attention_64.bin %t.decode_attention_64.bin
// RUN: minitpu-translate --import-minitpu-image %images/gemm_3072x768_gelu_bias.bin -o %t.gemm_3072x768_gelu_bias.mlir && minitpu-translate --export-minitpu-image %t.gemm_3072x768_gelu_bias.mlir -o %t.gemm_3072x768_gelu_bias.bin && cmp %images/gemm_3072x768_gelu_bias.bin %t.gemm_3072x768_gelu_bias.bin
// RUN: minitpu-translate --import-minitpu-image %images/gemm_768x2304_bias.bin -o %t.gemm_768x2304_bias.mlir && minitpu-translate --export-minitpu-image %t.gemm_768x2304_bias.mlir -o %t.gemm_768x2304_bias.bin && cmp %images/gemm_768x2304_bias.bin %t.gemm_768x2304_bias.bin
// RUN: minitpu-translate --import-minitpu-image %images/gemm_768x3072_bias.bin -o %t.gemm_768x3072_bias.mlir && minitpu-translate --export-minitpu-image %t.gemm_768x3072_bias.mlir -o %t.gemm_768x3072_bias.bin && cmp %images/gemm_768x3072_bias.bin %t.gemm_768x3072_bias.bin
// RUN: minitpu-translate --import-minitpu-image %images/gemm_768x768_bias.bin -o %t.gemm_768x768_bias.mlir && minitpu-translate --export-minitpu-image %t.gemm_768x768_bias.mlir -o %t.gemm_768x768_bias.bin && cmp %images/gemm_768x768_bias.bin %t.gemm_768x768_bias.bin
// RUN: minitpu-translate --import-minitpu-image %images/halt.bin -o %t.halt.mlir && minitpu-translate --export-minitpu-image %t.halt.mlir -o %t.halt.bin && cmp %images/halt.bin %t.halt.bin
// RUN: minitpu-translate --import-minitpu-image %images/layernorm_768.bin -o %t.layernorm_768.mlir && minitpu-translate --export-minitpu-image %t.layernorm_768.mlir -o %t.layernorm_768.bin && cmp %images/layernorm_768.bin %t.layernorm_768.bin

// RUN: minitpu-translate --import-minitpu-image %images/gemm_768x768_bias.bin | FileCheck %s --check-prefix=GEMM
// GEMM-LABEL: minitpuisa.program @gemm_768x768_bias
// GEMM:         minitpuisa.vmemld slot 0, vmem 0, rows 128, base s{{[0-3]}}, stride s{{[0-3]}}
// GEMM:         minitpuisa.flush_slot 1
// GEMM:         minitpuisa.lbegin 48
// GEMM:         minitpuisa.vmatpush
// GEMM:         minitpuisa.vmatpop
// GEMM:         minitpuisa.halt

// A data file beside an image is refused rather than read as instructions.
// RUN: not minitpu-translate --import-minitpu-image %images/attention_64_constants.bin 2>&1 | FileCheck %s --check-prefix=DATA
// DATA: attention_64_constants bundle 0"): Unable to decode the bundle exactly.
