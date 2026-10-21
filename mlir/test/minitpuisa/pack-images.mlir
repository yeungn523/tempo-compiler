// Every GPT-2 image's emitted stream, recorded under Inputs/schedule, packs to the bytes the retired Python assembler's
// pack() made of it, and layernorm_768, the image built packed, reaches its recorded image through the packer and the
// scheduler together. The attention streams are left out: pack() moved a vtxin past the
// vtxouts draining the tile before it, which their hand placement never ships, and this packer keeps every transpose
// in program order.

// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/add_768.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.add_768.bin && cmp %t.add_768.bin %S/../Inputs/schedule/add_768.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/add_bias_2304.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.add_bias_2304.bin && cmp %t.add_bias_2304.bin %S/../Inputs/schedule/add_bias_2304.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/add_bias_3072.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.add_bias_3072.bin && cmp %t.add_bias_3072.bin %S/../Inputs/schedule/add_bias_3072.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/add_bias_768.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.add_bias_768.bin && cmp %t.add_bias_768.bin %S/../Inputs/schedule/add_bias_768.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/copy_3072x32.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.copy_3072x32.bin && cmp %t.copy_3072x32.bin %S/../Inputs/schedule/copy_3072x32.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/copy_96x1024.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.copy_96x1024.bin && cmp %t.copy_96x1024.bin %S/../Inputs/schedule/copy_96x1024.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/gemm_3072x768_gelu.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.gemm_3072x768_gelu.bin && cmp %t.gemm_3072x768_gelu.bin %S/../Inputs/schedule/gemm_3072x768_gelu.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/gemm_768x2304.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.gemm_768x2304.bin && cmp %t.gemm_768x2304.bin %S/../Inputs/schedule/gemm_768x2304.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/gemm_768x3072.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.gemm_768x3072.bin && cmp %t.gemm_768x3072.bin %S/../Inputs/schedule/gemm_768x3072.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/gemm_768x768.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.gemm_768x768.bin && cmp %t.gemm_768x768.bin %S/../Inputs/schedule/gemm_768x768.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/halt.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.halt.bin && cmp %t.halt.bin %S/../Inputs/schedule/halt.packed.bin
// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/layernorm_768.emitted.bin | minitpu-opt -minitpuisa-pack | minitpu-translate --export-minitpu-image -o %t.layernorm_768.bin && cmp %t.layernorm_768.bin %S/../Inputs/schedule/layernorm_768.packed.bin

// RUN: minitpu-translate --import-minitpu-image %S/../Inputs/schedule/layernorm_768.emitted.bin | minitpu-opt -minitpuisa-pack -minitpuisa-schedule | minitpu-translate --export-minitpu-image -o %t.layernorm_768.board.bin && cmp %t.layernorm_768.board.bin %S/../Inputs/schedule/layernorm_768.scheduled.bin
