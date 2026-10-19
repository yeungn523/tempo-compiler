// A minitpu.matmul expands into the machine operations that compute it: the zero tile and the resident activations
// staged, then per column tile the weight chunks streamed double buffered, each k-tile loading a weight tile, pushing
// the eight token groups and popping eight results into the accumulators, and the column tile stored and drained.
// The whole expansion then allocates, lowers and schedules.

// RUN: minitpu-opt %s -split-input-file -pass-pipeline='builtin.module(minitpu.kernel(minitpu-tile-to-vector))' | FileCheck %s
// RUN: minitpu-opt %s -split-input-file -pass-pipeline='builtin.module(minitpu.kernel(minitpu-tile-to-vector,minitpu-allocate-vmem,minitpu-allocate-registers),convert-minitpu-to-isa,minitpuisa.program(minitpuisa-schedule))' | FileCheck %s --check-prefix=PROGRAM

// CHECK-LABEL: minitpu.kernel @gemm
// CHECK:       minitpu.alloc : !minitpu.vmem<16>
// CHECK:       minitpu.alloc : !minitpu.vmem<512>
// CHECK:       minitpu.vmemld {{.*}} rows 16, slot 0
// CHECK:       minitpu.vmemld {{.*}} rows 128, slot 0
// CHECK:       minitpu.vmemld {{.*}} rows 128, slot 1
// CHECK:       scf.for
// CHECK:         scf.for
// CHECK-COUNT-4: minitpu.vld
// CHECK:           minitpu.vmatload
// CHECK-COUNT-8:   minitpu.vmatpush
// CHECK-COUNT-8:   minitpu.vmatpop
// CHECK:           minitpu.vadd
// CHECK:         minitpu.vmemst {{.*}} rows 32
// CHECK-NOT:   minitpu.matmul
// CHECK-NOT:   minitpu.vgelu
// PROGRAM-LABEL: minitpuisa.program @gemm
// PROGRAM:       minitpuisa.vmatload
// PROGRAM:       minitpuisa.halt
minitpu.kernel @gemm {
  %activations = minitpu.arg 0
  %weights = minitpu.arg 1
  %drain = minitpu.arg 2
  minitpu.matmul %activations, %weights -> %drain tokens 32 depth 256 columns 32
}

// -----

// An activating product passes the resident activations through the GELU in place once, ahead of every column tile.

// CHECK-LABEL: minitpu.kernel @gemm_gelu
// CHECK:       scf.for
// CHECK:         minitpu.vld
// CHECK-NEXT:    minitpu.vgelu
// CHECK-NEXT:    minitpu.vst
// CHECK:       scf.for
// CHECK-NOT:   minitpu.vgelu
minitpu.kernel @gemm_gelu {
  %activations = minitpu.arg 0
  %weights = minitpu.arg 1
  %drain = minitpu.arg 2
  minitpu.matmul %activations, %weights -> %drain tokens 32 depth 256 columns 32 activate
}
