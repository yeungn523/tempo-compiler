// The scheduler spaces each consumer by its producer's latency from codegen/minitpuisa.td, and bridges a gap past one
// delay field with empty bundles. Each expectation matches what codegen/asm.py's schedule() places for the same stream.

// RUN: minitpu-opt %s -split-input-file -minitpuisa-schedule | FileCheck %s

// CHECK-LABEL: minitpuisa.program @latencies
minitpuisa.program @latencies {
  // CHECK: minitpuisa.bundle delay 1 {
  // CHECK-NEXT: minitpuisa.smovi s1, 64
  minitpuisa.bundle { minitpuisa.smovi s1, 64 }
  // CHECK: minitpuisa.bundle {
  // CHECK-NEXT: minitpuisa.vmemld
  minitpuisa.bundle { minitpuisa.vmemld slot 0, vmem 0, rows 16, base s1, stride s2 }
  minitpuisa.bundle { minitpuisa.flush_slot 1 }
  // CHECK: minitpuisa.bundle delay 6 {
  // CHECK-NEXT: minitpuisa.vld v1, row 0
  minitpuisa.bundle { minitpuisa.vld v1, row 0 }
  // CHECK: minitpuisa.bundle delay 5 {
  // CHECK-NEXT: minitpuisa.vadd v3, v1, v2
  minitpuisa.bundle { minitpuisa.vadd v3, v1, v2 }
  // CHECK: minitpuisa.bundle delay 15 {
  // CHECK-NEXT: minitpuisa.vredsum v4, v3
  minitpuisa.bundle { minitpuisa.vredsum v4, v3 }
  minitpuisa.bundle { minitpuisa.vexp v5, v4 }
  minitpuisa.bundle { minitpuisa.halt }
}

// -----

// CHECK-LABEL: minitpuisa.program @push_then_pop
minitpuisa.program @push_then_pop {
  // CHECK: minitpuisa.bundle delay 18 {
  // CHECK-NEXT: minitpuisa.vmatload v4
  minitpuisa.bundle { minitpuisa.vmatload v4 }
  // CHECK: minitpuisa.bundle {
  // CHECK-NEXT: minitpuisa.vmatpush v0
  // CHECK: minitpuisa.bundle delay 63 {
  // CHECK-NEXT: }
  // CHECK-NEXT: minitpuisa.bundle delay 63 {
  // CHECK-NEXT: }
  // CHECK-NEXT: minitpuisa.bundle {
  // CHECK-NEXT: minitpuisa.vmatpop v8
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  minitpuisa.bundle { minitpuisa.vmatpop v8 }
  minitpuisa.bundle { minitpuisa.halt }
}

// -----

// A write to a register a matrix command still streams waits until the command has read it: vmatload reads v4..v7
// over its 18-cycle occupancy, so a vld into v4, whose write lands 6 cycles after issue, issues no earlier than 13.

// CHECK-LABEL: minitpuisa.program @write_under_a_stream
minitpuisa.program @write_under_a_stream {
  // CHECK: minitpuisa.bundle delay 12 {
  // CHECK-NEXT: minitpuisa.vmatload v4
  minitpuisa.bundle { minitpuisa.vmatload v4 }
  minitpuisa.bundle { minitpuisa.vld v4, row 0 }
  minitpuisa.bundle { minitpuisa.halt }
}
