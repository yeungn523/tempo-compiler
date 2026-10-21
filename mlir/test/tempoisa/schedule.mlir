// The scheduler spaces each consumer by its producer's latency from tempoisa/isa.td, and bridges a gap past one
// delay field with empty bundles. Each expectation matches what the retired Python assembler placed for the same
// stream.

// RUN: tempo-opt %s -split-input-file -tempoisa-schedule | FileCheck %s

// CHECK-LABEL: tempoisa.program @latencies
tempoisa.program @latencies {
  // CHECK: tempoisa.bundle delay 1 {
  // CHECK-NEXT: tempoisa.smovi s1, 64
  tempoisa.bundle { tempoisa.smovi s1, 64 }
  // CHECK: tempoisa.bundle {
  // CHECK-NEXT: tempoisa.vmemld
  tempoisa.bundle { tempoisa.vmemld slot 0, vmem 0, rows 16, base s1, stride s2 }
  tempoisa.bundle { tempoisa.flush_slot 1 }
  // CHECK: tempoisa.bundle delay 6 {
  // CHECK-NEXT: tempoisa.vld v1, row 0
  tempoisa.bundle { tempoisa.vld v1, row 0 }
  // CHECK: tempoisa.bundle delay 5 {
  // CHECK-NEXT: tempoisa.vadd v3, v1, v2
  tempoisa.bundle { tempoisa.vadd v3, v1, v2 }
  // CHECK: tempoisa.bundle delay 15 {
  // CHECK-NEXT: tempoisa.vredsum v4, v3
  tempoisa.bundle { tempoisa.vredsum v4, v3 }
  tempoisa.bundle { tempoisa.vexp v5, v4 }
  tempoisa.bundle { tempoisa.halt }
}

// -----

// CHECK-LABEL: tempoisa.program @push_then_pop
tempoisa.program @push_then_pop {
  // CHECK: tempoisa.bundle delay 18 {
  // CHECK-NEXT: tempoisa.vmatload v4
  tempoisa.bundle { tempoisa.vmatload v4 }
  // CHECK: tempoisa.bundle {
  // CHECK-NEXT: tempoisa.vmatpush v0
  // CHECK: tempoisa.bundle delay 63 {
  // CHECK-NEXT: }
  // CHECK-NEXT: tempoisa.bundle delay 63 {
  // CHECK-NEXT: }
  // CHECK-NEXT: tempoisa.bundle {
  // CHECK-NEXT: tempoisa.vmatpop v8
  tempoisa.bundle { tempoisa.vmatpush v0 }
  tempoisa.bundle { tempoisa.vmatpop v8 }
  tempoisa.bundle { tempoisa.halt }
}

// -----

// A write to a register a matrix command still streams waits until the command has read it: vmatload reads v4..v7
// over its 18-cycle occupancy, so a vld into v4, whose write lands 6 cycles after issue, issues no earlier than 13.

// CHECK-LABEL: tempoisa.program @write_under_a_stream
tempoisa.program @write_under_a_stream {
  // CHECK: tempoisa.bundle delay 12 {
  // CHECK-NEXT: tempoisa.vmatload v4
  tempoisa.bundle { tempoisa.vmatload v4 }
  tempoisa.bundle { tempoisa.vld v4, row 0 }
  tempoisa.bundle { tempoisa.halt }
}

// -----

// A vld reading rows a vst wrote issues 9 cycles after the vst, the VMEM turnaround, while one reading other rows
// issues at once. A row a loop walks covers every value its induction variable takes, so a store across rows 64
// through 115 holds a load of row 100.

// CHECK-LABEL: tempoisa.program @vmem_turnaround
tempoisa.program @vmem_turnaround {
  // CHECK: tempoisa.bundle delay 8 {
  // CHECK-NEXT: tempoisa.vst v1, row 0
  tempoisa.bundle { tempoisa.vst v1, row 0 }
  // CHECK: tempoisa.bundle {
  // CHECK-NEXT: tempoisa.vld v2, row 0
  tempoisa.bundle { tempoisa.vld v2, row 0 }
  // CHECK: tempoisa.bundle {
  // CHECK-NEXT: tempoisa.vst v3, row 0
  tempoisa.bundle { tempoisa.vst v3, row 0 }
  // CHECK: tempoisa.bundle {
  // CHECK-NEXT: tempoisa.vld v4, row 8
  tempoisa.bundle { tempoisa.vld v4, row 8 }
  tempoisa.bundle { tempoisa.lbegin 4 }
  // CHECK: tempoisa.bundle delay 8 {
  // CHECK-NEXT: tempoisa.vst v5, row 64 agu shift 4
  tempoisa.bundle { tempoisa.vst v5, row 64 agu shift 4 }
  tempoisa.bundle { tempoisa.vld v6, row 100 }
  tempoisa.bundle { tempoisa.loop_end }
  tempoisa.bundle { tempoisa.halt }
}

// -----

// The transpose array is staged 2 cycles apart, drained 6 cycles after the last stage, and drained 6 cycles apart.

// CHECK-LABEL: tempoisa.program @transpose
tempoisa.program @transpose {
  // CHECK: tempoisa.bundle delay 1 {
  // CHECK-NEXT: tempoisa.vtxin v1, quarter 0
  // CHECK: tempoisa.bundle delay 5 {
  // CHECK-NEXT: tempoisa.vtxin v2, quarter 1
  // CHECK: tempoisa.bundle delay 5 {
  // CHECK-NEXT: tempoisa.vtxout v3, quarter 0
  tempoisa.bundle { tempoisa.vtxin v1, quarter 0 }
  tempoisa.bundle { tempoisa.vtxin v2, quarter 1 }
  tempoisa.bundle { tempoisa.vtxout v3, quarter 0 }
  tempoisa.bundle { tempoisa.vtxout v4, quarter 1 }
  tempoisa.bundle { tempoisa.halt }
}
