// Every instruction format parses, verifies, and prints back in the dialect's own syntax.

// RUN: tempo-opt %s | tempo-opt | FileCheck %s

// CHECK-LABEL: tempoisa.program @formats
tempoisa.program @formats {
  // CHECK: tempoisa.bundle {
  // CHECK-NEXT: tempoisa.vmemld slot 1, vmem 8192, rows 128, base s1, stride s2
  tempoisa.bundle {
    tempoisa.vmemld slot 1, vmem 8192, rows 128, base s1, stride s2
  }
  // CHECK: tempoisa.bundle delay 63 {
  // CHECK-NEXT: tempoisa.vredmax v31, v0
  // CHECK-NEXT: tempoisa.vmatload v4
  // CHECK-NEXT: tempoisa.vst v2, row 8191 agu level 3 shift 7 mask 1
  // CHECK-NEXT: tempoisa.saddi s3, s0, -32768
  // CHECK-NEXT: tempoisa.lbegin 65535 lo 15 step 15
  tempoisa.bundle delay 63 {
    tempoisa.vredmax v31, v0
    tempoisa.vmatload v4
    tempoisa.vst v2, row 8191 agu level 3 shift 7 mask 1
    tempoisa.saddi s3, s0, -32768
    tempoisa.lbegin 65535 lo 15 step 15
  }
  // CHECK: tempoisa.bundle {
  // CHECK-NEXT: tempoisa.vtxin v7, quarter 3
  // CHECK-NEXT: tempoisa.vld v1, row 0{{$}}
  // CHECK-NEXT: tempoisa.smov_arg s0, arg 3
  // CHECK-NEXT: tempoisa.flush_slot 3
  tempoisa.bundle delay 0 {
    tempoisa.vtxin v7, quarter 3
    tempoisa.vld v1, row 0 level 0 shift 0 mask 15
    tempoisa.smov_arg s0, arg 3
    tempoisa.flush_slot 3
  }
  tempoisa.bundle {
    tempoisa.halt
  }
}
