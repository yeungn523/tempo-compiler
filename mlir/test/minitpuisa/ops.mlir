// Every instruction format parses, verifies, and prints back in the dialect's own syntax.

// RUN: minitpu-opt %s | minitpu-opt | FileCheck %s

// CHECK-LABEL: minitpuisa.program @formats
minitpuisa.program @formats {
  // CHECK: minitpuisa.bundle {
  // CHECK-NEXT: minitpuisa.vmemld slot 1, vmem 8192, rows 128, base s1, stride s2
  minitpuisa.bundle {
    minitpuisa.vmemld slot 1, vmem 8192, rows 128, base s1, stride s2
  }
  // CHECK: minitpuisa.bundle delay 63 {
  // CHECK-NEXT: minitpuisa.vredmax v31, v0
  // CHECK-NEXT: minitpuisa.vmatload v4
  // CHECK-NEXT: minitpuisa.vst v2, row 8191 agu level 3 shift 7 mask 1
  // CHECK-NEXT: minitpuisa.saddi s3, s0, -32768
  // CHECK-NEXT: minitpuisa.lbegin 65535 lo 15 step 15
  minitpuisa.bundle delay 63 {
    minitpuisa.vredmax v31, v0
    minitpuisa.vmatload v4
    minitpuisa.vst v2, row 8191 agu level 3 shift 7 mask 1
    minitpuisa.saddi s3, s0, -32768
    minitpuisa.lbegin 65535 lo 15 step 15
  }
  // CHECK: minitpuisa.bundle {
  // CHECK-NEXT: minitpuisa.vtxin v7, quarter 3
  // CHECK-NEXT: minitpuisa.vld v1, row 0{{$}}
  // CHECK-NEXT: minitpuisa.smov_arg s0, arg 3
  // CHECK-NEXT: minitpuisa.flush_slot 3
  minitpuisa.bundle delay 0 {
    minitpuisa.vtxin v7, quarter 3
    minitpuisa.vld v1, row 0 level 0 shift 0 mask 15
    minitpuisa.smov_arg s0, arg 3
    minitpuisa.flush_slot 3
  }
  minitpuisa.bundle {
    minitpuisa.halt
  }
}
