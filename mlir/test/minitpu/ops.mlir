// A kernel states its machine operations over values: registers are SSA values, VMEM regions are buffers, loops are
// scf.for and scalar arithmetic is arith. Every form parses, verifies, and prints back.

// RUN: minitpu-opt %s | minitpu-opt | FileCheck %s

// CHECK-LABEL: minitpu.kernel @gemm_step
minitpu.kernel @gemm_step {
  // CHECK: %[[ACT:.*]] = minitpu.alloc : !minitpu.vmem<64>
  // CHECK: %[[W:.*]] = minitpu.alloc {base = 64 : i32} : !minitpu.vmem<16>
  %act = minitpu.alloc : !minitpu.vmem<64>
  %w = minitpu.alloc {base = 64 : i32} : !minitpu.vmem<16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %c48 = arith.constant 48 : index
  // CHECK: %[[P:.*]] = minitpu.arg 0
  %p = minitpu.arg 0
  %stride = arith.constant 32 : i32
  // CHECK: minitpu.vmemld %[[ACT]][%{{.*}}], base %[[P]], stride %{{.*}}, rows 64, slot 0 : !minitpu.vmem<64>
  minitpu.vmemld %act[%c0], base %p, stride %stride, rows 64, slot 0 : !minitpu.vmem<64>
  minitpu.flush_slot 1
  %w0 = minitpu.vld %w[%c0] : !minitpu.vmem<16>
  %w1 = minitpu.vld %w[%c4] : !minitpu.vmem<16>
  %c8 = arith.constant 8 : index
  %w2 = minitpu.vld %w[%c8] : !minitpu.vmem<16>
  %w3 = minitpu.vld %w[%c12] : !minitpu.vmem<16>
  // CHECK: minitpu.vmatload
  minitpu.vmatload %w0, %w1, %w2, %w3
  %zero = minitpu.vld %act[%c0] mask 1 : !minitpu.vmem<64>
  // CHECK: scf.for
  %sum = scf.for %k = %c0 to %c48 step %c1 iter_args(%acc = %zero) -> (!minitpu.vreg) {
    %row = arith.muli %k, %c4 : index
    %a = minitpu.vld %act[%row] : !minitpu.vmem<64>
    minitpu.vmatpush %a
    // CHECK: minitpu.vmatpop {minitpu.reg = 8 : i32}
    %r = minitpu.vmatpop {minitpu.reg = 8 : i32}
    %next = minitpu.vadd %acc, %r
    scf.yield %next : !minitpu.vreg
  }
  // CHECK: minitpu.vgelu
  %g = minitpu.vgelu %sum
  minitpu.vst %g, %act[%c0] : !minitpu.vmem<64>
}
