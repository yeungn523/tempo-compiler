// A kernel states its machine operations over values: registers are SSA values, VMEM regions are buffers, loops are
// scf.for and scalar arithmetic is arith. Every form parses, verifies, and prints back.

// RUN: tempo-opt %s | tempo-opt | FileCheck %s

// CHECK-LABEL: tempo.kernel @gemm_step
tempo.kernel @gemm_step {
  // CHECK: %[[ACT:.*]] = tempo.alloc : !tempo.vmem<64>
  // CHECK: %[[W:.*]] = tempo.alloc {base = 64 : i32} : !tempo.vmem<16>
  %act = tempo.alloc : !tempo.vmem<64>
  %w = tempo.alloc {base = 64 : i32} : !tempo.vmem<16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c12 = arith.constant 12 : index
  %c48 = arith.constant 48 : index
  // CHECK: %[[P:.*]] = tempo.arg 0
  %p = tempo.arg 0
  %stride = arith.constant 32 : i32
  // CHECK: tempo.vmemld %[[ACT]][%{{.*}}], base %[[P]], stride %{{.*}}, rows 64, slot 0 : !tempo.vmem<64>
  tempo.vmemld %act[%c0], base %p, stride %stride, rows 64, slot 0 : !tempo.vmem<64>
  tempo.flush_slot 1
  %w0 = tempo.vld %w[%c0] : !tempo.vmem<16>
  %w1 = tempo.vld %w[%c4] : !tempo.vmem<16>
  %c8 = arith.constant 8 : index
  %w2 = tempo.vld %w[%c8] : !tempo.vmem<16>
  %w3 = tempo.vld %w[%c12] : !tempo.vmem<16>
  // CHECK: tempo.vmatload
  tempo.vmatload %w0, %w1, %w2, %w3
  %zero = tempo.vld %act[%c0] mask 1 : !tempo.vmem<64>
  // CHECK: scf.for
  %sum = scf.for %k = %c0 to %c48 step %c1 iter_args(%acc = %zero) -> (!tempo.vreg) {
    %row = arith.muli %k, %c4 : index
    %a = tempo.vld %act[%row] : !tempo.vmem<64>
    tempo.vmatpush %a
    // CHECK: tempo.vmatpop {tempo.reg = 8 : i32}
    %r = tempo.vmatpop {tempo.reg = 8 : i32}
    %next = tempo.vadd %acc, %r
    scf.yield %next : !tempo.vreg
  }
  // CHECK: tempo.vgelu
  %g = tempo.vgelu %sum
  tempo.vst %g, %act[%c0] : !tempo.vmem<64>
}
