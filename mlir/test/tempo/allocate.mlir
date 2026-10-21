// The allocators name every unpinned buffer's rows and every unpinned value's register. Buffers take the next free rows
// in program order, stepping over pinned ones, and values take the free register idle the longest, never-used registers
// first, each argument read just before its first use. A loop-carried value shares one register with its initial value
// and result, an initial value read elsewhere is copied first, and a vmatload's four operands take consecutive
// registers.

// RUN: tempo-opt %s -split-input-file -pass-pipeline='builtin.module(tempo.kernel(tempo-allocate-vmem,tempo-allocate-registers))' | FileCheck %s

// CHECK-LABEL: tempo.kernel @accumulate
tempo.kernel @accumulate {
  // CHECK: tempo.alloc {base = 0 : i32} : !tempo.vmem<16>
  // CHECK: tempo.alloc {base = 32 : i32} : !tempo.vmem<8>
  // CHECK: tempo.alloc {base = 40 : i32} : !tempo.vmem<4>
  %buffer = tempo.alloc : !tempo.vmem<16>
  %pinned = tempo.alloc {base = 16 : i32} : !tempo.vmem<16>
  %after = tempo.alloc : !tempo.vmem<8>
  %last = tempo.alloc : !tempo.vmem<4>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: arith.constant {tempo.reg = 0 : i32} 32 : i32
  // CHECK: tempo.arg 0 {tempo.reg = 1 : i32}
  %pointer = tempo.arg 0
  %stride = arith.constant 32 : i32
  tempo.vmemld %buffer[%c0], base %pointer, stride %stride, rows 16, slot 0 : !tempo.vmem<16>
  tempo.flush_slot 1
  // CHECK: tempo.vld %{{.*}}[%{{.*}}] {tempo.reg = 0 : i32}
  %initial = tempo.vld %buffer[%c0] : !tempo.vmem<16>
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%accumulated = %initial) -> (!tempo.vreg) {
    %row = arith.muli %i, %c4 : index
    // CHECK: tempo.vld %{{.*}}[%{{.*}}] {tempo.reg = 1 : i32}
    // CHECK: tempo.vadd %{{.*}}, %{{.*}} {tempo.reg = 0 : i32}
    %value = tempo.vld %buffer[%row] : !tempo.vmem<16>
    %next = tempo.vadd %accumulated, %value
    scf.yield %next : !tempo.vreg
  }
  tempo.vst %sum, %buffer[%c0] : !tempo.vmem<16>
}

// -----

// CHECK-LABEL: tempo.kernel @initial_value_read_twice
tempo.kernel @initial_value_read_twice {
  %buffer = tempo.alloc : !tempo.vmem<16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[INITIAL:.*]] = tempo.vld %{{.*}}[%{{.*}}] {tempo.reg = 0 : i32}
  // CHECK: %[[COPY:.*]] = tempo.vmov %[[INITIAL]] {tempo.reg = 1 : i32}
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[COPY]])
  %initial = tempo.vld %buffer[%c0] : !tempo.vmem<16>
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%accumulated = %initial) -> (!tempo.vreg) {
    %next = tempo.vexp %accumulated
    scf.yield %next : !tempo.vreg
  }
  %both = tempo.vadd %sum, %initial
  tempo.vst %both, %buffer[%c0] : !tempo.vmem<16>
}

// -----

// CHECK-LABEL: tempo.kernel @weights_in_a_row
tempo.kernel @weights_in_a_row {
  %buffer = tempo.alloc : !tempo.vmem<16>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  // CHECK: tempo.vld %{{.*}}[%{{.*}}] {tempo.reg = 0 : i32}
  // CHECK: tempo.vld %{{.*}}[%{{.*}}] {tempo.reg = 1 : i32}
  // CHECK: tempo.vld %{{.*}}[%{{.*}}] {tempo.reg = 2 : i32}
  // CHECK: tempo.vld %{{.*}}[%{{.*}}] {tempo.reg = 3 : i32}
  // CHECK: tempo.vld %{{.*}}[%{{.*}}] {tempo.reg = 4 : i32}
  // CHECK: tempo.vld %{{.*}}[%{{.*}}] {tempo.reg = 5 : i32}
  // CHECK: tempo.vld %{{.*}}[%{{.*}}] {tempo.reg = 6 : i32}
  // CHECK: tempo.vadd %{{.*}}, %{{.*}} {tempo.reg = 7 : i32}
  %activation = tempo.vld %buffer[%c0] : !tempo.vmem<16>
  %bias = tempo.vld %buffer[%c4] : !tempo.vmem<16>
  %other = tempo.vld %buffer[%c8] : !tempo.vmem<16>
  %w0 = tempo.vld %buffer[%c0] : !tempo.vmem<16>
  %w1 = tempo.vld %buffer[%c4] : !tempo.vmem<16>
  %w2 = tempo.vld %buffer[%c8] : !tempo.vmem<16>
  %w3 = tempo.vld %buffer[%c12] : !tempo.vmem<16>
  tempo.vmatload %w0, %w1, %w2, %w3
  tempo.vmatpush %activation
  %sum = tempo.vadd %bias, %other
  tempo.vst %sum, %buffer[%c0] : !tempo.vmem<16>
}
