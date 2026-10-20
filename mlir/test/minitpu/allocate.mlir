// The allocators name every unpinned buffer's rows and every unpinned value's register. Buffers take the next free rows
// in program order, stepping over pinned ones, and values take the free register idle the longest, never-used registers
// first, each argument read just before its first use. A loop-carried value shares one register with its initial value
// and result, an initial value read elsewhere is copied first, and a vmatload's four operands take consecutive
// registers.

// RUN: minitpu-opt %s -split-input-file -pass-pipeline='builtin.module(minitpu.kernel(minitpu-allocate-vmem,minitpu-allocate-registers))' | FileCheck %s

// CHECK-LABEL: minitpu.kernel @accumulate
minitpu.kernel @accumulate {
  // CHECK: minitpu.alloc {base = 0 : i32} : !minitpu.vmem<16>
  // CHECK: minitpu.alloc {base = 32 : i32} : !minitpu.vmem<8>
  // CHECK: minitpu.alloc {base = 40 : i32} : !minitpu.vmem<4>
  %buffer = minitpu.alloc : !minitpu.vmem<16>
  %pinned = minitpu.alloc {base = 16 : i32} : !minitpu.vmem<16>
  %after = minitpu.alloc : !minitpu.vmem<8>
  %last = minitpu.alloc : !minitpu.vmem<4>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: arith.constant {minitpu.reg = 0 : i32} 32 : i32
  // CHECK: minitpu.arg 0 {minitpu.reg = 1 : i32}
  %pointer = minitpu.arg 0
  %stride = arith.constant 32 : i32
  minitpu.vmemld %buffer[%c0], base %pointer, stride %stride, rows 16, slot 0 : !minitpu.vmem<16>
  minitpu.flush_slot 1
  // CHECK: minitpu.vld %{{.*}}[%{{.*}}] {minitpu.reg = 0 : i32}
  %initial = minitpu.vld %buffer[%c0] : !minitpu.vmem<16>
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%accumulated = %initial) -> (!minitpu.vreg) {
    %row = arith.muli %i, %c4 : index
    // CHECK: minitpu.vld %{{.*}}[%{{.*}}] {minitpu.reg = 1 : i32}
    // CHECK: minitpu.vadd %{{.*}}, %{{.*}} {minitpu.reg = 0 : i32}
    %value = minitpu.vld %buffer[%row] : !minitpu.vmem<16>
    %next = minitpu.vadd %accumulated, %value
    scf.yield %next : !minitpu.vreg
  }
  minitpu.vst %sum, %buffer[%c0] : !minitpu.vmem<16>
}

// -----

// CHECK-LABEL: minitpu.kernel @initial_value_read_twice
minitpu.kernel @initial_value_read_twice {
  %buffer = minitpu.alloc : !minitpu.vmem<16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[INITIAL:.*]] = minitpu.vld %{{.*}}[%{{.*}}] {minitpu.reg = 0 : i32}
  // CHECK: %[[COPY:.*]] = minitpu.vmov %[[INITIAL]] {minitpu.reg = 1 : i32}
  // CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[COPY]])
  %initial = minitpu.vld %buffer[%c0] : !minitpu.vmem<16>
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%accumulated = %initial) -> (!minitpu.vreg) {
    %next = minitpu.vexp %accumulated
    scf.yield %next : !minitpu.vreg
  }
  %both = minitpu.vadd %sum, %initial
  minitpu.vst %both, %buffer[%c0] : !minitpu.vmem<16>
}

// -----

// CHECK-LABEL: minitpu.kernel @weights_in_a_row
minitpu.kernel @weights_in_a_row {
  %buffer = minitpu.alloc : !minitpu.vmem<16>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  // CHECK: minitpu.vld %{{.*}}[%{{.*}}] {minitpu.reg = 0 : i32}
  // CHECK: minitpu.vld %{{.*}}[%{{.*}}] {minitpu.reg = 1 : i32}
  // CHECK: minitpu.vld %{{.*}}[%{{.*}}] {minitpu.reg = 2 : i32}
  // CHECK: minitpu.vld %{{.*}}[%{{.*}}] {minitpu.reg = 3 : i32}
  // CHECK: minitpu.vld %{{.*}}[%{{.*}}] {minitpu.reg = 4 : i32}
  // CHECK: minitpu.vld %{{.*}}[%{{.*}}] {minitpu.reg = 5 : i32}
  // CHECK: minitpu.vld %{{.*}}[%{{.*}}] {minitpu.reg = 6 : i32}
  // CHECK: minitpu.vadd %{{.*}}, %{{.*}} {minitpu.reg = 7 : i32}
  %activation = minitpu.vld %buffer[%c0] : !minitpu.vmem<16>
  %bias = minitpu.vld %buffer[%c4] : !minitpu.vmem<16>
  %other = minitpu.vld %buffer[%c8] : !minitpu.vmem<16>
  %w0 = minitpu.vld %buffer[%c0] : !minitpu.vmem<16>
  %w1 = minitpu.vld %buffer[%c4] : !minitpu.vmem<16>
  %w2 = minitpu.vld %buffer[%c8] : !minitpu.vmem<16>
  %w3 = minitpu.vld %buffer[%c12] : !minitpu.vmem<16>
  minitpu.vmatload %w0, %w1, %w2, %w3
  minitpu.vmatpush %activation
  %sum = minitpu.vadd %bias, %other
  minitpu.vst %sum, %buffer[%c0] : !minitpu.vmem<16>
}
