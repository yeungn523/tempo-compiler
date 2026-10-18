// Every form the machine is unable to run is refused with its own diagnostic rather than lowered approximately.

// RUN: minitpu-opt %s -split-input-file -convert-minitpu-to-isa -verify-diagnostics

minitpu.kernel @value_with_no_register {
  %buffer = minitpu.alloc {base = 0 : i32} : !minitpu.vmem<16>
  %c0 = arith.constant 0 : index
  %v = minitpu.vld %buffer[%c0] {minitpu.reg = 1 : i32} : !minitpu.vmem<16>
  // expected-error @below {{reads a value with no register}}
  %w = minitpu.vexp %v
}

// -----

minitpu.kernel @buffer_with_no_row {
  %buffer = minitpu.alloc : !minitpu.vmem<16>
  %c0 = arith.constant 0 : index
  // expected-error @below {{reads a buffer with no base row}}
  %v = minitpu.vld %buffer[%c0] {minitpu.reg = 1 : i32} : !minitpu.vmem<16>
}

// -----

minitpu.kernel @row_the_agu_cannot_add {
  %buffer = minitpu.alloc {base = 0 : i32} : !minitpu.vmem<64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %row = arith.muli %i, %c3 : index
    // expected-error @below {{is unable to address its row}}
    %v = minitpu.vld %buffer[%row] {minitpu.reg = 1 : i32} : !minitpu.vmem<64>
  }
}

// -----

minitpu.kernel @loop_value_changing_register {
  %buffer = minitpu.alloc {base = 0 : i32} : !minitpu.vmem<16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %initial = minitpu.vld %buffer[%c0] {minitpu.reg = 2 : i32} : !minitpu.vmem<16>
  // expected-error @below {{A loop-carried value must stay in one register}}
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%accumulated = %initial) -> (!minitpu.vreg) {
    %next = minitpu.vexp %accumulated {minitpu.reg = 5 : i32}
    scf.yield %next : !minitpu.vreg
  }
}

// -----

minitpu.kernel @weights_not_consecutive {
  %buffer = minitpu.alloc {base = 0 : i32} : !minitpu.vmem<16>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %w0 = minitpu.vld %buffer[%c0] {minitpu.reg = 16 : i32} : !minitpu.vmem<16>
  %w1 = minitpu.vld %buffer[%c4] {minitpu.reg = 17 : i32} : !minitpu.vmem<16>
  %w2 = minitpu.vld %buffer[%c8] {minitpu.reg = 20 : i32} : !minitpu.vmem<16>
  %w3 = minitpu.vld %buffer[%c12] {minitpu.reg = 19 : i32} : !minitpu.vmem<16>
  // expected-error @below {{The four quarters must sit in consecutive registers from v16}}
  minitpu.vmatload %w0, %w1, %w2, %w3
}

// -----

minitpu.kernel @descriptor_walking_its_row {
  %buffer = minitpu.alloc {base = 0 : i32} : !minitpu.vmem<64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %pointer = minitpu.arg 0 {minitpu.reg = 0 : i32}
  %stride = arith.constant {minitpu.reg = 3 : i32} 32 : i32
  scf.for %i = %c0 to %c4 step %c1 {
    %row = arith.muli %i, %c4 : index
    // expected-error @below {{its row must be a constant}}
    minitpu.vmemld %buffer[%row], base %pointer, stride %stride, rows 4, slot 0 : !minitpu.vmem<64>
  }
}
