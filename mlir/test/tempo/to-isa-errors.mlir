// Every form the machine is unable to run is refused with its own diagnostic rather than lowered approximately.

// RUN: tempo-opt %s -split-input-file -convert-tempo-to-isa -verify-diagnostics

tempo.kernel @value_with_no_register {
  %buffer = tempo.alloc {base = 0 : i32} : !tempo.vmem<16>
  %c0 = arith.constant 0 : index
  %v = tempo.vld %buffer[%c0] {tempo.reg = 1 : i32} : !tempo.vmem<16>
  // expected-error @below {{reads a value with no register}}
  %w = tempo.vexp %v
}

// -----

tempo.kernel @buffer_with_no_row {
  %buffer = tempo.alloc : !tempo.vmem<16>
  %c0 = arith.constant 0 : index
  // expected-error @below {{reads a buffer with no base row}}
  %v = tempo.vld %buffer[%c0] {tempo.reg = 1 : i32} : !tempo.vmem<16>
}

// -----

tempo.kernel @row_the_agu_cannot_add {
  %buffer = tempo.alloc {base = 0 : i32} : !tempo.vmem<64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %row = arith.muli %i, %c3 : index
    // expected-error @below {{is unable to address its row}}
    %v = tempo.vld %buffer[%row] {tempo.reg = 1 : i32} : !tempo.vmem<64>
  }
}

// -----

tempo.kernel @loop_value_changing_register {
  %buffer = tempo.alloc {base = 0 : i32} : !tempo.vmem<16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %initial = tempo.vld %buffer[%c0] {tempo.reg = 2 : i32} : !tempo.vmem<16>
  // expected-error @below {{A loop-carried value must stay in one register}}
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%accumulated = %initial) -> (!tempo.vreg) {
    %next = tempo.vexp %accumulated {tempo.reg = 5 : i32}
    scf.yield %next : !tempo.vreg
  }
}

// -----

tempo.kernel @weights_not_consecutive {
  %buffer = tempo.alloc {base = 0 : i32} : !tempo.vmem<16>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %w0 = tempo.vld %buffer[%c0] {tempo.reg = 16 : i32} : !tempo.vmem<16>
  %w1 = tempo.vld %buffer[%c4] {tempo.reg = 17 : i32} : !tempo.vmem<16>
  %w2 = tempo.vld %buffer[%c8] {tempo.reg = 20 : i32} : !tempo.vmem<16>
  %w3 = tempo.vld %buffer[%c12] {tempo.reg = 19 : i32} : !tempo.vmem<16>
  // expected-error @below {{The four quarters must sit in consecutive registers from v16}}
  tempo.vmatload %w0, %w1, %w2, %w3
}

// -----

tempo.kernel @descriptor_walking_its_row {
  %buffer = tempo.alloc {base = 0 : i32} : !tempo.vmem<64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %pointer = tempo.arg 0 {tempo.reg = 0 : i32}
  %stride = arith.constant {tempo.reg = 3 : i32} 32 : i32
  scf.for %i = %c0 to %c4 step %c1 {
    %row = arith.muli %i, %c4 : index
    // expected-error @below {{its row must be a constant}}
    tempo.vmemld %buffer[%row], base %pointer, stride %stride, rows 4, slot 0 : !tempo.vmem<64>
  }
}
