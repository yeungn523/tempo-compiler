// Every field and buffer rule is refused with its own diagnostic.

// RUN: tempo-opt %s -split-input-file -verify-diagnostics

tempo.kernel @descriptor_past_its_buffer {
  %buffer = tempo.alloc : !tempo.vmem<16>
  %c8 = arith.constant 8 : index
  %p = tempo.arg 0
  %stride = arith.constant 32 : i32
  // expected-error @below {{A descriptor must stay inside its buffer, which holds 16 rows}}
  tempo.vmemld %buffer[%c8], base %p, stride %stride, rows 16, slot 0 : !tempo.vmem<16>
}

// -----

tempo.kernel @vector_register_past_the_file {
  // expected-error @below {{is unable to live in v32. It must name one of the 32 vector registers}}
  %r = tempo.vmatpop {tempo.reg = 32 : i32}
}

// -----

tempo.kernel @scalar_register_past_the_file {
  // expected-error @below {{is unable to live in s4. It must name one of the 4 scalar registers}}
  %p = tempo.arg 0 {tempo.reg = 4 : i32}
}

// -----

tempo.kernel @register_on_an_index {
  // expected-error @below {{Only a vector or i32 scalar value lives in a register}}
  %c0 = arith.constant {tempo.reg = 0 : i32} 0 : index
}

// -----

tempo.kernel @mask_naming_no_sublane {
  %buffer = tempo.alloc : !tempo.vmem<16>
  %c0 = arith.constant 0 : index
  // expected-error @below {{attribute 'mask' failed to satisfy constraint}}
  %v = tempo.vld %buffer[%c0] mask 0 : !tempo.vmem<16>
}
