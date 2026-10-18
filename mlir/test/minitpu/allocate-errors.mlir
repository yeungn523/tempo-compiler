// A kernel whose buffers or live values exceed what the machine holds is refused, naming the shortfall.

// RUN: minitpu-opt %s -split-input-file -pass-pipeline='builtin.module(minitpu.kernel(minitpu-allocate-vmem,minitpu-allocate-registers))' -verify-diagnostics

minitpu.kernel @buffers_past_a_slot {
  // expected-note @below {{buffer of 8000 rows}}
  %first = minitpu.alloc : !minitpu.vmem<8000>
  // expected-error @below {{A kernel's buffers must fit one VMEM slot's 8192 rows}}
  // expected-note @below {{buffer of 400 rows}}
  %second = minitpu.alloc : !minitpu.vmem<400>
}

// -----

minitpu.kernel @more_scalars_than_registers {
  %a = minitpu.arg 0
  %b = minitpu.arg 1
  %c = minitpu.arg 2
  %d = minitpu.arg 3
  // expected-error @below {{is unable to get a scalar register. 5 values are live here, where the file holds 4}}
  %e = arith.constant 32 : i32
  %buffer = minitpu.alloc : !minitpu.vmem<16>
  %c0 = arith.constant 0 : index
  minitpu.vmemld %buffer[%c0], base %a, stride %e, rows 1, slot 0 : !minitpu.vmem<16>
  minitpu.vmemld %buffer[%c0], base %b, stride %e, rows 1, slot 1 : !minitpu.vmem<16>
  minitpu.flush_slot 3
  minitpu.vmemld %buffer[%c0], base %c, stride %d, rows 1, slot 0 : !minitpu.vmem<16>
  minitpu.flush_slot 1
}
