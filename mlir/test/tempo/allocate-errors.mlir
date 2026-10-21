// A kernel whose buffers or live values exceed what the machine holds is refused, naming the shortfall.

// RUN: tempo-opt %s -split-input-file -pass-pipeline='builtin.module(tempo.kernel(tempo-allocate-vmem,tempo-allocate-registers))' -verify-diagnostics

tempo.kernel @buffers_past_a_slot {
  // expected-note @below {{buffer of 8000 rows}}
  %first = tempo.alloc : !tempo.vmem<8000>
  // expected-error @below {{A kernel's buffers must fit one VMEM slot's 8192 rows}}
  // expected-note @below {{buffer of 400 rows}}
  %second = tempo.alloc : !tempo.vmem<400>
}

// -----

tempo.kernel @more_scalars_than_registers {
  %e = arith.constant 32 : i32
  %a = tempo.arg 0
  %b = tempo.arg 1
  %c = tempo.arg 2
  // expected-error @below {{is unable to get a scalar register. 5 values are live here, where the file holds 4}}
  %d = tempo.arg 3
  %buffer = tempo.alloc : !tempo.vmem<16>
  %c0 = arith.constant 0 : index
  tempo.vmemld %buffer[%c0], base %a, stride %e, rows 1, slot 0 : !tempo.vmem<16>
  tempo.vmemld %buffer[%c0], base %b, stride %c, rows 1, slot 1 : !tempo.vmem<16>
  tempo.flush_slot 3
  tempo.vmemld %buffer[%c0], base %d, stride %e, rows 1, slot 0 : !tempo.vmem<16>
  tempo.flush_slot 1
  tempo.vmemld %buffer[%c0], base %a, stride %b, rows 1, slot 0 : !tempo.vmem<16>
  tempo.flush_slot 1
  tempo.vmemld %buffer[%c0], base %c, stride %d, rows 1, slot 0 : !tempo.vmem<16>
  tempo.flush_slot 1
}
