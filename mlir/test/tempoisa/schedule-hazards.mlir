// Every hazard no delay repairs is refused before the scheduler places any delay.

// RUN: tempo-opt %s -split-input-file -tempoisa-schedule -verify-diagnostics

tempoisa.program @two_port_a_readers {
  // expected-error @below {{must claim VREG read port A on separate cycles}}
  tempoisa.bundle {
    tempoisa.vadd v1, v2, v3
    tempoisa.vst v4, row 0
  }
  tempoisa.bundle { tempoisa.halt }
}

// -----

tempoisa.program @write_beside_pop {
  tempoisa.bundle { tempoisa.vmatpush v0 }
  // expected-error @below {{must be the only VREG writer in its bundle}}
  tempoisa.bundle {
    tempoisa.vmatpop v8
    tempoisa.vld v1, row 0
  }
  tempoisa.bundle { tempoisa.halt }
}

// -----

tempoisa.program @fifo_overflow {
  tempoisa.bundle { tempoisa.vmatpush v0 }
  tempoisa.bundle { tempoisa.vmatpush v0 }
  tempoisa.bundle { tempoisa.vmatpush v0 }
  tempoisa.bundle { tempoisa.vmatpush v0 }
  tempoisa.bundle { tempoisa.vmatpush v0 }
  tempoisa.bundle { tempoisa.vmatpush v0 }
  tempoisa.bundle { tempoisa.vmatpush v0 }
  tempoisa.bundle { tempoisa.vmatpush v0 }
  // expected-error @below {{The MXU output FIFO must hold every outstanding result}}
  tempoisa.bundle { tempoisa.vmatpush v0 }
  tempoisa.bundle { tempoisa.halt }
}

// -----

tempoisa.program @flush_of_unused_slot {
  tempoisa.bundle { tempoisa.vmemld slot 0, vmem 0, rows 1, base s0, stride s1 }
  // expected-error @below {{A flush must wait only on DMA slots a descriptor issues on}}
  tempoisa.bundle { tempoisa.flush_slot 2 }
  tempoisa.bundle { tempoisa.halt }
}

// -----

tempoisa.program @slot_reused_unflushed {
  tempoisa.bundle { tempoisa.vmemld slot 0, vmem 0, rows 1, base s0, stride s1 }
  // expected-error @below {{A slot must be flushed before it carries another descriptor}}
  tempoisa.bundle { tempoisa.vmemld slot 0, vmem 16, rows 1, base s0, stride s1 }
  tempoisa.bundle { tempoisa.halt }
}
