// Every hazard no delay repairs is refused before the scheduler places any delay.

// RUN: minitpu-opt %s -split-input-file -minitpuisa-schedule -verify-diagnostics

minitpuisa.program @two_port_a_readers {
  // expected-error @below {{must claim VREG read port A on separate cycles}}
  minitpuisa.bundle {
    minitpuisa.vadd v1, v2, v3
    minitpuisa.vst v4, row 0
  }
  minitpuisa.bundle { minitpuisa.halt }
}

// -----

minitpuisa.program @write_beside_pop {
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  // expected-error @below {{must be the only VREG writer in its bundle}}
  minitpuisa.bundle {
    minitpuisa.vmatpop v8
    minitpuisa.vld v1, row 0
  }
  minitpuisa.bundle { minitpuisa.halt }
}

// -----

minitpuisa.program @fifo_overflow {
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  // expected-error @below {{The MXU output FIFO must hold every outstanding result}}
  minitpuisa.bundle { minitpuisa.vmatpush v0 }
  minitpuisa.bundle { minitpuisa.halt }
}

// -----

minitpuisa.program @flush_of_unused_slot {
  minitpuisa.bundle { minitpuisa.vmemld slot 0, vmem 0, rows 1, base s0, stride s1 }
  // expected-error @below {{A flush must wait only on DMA slots a descriptor issues on}}
  minitpuisa.bundle { minitpuisa.flush_slot 2 }
  minitpuisa.bundle { minitpuisa.halt }
}

// -----

minitpuisa.program @slot_reused_unflushed {
  minitpuisa.bundle { minitpuisa.vmemld slot 0, vmem 0, rows 1, base s0, stride s1 }
  // expected-error @below {{A slot must be flushed before it carries another descriptor}}
  minitpuisa.bundle { minitpuisa.vmemld slot 0, vmem 16, rows 1, base s0, stride s1 }
  minitpuisa.bundle { minitpuisa.halt }
}
