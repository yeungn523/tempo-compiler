// Every rule a bundle, a program, or an instruction field holds is refused with its own diagnostic.

// RUN: minitpu-opt %s -split-input-file -verify-diagnostics

minitpuisa.program @two_in_one_slot {
  minitpuisa.bundle {
    minitpuisa.vadd v1, v2, v3
    // expected-error @below {{is unable to claim the E slot}}
    minitpuisa.vsub v1, v2, v3
  }
  minitpuisa.bundle {
    minitpuisa.halt
  }
}

// -----

minitpuisa.program @descriptor_shares_bundle {
  minitpuisa.bundle {
    // expected-error @below {{is unable to share its bundle}}
    minitpuisa.vmemld slot 0, vmem 0, rows 1, base s0, stride s1
    minitpuisa.vadd v1, v2, v3
  }
  minitpuisa.bundle {
    minitpuisa.halt
  }
}

// -----

// expected-error @below {{must end in a bundle carrying halt}}
minitpuisa.program @never_halts {
  minitpuisa.bundle {
    minitpuisa.vadd v1, v2, v3
  }
}

// -----

minitpuisa.program @register_past_the_file {
  minitpuisa.bundle {
    // expected-error @below {{attribute 'reg_idx' failed to satisfy constraint}}
    minitpuisa.vmatpush v32
  }
  minitpuisa.bundle {
    minitpuisa.halt
  }
}

// -----

minitpuisa.program @descriptor_moving_no_rows {
  minitpuisa.bundle {
    // expected-error @below {{attribute 'rows' failed to satisfy constraint}}
    minitpuisa.vmemld slot 0, vmem 0, rows 0, base s0, stride s1
  }
  minitpuisa.bundle {
    minitpuisa.halt
  }
}

// -----

minitpuisa.program @delay_past_its_field {
  // expected-error @below {{attribute 'delay' failed to satisfy constraint}}
  minitpuisa.bundle delay 64 {
    minitpuisa.halt
  }
}
