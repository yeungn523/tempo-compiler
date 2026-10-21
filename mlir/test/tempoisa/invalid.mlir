// Every rule a bundle, a program, or an instruction field holds is refused with its own diagnostic.

// RUN: tempo-opt %s -split-input-file -verify-diagnostics

tempoisa.program @two_in_one_slot {
  tempoisa.bundle {
    tempoisa.vadd v1, v2, v3
    // expected-error @below {{is unable to claim the E slot}}
    tempoisa.vsub v1, v2, v3
  }
  tempoisa.bundle {
    tempoisa.halt
  }
}

// -----

tempoisa.program @descriptor_shares_bundle {
  tempoisa.bundle {
    // expected-error @below {{is unable to share its bundle}}
    tempoisa.vmemld slot 0, vmem 0, rows 1, base s0, stride s1
    tempoisa.vadd v1, v2, v3
  }
  tempoisa.bundle {
    tempoisa.halt
  }
}

// -----

// expected-error @below {{must end in a bundle carrying halt}}
tempoisa.program @never_halts {
  tempoisa.bundle {
    tempoisa.vadd v1, v2, v3
  }
}

// -----

tempoisa.program @register_past_the_file {
  tempoisa.bundle {
    // expected-error @below {{attribute 'reg_idx' failed to satisfy constraint}}
    tempoisa.vmatpush v32
  }
  tempoisa.bundle {
    tempoisa.halt
  }
}

// -----

tempoisa.program @descriptor_moving_no_rows {
  tempoisa.bundle {
    // expected-error @below {{attribute 'rows' failed to satisfy constraint}}
    tempoisa.vmemld slot 0, vmem 0, rows 0, base s0, stride s1
  }
  tempoisa.bundle {
    tempoisa.halt
  }
}

// -----

tempoisa.program @delay_past_its_field {
  // expected-error @below {{attribute 'delay' failed to satisfy constraint}}
  tempoisa.bundle delay 64 {
    tempoisa.halt
  }
}
