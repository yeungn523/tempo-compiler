// A kernel with every register and row named lowers one bundle per machine operation. A loop index times a power of
// two becomes the AGU's induction term, the loop becomes lbegin and loop_end, and the accumulator stays in its
// register. Scheduled, the program is the image the retired Python assembler built from the same instructions. An
// empty kernel lowers to the halt alone, the shortest program the loader takes.

// RUN: tempo-opt %s -convert-tempo-to-isa | FileCheck %s
// RUN: tempo-opt %s -convert-tempo-to-isa -tempoisa-schedule | FileCheck %s --check-prefix=SCHEDULED

// CHECK-LABEL: tempoisa.program @accumulate
// CHECK:      tempoisa.smov_arg s0, arg 0
// CHECK:      tempoisa.smovi s3, 32
// CHECK:      tempoisa.vmemld slot 0, vmem 0, rows 16, base s0, stride s3
// CHECK:      tempoisa.flush_slot 1
// CHECK:      tempoisa.vld v2, row 0
// CHECK:      tempoisa.lbegin 4
// CHECK:      tempoisa.vld v1, row 0 agu shift 2
// CHECK:      tempoisa.vadd v2, v2, v1
// CHECK:      tempoisa.loop_end
// CHECK:      tempoisa.vst v2, row 0
// CHECK:      tempoisa.halt
// SCHEDULED:  tempoisa.bundle delay 6 {
// SCHEDULED-NEXT: tempoisa.vld v1, row 0 agu shift 2
tempo.kernel @accumulate {
  %buffer = tempo.alloc {base = 0 : i32} : !tempo.vmem<16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %pointer = tempo.arg 0 {tempo.reg = 0 : i32}
  %stride = arith.constant {tempo.reg = 3 : i32} 32 : i32
  tempo.vmemld %buffer[%c0], base %pointer, stride %stride, rows 16, slot 0 : !tempo.vmem<16>
  tempo.flush_slot 1
  %initial = tempo.vld %buffer[%c0] {tempo.reg = 2 : i32} : !tempo.vmem<16>
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%accumulated = %initial) -> (!tempo.vreg) {
    %row = arith.muli %i, %c4 : index
    %value = tempo.vld %buffer[%row] {tempo.reg = 1 : i32} : !tempo.vmem<16>
    %next = tempo.vadd %accumulated, %value {tempo.reg = 2 : i32}
    scf.yield %next : !tempo.vreg
  }
  tempo.vst %sum, %buffer[%c0] : !tempo.vmem<16>
}

// CHECK-LABEL: tempoisa.program @halt
// CHECK-NEXT:    tempoisa.bundle {
// CHECK-NEXT:      tempoisa.halt
// CHECK-NEXT:    }
// CHECK-NEXT:  }
tempo.kernel @halt {
}
