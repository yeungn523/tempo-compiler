// A kernel with every register and row named lowers one bundle per machine operation. A loop index times a power of
// two becomes the AGU's induction term, the loop becomes lbegin and loop_end, and the accumulator stays in its
// register. Scheduled, the program is the image codegen/asm.py assembles from the same instructions.

// RUN: minitpu-opt %s -convert-minitpu-to-isa | FileCheck %s
// RUN: minitpu-opt %s -convert-minitpu-to-isa -minitpuisa-schedule | FileCheck %s --check-prefix=SCHEDULED

// CHECK-LABEL: minitpuisa.program @accumulate
// CHECK:      minitpuisa.smov_arg s0, arg 0
// CHECK:      minitpuisa.smovi s3, 32
// CHECK:      minitpuisa.vmemld slot 0, vmem 0, rows 16, base s0, stride s3
// CHECK:      minitpuisa.flush_slot 1
// CHECK:      minitpuisa.vld v2, row 0
// CHECK:      minitpuisa.lbegin 4
// CHECK:      minitpuisa.vld v1, row 0 agu shift 2
// CHECK:      minitpuisa.vadd v2, v2, v1
// CHECK:      minitpuisa.loop_end
// CHECK:      minitpuisa.vst v2, row 0
// CHECK:      minitpuisa.halt
// SCHEDULED:  minitpuisa.bundle delay 6 {
// SCHEDULED-NEXT: minitpuisa.vld v1, row 0 agu shift 2
minitpu.kernel @accumulate {
  %buffer = minitpu.alloc {base = 0 : i32} : !minitpu.vmem<16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %pointer = minitpu.arg 0 {minitpu.reg = 0 : i32}
  %stride = arith.constant {minitpu.reg = 3 : i32} 32 : i32
  minitpu.vmemld %buffer[%c0], base %pointer, stride %stride, rows 16, slot 0 : !minitpu.vmem<16>
  minitpu.flush_slot 1
  %initial = minitpu.vld %buffer[%c0] {minitpu.reg = 2 : i32} : !minitpu.vmem<16>
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%accumulated = %initial) -> (!minitpu.vreg) {
    %row = arith.muli %i, %c4 : index
    %value = minitpu.vld %buffer[%row] {minitpu.reg = 1 : i32} : !minitpu.vmem<16>
    %next = minitpu.vadd %accumulated, %value {minitpu.reg = 2 : i32}
    scf.yield %next : !minitpu.vreg
  }
  minitpu.vst %sum, %buffer[%c0] : !minitpu.vmem<16>
}
