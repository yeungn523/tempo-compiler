// The packer keeps every vtxin and vtxout in program order, since they share the transpose staging array, which no
// register names: a vtxout reading no register stays behind the vtxin staging the tile it drains, however early its
// registers would let it issue.

// RUN: tempo-opt %s -tempoisa-pack | FileCheck %s

// CHECK-LABEL: tempoisa.program @transpose_order
// CHECK:       tempoisa.vtxin v1, quarter 0
// CHECK:       tempoisa.vtxout v5, quarter 0
tempoisa.program @transpose_order {
  tempoisa.bundle { tempoisa.vld v1, row 0 }
  tempoisa.bundle { tempoisa.vtxin v1, quarter 0 }
  tempoisa.bundle { tempoisa.vtxout v5, quarter 0 }
  tempoisa.bundle { tempoisa.halt }
}
