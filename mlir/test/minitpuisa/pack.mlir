// The packer keeps every vtxin and vtxout in program order, since they share the transpose staging array, which no
// register names: a vtxout reading no register stays behind the vtxin staging the tile it drains, however early its
// registers would let it issue.

// RUN: minitpu-opt %s -minitpuisa-pack | FileCheck %s

// CHECK-LABEL: minitpuisa.program @transpose_order
// CHECK:       minitpuisa.vtxin v1, quarter 0
// CHECK:       minitpuisa.vtxout v5, quarter 0
minitpuisa.program @transpose_order {
  minitpuisa.bundle { minitpuisa.vld v1, row 0 }
  minitpuisa.bundle { minitpuisa.vtxin v1, quarter 0 }
  minitpuisa.bundle { minitpuisa.vtxout v5, quarter 0 }
  minitpuisa.bundle { minitpuisa.halt }
}
