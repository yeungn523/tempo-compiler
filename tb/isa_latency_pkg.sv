// Copyright MiniTPU Authors
// SPDX-License-Identifier: Apache-2.0

// Run `python3 tools/gen_isa_doc.py --write` to generate.
//
// Concepts: ISA latency contract, writeback offset, generated artifact.
package isa_latency_pkg;
  localparam int unsigned DOC_WB_W_ALU = 5;
  localparam int unsigned DOC_WB_W_SFU = 7;
  localparam int unsigned DOC_WB_W_REDUCE = 15;
  localparam int unsigned DOC_WB_W_LANERED = 11;
  localparam int unsigned DOC_WB_W_VLD = 6;
  localparam int unsigned DOC_WB_W_MPOP_FIRST = 3;
  localparam int unsigned DOC_WB_W_MPOP_LAST = 6;
  localparam int unsigned DOC_MXU_OUTPUT_FIFO_DEPTH = 32;
  localparam int unsigned DOC_LB_CAP = 24;
  localparam int unsigned DOC_IRAM_DEPTH = 4096;
  localparam int unsigned DOC_DELAY_W = 6;
  localparam int unsigned DOC_DMA_SLOT_COUNT = 2;
  localparam int unsigned DOC_E_VADD = 1;
  localparam int unsigned DOC_E_VSUB = 2;
  localparam int unsigned DOC_E_VMUL = 3;
  localparam int unsigned DOC_E_VMAX = 11;
  localparam int unsigned DOC_E_VMIN = 12;
  localparam int unsigned DOC_E_VMOV = 4;
  localparam int unsigned DOC_E_VGELU = 5;
  localparam int unsigned DOC_E_VEXP = 6;
  localparam int unsigned DOC_E_VRECIP = 7;
  localparam int unsigned DOC_E_VRSQRT = 8;
  localparam int unsigned DOC_E_VREDSUM = 9;
  localparam int unsigned DOC_E_VREDMAX = 10;
  localparam int unsigned DOC_E_VLANERED = 15;
  localparam int unsigned DOC_E_VTXIN = 13;
  localparam int unsigned DOC_E_VTXOUT = 14;
  localparam int unsigned DOC_M_VMATLOAD = 1;
  localparam int unsigned DOC_M_VMATPUSH = 2;
  localparam int unsigned DOC_M_VMATPOP = 3;
  localparam int unsigned DOC_SMOVI = 0;
  localparam int unsigned DOC_SMOV_ARG = 1;
  localparam int unsigned DOC_SADDI = 2;
  localparam int unsigned DOC_SMAC = 3;
  localparam int unsigned DOC_SSHL = 4;
  localparam int unsigned DOC_F_LBEGIN = 1;
  localparam int unsigned DOC_F_LEND = 2;
  localparam int unsigned DOC_F_FLUSH = 3;
  localparam int unsigned DOC_F_HALT = 4;
  localparam int unsigned DOC_F_VMEMSWAP = 5;
endpackage : isa_latency_pkg
