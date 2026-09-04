"""minitpu ABI — single source of truth for the host<->FPGA interface.

This module is a design contract between hardware and software.  Any change
to the CMD-window layout, command opcodes, value semantics (units), or the
descriptor wire format must be reflected here AND must bump ABI_VERSION.

Counterpart for the program<->compute-engine encoding: minitpu/isa (ISA).

Authoritative RTL:
  hardware/rtl/minitpu_slave_axi_lite.v    — AXI-Lite register slave (CMD window)
  hardware/rtl/uncore/command_unit.sv    — CMD-window decode + route (COPY / LAUNCH)
  hardware/rtl/uncore/dm_axi_bridge.sv      — device-memory AXI address computation

Generated SV:   hardware/rtl/regmap_pkg.sv, hardware/rtl/minitpu_modes_pkg.sv
  Regenerate:   make -C hardware/rtl generate-sv-pkgs

Runtime consumer:   minitpu-runtime/runtime/pynq_host.py
  Imports:      ABI_VERSION, AXI_REG, DOORBELL_BIT, CmdOpc, CmdDir,
                REG_SEMANTICS, COPY_DESCRIPTOR, LAUNCH_DESCRIPTOR


===============================================================================
ABI Version Policy
===============================================================================

ABI_VERSION is a uint32 constant baked into the bitstream (regmap_pkg.sv).

NB (as-built, v8+) - ABI_VERSION IS NOT ENFORCEABLE AT RUNTIME.  Read this
before arguing about whether some change deserves a bump:

  - The device exposes NO readable abi_version CSR.  The v7 one was RETIRED in
    the v8 cutover: its address (0x1C) became the writable cmd_w7, and
    minitpu_slave_axi_lite.v now takes the parameter only to keep it from being
    flagged unused (`wire [31:0] _unused_abi_version = ABI_VERSION;`).  It is
    listed under "Deferred" below.
  - The runtime performs NO version check.  There is nothing to check against:
    no register carries the device's version.
  - Therefore ABI_VERSION today gates "rebuild the bitstream + redeploy the
    runtime" BY CONVENTION ONLY.  It is a source-side marker for humans and
    CI; it cannot make a host detect that it is talking to a stale bitstream,
    and bumping it does not add that ability.

Re-adding an abi_version CSR above 0x24 (additive) would make the constant
enforceable, and a host could then refuse to drive a mismatched bitstream.
That is a DELIBERATE OPEN ITEM, not an oversight - see "Deferred" below.  Until
it lands, weigh a proposed bump on what it actually buys: it costs a bitstream
rebuild and a runtime redeploy, and buys a marker, not a check.

  LANDED (#280): the ``build_id`` RO CSR @0x58 now makes the identity readable.
  It is the deferred abi_version/build_info identity CSR (see "Deferred"),
  re-added additively above 0x24 (so NO ABI_VERSION bump).  BUILD_ID embeds
  Board builds stamp BUILD_ID from the Git commit, so a host can correlate a
  loaded bitstream with its exact build provenance - which turns the #383 hazard
  (launch_status @0x3C silently reads 0 on a bitstream that lacks it) into a
  CHECKABLE identity mismatch instead of a silent no-op.  See the "Device build
  identity" section below for the value layout and the host-side check.

A bump is REQUIRED for any of:
  - Changing the CMD-window word layout (AXI_REG) or the doorbell bit
  - Changing a descriptor field (COPY_DESCRIPTOR / LAUNCH_DESCRIPTOR)
  - Adding / removing / renumbering CmdOpc or CmdDir values
  - Changing register value units / semantics (REG_SEMANTICS)

A bump is NOT required for:
  - Comments, notes, or cross-references in this file
  - Adding tests

After bumping, run `make -C hardware/rtl generate-sv-pkgs` then rebuild the bitstream.
Regenerate the snapshot: `make -C hardware/rtl abi-snapshot`.


===============================================================================
Control plane: encoded 2-kind command descriptor (ABI spec `docs/github-markdowns/22_hw_abi.md`, former charter §9)
===============================================================================

Host<->device is a single write-only CMD window of 8 x u32 words (W0..W7,
32 bytes) plus a self-clearing doorbell.  The host stages a command descriptor
into W0..W7 and rings the doorbell (bit in W0); the persistent command
unit (command_unit.sv) latches the window on the doorbell edge,
decodes W0's opcode, and routes the command:

  COPY   -> copy engine (host<->device memcpy; subsumes v7 WRITE_BRAM /
            READ_BRAM / WRITE_IRAM modes)
  LAUNCH -> core  (subsumes v7 COMPUTE mode)

Completion is polled on two read-only status words:
  instr_ready   RO — 1 when all sub-FSMs are idle (command accepted + retired)
  dma_slot_done RO — async-DMA slot completion bits

Backpressure is a host-API return (-EBUSY); there is no dedicated register.
There is exactly one bump (v7 -> v8): a clean cutover to the encoded window.

CMD window word map (byte offset = W index x 4):

  Word  Offset  COPY body               LAUNCH body
  ----  ------  ----------------------  ----------------------------------
  W0    0x00    header (see below)      header (same)
  W1    0x04    host_phys[31:0]         kbin_addr[31:0]
  W2    0x08    host_phys[63:32]        kbin_addr[63:32]
  W3    0x0C    dev_addr[31:0] (abs)    kbin_size[31:0]  (bytes)
  W4    0x10    len[31:0]  (bytes)      kernarg_ptr[31:0]   reserved [serving]
  W5    0x14    reserved                kernarg_ptr[63:32]  reserved [serving]
  W6    0x18    reserved                grid_x[31:0]        reserved [v3]
  W7    0x1C    reserved                grid_y[31:0]        reserved [v3]

W0 header bit layout:

  W0[3:0]   OPC       command opcode (CmdOpc: COPY / LAUNCH)
  W0[4]     DOORBELL  self-clearing trigger (DOORBELL_BIT)
  W0[7:5]   DIR/TGT   direction/target qualifier (CmdDir for COPY)
  W0[31:8]  reserved

  NOTE: control_plane.tex writes the header as {OPC[3:0], DIR/TGT[7:4]}.  The
  v7 self-clearing doorbell lives at slv_reg0[4] (uncore page `docs/github-markdowns/17_hw_uncore.md`, former charter §4), which lands
  inside that nominal [7:4] field.  This module keeps the doorbell at bit 4
  (preserving the existing auto-clear datapath) and places DIR/TGT at [7:5].
  TODO(abi-v8): the exact DIR/TGT bit positions vs the doorbell bit are OPEN
  pending ratification of docs/github-markdowns/22_hw_abi.md; treat [7:5] as
  provisional.

Status words (read-only), placed immediately above the CMD window:

  Offset  Name           Access  Meaning
  ------  -------------  ------  ------------------------------------------
  0x20    instr_ready    RO      1 when all sub-FSMs idle (completion + full)
  0x24    dma_slot_done  RO      async-DMA slot-done bits [1:0]

Deferred (additive later, NOT part of the minimal v1 surface): abi_version
CSR, build_info CSR, perf counters, the v2 phase-bit completion ring.  These
were live in v7 and may be re-added above 0x24 without a semantic bump.

  SCOPE of that carve-out: it covers RE-ADDING one of the retired v7 CSRs
  named above, whose semantics the host already knows.  It does NOT cover a
  genuinely NEW register.  A new RO status word that the host branches on reads
  back 0 on an older bitstream that lacks the address, so the host silently
  takes the "all clear" path - a semantic difference between two bitstreams,
  which is exactly the kind of thing ABI_VERSION is meant to mark.  So: a
  re-added v7 CSR is squarely inside the carve-out; a new register is not.

  launch_status @0x3C is such a new register (it re-uses the address the v7
  devmem_base vacated, but its semantics are new), and it was deliberately NOT
  bumped for.  It briefly was (8 -> 9), and that bump was reverted: the
  justification for it was that a host could otherwise not tell a v8 bitstream
  from a v9 one, and that justification is FALSE.  Per the NB above there is no
  readable abi_version CSR and the runtime checks nothing, so the bump could not
  buy the detection it was sold on - it only forced a rebuild + redeploy and
  carried a `breaking` label for a marker with no mechanism behind it.  The
  0x3C-reads-0-on-an-older-bitstream hazard is real and remains; it is handled
  where it can actually be handled (host-side provisioning of a matching
  build), and it is the concrete motivation for re-adding the abi_version CSR.
  Revisit this rule once that CSR exists: at that point a new register SHOULD
  bump, because a bump would finally be enforceable.
"""

import struct
from typing import Dict, List, NamedTuple, Tuple  # noqa: UP035

# ===========================================================================
# ABI version
# ===========================================================================

ABI_VERSION: int = 9
"""Source-side version constant (baked into the bitstream via regmap_pkg.sv).

Despite the name, the device does NOT report this at runtime and the runtime
does NOT verify it: there is no readable abi_version CSR (the v7 one at 0x1C was
retired in the v8 cutover), so there is nothing to check against.  Matching a
host to the bitstream it drives is a provisioning responsibility, not a runtime
check.  See the "ABI Version Policy" NB in the module docstring above.

History:
  1..6 — see git history; incremental v7-family surface (per-field CSRs +
         minitpuMode dispatch + AXI-Stream DMA + program registry).
  7 — final as-built v7: per-field CSR register map (minitpu_mode/addr_*/length/
      kernel_arg/program_id/pool_base/devmem_base + perf + build_info),
      the minitpuMode 9-mode dispatch, and the PROGRAM_IDS kernel registry.
  8 - encoded 2-kind command descriptor (ABI spec `docs/github-markdowns/22_hw_abi.md`, former charter §9).  Host<->device is a
      write-only CMD window (W0..W7) + a self-clearing doorbell, decoded by
      command_unit.sv.  COPY subsumes WRITE_BRAM/READ_BRAM/WRITE_IRAM;
      LAUNCH subsumes COMPUTE.  Removed: the per-field CSRs, the minitpuMode modes,
      the memory-tile host modes (DM_TO_MT/MT_TO_DM/MT_TO_VMEM/VMEM_TO_MT +
      addr_mt), the PROGRAM_IDS registry.  Deferred (additive later):
      abi_version/build_info CSRs, perf counters.  One deliberate bump; clean
      cutover to a new C/C++ runtime.

NOT bumped (recorded so it is not re-litigated): the RO launch_status CSR
@0x3C, added during silicon bring-up.  It is a genuinely new register, so the
"re-added v7 CSRs" carve-out does not cover it, and it was briefly bumped to 9
on that reasoning.  Reverted: the bump was justified by "a host cannot
otherwise tell a v8 bitstream from a v9 one", which is false - no readable
abi_version CSR exists and the runtime checks nothing (see the NB in the module
docstring), so the bump bought no detection while forcing a rebuild + redeploy.
The underlying hazard (0x3C reads 0 on a bitstream without it, so the
minitpuLaunchKernel guard silently passes) is real and is handled host-side by
provisioning a matching build.  Bump for a new register once the abi_version
CSR is re-added and the bump can actually be enforced.
"""

# ===========================================================================
# Device build identity - build_id CSR (RO @0x58; #280 / #383)
# ===========================================================================
#
# A device-identity register (standard practice: PCIe revision_id, XRT UUID,
# SoC IDCODE).  The host reads build_id to learn WHAT bitstream is actually
# loaded and can refuse to drive a mismatch.  Additive RO above 0x24, so it does
# NOT bump ABI_VERSION (it is the deferred abi_version/build_info identity CSR
# named in the carve-out; regenerate the snapshot only).

BUILD_ID_MAGIC: int = 0xB1
"""High byte [31:24] of BUILD_ID: a fixed self-describing tag ("build id") so a
host can validate that a read at the build_id offset is a real device identity
and not stray bus data.  A bitstream WITHOUT the build_id CSR reads 0x00000000
at that offset (the AXI-Lite slave's reserved default), which fails this magic
check - that is the concrete mismatch signal #383 needs."""

BUILD_ID: int = BUILD_ID_MAGIC << 24
"""Device identity constant baked into the bitstream (regmap_pkg::BUILD_ID),
read-only at AXI_REG['build_id'] (0x58).  Answers "what bitstream did I load?"
so the host can refuse a mismatch (#280), and makes the launch_status-reads-0
hazard (#383) a CHECKABLE identity mismatch instead of a silent no-op.

Layout (u32):
  [31:24] BUILD_ID_MAGIC (0xB1)   fixed self-describing tag
  [23:0]  Git SHA-1[0:24]                 first 24 bits of the source commit

The tracked package deliberately contains the `0xB1000000` placeholder. The
board build stamps an output-local package with the current clean `HEAD` and
writes the full SHA-1 to `build_identity/build_identity.json` beside its build
artifacts. That sidecar is the authoritative mapping, avoiding any ambiguity
from the 24-bit hardware prefix. There is no manually managed build serial.

Host check: require `(build_id >> 24) == 0xB1`, then resolve the low 24 bits
through the provenance sidecar for the deployed artifact.
Additive RO above 0x24 -> NO ABI_VERSION bump (deferred identity CSR carve-out
in the module docstring); regenerate the snapshot only: make -C hardware/rtl abi-snapshot.
"""


# ===========================================================================
# Wire format — CMD window + status
# ===========================================================================

# CMD window is 8 write-only words (W0..W7) at 0x00..0x1C, followed by two
# read-only status words.  Names are stable ABI keys used by the runtime.
AXI_REG: Dict[str, int] = {  # noqa: UP006
    "cmd_w0": 0x00,  # header: [3:0] OPC, [4] doorbell, [7:5] dir/tgt, [31:8] rsvd
    "cmd_w1": 0x04,  # COPY host_phys[31:0]   / LAUNCH kbin_addr[31:0]
    "cmd_w2": 0x08,  # COPY host_phys[63:32]  / LAUNCH kbin_addr[63:32]
    "cmd_w3": 0x0C,  # COPY dev_addr[31:0]    / LAUNCH kbin_size[31:0]
    "cmd_w4": 0x10,  # COPY len[31:0]         / LAUNCH kernarg_ptr[31:0]  (rsvd)
    "cmd_w5": 0x14,  # reserved               / LAUNCH kernarg_ptr[63:32] (rsvd)
    "cmd_w6": 0x18,  # reserved               / LAUNCH grid_x[31:0]       (rsvd)
    "cmd_w7": 0x1C,  # reserved               / LAUNCH grid_y[31:0]       (rsvd)
    "instr_ready": 0x20,  # RO: 1 when all sub-FSMs idle (completion + pollable-full)
    "dma_slot_done": 0x24,  # RO: async-DMA slot-done bits [1:0]
    # RFC P1 (kernarg arg-mode): the 4-entry kernel_arg CSR (param slots 0..3) the
    # host WRITES before LAUNCH so an arg-mode kernel binds its DRAM base/stride
    # from runtime values (minitpu_slave_axi_lite.v slv_reg9..12).  These are WRITE
    # aliases at addresses whose READ side is a pre-existing RO status word
    # (kernel_arg_0 WRITE @0x24 <-> dma_slot_done READ @0x24; _1/_2 @0x28/0x2C
    # <-> copy_status/copy_cdmasr READ; _3 @0x30 is a fresh WO slot).  Naming
    # them (they were referenced by name in this file since v2 but had no AXI_REG
    # entry) is a pure derivation so the host reads AXI_REG[...] instead of a raw
    # hex literal; the READ map is unchanged.  Additive at/above 0x24 with an
    # unchanged read map -> NO ABI_VERSION bump (regenerate the snapshot only).
    "kernel_arg_0": 0x24,  # WO: param slot 0 (slv_reg9; 0x24 READ stays dma_slot_done)
    "kernel_arg_1": 0x28,  # WO: param slot 1 (slv_reg10; 0x28 READ stays copy_status)
    "kernel_arg_2": 0x2C,  # WO: param slot 2 (slv_reg11; 0x2C READ stays copy_cdmasr)
    "kernel_arg_3": 0x30,  # WO: param slot 3 (slv_reg12; fresh WO slot)
    # Copy-engine diagnostic status (RO, additive above 0x24; silicon bring-up).
    # Additive-above-0x24 status words do NOT bump ABI_VERSION (see the module
    # docstring "Deferred ... may be re-added above 0x24 without a semantic bump").
    "copy_status": 0x28,  # RO: [0]=err [1]=oob [2]=timeout [3]=done [4]=busy [5]=reset_timeout [15:8]=xfers
    "copy_cdmasr": 0x2C,  # RO: last CDMASR value the copy engine read during POLL
    # LAUNCH status (RO, above 0x24; silicon bring-up).  Occupies the address
    # slot the retired v7 devmem_base register vacated, but it is a NEW register
    # rather than a re-added v7 CSR, so the carve-out does not cover it.  It
    # still does NOT bump ABI_VERSION: a bump is unenforceable today (no
    # readable abi_version CSR), so it would cost a rebuild + redeploy and buy
    # nothing.  See the carve-out scope note in the module docstring - it
    # records the reverted 8 -> 9 bump and when to revisit.
    "launch_status": 0x3C,  # RO: [0]=launch_error (sticky; last LAUNCH aborted on a failed kbin load)
    # Core performance counters (RO, additive above 0x24; does NOT bump ABI_VERSION).
    # These are the only DEVICE-side timing source: host wall-clock around a
    # launch also measures COPY, doorbell polling and Python overhead.
    "perf_cnt_cycles": 0x40,  # RO: free-running core cycle counter
    "perf_cnt_instrs": 0x44,  # RO: bundles issued (bundle_issued_o pulses)
    # DMA sticky error status (RO, additive above 0x24; does NOT bump ABI_VERSION).
    "dma_err": 0x4C,  # RO: [0]=dma_err, [1]=resp_error, [2]=watchdog, [3]=rsp_seen, [4]=bridge_to, [5]=ar_seen, [6]=r_seen, [7]=aw_seen, [15:8]=err_slot, [17:16]=resp, [18]=b_seen
    # DMA bandwidth counter (RO, additive above 0x24; does NOT bump ABI_VERSION).
    "beat_count": 0x50,  # RO: free-running count of 256-bit DM words moved (loads+stores)
    # DMA overlap observable (RO, additive above 0x24; does NOT bump ABI_VERSION).
    "dma_overlap": 0x54,  # RO: [7:0]=max concurrent non-IDLE slots, [31:8]=>=2-non-IDLE dwell cycles
    # Device build identity (RO, additive above 0x24; does NOT bump ABI_VERSION).
    # [31:24]=BUILD_ID_MAGIC (0xB1), [23:0]=Git SHA-1 prefix (board-stamped).
    "build_id": 0x58,  # RO: baked device identity (regmap_pkg::BUILD_ID); host reads to refuse a mismatch (#280/#383)
}

CMD_WINDOW_WORDS: int = 8
"""Number of u32 words in the write-only CMD window (W0..W7)."""

DOORBELL_BIT: int = 0x10
"""Self-clearing doorbell — bit 4 of cmd_w0 (W0[4]).

Auto-clears within 1-2 cycles of the command unit latching the window.
Preserved bit position from v7 (slv_reg0[4]); see the W0 header note in the
module docstring for the DIR/TGT overlap TODO(abi-v8).
"""

LAUNCH_ERROR_BIT: int = 0x1
"""Sticky launch_error - bit 0 of the RO ``launch_status`` CSR (0x3C).

Set when the command_unit aborted the last LAUNCH because the iram_loader
flagged ``load_error`` (oversize kbin clamped past IRAM_DEPTH, or a sub-word
kbin that loaded nothing).  The tile is NOT started in that case and the
command retires normally, so ``instr_ready`` returns to 1 and a completion poll
alone cannot distinguish an aborted LAUNCH from a successful one: the host MUST
read this bit to tell them apart (minitpuLaunchKernel does).  Cleared when the
next LAUNCH is accepted (keyed on OPC_LAUNCH specifically, not on every
accepted command, so an intervening COPY cannot wipe a still-unread status).

LIMIT - what this bit does NOT cover (FPGA_COPY_ONLY builds):
A COPY-only first-light bitstream has no core and no iram_loader, so
the command_unit retires OPC_LAUNCH immediately without executing anything
and launch_status stays 0.  A host cannot distinguish "the kernel ran fine"
from "this build has no tile to run it on" - both look like success.  This is
deliberate and NOT fixed by setting launch_error: that bit means "the loader
failed to load this kernel binary", and overloading it to also mean "this build
cannot run kernels at all" would destroy its diagnostic value.  Detecting a
COPY-only bitstream is the host's job via the build/ABI identity it provisioned,
not via launch_error.  See command_unit.sv (the FPGA_COPY_ONLY arm
of OPC_LAUNCH) and minitpu.sv (where the tile is gated out).
"""


class CmdOpc:
    """W0[3:0] command opcode: the COPY/LAUNCH discriminator (ABI spec `docs/github-markdowns/22_hw_abi.md`, former charter §9).

    TODO(abi-v8): numeric OPC values are not fixed by any ratified tracked
    file (control_plane.tex is unwritten).  These are the working assignments
    (COPY=0, LAUNCH=1) asserted by the cutover task; ratify before tapeout.
    """

    COPY = 0x0
    LAUNCH = 0x1


class CmdDir:
    """W0[7:5] direction/target qualifier for a COPY command.

    TODO(abi-v8): exact bit positions inside W0 (vs the doorbell at bit 4) and
    a separate target sub-field (DEVMEM vs IRAM) are OPEN pending
    control_plane.tex.  Only the direction enum is fixed here.
    """

    H2D = 0  # host -> device
    D2H = 1  # device -> host


# ===========================================================================
# Register semantics
# ===========================================================================


class RegSpec(NamedTuple):
    width: int  # declared bit-width
    units: str  # canonical units string
    access: str = "RW"  # "RW", "RO", or "WO" (write-only CMD window)
    note: str = ""  # RTL cross-reference


REG_SEMANTICS: Dict[str, RegSpec] = {  # noqa: UP006
    "cmd_w0": RegSpec(
        width=32,
        units="cmd_header",
        access="WO",
        note="[3:0]=OPC (CmdOpc), [4]=DOORBELL_BIT, [7:5]=DIR/TGT (CmdDir), [31:8]=rsvd; command_unit.sv latches on doorbell edge",
    ),
    "cmd_w1": RegSpec(
        width=32,
        units="cmd_body",
        access="WO",
        note="COPY host_phys[31:0] / LAUNCH kbin_addr[31:0]",
    ),
    "cmd_w2": RegSpec(
        width=32,
        units="cmd_body",
        access="WO",
        note="COPY host_phys[63:32] / LAUNCH kbin_addr[63:32]",
    ),
    "cmd_w3": RegSpec(
        width=32,
        units="cmd_body",
        access="WO",
        note="COPY dev_addr[31:0] absolute device phys addr / LAUNCH kbin_size[31:0] (bytes)",
    ),
    "cmd_w4": RegSpec(
        width=32,
        units="cmd_body",
        access="WO",
        note="COPY len[31:0] (bytes) / LAUNCH kernarg_ptr[31:0] reserved [serving]",
    ),
    "cmd_w5": RegSpec(
        width=32,
        units="cmd_body",
        access="WO",
        note="reserved / LAUNCH kernarg_ptr[63:32] reserved [serving]",
    ),
    "cmd_w6": RegSpec(
        width=32, units="cmd_body", access="WO", note="reserved / LAUNCH grid_x[31:0] reserved [v3]"
    ),
    "cmd_w7": RegSpec(
        width=32, units="cmd_body", access="WO", note="reserved / LAUNCH grid_y[31:0] reserved [v3]"
    ),
    "instr_ready": RegSpec(
        width=1,
        units="status",
        access="RO",
        note="1 when all sub-FSMs idle; completion + pollable-full backpressure signal.",
    ),
    "dma_slot_done": RegSpec(
        width=32,
        units="bitmask",
        access="RO",
        note="async-DMA slot-done bits [1:0]; bit i=1 when the async-flagged DMA for slot i completes; cleared on flush.slot exit.",
    ),
    "kernel_arg_0": RegSpec(
        width=32,
        units="kernarg",
        access="WO",
        note="RFC P1 arg-mode param slot 0 (WO @0x24). Host writes before LAUNCH; latched into kernel_arg_csr[31:0] and consumed by tile_sequencer.sv param_csr[0]. WRITE-only alias: the 0x24 READ side is the RO dma_slot_done status word (unchanged). minitpu_slave_axi_lite.v slv_reg9.",
    ),
    "kernel_arg_1": RegSpec(
        width=32,
        units="kernarg",
        access="WO",
        note="RFC P1 arg-mode param slot 1 (WO @0x28). Latched into kernel_arg_csr[63:32] (param_csr[1]). WRITE-only alias: the 0x28 READ side is the RO copy_status word (unchanged). minitpu_slave_axi_lite.v slv_reg10.",
    ),
    "kernel_arg_2": RegSpec(
        width=32,
        units="kernarg",
        access="WO",
        note="RFC P1 arg-mode param slot 2 (WO @0x2C). Latched into kernel_arg_csr[95:64] (param_csr[2]). WRITE-only alias: the 0x2C READ side is the RO copy_cdmasr word (unchanged). minitpu_slave_axi_lite.v slv_reg11.",
    ),
    "kernel_arg_3": RegSpec(
        width=32,
        units="kernarg",
        access="WO",
        note="RFC P1 arg-mode param slot 3 (WO @0x30). Latched into kernel_arg_csr[127:96] (param_csr[3]). Fresh WO slot (no RO aliasing). minitpu_slave_axi_lite.v slv_reg12.",
    ),
    "copy_status": RegSpec(
        width=32,
        units="bitmask",
        access="RO",
        note="copy-engine per-COPY diagnostic (silicon bring-up): [0]=st_err, [1]=st_oob, [2]=st_timeout, [3]=st_done, [4]=busy, [5]=st_reset_timeout (CDMA soft-reset never self-cleared), [15:8]=transfer count. Latched on each COPY, held until next start. axi_cdma_ctrl.sv. NB: SA/DA must be 32-byte (256-bit) aligned (no DRE at 256-bit); a misalignment raises DMAIntErr which the per-COPY soft-reset preamble clears.",
    ),
    "launch_status": RegSpec(
        width=32,
        units="bitmask",
        access="RO",
        note="LAUNCH status (silicon bring-up): [0]=launch_error (sticky), set when the last LAUNCH was aborted because the iram_loader flagged load_error (oversize kernel clamped past IRAM_DEPTH, dropping the tail incl. HALT; or a sub-word-size kbin that loaded nothing). The command_unit does NOT start the tile in that case (no start_launch) and retires, so the host reads this instead of hitting the launch timeout. Cleared when the next LAUNCH is accepted (keyed on OPC_LAUNCH specifically, not on every accepted command, so an intervening COPY cannot wipe a still-unread status). command_unit.sv launch_error_r. LIMIT: in a FPGA_COPY_ONLY build there is no tile and no loader, so LAUNCH retires without executing and this word stays 0 - a host cannot tell 'ran fine' from 'this build has no tile'; that is the host's job via build/ABI identity, deliberately not overloaded onto launch_error (see LAUNCH_ERROR_BIT).",
    ),
    "copy_cdmasr": RegSpec(
        width=32,
        units="bitmask",
        access="RO",
        note="copy-engine diagnostic: last CDMASR (PG034) value read during CE_POLL: bit1=Idle, bit12=IOC_Irq, bit4/5/6=DMAIntErr/SlvErr/DecErr, bit14=Err_Irq. axi_cdma_ctrl.sv.",
    ),
    "dma_err": RegSpec(
        width=32,
        units="bitmask",
        access="RO",
        note=(
            "core DMA sticky error (additive above 0x24; no ABI bump). "
            "The cause bits are additive INSIDE this existing RO word: [0] and "
            "[15:8] keep the exact meaning and position they had when first "
            "deployed, so no ABI_VERSION bump and no snapshot change (the ABI "
            "surface hash covers units/access/width, not this note). Layout: "
            "[0]=dma_err (sticky fault); "
            "[1]=cause resp_error (first fault was dm_rsp_resp != OKAY, i.e. a "
            "SLVERR/DECERR actually came back: the request reached AXI and was "
            "rejected or mis-decoded); "
            "[2]=cause watchdog (first fault was the bridge-silence timeout: work "
            "outstanding but no dm_rsp_valid for TIMEOUT_CYCLES, i.e. the request "
            "never reached AXI, or the response path is broken); "
            "[1] and [2] are one-hot when [0] is set; watchdog wins a same-cycle "
            "tie (matches how err_slot already latches); "
            "[3]=rsp_seen (sticky: ANY dm_rsp_valid seen since rst_n; 0 means the "
            "response path never delivered anything to the DMA at all); "
            "[4]=bridge_to (sticky: dm_axi_bridge's OWN watchdog fabricated a "
            "synthetic SLVERR into the DMA); "
            "[5]=ar_seen (sticky: an m00_axi AR handshake ever completed); "
            "[6]=r_seen (sticky: the slave ever returned read data, RVALID); "
            "[7]=aw_seen (sticky: an m00_axi AW handshake ever completed); "
            "[15:8]=err_slot (async-DMA slot of the first fault); "
            "[17:16]=err_resp (captured dm_rsp_resp[1:0]; meaningful only when [1] "
            "is set, forced to 0 when [2] wins); "
            "[18]=b_seen (sticky: the slave ever returned a write response, BVALID); "
            "[31:19]=reserved 0. "
            "All fault bits latch on the FIRST fault and stay sticky until rst_n; "
            "the *_seen / bridge_to taps are sticky from rst_n and are NOT cleared "
            "by a watchdog flush. "
            "IMPORTANT: dm_axi_bridge's watchdog force-completes a stuck head by "
            "driving a SYNTHETIC SLVERR at the DMA port, so a SILENT slave reaches "
            "the DMA as resp_error, not as silence; with this bridge in the path "
            "[2] is effectively unreachable (both TIMEOUT_CYCLES are 1024 and the "
            "bridge always wins). Triage: [1]=1 with [4]=1 means the SLVERR was "
            "MANUFACTURED (slave went silent); [1]=1 with [4]=0 means a genuine "
            "SLVERR/DECERR came back (see [17:16]); [5]=0 means the read address "
            "never left on m00_axi at all; [5]=1 with [6]=0 means the address left "
            "and the slave never returned data. "
            "dma.sv / dm_axi_bridge.sv -> command_unit.sv -> "
            "minitpu_slave_axi_lite.v @0x4C."
        ),
    ),
    "perf_cnt_cycles": RegSpec(
        width=32,
        units="cycles",
        access="RO",
        note="free-running core cycle counter (additive above 0x24; no ABI bump). Reset to 0 on rst_n. The device-side time base: pair with beat_count for achieved DDR bandwidth, or with perf_cnt_instrs for issue density. src/common/perf_counters.sv -> minitpu_core.sv -> minitpu_slave_axi_lite.v @0x40.",
    ),
    "perf_cnt_instrs": RegSpec(
        width=32,
        units="count",
        access="RO",
        note="bundles issued (bundle_issued_o pulses, i.e. state_is_exec_dispatch). Reset to 0 on rst_n. perf_cnt_instrs / perf_cnt_cycles is the achieved issue rate, which is 1.0 only when nothing is spacing the schedule. src/common/perf_counters.sv -> minitpu_core.sv -> minitpu_slave_axi_lite.v @0x44.",
    ),
    "beat_count": RegSpec(
        width=32,
        units="count",
        access="RO",
        note="core DMA bandwidth counter (additive above 0x24; no ABI bump): free-running count of 256-bit DM WORDS successfully moved (loads + stores). Host bandwidth = beat_count * 32 B / (perf_cnt_cycles / f_clk). One increment per DM word that actually leaves the DMA (a load beat written to VMEM, or a store word whose write completion returned OKAY); errored beats are excluded. Reset to 0 on rst_n, like dma_err / perf counters. dma.sv -> command_unit.sv -> minitpu_slave_axi_lite.v @0x50.",
    ),
    "dma_overlap": RegSpec(
        width=32,
        units="count",
        access="RO",
        note="core DMA slot-overlap observable (additive above 0x24; no ABI bump): MEASURED concurrency proof for the async multi-slot DMA. A slot is 'non-IDLE' iff its FSM is S_PENDING or S_ACTIVE (has outstanding work); S_DONE is EXCLUDED so a slot idling in S_DONE awaiting its flush.slot W1C cannot make a serialised kernel read as overlapped. [7:0]=overlap_max: sticky MAX number of slots simultaneously non-IDLE in one cycle (reads 1 for any serialised run; >=2 ONLY if two descriptors were genuinely in flight at once). [31:8]=overlap_cyc: saturating count of cycles with >=2 slots non-IDLE (overlap dwell). Sticky from rst_n, like beat_count; each kernel run resets it via the tile reset. This is what turns 'overlap happened' from a sim claim into a silicon measurement. dma.sv -> command_unit.sv -> minitpu_slave_axi_lite.v @0x54.",
    ),
    "build_id": RegSpec(
        width=32,
        units="identity",
        access="RO",
        note=(
            "device build-identity constant (additive above 0x24; no ABI bump): "
            "baked from regmap_pkg::BUILD_ID (constgen from minitpu.abi.BUILD_ID). "
            "Answers 'what bitstream did I load?' so the host can refuse a mismatch "
            "(#280) and the launch_status-reads-0 hazard (#383) becomes a checkable "
            "identity mismatch rather than a silent no-op. Layout: [31:24]=magic "
            "0xB1 (self-describing tag; a build WITHOUT this CSR reads 0 here and "
            "fails the magic check), [23:0]=Git SHA-1 prefix stamped from a clean "
            "HEAD. The build_identity.json sidecar records the full SHA-1, so it "
            "resolves the shortened hardware value unambiguously. Static (not reset-cleared): a module parameter wired "
            "straight to the read mux, not a live counter. regmap_pkg.sv -> minitpu.sv "
            "-> minitpu_slave_axi_lite.v @0x58 (read index 5'h16)."
        ),
    ),
}

# ===========================================================================
# Command descriptor wire format (replaces the v7 per-mode PROTOCOL table)
# ===========================================================================


class CmdField(NamedTuple):
    """One field within the CMD-window descriptor for a given command kind.

    word     — W index (0..7).
    lo       — bit offset within that 32-bit word.
    width    — field width in bits.
    name     — logical field name.
    status   — "v1" (live), "serving" (reserved for the serving milestone),
               or "v3" (reserved for multi-tile grids).  Reserved fields are
               present in the wire layout but ignored by v1 RTL.
    """

    word: int
    lo: int
    width: int
    name: str
    status: str = "v1"


# W0 header sub-fields (shared by both kinds).
CMD_HEADER: Tuple[CmdField, ...] = (  # noqa: UP006
    CmdField(0, 0, 4, "opc"),
    CmdField(0, 4, 1, "doorbell"),
    # TODO(abi-v8): dir/tgt bit positions provisional (see docstring).
    CmdField(0, 5, 3, "dir_tgt"),
)
ABI_VERSION: int = 9

# BUILD_ID: board-stamped device identity CSR (RO 0x58).
# [31:24]=magic 0xB1, [23:0]=Git SHA-1 prefix. The source value is a placeholder.
BUILD_ID_MAGIC = 0xB1
BUILD_ID: int = BUILD_ID_MAGIC << 24

# COPY body: a host<->device memcpy request (ABI spec `docs/github-markdowns/22_hw_abi.md`, former charter §9).
#
# Alignment: for the v1 256-bit AXI CDMA path, both ``host_phys`` (W1/W2) and
# ``dev_addr`` (W3/W4) must be 32-byte aligned.  The CDMA has no Data Realignment
# Engine at 256-bit (PG034), so a misaligned address raises DMAIntErr (no data
# moved, sticky until a copy-engine soft reset).  The host runtime
# (runtime.memcpy) enforces this before issuing the descriptor.  ``len``
# need not be aligned (the CDMA handles a partial final beat).
COPY_DESCRIPTOR: Tuple[CmdField, ...] = CMD_HEADER + (  # noqa: UP006
    CmdField(1, 0, 32, "host_phys_lo"),
    CmdField(2, 0, 32, "host_phys_hi"),
    CmdField(3, 0, 32, "dev_addr_lo"),
    CmdField(4, 0, 32, "dev_addr_hi"),
    CmdField(5, 0, 32, "len"),
)

# LAUNCH body: run a kernel binary already resident (or to be loaded) in device
# memory.  kernarg_ptr / grid are reserved (not consumed by v1 RTL).
LAUNCH_DESCRIPTOR: Tuple[CmdField, ...] = CMD_HEADER + (  # noqa: UP006
    CmdField(1, 0, 32, "kbin_addr_lo"),
    CmdField(2, 0, 32, "kbin_addr_hi"),
    CmdField(3, 0, 32, "kbin_size"),
    CmdField(4, 0, 32, "kernarg_ptr_lo", "serving"),
    CmdField(5, 0, 32, "kernarg_ptr_hi", "serving"),
    CmdField(6, 0, 32, "grid_x", "v3"),
    CmdField(7, 0, 32, "grid_y", "v3"),
)

# Kind -> descriptor, keyed by CmdOpc value.
CMD_DESCRIPTORS: Dict[int, Tuple[CmdField, ...]] = {  # noqa: UP006
    CmdOpc.COPY: COPY_DESCRIPTOR,
    CmdOpc.LAUNCH: LAUNCH_DESCRIPTOR,
}


def build_cmd_words(
    descriptor: Tuple[CmdField, ...],  # noqa: UP006
    values: Dict[str, int],  # noqa: UP006
) -> List[int]:  # noqa: UP006
    """Pack named field values into the 8-word CMD window per a descriptor.

    ``descriptor`` is one of COPY_DESCRIPTOR / LAUNCH_DESCRIPTOR (or CMD_HEADER);
    ``values`` maps field names (``opc``, ``host_phys_lo``, ...) to integers.
    Returns a list of ``CMD_WINDOW_WORDS`` u32 words (W0..W7), each field placed
    at its declared ``(word, lo, width)`` and width-masked.  This is the single
    place that turns the abi.py descriptor layout into wire words, so the host
    runtime never hardcodes offsets or bit positions.

    The doorbell bit is NOT set here: the caller writes W1..W7 first, then rings
    the doorbell by OR-ing ``DOORBELL_BIT`` into W0 as the final write (so the
    command unit latches a fully-staged window on the doorbell edge).

    Unknown ``values`` keys, or a value that does not fit its field width, raise
    ValueError.  Adding this helper is a pure derivation over the existing
    descriptor tables (no wire-format change), so it does NOT bump ABI_VERSION.
    """
    fields = {f.name: f for f in descriptor}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(
            f"build_cmd_words: unknown field(s) {sorted(unknown)} for this "
            f"descriptor; known fields: {sorted(fields)}"
        )
    words = [0] * CMD_WINDOW_WORDS
    for name, val in values.items():
        f = fields[name]
        if not (0 <= val < (1 << f.width)):
            raise ValueError(
                f"build_cmd_words: value {val} for field '{name}' does not fit {f.width} bits"
            )
        words[f.word] |= (val & ((1 << f.width) - 1)) << f.lo
    return words


DMA_SLOT_DONE_CSR: int = AXI_REG["dma_slot_done"]
"""AXI-Lite address of the read-only async-DMA slot-done status word (0x24).

Test-bench code may read this to observe DMA completion without stalling the
sequencer (the sequencer's flush.slot uses an internal semaphore, not this CSR).
"""


# ===========================================================================
# D4 Patch Table — compiler<->runtime contract
# ===========================================================================
#
# D4's correctness mechanism.  Loop bounds are **literal-only** (no
# hardware/sim multiply at loop entry).  The compiler emits ONE program whose
# grid-dependent literals (today: only the FA causal phase-1 `loop.begin` `hi`)
# are placeholders, plus a patch table appended after the program + literal
# pool.  Before each grid launch the host (`minitpu-runtime`) recomputes those
# literals from the launch's grid and IRAM-patches them in place — so a single
# binary serves every `pid_m` without per-launch recompilation.
#
# This is the entire compiler<->runtime contract for D4; nothing else is
# patched.  A program with **zero** records is launch-invariant.  The compiler
# side (binary producer) emits records in this exact wire form; this module is
# the SSOT for that form and the runtime apply semantics.  The two MUST agree
# byte-for-byte (the binary producer `software.toolchain.asm.assembler`, whose
# `PatchRecord`/`PatchTable` frame these exact constants).
#
# Wire form (little-endian), per record — fixed 16 bytes:
#
#   Field         | Width | Meaning
#   --------------|-------|-----------------------------------------------
#   insn_index    | u32   | index of the 64-bit instruction word to patch
#   operand_slot  | u8    | which 16-bit operand field: 0=op0,1=op1,2=op2
#   expr_kind     | u8    | 0x01 = GRID_LINEAR (only kind)
#   grid_axis     | u8    | 0=program_id_x, 1=_y, 2=_z (FA phase-1 = x)
#   _pad          | u8    | reserved, must be 0
#   scale         | u32   | compile-time constant (FA: BLOCK_N)
#   addend        | i32   | constant offset (FA: 0)
#
# Section framing (after program ‖ literal pool):
#   PATCH_SECTION_MAGIC (4 B) ‖ count:u32 (LE) ‖ count × 16-byte records
#
# Runtime apply semantics, before EACH grid launch, for every record:
#   value = grid[grid_axis] * scale + addend
#   assert 0 <= value < 2**16          # must fit a 16-bit operand field
#   write `value` into the `operand_slot` 16-bit field of program word
#     `insn_index` in IRAM
#
# A bump of ABI_VERSION is REQUIRED if this wire form or these semantics
# change (it is part of the host<->FPGA interface contract).

PATCH_SECTION_MAGIC: bytes = b"MINITPUP"

# expr_kind values (only one kind exists).
EXPR_GRID_LINEAR: int = 0x01  # value = grid[grid_axis] * scale + addend

# grid_axis values.
GRID_AXIS_X: int = 0  # program_id_x  (FA causal phase-1 = pid_m)
GRID_AXIS_Y: int = 1
GRID_AXIS_Z: int = 2

# struct format for one record (matches the table above, little-endian).
_PATCH_RECORD_FMT: str = "<IBBBBIi"  # insn_index slot kind axis _pad scale addend
PATCH_RECORD_SIZE: int = struct.calcsize(_PATCH_RECORD_FMT)
assert PATCH_RECORD_SIZE == 16, PATCH_RECORD_SIZE


class PatchRecord(NamedTuple):
    """One §10 runtime-patch record (16 bytes, little-endian).

    See module section "D4 Patch Table" for the full wire form and the
    runtime apply semantics.  `pack()`/`unpack()` are the canonical
    (de)serializers; `resolve()` evaluates the GRID_LINEAR expression for a
    given grid tuple.
    """

    insn_index: int  # u32
    operand_slot: int  # u8  — 0=op0, 1=op1, 2=op2
    expr_kind: int = EXPR_GRID_LINEAR  # u8
    grid_axis: int = GRID_AXIS_X  # u8
    scale: int = 1  # u32 — FA: BLOCK_N
    addend: int = 0  # i32 — FA: 0

    def pack(self) -> bytes:
        if not (0 <= self.insn_index <= 0xFFFF_FFFF):
            raise ValueError(f"patch insn_index out of u32 range: {self.insn_index}")
        if self.operand_slot not in (0, 1, 2):
            raise ValueError(f"patch operand_slot must be 0/1/2: {self.operand_slot}")
        if self.expr_kind != EXPR_GRID_LINEAR:
            raise ValueError(f"patch expr_kind: only GRID_LINEAR (0x01); got {self.expr_kind:#x}")
        if self.grid_axis not in (GRID_AXIS_X, GRID_AXIS_Y, GRID_AXIS_Z):
            raise ValueError(f"patch grid_axis must be 0/1/2: {self.grid_axis}")
        if not (0 <= self.scale <= 0xFFFF_FFFF):
            raise ValueError(f"patch scale out of u32 range: {self.scale}")
        if not (-(1 << 31) <= self.addend < (1 << 31)):
            raise ValueError(f"patch addend out of i32 range: {self.addend}")
        return struct.pack(
            _PATCH_RECORD_FMT,
            self.insn_index,
            self.operand_slot,
            self.expr_kind,
            self.grid_axis,
            0,  # _pad
            self.scale,
            self.addend,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "PatchRecord":
        insn_index, slot, kind, axis, _pad, scale, addend = struct.unpack(
            _PATCH_RECORD_FMT, buf[:PATCH_RECORD_SIZE]
        )
        return cls(
            insn_index=insn_index,
            operand_slot=slot,
            expr_kind=kind,
            grid_axis=axis,
            scale=scale,
            addend=addend,
        )

    def resolve(self, grid: Tuple[int, int, int]) -> int:  # noqa: UP006
        """value = grid[grid_axis] * scale + addend; assert it fits u16."""
        value = grid[self.grid_axis] * self.scale + self.addend
        if not (0 <= value < (1 << 16)):
            raise ValueError(
                f"D4 patch value out of 16-bit operand range: {value} "
                f"(grid={grid}, axis={self.grid_axis}, scale={self.scale}, "
                f"addend={self.addend})"
            )
        return value


def parse_patch_table(blob: bytes) -> List[PatchRecord]:  # noqa: UP006
    """Parse the patch-table section (`magic ‖ count:u32 ‖ records`).

    `blob` is the patch section (the trailing bytes of the compiler binary,
    after program ‖ literal pool).  Returns [] for a launch-invariant program
    (count == 0).
    """
    if blob[:4] != PATCH_SECTION_MAGIC:
        raise ValueError(f"patch table: bad magic {blob[:4]!r}, expected {PATCH_SECTION_MAGIC!r}")
    (count,) = struct.unpack("<I", blob[4:8])
    recs: List[PatchRecord] = []  # noqa: UP006
    off = 8
    for _ in range(count):
        recs.append(PatchRecord.unpack(blob[off : off + PATCH_RECORD_SIZE]))
        off += PATCH_RECORD_SIZE
    return recs


def apply_patch_table(
    program_words: List[int],  # noqa: UP006
    records: List[PatchRecord],  # noqa: UP006
    grid: Tuple[int, int, int],  # noqa: UP006
) -> List[int]:  # noqa: UP006
    """Apply every §10 record for one grid launch (in place, also returned).

    For each record: compute `value = grid[grid_axis]*scale + addend`
    (range-checked to 16 bits), then overwrite the `operand_slot` 16-bit
    field of `program_words[insn_index]`.  64-bit word layout (see
    § Instruction word in minitpu/isa): ``opcode:8 │ flags:8 │ op0:16 │ op1:16 │ op2:16`` — op0=[47:32],
    op1=[31:16], op2=[15:0].  The host calls this before launching the grid.
    """
    _SLOT_SHIFT = (32, 16, 0)  # op0, op1, op2
    for r in records:
        value = r.resolve(grid)
        shift = _SLOT_SHIFT[r.operand_slot]
        w = program_words[r.insn_index]
        w &= ~(0xFFFF << shift) & 0xFFFF_FFFF_FFFF_FFFF
        w |= (value & 0xFFFF) << shift
        program_words[r.insn_index] = w
    return program_words


# ===========================================================================
# v2 binary parser (MINITPUH header pointer replaces magic-string scan)
# ===========================================================================

PATCH_HEADER_MAGIC: bytes = b"MINITPUH"
"""4-byte magic at byte 0 of the binary blob signals a v2 binary with a header pointer.

v2 binary layout:
  [0:3]   PATCH_HEADER_MAGIC  b"MINITPUH"
  [4:7]   patch_table_offset  u32 LE  (byte offset from blob start to PATCH_SECTION_MAGIC)
  [8:]    program instructions (64-bit words, little-endian)
  [patch_table_offset:]  PATCH_SECTION_MAGIC ‖ count:u32 ‖ records

v1 binary layout (no header; magic-string scan):
  [0:]    program instructions
  [end:]  PATCH_SECTION_MAGIC ‖ count:u32 ‖ records
"""

PATCH_HEADER_SIZE: int = 8  # 4 magic + 4 offset


def parse_binary_v2(blob: bytes) -> Tuple[List[int], List[PatchRecord]]:  # noqa: UP006
    """Parse a binary blob (v1 or v2) into (program_words, patch_records).

    Handles both v2 (MINITPUH header pointer) and v1 (magic-string scan) layouts.
    Only EXPR_GRID_LINEAR records are returned; variable-length dispatch uses
    inline kernargs (LAUNCH kernarg_ptr, reserved [serving]).
    """
    if blob[:4] == PATCH_HEADER_MAGIC:
        (patch_offset,) = struct.unpack("<I", blob[4:8])
        instr_bytes = blob[8:patch_offset]
        patch_section = blob[patch_offset:]
    elif blob[:4] == PATCH_SECTION_MAGIC:
        raise ValueError("bare patch section passed; expected full binary blob")
    else:
        idx = blob.find(PATCH_SECTION_MAGIC)
        if idx < 0:
            instr_bytes = blob
            patch_section = PATCH_SECTION_MAGIC + struct.pack("<I", 0)
        else:
            instr_bytes = blob[:idx]
            patch_section = blob[idx:]

    n_words = len(instr_bytes) // 8
    program_words = list(struct.unpack(f"<{n_words}Q", instr_bytes[: n_words * 8]))
    records = _parse_patch_section_v2(patch_section)
    return program_words, records


def _parse_patch_section_v2(section: bytes) -> List[PatchRecord]:  # noqa: UP006
    """Parse the patch section; returns only EXPR_GRID_LINEAR PatchRecords."""
    if section[:4] != PATCH_SECTION_MAGIC:
        raise ValueError(f"bad patch magic {section[:4]!r}")
    (count,) = struct.unpack("<I", section[4:8])
    records: List[PatchRecord] = []  # noqa: UP006
    off = 8
    for _ in range(count):
        raw = section[off : off + PATCH_RECORD_SIZE]
        insn_index, slot, kind, axis, _pad, scale, addend = struct.unpack(_PATCH_RECORD_FMT, raw)
        if kind == EXPR_GRID_LINEAR:
            records.append(
                PatchRecord(
                    insn_index=insn_index,
                    operand_slot=slot,
                    expr_kind=kind,
                    grid_axis=axis,
                    scale=scale,
                    addend=addend,
                )
            )
        else:
            raise ValueError(
                f"unknown expr_kind {kind:#x} at record offset {off} — "
                "v2 only supports EXPR_GRID_LINEAR; variable loop bounds "
                "use inline kernargs (LAUNCH kernarg_ptr)"
            )
        off += PATCH_RECORD_SIZE
    return records


def apply_patch_table_v2(
    program_words: List[int],  # noqa: UP006
    records: List[PatchRecord],  # noqa: UP006
    grid: Tuple[int, int, int],  # noqa: UP006
) -> List[int]:  # noqa: UP006
    """Apply all patch records for one launch (EXPR_GRID_LINEAR only)."""
    _SLOT_SHIFT = (32, 16, 0)
    for r in records:
        value = r.resolve(grid)
        shift = _SLOT_SHIFT[r.operand_slot]
        w = program_words[r.insn_index]
        w &= ~(0xFFFF << shift) & 0xFFFF_FFFF_FFFF_FFFF
        w |= (value & 0xFFFF) << shift
        program_words[r.insn_index] = w
    return program_words


# ===========================================================================
# RTL elaboration continuity — legacy sub-FSM mode encodings
# ===========================================================================
#
# TODO(abi-v8): the v7 host-dispatch sub-FSM uncore_io_tile/dma_engine.sv still
# references these MODE_* localparams in its case statement.  They are retired
# from the HOST ABI surface above (no CmdOpc/CmdDir maps to them and the runtime
# never writes them), but that sub-FSM is not yet rewritten, so constgen keeps
# emitting the localparams to keep minitpu_modes_pkg.sv-consuming modules elaborating.
# Delete this map once dma_engine is retired or folded into the copy engine.
#
# The MT/tma_engine data path (modes DM_TO_MT/MT_TO_DM/MT_TO_VMEM/VMEM_TO_MT =
# 5/6/7/8) is retired (issue #158): memory_tile.sv / mem_tile_ctrl.sv /
# tma_engine.sv are deleted and their mode encodings are dropped here.  Only the
# modes dma_engine still decodes (IDLE/WRITE_BRAM/READ_BRAM/COMPUTE/WRITE_IRAM)
# remain.
_LEGACY_RTL_MODES: Dict[str, int] = {  # noqa: UP006
    "IDLE": 0,
    "WRITE_BRAM": 1,
    "READ_BRAM": 2,
    "COMPUTE": 3,
    "WRITE_IRAM": 4,
}
