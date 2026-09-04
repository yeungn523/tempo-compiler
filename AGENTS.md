# minitpu-compiler — guidance for coding agents

`CLAUDE.md` is a symlink to this file.

The rules below are the ones that will cost you a board run if you ignore
them.

## The one thing that matters most

This machine has no hazard interlocks and no bounds checks: a wrong schedule and
a wrong address both return plausible numbers rather than faulting. **Check at
the cheapest rung that can catch the bug class you are risking.** In the source
repository that ladder is numpy → simulation → board. **Here the middle rung is
gone** — no `src/`, no `tb/` runners — so it is numpy at the exact row indices
(seconds), then a board. That makes the numpy rung load-bearing: a check that
exercises one shape at one offset can print EXACT while the addressing is wrong
everywhere else, and there is no simulator to catch it before the hardware does.

## Fast path

```sh
python3 tools/gen_isa_doc.py --check     # ISA artifacts agree with the SSOT
cd board_package && make check           # package integrity + host unit tests
python3 tools/sim_triton_kernel.py       # compile a GEMM, scp it to the board
```

`make check` runs `minitpu/scripts/check_package.py` (which assembles
`single_mul_add`, so it catches a truncated source) and the four unit test
modules in `board_package/tests/`. Neither needs hardware.

## What compiles today

GEMM and GEMM+GELU, through `tools/sim_triton_kernel.py`. Nothing else.
`read_program` returns matmuls only; `_EPILOGUES` has two entries. Do not
describe this as a working compiler for anything wider without running it.

`gpt2_kernels.py`'s attention is **materialized, not flash**, and there is no
flash-attention reference anywhere here to check one against.

## Hazards the compiler owns

- **`_decode` models VREG hazards only.** A `saddi` decodes to no reads and no
  writes, and MXU occupancy is not in it either — SREG RAW, WAW and VMEM
  ownership are *compiler* responsibilities, and `kernels.py`'s builders
  hand-space them (`_advance` leaves two bundles for an uninterlocked scalar
  writeback). `schedule()` only fills delays; it never reorders. **Any pass that
  moves bundles must treat S and M slots as immovable until `_decode` is
  extended.** Ignoring this cost two board runs returning NaN while every
  offline check passed.
- **Flush what you issued.** `flush_slot`'s mask is per DMA slot and it blocks
  until `dma_slot_done` covers every bit in it. A kernel that issues one
  descriptor on slot 0 and flushes mask 3 waits forever on a slot that never had
  a descriptor — which wedges the device.
- Treat VREG port ownership, loop-buffer replay and DMA alignment as
  architectural hazards. Add a test that exercises the exact scheduling or
  alignment boundary before changing them.

## Generated artifacts and single sources of truth

- `docs/isa_latency.json` generates `docs/ISA_AND_INTERFACES.md`,
  `tb/isa_latency_pkg.sv` and `board_package/isa_model.py`. After editing it run
  `python3 tools/gen_isa_doc.py --write`; `--check` fails on a stale artifact.
- **That check is weaker here than upstream.** `tb/tb_isa_conformance.sv`, which
  asserts the published offsets equal `sequencer_pkg`'s `WB_W_*`, is not in this
  tree. `--check` proves the three artifacts agree with each other and nothing
  about the hardware. **Change the ISA in `minitpu-tmp` first**, where the RTL
  can contradict you, then port the JSON.
- `board_package/minitpu/platform.json` is the DMEM aperture map that ships with
  the bitstream; `runtime/address_space.py` mirrors it only as a fallback. The
  upstream `tools/check_platform_consistency.py` needs `fpga/` and is not here.

## Board flow

- Deploy with `tools/deploy_board.sh <octet> [--program]`. It has an md5 gate per
  file. Use it rather than ad-hoc `scp`: a `tar | ssh` deploy once reported
  success while writing zero-byte files, and every board check passed against
  the stale copies already there.
- On the board: `sudo make -C board_package minitpu` programs the PL, then
  `make -C board_package run` or `kernels`. **Run `sudo make minitpu` from the
  deployed package, not an older copy** — that target sets `PL_CLK_MHZ=187`, and
  a board programmed without it runs the fabric at 99.999 MHz where every check
  still passes and every time is about 1.8x too slow.
- Check the stamped build identity (`make -C board_package identity`) before
  relying on results from an already-programmed board.
- A host process killed mid-launch **wedges the device** until the PL is
  reprogrammed — no timeout, no error. If a board goes quiet, reprogram first.
- Never print `board.cred` or copy it into artifacts or logs.
- The Makefile was ported unmodified and names tools that are not here. Working
  targets: `help`, `minitpu`, `provision`, `run`, `check`, `identity`,
  `kernels`. The rest will fail.

## Handoff

Report the exact commands run and whether they were host or board results.
Distinguish a claim that was **measured** from one that was **derived** — an
estimate in this project's history predicted 1.75x and measured 1.00x.
