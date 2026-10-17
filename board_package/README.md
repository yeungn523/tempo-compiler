# MINITPU on the ZCU104

Everything needed to program the PL, run GPT-2 small's layers on it, and check the
answers against a host reference. The board carries neither Triton nor the
compiler, so every image it launches is compiled on a host and arrives in
`images/`.

```sh
make check           # the package is complete (bitstream, overlay, host tests)
sudo make minitpu    # set the clock, program the PL, create the runtime devices
make kernels         # the board checks
```

A board erases the home directory whenever no session holds it, so after every
login run `tools/prepare_board.sh <netid> <octet>` on the host with a session held
open on the board. It exports GPT-2's weights and tokenizer, builds every image
and deploys them with this package.

**Run `sudo make minitpu` from THIS package, not a copy already on the board.**
The clock is set by this Makefile's `minitpu` target and a board that programs
the PL without it runs the fabric at 99.999 MHz — where every check still
passes and every time is about 1.8x too slow. `tools/deploy_board.sh` ships the
Makefile under its md5 gate for exactly that reason.

`sudo make minitpu` is needed after boot or whenever the PL is replaced. It
loads `minitpu/minitpu_zcu104.bit.bin`, applies `minitpu/overlay.dtso`, and
exposes the MINITPU as a UIO device plus a contiguous DMA buffer, falling back
to the bundled kernel-matched `minitpu/u-dma-buf.ko` when no in-tree module
fits. Everything after that runs as a normal user.

## What the board checks prove

`make kernels` runs eight checks. Each compares the device against a host
reference that rounds where the device rounds, so the residual is the arithmetic
the machine actually does, not a tolerance covering an unmodelled effect. Every
GEMM, add, LayerNorm, attention and cache copy a check launches is a compiled
image from `images/`, each GEMM summing the whole contraction in the device's
BF16 accumulators. A decode step uses the image built for the shortest whole
number of token blocks holding its cache, and masks the keys past the cache's
length with tiles the board writes for that step.

Summing on the device roughly doubles a deep product's error against summing
per-group partials on the host, which is why Full GEMM sits at 0.86 % against
its 1 % bar.

| Check | Shape | Result |
| --- | --- | --- |
| Halt, COPY round trip | — | control-path smoke tests |
| Full GEMM | `[32,768] @ [768,3072]` | 0.86 % |
| Elementwise add | `[32,768] + [32,768]` | exact |
| Decode attention, all heads | 12 heads × 1 query over 48 + 1 cached keys, compiled | not yet run on the board |
| LayerNorm | 32 × 768, resident | 0.49 % |
| KV cache | 64 prefill tokens + 1 decode token, 1536 features | exact |
| **Transformer block** | 64 × 768, 12 causal heads, resident | **1.28 %** |

The GEMM check runs **two** operand sets. The structured one is exactly
representable and pins the kernel bit for bit; the varying one differs along
every axis the kernel indexes. Both are needed: an earlier version of this check
used only structured operands whose token rows were identical and whose weights
were constant along the contraction, and it passed at `rtol=0, atol=0` for weeks
while the kernel was dropping every result that overflowed the MXU's output
FIFO. `assert_operands_discriminating` now rejects an operand set that cannot
observe a token, contraction-block or output-column confusion.

## What runs where

The PL runs all eighteen operations in each transformer layer: LayerNorm,
projections, attention, GELU, biases, and residuals. The block's attention is one
compiled launch covering every head and token block, reading Q, K and V where the
QKV projection drained them. Activations remain in DMEM from the body's entry upload through its
exit download. Projection biases are stored as one row and broadcast by
stride-zero VMEM reads rather than materialised for each token shape.

The PS performs all launch orchestration. The embedding, the final LayerNorm, the
tied language head and the generation loop are not yet in this package.

The compiled attention image fixes its sequence length when it is built, so the
transformer block runs only at the lengths `tools/build_gpt2_images.py` lists,
currently 64 tokens.

## Files

The package is three layers. Each one only knows about the one below it.

| Path | Purpose |
| --- | --- |
| | **The kernels** |
| `gpt2_kernels.py` | **Entry point for the checks.** Host drivers, layer references, the board suite. `make kernels`. |
| `emit.py` | The templates the compiler emits every image's bundles from. |
| `images/` | The compiled images and the manifest describing them, written by `tools/build_gpt2_images.py`. |
| `bf16.py` | Device geometry, BF16 packing, operand layouts, host references. Touches no device, so it can be checked by inspection. |
| `layout.py` | Where a tensor's elements live and what that costs: the DMEM tiling ABI and the VMEM allocator. |
| | **The assembler** |
| `asm.py` | Fills delay fields, and **rejects** what no delay can fix. |
| `isa_model.py` | The per-opcode scheduling facts `asm.py` schedules against. Generated — run `tools/gen_isa_doc.py --write`, never edit by hand. |
| | **The board** |
| `minitpu/` | Bitstream, device-tree overlay, PL bring-up scripts. |
| `minitpu_runtime/runtime/` | Board runtime: UIO, udmabuf, COPY, LAUNCH. |
| `minitpu_runtime/minitpu/` | Hardware ABI, register map, launch limits. |
| | **Examples and tests** |
| `run_compiled_kernel.py` | Launches the image the compiler produced. `kernel_entry_point.py` emits it, and it is copied here as `triton_kernel.bin`. |
| `tests/` | Host-only tests: assembler encodings, scheduling distances, the operand rule, and the compiler held to the hand-written images it replaced, frozen in `tests/golden/`. `make check` runs them. |

`run_compiled_kernel.py` launches a pre-built `triton_kernel.bin` against the
explicit allocate/COPY/LAUNCH primitives and reduces the drain on the host.

## Reading `asm.py` first

The machine issues one bundle per cycle and holds the fetch head for exactly one
reason: the delay field. There is no scoreboard, no replay, no stall on a full
buffer, no arbiter that makes a loser wait. **A schedule that breaks a rule is
not reported — it produces a number.** So the assembler is the only place a rule
can be enforced on every image, and `schedule()` is where to look:

- it models writeback-port occupancy, VREG read-port A ownership, matrix
  controller occupancy, per-push MXU result latency, and VMEM bank conflicts;
- it **rejects** two port-A consumers in one bundle, and a `vmatpush` burst that
  would overflow the MXU output FIFO. Neither is fixable by spacing — one has no
  gap to insert, the other would need reordering — so raising is the only honest
  answer;
- `analyze_stalls` attributes every waiting cycle to the constraint that caused
  it, which is how the open hardware items are ranked.

`compiler/minitpuisa.td` in the repo is the single source of truth for these
numbers; `tb_isa_conformance.sv` holds the RTL to it and
`tools/gen_isa_doc.py --check` holds this package's constants to it.

## Writing a kernel

The `@minitpu.asm` decorator runs a Python function as an assembler, emitting one
128-bit VLIW bundle per `minitpu.bundle` call. Every program must end in `F:halt`.

```python
@minitpu.asm
def my_program():
    minitpu.bundle(f=minitpu.halt())
```

Call an assembled program with host arrays and it hides image staging, pinned
host and device allocation, the COPY commands, argument binding, LAUNCH, the
output COPY and cleanup:

```python
bf16_mul_add(h_src1, h_src2, h_dest)
```

Arguments are C-contiguous `numpy.uint16` arrays — the BF16 wire format. Every
argument before the last is an input; the last is the output. On a board whose
UIO node is named differently, use `@minitpu(uio_name="my-minitpu")`.
