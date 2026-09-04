# MINITPU on the ZCU104

Everything needed to program the PL, run GPT-2 arithmetic on it, and check the
answers against a host reference. Copy this directory to a board and run it
there; nothing here is pre-built, every kernel is assembled from Python each
time.

```sh
make check           # the package is complete (bitstream, overlay, host tests)
sudo make minitpu    # set the clock, program the PL, create the runtime devices
make reproduce       # every headline claim in the root README, in order (~6 min)
```

`make reproduce` runs `identity`, `kernels`, `batch`, `boundary`, `throughput`
and `chat`. Each is also a target on its own — `make help` lists them with what
they cost. **They are the claims:** every number in the root README's table comes
from one of these commands, so if a number there is wrong, one of these prints a
different one.

The weight-dependent targets (`batch`, `boundary`, `throughput`, `chat`) need
`gpt2_weights.npz`, which is 475 MB and not in the repository. Build it on a host
with torch and copy it here:

```sh
python3 tools/export_gpt2_weights.py --out build/gpt2_weights.npz
scp build/gpt2_weights.npz <board>:board_package/
```

`make identity` and `make kernels` need no checkpoint, so a fresh board can prove
it runs the right bitstream and passes 16/16 before the copy finishes.

`make chat` additionally needs the GPT-2 tokenizer, which is not in the board's
Python:

```sh
TMPDIR=$HOME/piptmp python3 -m pip install --user --no-cache-dir tokenizers
```

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

## The headline

**GPT-2 124M answers a prompt with every transformer-body operation on the PL.**

```
$ make chat ARGS='--prompt "The capital of France is" --temperature 0 --max-new-tokens 2'
body: MINITPU PL (16x16 BF16 systolic array, ZCU104)
layers=12 heads=12 model=768 head_dim=64 vocab=50257
 the capital
  [2 tokens: prefill 1.1 s, 1.10 tok/s decode]
```

Measured on `.187` at 187.5 MHz. Every projection in all twelve layers, both
LayerNorms, GELU, attention, residual, and bias operation runs on the device.
Embedding, final LayerNorm, and the tied language head run on the PS.

A warm transformer-body pass has two activation transfers, at entry and exit.
That does **not** mean the interval is pure MiniTPU cycles: the host issues every
launch, while cold and new-shape passes also stage images, constants, masks, and
scratch buffers. These costs are measured separately in `SESSION_CHECKLIST.md`.

## What the board checks prove

`make kernels` runs sixteen checks. Each compares the device against a host
reference that rounds where the device rounds, so the residual is the arithmetic
the machine actually does, not a tolerance covering an unmodelled effect.

| Check | Shape | Result |
| --- | --- | --- |
| Halt, command interleave, COPY round trip | — | control-path smoke tests |
| BF16 mul-add | 32×32 | exact |
| Full GEMM | `[32,768] @ [768,3072]` | 0.47 % |
| Feed-forward layer | `[32,768] → [32,3072] → [32,768]` | 0.72 % |
| Row-wise softmax | 16 × 64 | 0.45 % |
| Single attention head | `softmax(QKᵀ/√d)V`, 64 × 64 | 0.89 % |
| Elementwise add | `[32,768] + [32,768]` | exact |
| Weight residency | packed per call vs resident | 2.8× faster, identical |
| Group sum on device | host fp32 vs device BF16 | 1.08× faster, 1.81× error |
| Prefill attention, all heads | 12 heads × 20 causal queries | 0.54 % |
| Decode attention, all heads | 12 heads × 1 query, 48 keys | 0.67 % |
| LayerNorm | 8 × 768 | 0.42 % |
| **Transformer block** | 64 × 768, 12 heads | **0.68 %** |

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
projections, attention reshaping and transpose, softmax, GELU, biases, and
residuals. Activations remain in DMEM from the body's entry upload through its
exit download. Projection biases are stored as one row and broadcast by
stride-zero VMEM reads rather than materialised for each token shape.

The PS performs embedding, final LayerNorm, the tied language head, sampling, and
all launch orchestration. First-use setup transfers still occur; only a warm pass
is down to the two activation transfers.

Two costs are worth knowing because they are temporary, not inherent:

- Attention pads its 64-deep contraction to 192, because the looped GEMM
  accumulates a whole number of `_LOOP_BLOCKS × _BLOCK_DEPTH` before draining.
  Exact — the extra terms are zero — but it moves three times the bytes it needs.
- The attention row limit is 320 padded rows, set by its current VMEM region map.

## Files

The package is three layers. Each one only knows about the one below it.

| Path | Purpose |
| --- | --- |
| | **The kernels** |
| `gpt2_kernels.py` | **Entry point for the checks.** Host drivers, layer references, the board suite. `make kernels`. |
| `kernels.py` | Every assembled program. The machine has no hazard interlocks, so each schedule is a correctness argument and the comments carry it. |
| `bf16.py` | Device geometry, BF16 packing, operand layouts, host references. Touches no device, so it can be checked by inspection. |
| `layout.py` | Where a tensor's elements live and what that costs: the DMEM tiling ABI and the VMEM allocator. |
| `dsl.py` | The kernel surface. The programmer writes layouts and tiling; every descriptor and `vld`/`vst` field is derived. |
| | **The assembler** |
| `asm.py` | Fills delay fields, and **rejects** what no delay can fix. |
| `isa_model.py` | The per-opcode scheduling facts `asm.py` schedules against. Generated — run `tools/gen_isa_doc.py --write`, never edit by hand. |
| | **The board** |
| `minitpu/` | Bitstream, device-tree overlay, PL bring-up scripts. |
| `minitpu_runtime/runtime/` | Board runtime: UIO, udmabuf, COPY, LAUNCH. |
| `minitpu_runtime/minitpu/` | Hardware ABI, register map, launch limits. |
| | **Examples and tests** |
| `run_example_kernel.py` | A hand-written example program, meant to be opened and changed. `make run`. |
| `run_compiled_kernel.py` | Launches the image the compiler produced. `tools/sim_triton_kernel.py` compiles it and copies it here. |
| `tests/` | Host-only tests: assembler encodings, scheduling distances, the operand rule. `make check` runs them. |

The two example runners are not variants of each other: `run_example_kernel.py`
assembles its own source and calls it through the high-level callable, while
`run_compiled_kernel.py` launches a pre-built `triton_kernel.bin` against the
explicit allocate/COPY/LAUNCH primitives and reduces the drain on the host.

**The chat stack is not in this tree.** `minichat.py` and `run_gpt2_forward.py`
are described below and named by the `chat` target, but neither file ships here;
the same is true of most of the `reproduce` targets. `make help`, `minitpu`,
`provision`, `run`, `check`, `identity` and `kernels` are the ones that work.

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

`docs/isa_latency.json` in the repo is the single source of truth for these
numbers; `tb_isa_conformance.sv` holds the RTL to it and
`tools/gen_isa_doc.py --check` holds this package's constants to it.

## Writing a kernel

Edit `run_example_kernel.py`. The `@minitpu.asm` decorator runs the Python function as
an assembler, emitting one 128-bit VLIW bundle per `minitpu.bundle` call. Every
program must end in `F:halt`.

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

`run_example_kernel.py` keeps its own copy of the example program on purpose: it
is the file you open to see what one looks like. Nothing compares the two copies
directly; each is held to a numpy reference on the board instead, `check_mul_add`
in `gpt2_kernels.py` for `kernels.py`'s and `main` for its own. Keep it that way
when you edit either: the copies once disagreed because a DMA row pitch stayed at
64 bytes after BF16-native halved a row to 32, so every descriptor read every
other row, and a check against the other copy would not have caught it.

## The chat demo

```sh
make chat                                                   # the REPL
make chat ARGS='--prompt "The capital of France is" --temperature 0'
```

GPT-2 124M answering a prompt with **every projection in all twelve layers on the
PL**: fused QKV, the attention output projection, and both feed-forward products.
Greedy decoding reproduces HuggingFace's continuation token for token.

`run_gpt2_forward.py` supplies the body object used by `minichat.py`. The wrapper
keeps embedding, the final LayerNorm, tied language head, and sampling in numpy;
the tokenizer is the real GPT-2 BPE. Weights come from
`tools/export_gpt2_weights.py`, run on a host that has torch, and land here as
`gpt2_weights.npz`.

`tools/check_body_math.py` runs the same body with numpy in place of the device
and compares against HuggingFace: that is what pins the weight orientation, the
causal mask, the KV cache and the language head, none of which a single-kernel
board check can see.

`minichat.py` needs no torch on the board.
