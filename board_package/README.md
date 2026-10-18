# NPU Board Package

Everything needed to program the PL, run GPT-2 small's layers on it, and check the
answers against a host reference. The board carries neither Triton nor the
compiler, so every image it launches is compiled on a host and arrives in
`images/`.

```sh
make check           # the package is complete (bitstream, overlay, host tests)
sudo make minitpu    # set the clock, program the PL, create the runtime devices
make kernels         # the board checks
make generate        # continue a prompt with GPT-2 small
```

A board erases the home directory whenever no session holds it, so after every
login run `tools/prepare_board.sh <account> <board-id>` on the host with a session held
open on the board. It exports GPT-2's weights and tokenizer, builds every image
and deploys them with this package.

**Run `sudo make minitpu` from THIS package, not a copy already on the board.**
The clock is set by this Makefile's `minitpu` target and a board that programs
the PL without it runs the fabric at 99.999 MHz — where every check still
passes and every time is about 1.8x too slow. `tools/deploy_board.sh` ships the
Makefile under its md5 gate for exactly that reason.

`sudo make minitpu` is needed after boot or whenever the PL is replaced. It
loads `fpga/minitpu_zcu104.bit.bin`, applies `fpga/overlay.dtso`, and
exposes the NPU as a UIO device plus a contiguous DMA buffer, falling back
to the bundled kernel-matched `fpga/u-dma-buf.ko` when no in-tree module
fits. Everything after that runs as a normal user.

## What the board checks prove

`make kernels` runs ten checks. Each compares the device against a host
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
| GPT-2 prefill | "The capital of France is", 12 layers of real weights, each against its reference | not yet run on the board |
| GPT-2 decode | One greedy token decoded through 12 layers, each layer and both tokens against the reference | tokens match, worst layer 1.67 % |

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

The PS performs all launch orchestration, the embedding and the tied language head,
while the final LayerNorm runs on the NPU through the compiled LayerNorm image.
`prefill` runs a prompt through every layer and fills one KV cache per layer,
`decode_step` runs each new token through every layer over those caches, and
`generate` loops the two greedily. `make generate PROMPT="..." TOKENS=8` prints a
prompt's continuation.

The compiled attention image fixes its sequence length when it is built, so the
transformer block runs only at the lengths `tools/build_gpt2_images.py` lists,
currently 64 tokens.

## Files

| Path | Purpose |
| --- | --- |
| `gpt2_inference.py` | GPT-2 small's inference on the NPU, which includes the device tensors, the launches of every compiled image, the forward pass, and generation. `make generate` runs it. |
| `board_checks.py` | **Entry point for the checks.** The board suite `make kernels` runs. |
| `references.py` | The host references every check holds the device to. |
| `tokenizer.py` | GPT-2's byte-level BPE, reading the exported vocabulary and merges. |
| `bf16.py` | Device geometry, BF16 packing, operand layouts. Touches no device, so it can be checked by inspection. |
| `runtime/` | Board runtime: UIO, udmabuf, COPY, LAUNCH, and the hardware ABI they speak. |
| `images/` | The compiled images and the manifest describing them, written by `tools/build_gpt2_images.py`. |
| `weights/` | GPT-2's weights and tokenizer, written by `tools/export_gpt2_weights.py`. Not tracked. |
| `fpga/` | Bitstream, device-tree overlay, u-dma-buf module, PL bring-up scripts. |
| `tests/` | Host tests of the runtime, the drain, the operand rule and the tokenizer. `make check` runs them. |

The code generator that writes `images/`, with the assembler and the ISA model it schedules against, lives in
`codegen/` at the repository root and never reaches the board.
