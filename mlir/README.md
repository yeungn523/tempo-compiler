# mlir/

The compiler's MLIR pipeline: Triton's tt dialect in, MiniTPU images out, as one C++ tool (`minitpu-opt`) built
against the LLVM Triton 3.8.0 pins. It is the only compiler: `driver/` at the repository root runs it as the stages of
a Triton backend.

## Building

`pyproject.toml` pins everything the build needs. Its dependency groups carry the Python side, and its
`[tool.minitpu.toolchain]` table carries the C++ side: Triton's source at `v3.8.0`, for the tt dialect, and the LLVM build
that release pins (commit `5f07f818`), which Triton publishes prebuilt and no Python package carries.

```sh
pip install --group compiler --group mlir      # Triton, numpy, lit, clang-format, clang-tidy
python3 tools/toolchain/setup_toolchain.py               # fetches and verifies the Triton source and LLVM, prints the cmake line
<the cmake line it prints>
ninja -C mlir/build check-minitpu
```

`tools/toolchain/setup_toolchain.py` checksums the LLVM package, checks the revision it records, and checks that the pinned commit
is the one Triton's own `cmake/llvm-info.json` names. Configuring refuses an LLVM built from any other commit, so the
pin changes in `pyproject.toml` alone.

`driver/` runs this pipeline as the stages of a Triton backend: `driver.compile_kernel` takes a Triton kernel, or a
kernel's MLIR text, through minitpu-opt and minitpu-translate and returns its image with the manifest entry
`--export-minitpu-manifest` writes from its tile op, and the constants block `--export-minitpu-constants` writes for a
causal attention. `tools/board/build_images.py [--output <dir>]` writes the board's whole image set and its manifest
that way: the four GEMMs, each adding its bias, the two sums and the LayerNorm from their Triton sources in
`gpt2_triton.py`, the four attention images converted from one-head kernels into images walking all twelve heads, and
the two KV-cache copies and the halt, which have no Triton source and are stated as tile ops.
`tools/emulators/emulate_board_checks.py --images <dir>` runs the board's checks against that set. Both import Triton,
so they run under the environment the `compiler` group installs. `tools/board/prepare_board.sh <account> <board>`
builds and deploys it.

C++ is formatted by `.clang-format` and linted by `.clang-tidy`, both in this directory.

## Layout

| Path                          | Holds                                                                               |
|-------------------------------|-------------------------------------------------------------------------------------|
| `third_party/triton/`         | Triton's tt dialect, compiled from `TRITON_SOURCE_DIR` without the rest of Triton    |
| `include/minitpu/minitpu/`    | The minitpu dialect: machine operations over virtual registers (`!minitpu.vreg`), VMEM buffers and `scf.for` loops, and its passes |
| `lib/minitpu/`                | The dialect's verifiers, the conversion of a Triton kernel into the one tile op computing it, attention given its head count (`tt_to_minitpu.cpp`), the expansion of the `minitpu.matmul`, `minitpu.add`, `minitpu.layernorm`, `minitpu.copy` and `minitpu.attention` tile ops (`tile_to_vector.cpp`, over the layout in `tile_layout.h`), the VMEM and register allocators (`allocate.cpp`), and the lowering into minitpuisa (`to_isa.cpp`) |
| `include/minitpu/minitpuisa/` | The minitpuisa dialect: one op per instruction, its slots and opcodes read from the ISA contract `isa.td`, the single source every latency, capacity and opcode derives from, and the bundle bit layout (`encoding.h`) |
| `lib/minitpuisa/`             | The dialect's verifiers, the image import and export (`image.cpp`), and the `-minitpuisa-pack` and `-minitpuisa-schedule` passes over the traffic model they share (`traffic.h`) |
| `tools/minitpu-opt/`          | The `mlir-opt`-style driver every lit test runs                                     |
| `tools/minitpu-tblgen/`       | The TableGen driver whose `-gen-minitpu-sched` writes the scheduling facts from `isa.td` |
| `tools/minitpu-translate/`    | `--import-minitpu-image` reads a `.bin` into minitpuisa, `--export-minitpu-image` writes it back |
| `test/`                       | lit/FileCheck tests, one pass per file, including a byte-for-byte round trip of every board image |
| `test/Inputs/`                | Every GPT-2 small kernel's tt module, written by `tools/fixtures/export_gpt2_ttir.py`          |
| `test/Inputs/schedule/`       | The streams the retired Python assembler packed and scheduled for every GPT-2 image, which the packer and scheduler stay byte-identical to |
