# mlir/

The compiler's MLIR pipeline: Triton's tt dialect in, MiniTPU images out, as one C++ tool (`minitpu-opt`) built
against the LLVM Triton 3.8.0 pins. The Python compiler under `codegen/` stays the byte-for-byte oracle for each stage
until that stage replaces it.

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

C++ is formatted by `.clang-format` and linted by `.clang-tidy`, both in this directory.

## Layout

| Path                          | Holds                                                                               |
|-------------------------------|-------------------------------------------------------------------------------------|
| `third_party/triton/`         | Triton's tt dialect, compiled from `TRITON_SOURCE_DIR` without the rest of Triton    |
| `include/minitpu/minitpu/`    | The minitpu dialect: machine operations over virtual registers (`!minitpu.vreg`), VMEM buffers and `scf.for` loops, and its passes |
| `lib/minitpu/`                | The dialect's verifiers, the VMEM and register allocators (`allocate.cpp`), and the lowering into minitpuisa (`to_isa.cpp`) |
| `include/minitpu/minitpuisa/` | The minitpuisa dialect: one op per instruction, its slots and opcodes read from `codegen/minitpuisa.td`, and the bundle bit layout (`encoding.h`) |
| `lib/minitpuisa/`             | The dialect's verifiers, the image import and export (`image.cpp`), and the `-minitpuisa-pack` and `-minitpuisa-schedule` passes over the traffic model they share (`traffic.h`) |
| `tools/minitpu-opt/`          | The `mlir-opt`-style driver every lit test runs                                     |
| `tools/minitpu-tblgen/`       | The TableGen driver whose `-gen-minitpu-sched` writes the scheduling facts from `codegen/minitpuisa.td` |
| `tools/minitpu-translate/`    | `--import-minitpu-image` reads a `.bin` into minitpuisa, `--export-minitpu-image` writes it back |
| `test/`                       | lit/FileCheck tests, one pass per file, including a byte-for-byte round trip of every board image |
| `test/Inputs/`                | Every GPT-2 small kernel's tt module, written by `tools/fixtures/export_gpt2_ttir.py`          |
| `test/Inputs/schedule/`       | The streams `asm.py` scheduled for every GPT-2 image, written by `tools/fixtures/record_schedule_streams.py` |
