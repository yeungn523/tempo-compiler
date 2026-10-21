# tempo-compiler

Compiles GPT-2 small inference from Triton kernels into images for a systolic-array NPU whose RTL has no hardware
interlocks.

Summer' 26 @ Cornell CSL

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

___

## Detailed Description

The target NPU is a 16 by 16 BF16 systolic array with a VLIW vector unit, synthesized on a ZCU104. The compiler takes a
Triton kernel through Triton's own frontend to the tt dialect and converts it into a tempo tile op. An MLIR pipeline
lowers that op into the 128-bit VLIW bundles the board fetches, along with the manifest entry the board runtime
launches each image with. The images run on the board from `board_package/`, which holds the runtime, the GPT-2 layer
drivers, and the NPU's bitstream.

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Compiling a Kernel](#compiling-a-kernel)
  - [Running on the Board](#running-on-the-board)
  - [Simulating Against the RTL](#simulating-against-the-rtl)
- [Developers](#developers)
- [Authors](#authors)
- [License](#license)

___

## Dependencies

Building the compiler requires Python 3.11, git, [CMake](https://cmake.org/) 3.20 or above,
[Ninja](https://ninja-build.org/), and [Clang](https://clang.llvm.org/) with the lld linker on the system path. The
build fetches the rest of the C++ toolchain itself: Triton's v3.8.0 source, for the tt dialect, and the prebuilt LLVM
that release pins (commit `5f07f818`). `pyproject.toml` pins both, along with every Python dependency.

Running on the board requires a ZCU104 running PetaLinux. The simulation tools additionally require
[Verilator](https://www.veripool.org/verilator/) and a checkout of the RTL repository, named by the `minitpu_rtl`
environment variable or found as a sibling directory named `minitpu-tmp`.

___

## Installation

1. Download this repository to the local machine.
2. `cd` to the root directory of the repository.
3. Run `pip install --group compiler --group mlir --group test "tox>=4,<5" "tox-uv>=1,<2"` to install the Python
   dependencies and tox.
4. Run `tox -e build`. It fetches and verifies Triton's source and the pinned LLVM, then builds `tempo-opt`,
   `tempo-translate` and `tempo-tblgen` into `mlir/build/bin`.

___

## Usage

### Compiling a Kernel

A kernel is an ordinary `@triton.jit` function whose shapes are `tl.constexpr` parameters and whose pointers are
annotated with their element types. `driver.compile_kernel` compiles it from the repository root, and its keyword arguments
override the kernel's constexpr defaults:

```python
import pathlib

import triton
import triton.language as tl

from driver import compile_kernel


@triton.jit
def matmul(a: "*bf16", b: "*bf16", c: "*fp32", N: tl.constexpr = 768, K: tl.constexpr = 768):
    ...


compiled = compile_kernel(matmul, N=3072)             # compiles the 768 by 3072 product
compiled.write(pathlib.Path("board_package/images"))  # writes gemm_768x3072.bin
print(compiled.entry)  # {'kind': 'gemm', 'depth': 768, 'columns': 3072, 'activate': False, 'drain_words': 98560}
```

The compiler supports four types of kernels: a GEMM, optionally passing its input through the exact GELU and adding a
bias, attention, a LayerNorm, and an elementwise add. Each image covers one block of 32 tokens, which the host launches
once per block. Any other kernel raises `driver.ProgramError` with the reason it was refused.

`tools/board/build_images.py` compiles every image GPT-2 small launches from `gpt2_triton.py` and writes them, with
their manifest, into `board_package/images`.

### Running on the Board

On the host, with a session open on the board:

```bash
tools/board/prepare_board.sh <account> <board-id>  # exports GPT-2's weights, builds the images, deploys the package
```

On the board:

```bash
sudo make minitpu                                            # programs the PL and sets its clock
make kernels                                                 # runs the board checks
make generate PROMPT="The capital of France is" TOKENS=8     # prints GPT-2 small's greedy continuation
```

The board erases its home directory whenever no session holds it, so rerun `prepare_board.sh` after every login. See
[board_package/README.md](board_package/README.md) for what each check proves.

***Warning!*** Program the PL only through `sudo make minitpu`.

### Simulating Against the RTL

| Tool                                          | Checks                                                                           |
|-----------------------------------------------|----------------------------------------------------------------------------------|
| `tools/simulations/sim_gemm.py`               | A GEMM image against a float64 reference, from its dimensions or a Triton kernel |
| `tools/simulations/sim_layernorm.py`          | GPT-2's LayerNorm image against the host reference                               |
| `tools/simulations/sim_resident_attention.py` | The attention image over a sequence, or one decode step with `--decode`          |

`tools/simulations/sim_gemm.py --kernel my_kernels.py:matmul` compiles a Triton kernel and runs its image in one step.
Pass `--build-dir` to keep the Verilator build across runs.

___

## Development automation

Tox runs `build`, `test` and `memcheck` in order.

| Environment | Description                                                                                          |
|-------------|------------------------------------------------------------------------------------------------------|
| `build`     | Fetches and verifies the pinned toolchain, then configures and builds `mlir/` into `mlir/build`      |
| `test`      | Runs the lit suite, then the compiler's pytest suite and the board package's tests                   |
| `memcheck`  | Builds `mlir/` with AddressSanitizer into `mlir/build-asan` and runs the lit suite, failing on leaks |
| `clean`     | Removes both build directories                                                                       |

`tox -e clean,build` rebuilds from scratch. `tools/emulators/emulate_board_checks.py` runs the board's own checks on the
functional emulator.

C++ under `mlir/` is formatted by `mlir/.clang-format` and linted by `mlir/.clang-tidy`. The generated ISA tables in
`docs/ISA_AND_INTERFACES.md` and `tools/emulators/isa_model.py` are regenerated from `isa.td` with
`python3 tools/toolchain/gen_isa_doc.py --write`.

### Repository Layout

| Path                           | Holds                                                                                   |
|--------------------------------|-----------------------------------------------------------------------------------------|
| `driver/`                      | The Triton backend whose stages run the MLIR tools, and `compile_kernel`                |
| `gpt2_triton.py`               | The Triton kernels a GPT-2 small layer launches                                         |
| `mlir/include/tempo/tempo/`    | The tempo dialect: tile ops, machine operations over virtual registers, and its passes  |
| `mlir/include/tempo/tempoisa/` | The tempoisa dialect, one op per instruction, and the ISA contract `isa.td`             |
| `mlir/lib/tempo/`              | The Triton conversion, tile op expansion, allocators, ISA lowering, and manifest export |
| `mlir/lib/tempoisa/`           | The packer, the scheduler, and the image import and export                              |
| `mlir/tools/`                  | `tempo-opt`, `tempo-translate`, and `tempo-tblgen`                                      |
| `mlir/test/`                   | The lit suite, one pass per file, with every GPT-2 kernel's tt module under `Inputs/`   |
| `tests/`                       | The compiler's pytest suite                                                             |
| `tools/`                       | The image builder, board deployment, emulator, RTL simulations, and toolchain setup     |
| `board_package/`               | The board runtime, the GPT-2 drivers, the board checks, and the compiled images         |

___

## Authors

- Natalie Yeung ([yeungn523](https://github.com/yeungn523))
- Duke Gandin ([dgandin1](https://github.com/dgandin1))

___

## License

This project is licensed under the Apache 2.0 License: see the [LICENSE](LICENSE) file for details.
