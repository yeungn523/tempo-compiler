# minitpu-compiler

Compiles GPT-2 small inference from Triton kernels into images for a systolic-array NPU.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

___

## Detailed Description

The target NPU is a 16 by 16 BF16 systolic array with a VLIW vector unit, synthesized on a ZCU104. This compiler reads
each kernel of a GPT-2 small layer, written in Triton, through Triton's own frontend down to the tt dialect, converts it
into a minitpu tile op, and lowers that through the MLIR pipeline under `mlir/` into the 128-bit VLIW bundles the board
fetches, along with the manifest entry the runtime launches each image with. `driver/` runs those steps as the stages
of a Triton backend. The board carries no compiler, so the images are built on a host and run there from
`board_package/`, which holds the runtime and the layer drivers.

The NPU's RTL lives in a separate repository. This repository ships its synthesized design as the bitstream
`board_package/fpga/minitpu_zcu104.bit.bin`, which `sudo make minitpu` loads onto the board, so running on the
board needs no RTL checkout. Only the simulation tools read the RTL itself.

## Features

- Compiles every launch of a GPT-2 small layer: the four projections, the bias and residual adds, both LayerNorms,
  and causal flash attention covering every head and token block in one launch.
- Compiles decode attention over a KV cache that fills on the device, with images for 64, 256 and 1024 keys.
- Compiles GEMMs past GPT-2's shapes: an extent a kernel masks is padded with zeros to whole 16-wide tiles, and a
  contraction deeper than one VMEM bank holds runs in segments within one image.
- Derives every latency, capacity and opcode from `mlir/include/minitpu/minitpuisa/isa.td`, the implemented scheduling
  contract.
- Checks each image against the RTL in simulation and against a host reference on the board.
- Apache 2.0 License.

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [Running on the Board](#running-on-the-board)
  - [Simulating Against the RTL](#simulating-against-the-rtl)
  - [Running the Tests](#running-the-tests)
- [Authors](#authors)
- [License](#license)

___

## Dependencies

- [Triton](https://triton-lang.org/) 3.8.0, which compiling any kernel needs. `pyproject.toml` pins it with the rest of
  the Python dependencies, installed by `pip install --group compiler`, and pins the C++ toolchain `mlir/` builds against
  (see `mlir/README.md`).
- A checkout of the separate RTL repository, named by `minitpu_rtl`, for the simulation tools alone. The tools fall
  back to a sibling directory named `minitpu-tmp` when the variable is unset.
- [LLVM](https://llvm.org/) 17 through 20, supplying the `llvm-tblgen` that `tools/toolchain/gen_isa_doc.py` reads the ISA
  source through.
- A ZCU104 running PetaLinux, for everything under `board_package/` that launches.

___

## Installation

The repository runs from a checkout. Every tool extends `sys.path` to reach `driver/` and `board_package/` itself, so
each one runs from the repository root once Triton is installed.

___

## Usage

### Running on the Board

Hold a session open on the board, then run `tools/board/prepare_board.sh <account> <board-id>` on the host. The script
downloads GPT-2 small's weights and tokenizer, compiles every image the layers launch, and deploys them with the board
package. The board erases the home directory whenever no session holds it, so run the script after every login.

On the board, `sudo make minitpu` programs the PL, `make kernels` runs the board checks, and
`make generate PROMPT="The capital of France is" TOKENS=8` prints GPT-2 small's greedy continuation of a prompt. See
[board_package/README.md](board_package/README.md) for what each check proves.

***Warning!*** Program the PL through `sudo make minitpu` only. The target sets the PL clock first, and a PL programmed
without it runs at 99.999 MHz, where every check still passes and every launch is about 1.8 times too slow.

### Simulating Against the RTL

Each tool compiles its image through `driver/` and runs it in the RTL's kernel image testbench, which it finds in the
checkout `minitpu_rtl` names.

| Tool                                          | Checks                                                                        |
|-----------------------------------------------|-------------------------------------------------------------------------------|
| `tools/simulations/sim_gemm.py`               | A GEMM image against a float64 reference, from its dimensions or a Triton kernel |
| `tools/simulations/sim_layernorm.py`          | GPT-2's LayerNorm image against the host reference                            |
| `tools/simulations/sim_resident_attention.py` | The attention image over a sequence, or one decode step with `--decode`       |

Pass `--build-dir` to keep the Verilator build across runs.
`tools/simulations/sim_gemm.py --kernel gpt2_triton.py:gpt2_matmul` compiles a Triton kernel and runs its image in one
step.

### Running the Tests

`tox` builds mlir/, runs every suite below against that build, and repeats the lit suite against an AddressSanitizer
build that also fails on leaks. `tox -e build`, `tox -e test`, and `tox -e memcheck` run one step each, and
`tox -e clean,build` rebuilds from scratch. tox needs `pip install "tox>=4,<5" "tox-uv>=1,<2"`.

`ninja -C mlir/build check-minitpu` runs the lit suite, one pass per file. `pytest -q tests` runs the compiler's
Python suite, which compiles each kernel through `driver/` and checks its image on the functional emulator against a
reference. `python3 -m unittest discover -s board_package/tests -q` runs the board package's own tests, which
`make check` also runs on the board.

___

## Authors

- Natalie Yeung ([yeungn523](https://github.com/yeungn523))
- Duke Gandin

___

## License

This project is licensed under the Apache 2.0 License: see the [LICENSE](LICENSE) file for details.
