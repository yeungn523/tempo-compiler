# minitpu-compiler

Compiles GPT-2 small inference from Triton kernels into images for a systolic-array NPU.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

___

## Detailed Description

The target NPU is a 16 by 16 BF16 systolic array with a VLIW vector unit, synthesized on a ZCU104. This compiler reads
each kernel of a GPT-2 small layer, written in Triton, through Triton's own frontend down to the tt dialect, plans it
into a launch, and emits the 128-bit VLIW bundles the board fetches. The board carries no compiler, so the images are
built on a host and run there from `board_package/`, which holds the assembler, the runtime, and the layer drivers.

The NPU's RTL lives in a separate repository. This repository ships its synthesized design as the bitstream
`board_package/minitpu/minitpu_zcu104.bit.bin`, which `sudo make minitpu` loads onto the board, so running on the
board needs no RTL checkout. Only the simulation tools read the RTL itself.

## Features

- Compiles every launch of a GPT-2 small layer: the four projections, the bias and residual adds, both LayerNorms,
  and causal flash attention covering every head and token block in one launch.
- Compiles decode attention over a KV cache that fills on the device, one image per 32-key bucket up to 1024 keys.
- Derives every placement constant from `compiler/minitpuisa.td`, the implemented scheduling contract.
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

- [Triton](https://triton-lang.org/) 3.8.0, which compiling any kernel needs.
- A checkout of the separate RTL repository, named by `minitpu_rtl`, for the simulation tools alone. The tools fall
  back to a sibling directory named `minitpu-tmp` when the variable is unset.
- [LLVM](https://llvm.org/) 17 through 20, supplying the `llvm-tblgen` that `tools/gen_isa_doc.py` reads the ISA
  source through.
- A ZCU104 running PetaLinux, for everything under `board_package/` that launches.

___

## Installation

The repository runs from a checkout. Every tool extends `sys.path` to reach `compiler/` and `board_package/` itself, so
each one runs from the repository root once Triton is installed.

___

## Usage

### Running on the Board

Hold a session open on the board, then run `tools/prepare_board.sh <account> <board-id>` on the host. The script
downloads GPT-2 small's weights and tokenizer, compiles every image the layers launch, and deploys them with the board
package. The board erases the home directory whenever no session holds it, so run the script after every login.

On the board, `sudo make minitpu` programs the PL, `make kernels` runs the board checks, and
`make generate PROMPT="The capital of France is" TOKENS=8` prints GPT-2 small's greedy continuation of a prompt. See
[board_package/README.md](board_package/README.md) for what each check proves.

***Warning!*** Program the PL through `sudo make minitpu` only. The target sets the PL clock first, and a PL programmed
without it runs at 99.999 MHz, where every check still passes and every launch is about 1.8 times too slow.

### Simulating Against the RTL

| Tool                              | Checks                                                                       |
|-----------------------------------|------------------------------------------------------------------------------|
| `tools/sim_gemm.py`               | An emitted GEMM image against a float64 reference                            |
| `tools/sim_layernorm.py`          | The compiled LayerNorm against the hand-written image it replaced            |
| `tools/sim_resident_attention.py` | The compiled attention over a sequence, or one decode step with `--decode`   |

Pass `--build-dir` to keep the Verilator build across runs. `kernel_entry_point.py` compiles any Triton kernel in
`sample_triton.py` into an image and its packed operands for `tools/sim_gemm.py` to run.

### Running the Tests

`python3 -m unittest discover -s board_package/tests -t board_package/tests -q` runs the host tests, which cover the
assembler, the emitter, and the compiler from a Triton kernel through to the image its plan emits.

___

## Authors

- Natalie Yeung ([yeungn523](https://github.com/yeungn523))
- Duke Gandin

___

## License

This project is licensed under the Apache 2.0 License: see the [LICENSE](LICENSE) file for details.
