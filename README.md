# ECE 4950 -- minitpu-compiler

Compiler stack for GPT-2 small
___

## Detailed Description

This repository lowers Triton kernels onto the MINITPU, a 16 by 16 BF16 systolic array synthesized on a ZCU104. The
frontend runs Triton's own pipeline down to the tt dialect, the planner reads that printed module back into a launch
description, and the emitter turns the description into the 128 bit VLIW bundles the board fetches. Every generated
artifact derives from `compiler/minitpuisa.td`, which states the implemented scheduling contract, and `board_package/`
carries the assembler, the kernels, and the runtime that launches them on hardware. Nothing registers the backend with
Triton, because `compiler/frontend.py` constructs it directly and drives `make_ir` itself.

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
  - [LLVM TableGen](#llvm-tablegen)
  - [MLIR](#mlir)
  - [The RTL Checkout](#the-rtl-checkout)
- [Usage](#usage)
  - [Entry Points](#entry-points)
  - [Compiling a Kernel](#compiling-a-kernel)
  - [Simulating Against the RTL](#simulating-against-the-rtl)
  - [Regenerating the ISA Artifacts](#regenerating-the-isa-artifacts)
  - [Deploying to the Board](#deploying-to-the-board)
  - [Running the Tests](#running-the-tests)
- [API Documentation](#api-documentation)
- [Authors](#authors)
- [License](#license)

___

## Dependencies

This repository runs from a checkout rather than from an installed distribution, so each dependency below is installed
by hand. Only the first is needed to compile a kernel.

- [Triton](https://triton-lang.org/) 3.8.0 
- [LLVM](https://llvm.org/) 17 through 20, supplying the `llvm-tblgen` that `tools/gen_isa_doc.py` reads
  `compiler/minitpuisa.td` through.
- A checkout carrying the RTL and its testbench, which `tools/sim_gemm.py` drives.
- A ZCU104 running PetaLinux and carrying the MINITPU bitstream, for everything under `board_package/` that launches.

***Note,*** the scheduling records read through `llvm-tblgen` alone, so an MLIR build is not part of this dependency
set. `mlir-tblgen` becomes a requirement only when the dialect's operation definitions join those records in the same
file, because the definitions need TableGen's C++ backends and the records do not. Reaching the tt dialect needs no
MLIR build either, as Triton's wheel ships the MLIR libraries its frontend calls into.

___

## Installation

### LLVM TableGen

```sh
sudo apt install llvm-18
```

On Ubuntu 24.04 this places `llvm-tblgen-18` on the path. `tools/gen_isa_doc.py` searches for `llvm-tblgen`,
`llvm-tblgen-18`, `llvm-tblgen-17`, `llvm-tblgen-19`, and `llvm-tblgen-20` in that order, so the versioned binary is
found without further configuration. Set `LLVM_TBLGEN` to a binary's full path to override the search.

### MLIR

A full LLVM and MLIR toolchain is needed only to extend `compiler/minitpuisa.td` past its scheduling records with
dialect operation definitions. conda-forge ships one that carries both TableGen drivers.

```sh
mamba create -n minitpu-mlir -c conda-forge mlir llvmdev
export PATH="$HOME/miniconda3/envs/minitpu-mlir/bin:$PATH"
```

That environment provides `mlir-tblgen` and `llvm-tblgen` at LLVM 22.1.8, alongside the MLIR headers and libraries a
C++ TableGen backend links against. Keeping it in its own environment leaves the everyday `llvm-tblgen` search pointed
at the system package.

### The RTL Checkout

```sh
export MINITPU_RTL=/path/to/the/rtl/checkout
```

The RTL is not vendored here. `MINITPU_RTL` names a checkout carrying `tb/run_kernel_image.sh`, which is the testbench
`tools/sim_gemm.py` drives, and it falls back to a sibling directory named `minitpu-tmp` when unset. That checkout
carries the simulation environment its testbench builds in.

___

## Usage

### Entry Points

Every tool extends `sys.path` to reach `compiler/`, `board_package/`, and `board_package/minitpu_runtime/` on its own,
so each one runs from the repository root with no `PYTHONPATH` export.

| Entry point                            | Runs on | Purpose                                                          |
|----------------------------------------|---------|------------------------------------------------------------------|
| `tools/sim_triton_kernel.py`           | Host    | Compiles the Triton matmul and copies it to a board              |
| `tools/sim_gemm.py`                    | Host    | Runs an emitted GEMM image against the RTL and reports its error |
| `tools/gen_isa_doc.py`                 | Host    | Regenerates or checks every artifact derived from the ISA source |
| `tools/deploy_board.sh`                | Host    | Copies the board package over an md5 gate and programs the PL    |
| `board_package/Makefile`               | Board   | The documented board interface, listed by `make help`            |
| `board_package/run_compiled_kernel.py` | Board   | Launches the compiled image and checks its product               |
| `board_package/run_example_kernel.py`  | Board   | Assembles and runs the editable hand-written example program     |
| `board_package/gpt2_kernels.py`        | Board   | The sixteen board checks, also reachable as `make kernels`       |

### Compiling a Kernel

```sh
python3 tools/sim_triton_kernel.py
```

This compiles the `matmul_kernel` defined at the top of that file, writes `triton_kernel.bin` and
`triton_operands.npz` into `build/compiled_kernel/`, and copies both to the board over `scp`. It prints the bundle
count, the shape the module states, and the launch the planner settled on, then names the command that runs the image.
The copy step needs a key the board has authorized, and the image stays in `build/compiled_kernel/` when that copy
fails.

### Simulating Against the RTL

```sh
python3 tools/sim_gemm.py --rows 32 --depth 192 --columns 48
```

This emits the image one launch describes, runs it through the RTL testbench, and compares the drained product against
a float64 reference. `--fuse-groups` accumulates the groups on the device, `--loop-blocks` and `--block-n` change the
shape one block covers, and `--build-dir` keeps the simulation build across runs, which is what makes a sweep cheap.

### Regenerating the ISA Artifacts

```sh
python3 tools/gen_isa_doc.py --check
python3 tools/gen_isa_doc.py --write
```

`compiler/minitpuisa.td` is the single source of truth for the scheduling contract. `--write` regenerates the operation
table in `docs/ISA_AND_INTERFACES.md`, the localparams in `tb/isa_latency_pkg.sv`, and the per-opcode facts in
`board_package/isa_model.py`. `--check` exits non-zero when any of those, or any scheduling constant in
`board_package/asm.py`, disagrees with the source. Edit the TableGen source and regenerate rather than editing an
artifact, and run `--write` after a fresh clone, because `docs/` is untracked.

### Deploying to the Board

```sh
tools/deploy_board.sh 187
tools/deploy_board.sh 187 --program
```

The argument is the last octet of the board's address. Every file is copied and then verified by md5, because a
`tar | ssh` deploy once reported success while writing zero byte files and every board check passed against the stale
copies already there. `--program` additionally programs the PL and reads its password from `board.cred`.

Once the package is on the board, run its own targets there:

```sh
make check           # verify the package without touching hardware
sudo make minitpu    # set the PL clock, program the PL, provision the runtime devices
make kernels         # the sixteen board checks
make reproduce       # every headline claim, in order
```

***Warning!*** Run `sudo make minitpu` from the deployed package rather than an older copy on the board. The Makefile
sets the PL clock, and a board that programs the PL without that step runs the fabric at 99.999 MHz, where every check
still passes and every time is about 1.8 times too slow. See [board_package/README.md](board_package/README.md) for
what each target proves and what it costs.

### Running the Tests

```sh
python3 -m unittest discover -s board_package/tests -t board_package/tests -q
```

This runs 56 host tests covering the assembler encodings, the scheduling distances, the operand rule, the address
space, the drain, and the compiler path from a decorated Triton kernel through to the image its plan emits. The
compiler tests import Triton, so they run on the host rather than on the board.

___

## API Documentation

`docs/ISA_AND_INTERFACES.md` carries the scheduling contract for the synthesized profile, covering the per-operation
VREG requirements, the matrix command latencies, and the capacities a schedule is able to exceed. It is generated, so
produce it with `python3 tools/gen_isa_doc.py --write`. [board_package/README.md](board_package/README.md) documents
the board side, covering what runs on the PL, what each check proves, and how to write a kernel.

___

## Authors

- Natalie Yeung ([yeungn523](https://github.com/yeungn523))

___

## License

This project is licensed under the Apache 2.0 License: see the [LICENSE](LICENSE) file for details.
