# What was ported, and from where

Every file here was copied byte-identical from `minitpu-tmp` at commit
`2872ede` ("Added a generated machine model the assembler schedules against"),
branch `mlir`. Nothing was edited in transit; `AGENTS.md` and this file are the
only additions. The hierarchy is the source repository's, so a path here is the
same path there.

## The stack, bottom up

| Tier | Files | Role |
|---|---|---|
| 0 — ISA contract | `docs/isa_latency.json`, `tools/gen_isa_doc.py`, `board_package/isa_model.py`, `docs/ISA_AND_INTERFACES.md`, `tb/isa_latency_pkg.sv` | SSOT for slots, opcodes, writeback offsets and matrix occupancy, plus the three artifacts generated from it |
| 1 — assembler | `board_package/asm.py` | `AsmBuilder`/`AsmKernel`, 128-bit bundle encoding, `pack()` / `place()` / `schedule()` / `analyze_stalls()`, the `_decode` hazard model |
| 2 — geometry | `board_package/bf16.py` | tile/lane/VMEM constants, `pack_activations` / `pack_weights`, BF16 host references |
| 3 — emitters | `board_package/kernels.py` | `build_looped_gemm` and 8 more hand-written builders |
| 4 — frontend | `compiler/{__init__,backend,frontend,compile}.py` | Triton → tt dialect → structural read → `Plan` |
| 5 — runtime | `board_package/minitpu_runtime/` | `minitpu.abi` + `runtime.*`; the device driver, and a **compile-time** dependency via the `@minitpu.asm` decorator |
| 6 — drivers | `tools/sim_triton_kernel.py`, `board_package/run_triton.py`, `board_package/run_minitpu.py` | host compile+deploy, and the board-side launch |
| 7 — oracles/tests | `board_package/gpt2_kernels.py`, `board_package/tests/`, `tools/sim_kernel.py` | host references, the `DeviceTensor` layer, attention composition, and the 16-check board suite |
| — allocator seed | `board_package/layout.py`, `board_package/dsl.py` | `VmemAllocator`, `Tiled`, `conflict_free_pitch`, `Kernel`/`Ref`/`Access`. **Nothing on the Triton path imports these.** They are here because a real code generator needs a VMEM allocator and this is the closest thing to one |

The Python import closure is 18 local modules. Verified in this tree: every file
parses, every import resolves to stdlib / numpy / triton / a file present here,
and `python3 tools/gen_isa_doc.py --check` passes.

## What is deliberately absent

`src/`, `fpga/` and all of `tb/` except `isa_latency_pkg.sv`. This is a compiler
tree, not an RTL tree.

Two consequences worth stating outright:

- **`docs/isa_latency.json` is now an unverifiable mirror.** In the source
  repository `tb/tb_isa_conformance.sv` asserts the published writeback offsets
  equal `sequencer_pkg`'s `WB_W_*` localparams. That testbench is not here, so
  `gen_isa_doc.py --check` proves the three artifacts agree *with each other* and
  says nothing about whether they agree with the hardware. If the ISA moves,
  it moves in `minitpu-tmp` first.
- **No simulation rung.** `tools/sim_kernel.py` was ported for
  `build_operands` / `recover` / `report`, which are pure host code. Its
  `run_product` / `run_blocked_product` paths shell out to
  `tb/run_kernel_image.sh` and will not run here. That removes the middle rung
  of the check ladder: numpy → *simulation* → board. Plan on numpy then board.

Also absent: `minichat.py`, `miniserve_bridge.py`, `run_gpt2_forward.py`, and
the benchmark/demo tools. `board_package/Makefile` was ported unmodified and
still names them, so `chat`, `batch`, `boundary`, `throughput`, `memory`,
`layout` and `trace` will fail. Working targets: `help`, `minitpu`, `provision`,
`run`, `check`, `identity`, `kernels`.

## The state of the compiler, honestly

The Triton path is a frontend and a planner, not a code generator:

```
@triton.jit  →  compile_to_ttir  →  read_program → tuple[MatMul, ...]
             →  plan             →  Plan{Launch, Drain, Repack, Padding}
             →  kernels.build_looped_gemm(**five scalars)   ← the emitter
             →  asm: pack/place/schedule → 128-bit bundles → .bin
```

`Launch` carries `column_tiles`, `groups`, `activate`, `fuse_groups` and
`loop_blocks`. `plan()` *configures* one hand-written GEMM; it does not lower
IR. There is no IR→bundle code generation, no register allocation and no VMEM
allocation anywhere on this path.

**GEMM and GEMM+GELU compile end to end. Nothing else does.** `read_program`
returns matmuls only, and `_EPILOGUES` (`compiler/compile.py:35`) has exactly
two entries: empty, and gelu.

**Flash attention does not exist anywhere in this tree, in either form.** The
compiler cannot lower it — `_Graph` has no resolver for `tt.reduce`, `math.exp`
or a loop-carried running max/rescale, so such a kernel raises `ProgramError` at
the first unrecognized operation. And `gpt2_kernels.attention` is *materialized*
attention, not flash: fused QKV → transposed keys → a scores `DeviceTensor` →
explicit `_attention_mask` → `build_softmax_packed` → AV gather, run as twelve
grouped problems in one launch. There is no online softmax and no running
rescale to reuse, and no oracle to test a flash kernel against.

So `gpt2_kernels.py` is not part of the compiler. It is the working,
board-validated implementation the compiler is aimed at, plus the references
(`softmax_reference`, `layernorm_reference`, `attention_reference`,
`transformer_block_reference`) that would tell you when a compiled kernel
matched it.

## Where the work starts

- **Attention in the frontend**: extend `_Graph`'s resolver table in
  `compiler/compile.py` past matmul. `Guard`, `Padding`, `_ORDERING_PREDICATES`
  and `_PREDICATE_RESHAPES` already read bounds predicates, which is a start on
  causal masks but not on the softmax reduction.
- **A real backend**: something that consumes `Plan` and emits `AsmBuilder`
  bundles for a shape it was not written for, instead of selecting among
  `kernels.py`'s builders. `layout.py` is the allocator to build it on.
- **The scheduler's limit**: `asm.py`'s `_decode` models VREG hazards only.
  SREG RAW/WAW and VMEM ownership are compiler responsibilities that the
  builders hand-space, and `schedule()` only fills delays — it never reorders.
  Any pass that moves bundles must treat S and M slots as immovable until
  `_decode` is extended.
