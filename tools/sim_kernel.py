#!/usr/bin/env python3
"""Runs a Python-assembled MINITPU kernel in simulation and checks it against a host reference.

This is the driver half of ``tb/tb_kernel_image.sv``. It assembles an image with the same
assembler the board runtime uses, packs operands with the same packers, replays the whole thing
against the real ``minitpu.sv`` under Verilator, and compares the drained result to numpy.

Before this existed the kernel flow ran one way only: workloads were hand-written in SystemVerilog
and the assembled image was dumped out for the board, so nothing the Python assembler emitted had
ever executed in any simulation.

Examples:
    tools/sim_kernel.py                              # smallest shape, varying operands
    tools/sim_kernel.py --operands structured        # the operands the old board check used
    tools/sim_kernel.py --rounds 2                   # two launches, no reset between them
    tools/sim_kernel.py --column-tiles 4 --groups 2  # a larger product
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
from numpy.typing import NDArray

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "board_package"))
sys.path.insert(0, str(REPO / "board_package" / "minitpu_runtime"))

import asm  # noqa: E402
import gpt2_kernels as k  # noqa: E402
from gpt2_kernels import (  # noqa: E402
    _BLOCK_ACTIVATION_BYTES,
    _BLOCK_DEPTH,
    _BLOCK_TOKENS,
    _BLOCK_WEIGHT_BYTES,
    _ELEMENT_BYTES,
    _LOOP_BLOCKS,
    _TILE,
    _ZERO_TILE_WORDS,
    from_bf16_words,
    pack_activations,
    pack_weights,
)

_ERROR_LIMIT: float = 0.01
"""Relative error the BF16 partial sums explain, matching check_full_gemm's bound."""


def build_operands(
    tokens: int, depth: int, columns: int, *, kind: str, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Builds one operand pair.

    Notes:
        The two kinds are not interchangeable. ``structured`` reproduces what the board check used:
        exactly representable, but every token row is identical and every weight is constant along
        the contraction, so a kernel that confuses one token or one contraction block for another
        still matches. ``varying`` differs along every axis, which is what makes those confusions
        observable, at the cost of exactness.

    Args:
        tokens: Rows of the left operand.
        depth: The contraction.
        columns: Output features.
        kind: Either ``structured`` or ``varying``.
        seed: Seed for the varying operands, so two rounds can differ.

    Returns:
        The left and right operands, both on the BF16 grid.
    """
    if kind == "structured":
        left = (np.arange(tokens * depth, dtype=np.float32).reshape(tokens, depth) % 2)
        scales = (2.0 ** -(np.arange(columns) % 4)) / 16.0
        right = np.broadcast_to(scales, shape=(depth, columns)).astype(np.float32)
        return left, right
    rng = np.random.default_rng(seed)
    left = k._on_bf16_grid(values=rng.standard_normal((tokens, depth)).astype(np.float32))
    right = k._on_bf16_grid(values=rng.standard_normal((depth, columns)).astype(np.float32))
    # Fails loudly rather than reporting a vacuous pass, which is how the defect this harness
    # exists to catch survived every gate it had.
    k.assert_operands_discriminating(activations=left, weights=right)
    return left, right


def write_operands(
    directory: pathlib.Path,
    tag: str,
    left: NDArray[np.float32],
    right: NDArray[np.float32],
    drain_bytes: int,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Writes the three argument blobs one launch binds.

    Notes:
        Each operand carries one block of trailing zero slack, because the kernel's last prefetch
        reads one block past the final block. The drain blob leads with the zeroed clearing tile the
        accumulators load from, then poison, so a tile the kernel fails to write is visibly unwritten
        rather than silently zero.

    Args:
        directory: Where to write the blobs.
        tag: Suffix distinguishing one round's blobs from another's.
        left: The activation operand.
        right: The weight operand.
        drain_bytes: Size of the drain window, clearing tile included.

    Returns:
        The activation, weight and drain paths.
    """
    activation_path = directory / f"arg0_{tag}.bin"
    weight_path = directory / f"arg1_{tag}.bin"
    drain_path = directory / f"arg2_{tag}.bin"

    activations = pack_activations(activations=left).tobytes()
    activation_path.write_bytes(activations + bytes(_BLOCK_ACTIVATION_BYTES))
    weights = pack_weights(weights=right).tobytes()
    weight_path.write_bytes(weights + bytes(_BLOCK_WEIGHT_BYTES))

    zero_tile_bytes = _ZERO_TILE_WORDS * _ELEMENT_BYTES
    poison = np.full((drain_bytes - zero_tile_bytes) // _ELEMENT_BYTES, 0x7F80, dtype="<u2")
    drain_path.write_bytes(bytes(zero_tile_bytes) + poison.tobytes())
    return activation_path, weight_path, drain_path


def recover(drain: bytes, *, column_tiles: int, groups: int, columns: int) -> NDArray[np.float32]:
    """Recovers the product from a drained window, the way full_gemm does on the host.

    Args:
        drain: The dumped drain window, clearing tile included.
        column_tiles: Output column tiles the product covers.
        groups: Block groups each column tile drained.
        columns: Output features.

    Returns:
        The float32 product for one token block.
    """
    drain_words = _ZERO_TILE_WORDS + column_tiles * groups * _BLOCK_TOKENS * _TILE
    words = np.frombuffer(drain, dtype="<u2", count=drain_words)
    partials = from_bf16_words(words=words[_ZERO_TILE_WORDS:drain_words])
    summed = partials.reshape(column_tiles, groups, _BLOCK_TOKENS, _TILE).sum(axis=1, dtype=np.float32)
    return summed.transpose(1, 0, 2).reshape(_BLOCK_TOKENS, columns)


def report(tag: str, actual: NDArray[np.float32], expected: NDArray[np.float64]) -> bool:
    """Prints one round's verdict and returns whether it passed."""
    unwritten = int(np.isinf(actual).sum())
    error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    verdict = "PASS" if error < _ERROR_LIMIT and not unwritten else "FAIL"
    print(f"  {tag}: {verdict}  relative error {error:.4%}"
          + (f"  UNWRITTEN {unwritten} of {actual.size} elements still poisoned" if unwritten else ""))
    if verdict == "FAIL":
        print(f"      device[0,:4] {actual[0, :4]}")
        print(f"      host  [0,:4] {expected[0, :4]}")
        print(f"      device[7,:4] {actual[7, :4]}")
        print(f"      host  [7,:4] {expected[7, :4]}")
    return verdict == "PASS"


def run_product(
    left: NDArray[np.float32],
    right: NDArray[np.float32],
    *,
    activate: bool,
    rounds: int = 1,
    reset_between: bool = False,
    second: tuple[NDArray[np.float32], NDArray[np.float32]] | None = None,
    label: str = "",
    fuse_groups: bool = False,
    loop_blocks: int = _LOOP_BLOCKS,
    column_tiles: int | None = None,
    groups: int | None = None,
) -> list[NDArray[np.float32]]:
    """Runs one product through the simulator and returns each round's recovered result.

    Notes:
        Derives the kernel's trip counts from the operand shapes the same way full_gemm does, so the
        image simulated here is the image a board would launch, bundle for bundle.

    Args:
        left: The activation operand, whose rows are tokens and columns the contraction.
        right: The weight operand.
        activate: Whether each staged tile is activated before it enters the array.
        rounds: Launches of the same image, without a reset between them unless asked.
        reset_between: Whether to reset the device between rounds.
        second: Operands restaged before round 1, when it should see different data.
        label: Prefix for the progress line.
        fuse_groups: Whether every group accumulates on the device rather than draining a partial each.
        loop_blocks: Accumulation blocks one contraction group carries.
        column_tiles: Column tiles a caller expects the operands to cover, checked against the shapes rather than
            replacing them. A caller that planned the trip counts elsewhere passes them here so a disagreement fails
            instead of building an image for a shape the operands do not hold.
        groups: Contraction groups a caller expects, checked the same way.

    Returns:
        One recovered product per round.

    Raises:
        RuntimeError: If a round failed to reach HALT, or the harness could not run.
        ValueError: If a caller's expected trip counts disagree with the ones the operand shapes give.
    """
    tokens, depth = left.shape
    columns = right.shape[1]
    derived_groups = depth // (loop_blocks * _BLOCK_DEPTH)
    derived_column_tiles = columns // _TILE
    for name, expected, derived in (("column_tiles", column_tiles, derived_column_tiles),
                                    ("groups", groups, derived_groups)):
        if expected is not None and expected != derived:
            message = (
                f"Unable to run a product whose {name} the caller expects to be {expected}. The operands are "
                f"[{tokens},{depth}] by [{depth},{columns}] at {loop_blocks} blocks a group, which gives {derived}."
            )
            raise ValueError(message)
    groups = derived_groups
    column_tiles = derived_column_tiles
    drained = 1 if fuse_groups else groups
    drain_words = _ZERO_TILE_WORDS + column_tiles * drained * _BLOCK_TOKENS * _TILE
    drain_bytes = drain_words * _ELEMENT_BYTES

    kernel = k.build_looped_gemm(column_tiles=column_tiles, groups=groups, activate=activate,
                                 fuse_groups=fuse_groups, loop_blocks=loop_blocks)
    image = kernel.image.to_bytes(pad_to=4)
    print(f"{label}[{tokens},{depth}] @ [{depth},{columns}]  activate={activate}  "
          f"{len(image) // 16} bundles  rounds={rounds}")

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="sim_kernel."))
    image_path = scratch / "kernel.bin"
    image_path.write_bytes(image)

    # One block group is ~5.2k device cycles at the smallest shape; leave an order of
    # magnitude of headroom so a bound never masks a real hang as a plausible number.
    max_cycles = 200_000 + column_tiles * groups * 60_000
    plusargs = [f"+KERNEL_BIN={image_path}", f"+DUMP_BYTES={drain_bytes}",
                f"+ROUNDS={rounds}", f"+MAX_CYCLES={max_cycles}"]
    if reset_between:
        plusargs.append("+RESET_BETWEEN=1")

    dumps = []
    for index in range(rounds):
        operands = (left, right) if index == 0 or second is None else second
        suffix = "" if index == 0 else "B"
        activation_path, weight_path, drain_path = write_operands(
            scratch, f"r{index}", operands[0], operands[1], drain_bytes)
        plusargs += [f"+ARG0{suffix}={activation_path}", f"+ARG1{suffix}={weight_path}",
                     f"+ARG2{suffix}={drain_path}"]
        dump_path = scratch / f"dump_r{index}.bin"
        plusargs.append(f"+{'DUMP' if index == 0 else 'DUMPB'}={dump_path}")
        dumps.append(dump_path)

    depth = asm.mxu_output_fifo()[0]
    environment = {
        **os.environ,
        # One build directory per depth: the DUT differs, so sharing one would
        # silently run the previous depth's binary.
        "KERNEL_IMAGE_BUILD_DIR": str(REPO / ".build" / f"kernel_image_fifo{depth}"),
        "TB_TIMEOUT": os.environ.get("TB_TIMEOUT", "7200"),
        "MINITPU_FIFO_DEPTH": str(depth),
    }
    completed = subprocess.run([str(REPO / "tb" / "run_kernel_image.sh"), *plusargs],
                               cwd=REPO, env=environment, capture_output=True, text=True)
    transcript = completed.stdout + completed.stderr
    for line in transcript.splitlines():
        if line.startswith(("KERNEL_IMAGE_ROUND", "ERROR")):
            print("  " + line)
    if "=== TB_KERNEL_IMAGE PASS ===" not in transcript:
        print(transcript[-4000:])
        raise RuntimeError("tb_kernel_image did not reach HALT on every round")

    return [recover(path.read_bytes(), column_tiles=column_tiles, groups=drained, columns=columns)
            for path in dumps]


def run_blocked_product(
    left: NDArray[np.float32], right: NDArray[np.float32], *, activate: bool, label: str = ""
) -> NDArray[np.float32]:
    """Runs a product whose token count spans several launches, stacking the blocks.

    Notes:
        One launch covers `_BLOCK_TOKENS` token rows, which is what `full_gemm` loops over on a
        board. Simulation needs the same loop rather than the same image run twice, because each
        block stages different activations.

    Args:
        left: The activation operand.
        right: The weight operand.
        activate: Whether each staged tile is activated.
        label: Prefix for the progress line.

    Returns:
        The float32 product over every token block.
    """
    blocks = [
        run_product(left[first : first + _BLOCK_TOKENS], right, activate=activate,
                    label=f"{label}block {first // _BLOCK_TOKENS} ")[0]
        for first in range(0, left.shape[0], _BLOCK_TOKENS)
    ]
    return np.concatenate(blocks, axis=0)


def activated(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Applies the SFU's own GELU table, which is what the activated kernel applies."""
    return from_bf16_words(words=k.gelu_lut_reference(words=k.as_bf16_words(values=values)))


def run_ffn(tokens: int, model: int, hidden: int, seed: int) -> bool:
    """Runs one GPT-2 feed-forward layer in simulation, composed as full_ffn composes it.

    Notes:
        Two launches with a host summation between them, exactly as ``full_ffn`` runs on a board: the
        up projection, then the down projection with the activation riding its left operand. The
        reference quantises both operands to BF16 and applies the SFU's own table, so only the
        accumulate differs.

    Args:
        tokens: Rows the layer processes.
        model: The layer's input and output width.
        hidden: The intermediate width the up projection expands to.
        seed: Seed for the operands.

    Returns:
        Whether the layer matched the host reference.
    """
    rng = np.random.default_rng(seed)
    activations = k._on_bf16_grid(values=rng.standard_normal((tokens, model)).astype(np.float32))
    up_weight = k._on_bf16_grid(values=(rng.standard_normal((model, hidden)) * 0.14).astype(np.float32))
    down_weight = k._on_bf16_grid(values=(rng.standard_normal((hidden, model)) * 0.09).astype(np.float32))

    up = run_product(activations, up_weight, activate=False, label="up   ")[0]
    down = run_product(up, down_weight, activate=True, label="down ")[0]

    expected = (activated(activations.astype(np.float64) @ up_weight.astype(np.float64))
                .astype(np.float64) @ down_weight.astype(np.float64))
    return report(f"GPT-2 FFN [{tokens},{model}]->[{tokens},{hidden}]->[{tokens},{model}]",
                  down, expected)


def softmax_in_sim(scores: NDArray[np.float32]) -> NDArray[np.float32]:
    """Runs the row-wise softmax in simulation and returns what the device produced.

    Args:
        scores: One row per reduction, each one VREG wide.

    Returns:
        The device's float32 output.

    Raises:
        RuntimeError: If the harness did not reach HALT.
    """
    rows, width = scores.shape
    # A descriptor reaches _MAX_DESCRIPTOR_ROWS VMEM rows, which is this many
    # softmax rows; more than that streams inside ONE launch rather than splitting
    # into several, which is what naming the AGU's loop level made possible.
    per_chunk = k._MAX_DESCRIPTOR_ROWS // k._TILE_VREGS
    chunks, remainder = divmod(rows, per_chunk)
    if remainder or chunks == 0:
        chunks, per_chunk = 1, rows
    kernel = k.build_softmax_rows(per_chunk, chunks=chunks)
    image = kernel.image.to_bytes(pad_to=4)
    print(f"softmax {rows}x{width}  {len(image) // 16} bundles, "
          f"{chunks} chunk(s) of {per_chunk} in one launch")

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="sim_kernel."))
    image_path = scratch / "kernel.bin"
    image_path.write_bytes(image)
    scores_path = scratch / "scores.bin"
    scores_path.write_bytes(k.as_bf16_words(values=scores).reshape(-1).tobytes())
    out_bytes = rows * width * _ELEMENT_BYTES
    poison_path = scratch / "poison.bin"
    poison_path.write_bytes(np.full(rows * width, 0x7F80, dtype="<u2").tobytes())
    dump_path = scratch / "dump.bin"

    depth = asm.mxu_output_fifo()[0]
    completed = subprocess.run(
        [str(REPO / "tb" / "run_kernel_image.sh"),
         f"+KERNEL_BIN={image_path}", f"+ARG0={scores_path}", f"+ARG2={poison_path}",
         f"+DUMP={dump_path}", f"+DUMP_BYTES={out_bytes}", "+MAX_CYCLES=200000"],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ,
             "KERNEL_IMAGE_BUILD_DIR": str(REPO / ".build" / f"kernel_image_fifo{depth}"),
             "MINITPU_FIFO_DEPTH": str(depth),
             "TB_TIMEOUT": os.environ.get("TB_TIMEOUT", "7200")})
    transcript = completed.stdout + completed.stderr
    for line in transcript.splitlines():
        if line.startswith(("KERNEL_IMAGE_ROUND", "ERROR")):
            print("  " + line)
    if "=== TB_KERNEL_IMAGE PASS ===" not in transcript:
        print(transcript[-4000:])
        raise RuntimeError("tb_kernel_image did not reach HALT on the softmax")

    return from_bf16_words(
        words=np.frombuffer(dump_path.read_bytes(), dtype="<u2")).reshape(rows, width)


def run_layernorm(rows: int, seed: int) -> bool:
    """Runs LayerNorm over several hidden rows in simulation and checks it against the host."""
    rng = np.random.default_rng(seed)
    values = k._on_bf16_grid(
        values=(rng.standard_normal((rows, k._MODEL_WIDTH)) * 2.0).astype(np.float32))
    gamma = k._on_bf16_grid(values=(1.0 + 0.1 * rng.standard_normal(k._MODEL_WIDTH)).astype(np.float32))
    beta = k._on_bf16_grid(values=(0.1 * rng.standard_normal(k._MODEL_WIDTH)).astype(np.float32))

    kernel = k.build_layernorm_rows(rows)
    image = kernel.image.to_bytes(pad_to=4)
    print(f"layernorm {rows}x{k._MODEL_WIDTH}  {len(image) // 16} bundles")

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="sim_kernel."))
    (scratch / "kernel.bin").write_bytes(image)
    (scratch / "x.bin").write_bytes(k.as_bf16_words(values=values).reshape(-1).tobytes())
    (scratch / "c.bin").write_bytes(k.layernorm_constants(gamma=gamma, beta=beta).tobytes())
    out_bytes = rows * k._MODEL_WIDTH * _ELEMENT_BYTES
    (scratch / "poison.bin").write_bytes(
        np.full(rows * k._MODEL_WIDTH, 0x7F80, dtype="<u2").tobytes())
    dump_path = scratch / "dump.bin"

    depth = asm.mxu_output_fifo()[0]
    completed = subprocess.run(
        [str(REPO / "tb" / "run_kernel_image.sh"),
         f"+KERNEL_BIN={scratch/'kernel.bin'}", f"+ARG0={scratch/'x.bin'}",
         f"+ARG1={scratch/'c.bin'}", f"+ARG2={scratch/'poison.bin'}",
         f"+DUMP={dump_path}", f"+DUMP_BYTES={out_bytes}", "+MAX_CYCLES=2000000"],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ,
             "KERNEL_IMAGE_BUILD_DIR": str(REPO / ".build" / f"kernel_image_fifo{depth}"),
             "MINITPU_FIFO_DEPTH": str(depth),
             "TB_TIMEOUT": os.environ.get("TB_TIMEOUT", "7200")})
    transcript = completed.stdout + completed.stderr
    for line in transcript.splitlines():
        if line.startswith(("KERNEL_IMAGE_ROUND", "ERROR")):
            print("  " + line)
    if "=== TB_KERNEL_IMAGE PASS ===" not in transcript:
        print(transcript[-4000:])
        return False

    actual = from_bf16_words(
        words=np.frombuffer(dump_path.read_bytes(), dtype="<u2")).reshape(rows, k._MODEL_WIDTH)
    expected = k.layernorm_reference(rows=values, gamma=gamma, beta=beta)
    return report(f"layernorm {rows}x{k._MODEL_WIDTH}", actual, expected.astype(np.float64))


def run_softmax(rows: int, seed: int) -> bool:
    """Checks the row-wise softmax against the host reference."""
    rng = np.random.default_rng(seed)
    # A spread wide enough that the maximum matters: without the subtraction these would leave
    # exp's non-positive domain, which sfu.sv asserts on.
    scores = k._on_bf16_grid(
        values=(rng.standard_normal((rows, k._VREG_ELEMENTS)) * 4.0).astype(np.float32))
    actual = softmax_in_sim(scores)
    expected = k.softmax_reference(scores=scores)
    return report(f"softmax {rows}x{k._VREG_ELEMENTS}", actual, expected.astype(np.float64))


def run_attention(seed: int) -> bool:
    """Runs one attention head in simulation, composed as attention_head composes it.

    Notes:
        Three launches -- a product, a softmax, a product -- with the transpose and the contraction
        padding on the host, for the reasons `attention_head` documents.
    """
    width = k._VREG_ELEMENTS
    rng = np.random.default_rng(seed)
    query, key, value = (
        k._on_bf16_grid(values=rng.standard_normal((width, width)).astype(np.float32))
        for _ in range(3))

    contraction = k._LOOP_BLOCKS * k._BLOCK_DEPTH
    def pad_rows(values: NDArray[np.float32]) -> NDArray[np.float32]:
        out = np.zeros((contraction, values.shape[1]), dtype=np.float32)
        out[: values.shape[0]] = values
        return out

    def pad_columns(values: NDArray[np.float32]) -> NDArray[np.float32]:
        out = np.zeros((values.shape[0], contraction), dtype=np.float32)
        out[:, : values.shape[1]] = values
        return out

    scaled = (query / np.sqrt(width)).astype(np.float32)
    scores = run_blocked_product(pad_columns(scaled), pad_rows(key.T.copy()),
                                 activate=False, label="scores ")
    chunk = k._MAX_DESCRIPTOR_ROWS // k._TILE_VREGS
    weights = np.concatenate(
        [softmax_in_sim(scores[first : first + chunk]) for first in range(0, scores.shape[0], chunk)],
        axis=0)
    actual = run_blocked_product(pad_columns(weights), pad_rows(value),
                                 activate=False, label="values ")
    expected = k.attention_reference(query=query, key=key, value=value)
    return report(f"attention head {width}x{width}", actual, expected.astype(np.float64))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--column-tiles", type=int, default=1,
                        help="output column tiles, the outer loop's trip count")
    parser.add_argument("--groups", type=int, default=1,
                        help="block groups per column tile, the middle loop's trip count")
    parser.add_argument("--loop-blocks", type=int, default=_LOOP_BLOCKS,
                        help="accumulation blocks one contraction group carries, which must be even")
    parser.add_argument("--fuse-groups", action="store_true",
                        help="accumulate every group on the device instead of draining a partial each")
    parser.add_argument("--activate", action="store_true",
                        help="activate each staged tile through the SFU's GELU, as the down projection does")
    parser.add_argument("--operands", choices=("varying", "structured"), default="varying",
                        help="varying differs along every axis; structured is what the board check used")
    parser.add_argument("--rounds", type=int, default=1, choices=(1, 2),
                        help="launches of the same image; round 1 gets different operands")
    parser.add_argument("--reset-between", action="store_true",
                        help="reset the device between rounds, to A/B state that carries across launches")
    parser.add_argument("--seed", type=int, default=0, help="seed for varying operands")
    parser.add_argument("--softmax", type=int, default=None, metavar="ROWS",
                        help="run the row-wise softmax over this many VREG-wide rows")
    parser.add_argument("--layernorm", type=int, default=None, metavar="ROWS",
                        help="run LayerNorm over this many hidden rows")
    parser.add_argument("--attention", action="store_true",
                        help="run one attention head: a product, a softmax and a product")
    parser.add_argument("--fifo-depth", type=int, default=None,
                        help="build the DUT with this MXU output FIFO depth and assemble for it, to "
                             "price the depth before committing to a bitstream")
    parser.add_argument("--ffn", action="store_true",
                        help="run a whole GPT-2 feed-forward layer, composed as full_ffn composes it")
    parser.add_argument("--tokens", type=int, default=_BLOCK_TOKENS, help="FFN token rows")
    parser.add_argument("--model", type=int, default=768, help="FFN input and output width")
    parser.add_argument("--hidden", type=int, default=3072, help="FFN intermediate width")
    args = parser.parse_args()

    if args.fifo_depth is not None:
        # Both halves or neither: the RTL builds with this depth (below) and the assembler emits
        # for it. Setting one without the other is the defect this whole harness exists to catch.
        asm._MXU_OUTPUT_FIFO_DEPTH = args.fifo_depth
        print(f"MXU output FIFO depth overridden to {args.fifo_depth} "
              f"(RTL and assembler both; not the shipped default)")

    if args.layernorm is not None:
        return 0 if run_layernorm(args.layernorm, args.seed) else 1

    if args.attention:
        return 0 if run_attention(args.seed) else 1

    if args.softmax is not None:
        return 0 if run_softmax(args.softmax, args.seed) else 1

    if args.ffn:
        return 0 if run_ffn(args.tokens, args.model, args.hidden, args.seed) else 1

    tokens = _BLOCK_TOKENS
    depth = args.groups * args.loop_blocks * _BLOCK_DEPTH
    columns = args.column_tiles * _TILE
    left, right = build_operands(tokens, depth, columns, kind=args.operands, seed=args.seed)
    second = None
    if args.rounds == 2:
        second = build_operands(tokens, depth, columns, kind=args.operands, seed=args.seed + 1)

    results = run_product(left, right, activate=args.activate, rounds=args.rounds,
                          reset_between=args.reset_between, second=second,
                          fuse_groups=args.fuse_groups, loop_blocks=args.loop_blocks)

    passed = True
    for index, result in enumerate(results):
        operands = (left, right) if index == 0 or second is None else second
        staged = activated(operands[0]) if args.activate else operands[0]
        expected = staged.astype(np.float64) @ operands[1].astype(np.float64)
        passed &= report(f"round {index}", result, expected)

    if args.rounds == 2 and np.array_equal(results[0], results[1]):
        print("  round 1 returned round 0's result bit-for-bit, on different operands")
        passed = False
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
