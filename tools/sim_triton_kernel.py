"""Compiles one Triton matmul kernel for the MINITPU and copies its image to the board."""

from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np

_HERE: pathlib.Path = pathlib.Path(__file__).resolve().parent
"""Stores the path to the repository's tools directory."""

_BOARD = "petalinux@132.236.59.190"
"""Stores the account and address the compiled image is copied to."""

_BOARD_DIRECTORY = "/home/petalinux/board_package"
"""Stores the path on the board that receives the image and its operands."""

sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "board_package"))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from bf16 import (  # noqa: E402
    _BLOCK_ACTIVATION_BYTES,
    _BLOCK_TOKENS,
    _BLOCK_WEIGHT_BYTES,
    _ELEMENT_BYTES,
    build_operands,
    pack_activations,
    pack_weights,
)
from compiler import plan, read_program, signature_of, stage  # noqa: E402
from kernels import build_looped_gemm  # noqa: E402


@triton.jit
def matmul_kernel(
    activation_pointer,
    weight_pointer,
    output_pointer,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    M: tl.constexpr = 64,
    N: tl.constexpr = 48,
    K: tl.constexpr = 192,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 16,
    BLOCK_K: tl.constexpr = 64,
):
    """Multiplies an M by K activation by a K by N weight, one BLOCK_M by BLOCK_N output tile at a time.

    Notes:
        Walking the output in loops is what puts M and N in the compiled module. Reading either tile from
        tl.program_id would leave that axis to the launch instead, because a grid dimension is an argument rather
        than module content. The step of each loop is what names the axis it walks, so the token loop must step by
        BLOCK_M and the feature loop by BLOCK_N for the reader to tell them apart.
    """
    offs_k = tl.arange(0, BLOCK_K)
    for block in range(0, M, BLOCK_M):
        offs_m = block + tl.arange(0, BLOCK_M)
        for tile in range(0, N, BLOCK_N):
            offs_n = tile + tl.arange(0, BLOCK_N)
            activation_pointers = activation_pointer + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            weight_pointers = weight_pointer + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                accumulator += tl.dot(tl.load(activation_pointers), tl.load(weight_pointers))
                activation_pointers += BLOCK_K * stride_ak
                weight_pointers += BLOCK_K * stride_bk
            output_pointers = output_pointer + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(output_pointers, accumulator)


def main() -> int:
    """Compiles the kernel above, writes its image and operands, and copies them to the board.

    Returns:
        Zero once the image is on the board.

    Raises:
        RuntimeError: When scp fails to copy the image or its operands to the board.
    """
    constexprs = {parameter.name: parameter.default
                  for parameter in matmul_kernel.params if parameter.is_constexpr}
    matmuls = read_program(
        matmul_kernel,
        signature=signature_of(matmul_kernel, operands={"activation_pointer": "*bf16",
                                                        "weight_pointer": "*bf16",
                                                        "output_pointer": "*fp32"}),
        constexprs=constexprs,
    )
    matmul = matmuls[0]

    width = constexprs["N"]
    planned = plan(matmuls,
                   strides={"stride_am": matmul.reduction, "stride_ak": 1, "stride_bk": width, "stride_bn": 1,
                            "stride_cm": width, "stride_cn": 1})
    launch = planned.launches[0]
    rows = launch.row_blocks * _BLOCK_TOKENS
    columns = launch.column_tiles * matmul.columns

    image = build_looped_gemm(column_tiles=launch.column_tiles, groups=launch.groups, activate=launch.activate,
                              fuse_groups=launch.fuse_groups,
                              loop_blocks=launch.loop_blocks).image.to_bytes(pad_to=4)
    left, right = stage(*build_operands(rows, matmul.reduction, columns, kind="varying", seed=0),
                        matmul=matmul, launch=launch)

    # Reorders each operand into the traversal order the kernel walks. Runs on the host because the board is a slow
    # core, and because it packs against the same bf16 module the image was built from.
    # The drain is described by the plan, not derived here: the board reads the same numbers back out of the
    # npz, so a second derivation on this side is a second statement of one fact.
    drain = planned.drains[0]
    packed_activations = np.stack([pack_activations(activations=left[token : token + _BLOCK_TOKENS])
                                   for token in range(0, left.shape[0], _BLOCK_TOKENS)])
    packed_weights = pack_weights(weights=right)

    out = _HERE.parent / "build" / "compiled_kernel"
    out.mkdir(parents=True, exist_ok=True)
    (out / "triton_kernel.bin").write_bytes(image)
    # Derives every size the launch needs, so the board allocates and reshapes without deriving anything of its
    # own. The slack on each operand is the block the kernel prefetches past the last one.
    np.savez(
        out / "triton_operands.npz",
        activations=packed_activations,
        weights=packed_weights,
        expected=(left.astype(np.float64) @ right.astype(np.float64)).astype(np.float32),
        activation_bytes=packed_activations[0].nbytes + _BLOCK_ACTIVATION_BYTES,
        weight_bytes=packed_weights.nbytes + _BLOCK_WEIGHT_BYTES,
        drain_words=drain.words,
        drain_bytes=drain.words * _ELEMENT_BYTES,
        zero_tile_words=drain.zero_tile_words,
        # Records one partial per group per column tile, which is both the count of words to read back and the
        # shape they are summed over. The second axis is one when the device summed the groups itself.
        drain_shape=drain.shape,
    )

    print(f"compiled {rows}x{matmul.reduction} @ {matmul.reduction}x{columns}, {len(image) // 16} bundles")
    print(f"read     tile_loops={matmul.tile_loops}, so the module states {rows} tokens and {columns} features")
    print(f"launch   row_blocks={launch.row_blocks} column_tiles={launch.column_tiles} groups={launch.groups} "
          f"loop_blocks={launch.loop_blocks} activate={launch.activate} fuse_groups={launch.fuse_groups}")

    for name in ("triton_kernel.bin", "triton_operands.npz"):
        copied = subprocess.run(["scp", "-q", str(out / name), f"{_BOARD}:{_BOARD_DIRECTORY}/{name}"])
        if copied.returncode:
            message = (
                f"Unable to copy {name} to the board. scp exited {copied.returncode}, which is what it returns for "
                f"a key the board has not authorised. The image is written at {out} and waiting."
            )
            raise RuntimeError(message)
    print(f"copied   the image and its operands to {_BOARD}:{_BOARD_DIRECTORY}")
    print(f"run      ssh {_BOARD} 'cd board_package && python3 run_triton.py'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
