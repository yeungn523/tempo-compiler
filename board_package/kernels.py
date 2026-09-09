#!/usr/bin/env python3
"""Provides every assembled kernel, and the schedules that make each one legal.

The machine has no hazard interlocks, so a schedule is a correctness argument and the comments here
carry it. `asm.schedule` enforces what it can and rejects what it cannot fix.
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "minitpu_runtime"))

from collections.abc import Callable  # noqa: E402

import asm as asm_module  # noqa: E402
from asm import AGU_SHIFT_W  # noqa: E402
from bf16 import (  # noqa: E402,F401
    HERE,
    MAX_ATTENTION_ROWS,
    _BLOCK_ACTIVATION_BYTES,
    _BLOCK_DEPTH,
    _BLOCK_TOKENS,
    _BLOCK_WEIGHT_BYTES,
    _CORRUPTION_WORD,
    _ELEMENT_BYTES,
    _GELU_LIMIT_Q7,
    _GELU_TABLE_ENTRIES,
    _INTERLEAVE_PAYLOAD_BYTES,
    _INTERLEAVE_ROUNDS,
    _LAUNCH_REPEATS,
    _LN_EPSILON,
    _LN_ROW_CONST,
    _LN_ROW_OUT,
    _LN_ROW_X,
    _LOOP_BLOCKS,
    _MAX_DESCRIPTOR_ROWS,
    _ROW_TX_IN,
    _ROW_TX_OUT,
    _ROW_ADD_BIAS,
    _ROW_ADD_LEFT,
    _ROW_ADD_OUT,
    _ROW_ADD_RIGHT,
    _MODEL_WIDTH,
    _NUM_LANES,
    _NUM_SUBLANES,
    _ROW_ACTIVATION,
    _ROW_BYTES,
    _ROW_OUTPUT,
    _ROW_SCORES,
    _ROW_SOFTMAX_OUT,
    _ROW_WEIGHT,
    _ROW_ZERO,
    _STEPS,
    _STEP_TILES,
    _TILE,
    _TILE_ROWS,
    _TILE_VREGS,
    _TOKEN_GROUPS,
    _VMEM_ENTRIES_PER_LANE,
    _VMEM_SLOT_STRIDE,
    _VREG_ELEMENTS,
    _WIDTH_ROWS,
    _WIDTH_VREGS,
    _ZERO_TILE_BYTES,
    _ZERO_TILE_WORDS,
    _bf16_round,
    _fp32_abs_q7,
    _gelu_table,
    _on_bf16_grid,
    _single_launch_operands,
    as_bf16_words,
    assert_operands_discriminating,
    from_bf16_words,
    gelu_lut_reference,
    layernorm_constants,
    pack_activations,
    pack_gemm_input,
    pack_weights,
)
from runtime import MinitpuKernel, minitpu  # noqa: E402


@minitpu.asm
def single_halt() -> None:
    """Assembles the shortest valid program, a single halt bundle."""
    minitpu.bundle(f=minitpu.halt())


@minitpu.asm
def single_mul_add() -> None:
    """Assembles a 32x32 BF16 ``C = A * B + A`` program over four DMA-streamed tiles."""
    bundles = {}
    pc = 0
    tile_bytes = 16 * _ROW_BYTES
    for tile in range(4):
        bundles[pc] = {"s": minitpu.smov_arg(0, 0) if tile == 0 else minitpu.saddi(0, 0, tile_bytes)}; pc += 1
        # Carries the DMA row stride, which is one lane stripe. Naming it as a literal instead walks DRAM at whatever
        # that literal says, so a descriptor reads every other row and the result is plausible.
        bundles[pc] = {"s": minitpu.smovi(1, _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(2, 1) if tile == 0 else minitpu.saddi(2, 2, tile_bytes)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(3, 2) if tile == 0 else minitpu.saddi(3, 3, tile_bytes)}; pc += 1
        pc += 1  # Explicit NOP that preserves the verified DMA schedule.
        bundles[pc] = {"d": minitpu.dma_load(0, 0, 16, 0, 1)}; pc += 1
        bundles[pc] = {"d": minitpu.dma_load(1, 16, 16, 2, 1)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(3)}; pc += 1
        bundles[pc] = {"f": minitpu.lbegin(4)}; pc += 1
        bundles[pc] = {"x": minitpu.vld(0, 0, agu=True, shift=2)}
        bundles[pc + 1] = {"x": minitpu.vld(1, 16, agu=True, shift=2)}
        bundles[pc + 7] = {"v": minitpu.vmul(2, 0, 1)}
        bundles[pc + 13] = {"v": minitpu.vadd(3, 2, 0)}
        bundles[pc + 19] = {"x": minitpu.vst(3, 32, agu=True, shift=2), "f": minitpu.loop_end()}; pc += 20
        bundles[pc] = {"d": minitpu.dma_store(0, 32, 16, 3, 1)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
    bundles[pc] = {"f": minitpu.halt()}; pc += 1
    for index in range(pc):
        minitpu.bundle(**bundles.get(index, {}))


def build_looped_gemm(column_tiles: int, groups: int, *, activate: bool,
                      fuse_groups: bool = False, loop_blocks: int = _LOOP_BLOCKS,
                      batch: int = 1, batch_activation_bytes: int = 0
                      ) -> MinitpuKernel:
    """Assembles the tiled GEMM with each block's staging hidden behind the previous block's compute.

    Args:
        column_tiles: Output column tiles the product covers, being the outer loop's trip count.
        groups: Block groups each column tile drains, being the middle loop's trip count.
        activate: Determines whether each staged tile is activated before it enters the array.
        fuse_groups: Accumulate every group into the resident output tiles and drain once per
            column tile, instead of draining a partial per group for the host to sum in float32.
            The accumulation is then BF16 rather than float32, which measures 1.16x the relative
            error at GPT-2's contraction depth. What it buys is that the
            product never has to reach the host at all: one drain per column tile instead of
            `groups`, no host reduction, and an output the next launch can consume where it lies.
        loop_blocks: Blocks of `_BLOCK_DEPTH` one contraction group accumulates. The default makes
            a group 192 deep, which is why GPT-2's contractions must be multiples of 192; attention
            wants 64, which is this at 2.
        batch: INDEPENDENT products in one launch -- a grouped GEMM. This is the axis a
            block-diagonal operand was standing in for: attention runs 12 heads (and B sequences)
            that may not see each other, and encoding that as zeros off the diagonal costs a factor
            of `heads * B` in multiplies. As a loop level it costs a loop level.

            All three operands must be contiguous from one problem to the next, which is what makes
            the weight and drain pointers need no bookkeeping at all: they already walk forward.
        batch_activation_bytes: How far the activation pointer moves between problems. Needed
            because the activation pointer, unlike the other two, is REWOUND for every output column
            tile -- each column tile re-reads the whole left operand.

    Returns:
        The program, taking the packed activation, packed weight and drain pointers as its three arguments.

    Raises:
        ValueError: If a group does not carry an even number of blocks, which the two-way unroll needs.
    """
    emit_step = _emit_gelu_gemm_step if activate else _emit_gemm_step
    if loop_blocks % 2:
        message = (
            f"Unable to pipeline a group of {loop_blocks} blocks. Double buffering alternates the VMEM slot "
            f"every block and a descriptor names its slot outright, so a group must carry an even count."
        )
        raise ValueError(message)
    if batch < 1:
        message = f"Unable to run {batch} products in one launch. At least one."
        raise ValueError(message)

    def looped_gemm() -> None:
        """Accumulates every block of every column tile with staging overlapped onto compute."""
        ROW_BYTES = _ROW_BYTES
        ACTIVATION_ROWS = _TOKEN_GROUPS * _TILE
        WEIGHT_ROWS = _TILE
        OUTPUT_ROWS = _TOKEN_GROUPS * _TILE

        bundles = {}

        def put(slot: int, **slots: int) -> None:
            """Merges slots into one bundle so co-issue reads at the call site."""
            bundles.setdefault(slot, {}).update(slots)

        def stage(pc: int, step: int, vmem_slot: int) -> int:
            """Issues one step's pair of descriptors into the named VMEM slot.

            Each pointer advance follows the descriptor that reads it, since the scalar unit takes two bundles
            to write back and does not interlock.
            """
            row = vmem_slot * _VMEM_SLOT_STRIDE
            bundles[pc] = {"d": minitpu.dma_load(0, _ROW_ACTIVATION[step] + row, ACTIVATION_ROWS, 0, 3)}; pc += 1
            bundles[pc] = {"s": minitpu.saddi(0, 0, ACTIVATION_ROWS * ROW_BYTES)}; pc += 1
            bundles[pc] = {"d": minitpu.dma_load(1, _ROW_WEIGHT[step] + row, WEIGHT_ROWS, 2, 3)}; pc += 1
            bundles[pc] = {"s": minitpu.saddi(2, 2, WEIGHT_ROWS * ROW_BYTES)}; pc += 1
            return pc

        def emit_block(pc: int, compute_slot: int) -> int:
            """Emits one block, which accumulates out of compute_slot while the next lands in the other.

            Each step's leading flush collects the round issued one step earlier.
            """
            for step in range(_STEPS):
                bundles[pc] = {"f": minitpu.flush_slot(3)}; pc += 1
                pc = stage(pc, step, 1 - compute_slot)
                pc += emit_step(put, pc, activation_row=_ROW_ACTIVATION[step],
                                weight_row=_ROW_WEIGHT[step], close_loop=False)
            bundles[pc] = {"f": minitpu.vmemswap()}; pc += 1
            return pc

        pc = 0
        bundles[pc] = {"s": minitpu.smovi(3, ROW_BYTES)}; pc += 1             # sreg3 = the stride every packer writes
        bundles[pc] = {"s": minitpu.smov_arg(1, 2)}; pc += 1                  # sreg1 = drain buffer base
        pc += 1                                                           # scalar writeback latency
        bundles[pc] = {"d": minitpu.dma_load(0, _ROW_ZERO, _TILE, 1, 3)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(2, 1)}; pc += 1                  # sreg2 = packed weight pointer
        bundles[pc] = {"s": minitpu.saddi(1, 1, _ZERO_TILE_BYTES)}; pc += 1   # step past the clearing tile
        pc += 1
        bundles[pc] = {"s": minitpu.saddi(2, 2, _BLOCK_WEIGHT_BYTES)}; pc += 1
        pc += 1

        # Rewinds the activation pointer by SUBTRACTION, because a grouped launch cannot re-read arg0: the four scalar
        # registers hold three pointers and the row stride, so there is nowhere to keep a per-problem base, and arg0
        # names only the first problem. The body's net movement is known here because every advance is emitted here, so
        # the rewind needs no register and leaves the pointer free to walk on to the next problem.
        batched = batch > 1
        if batched:
            # Seeds the pointer OUTSIDE the batch loop. Seeding it inside resets every problem to the first one's
            # operand, which the first problem cannot tell you: it is the only one for which that is the right answer.
            bundles[pc] = {"s": minitpu.smov_arg(0, 0)}; pc += 1
            pc += 2
            bundles[pc] = {"f": minitpu.lbegin(batch)}; pc += 1

        bundles[pc] = {"f": minitpu.lbegin(column_tiles)}; pc += 1
        if not batched:
            bundles[pc] = {"s": minitpu.smov_arg(0, 0)}; pc += 1               # sreg0 rewinds to this column tile
            pc += 1                                                        # scalar writeback latency
        bundles[pc] = {"s": minitpu.saddi(0, 0, _BLOCK_ACTIVATION_BYTES)}; pc += 1
        pc += 1                                                           # scalar writeback latency

        def emit_clear(pc: int) -> int:
            """Clears the resident output tiles from the clearing tile, which stays resident."""
            for group in range(_TOKEN_GROUPS):
                for vreg in range(_TILE_VREGS):
                    put(slot=pc + _TILE_VREGS * group + vreg,
                        x=minitpu.vld(_TILE_VREGS * group + vreg,
                                      _ROW_ZERO + _NUM_SUBLANES * vreg))
            return pc + _TOKEN_GROUPS * _TILE_VREGS

        def emit_drain(pc: int) -> int:
            """Stores the resident output tiles and advances the drain pointer past them."""
            for group in range(_TOKEN_GROUPS):
                for vreg in range(_TILE_VREGS):
                    put(slot=pc + _TILE_VREGS * group + vreg,
                        x=minitpu.vst(_TILE_VREGS * group + vreg,
                                      _ROW_OUTPUT + _TILE_ROWS * group + _NUM_SUBLANES * vreg))
            pc += _TOKEN_GROUPS * _TILE_VREGS
            bundles[pc] = {"d": minitpu.dma_store(0, _ROW_OUTPUT, OUTPUT_ROWS, 1, 3)}; pc += 1
            bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
            bundles[pc] = {"s": minitpu.saddi(1, 1, OUTPUT_ROWS * ROW_BYTES)}; pc += 1
            return pc

        # Clears the accumulators once per column tile when fused, so they carry every group and the contraction sums on
        # the device. Unfused: they clear and drain per group, and the host sums the partials in float32.
        if fuse_groups:
            pc = emit_clear(pc)

        bundles[pc] = {"f": minitpu.lbegin(groups)}; pc += 1
        # Steps back and re-primes this group's first block. flush.slot blocks until dma_slot_done covers flush_mask_q,
        # then clears it (sequencer.sv:318, :662), and only a new descriptor re-asserts that bit.
        bundles[pc] = {"s": minitpu.saddi(0, 0, -_BLOCK_ACTIVATION_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.saddi(2, 2, -_BLOCK_WEIGHT_BYTES)}; pc += 1
        pc += 1                                                           # scalar writeback latency
        for step in range(_STEPS):
            pc = stage(pc, step, 0)
            if step + 1 < _STEPS:
                bundles[pc] = {"f": minitpu.flush_slot(3)}; pc += 1

        if not fuse_groups:
            pc = emit_clear(pc)

        bundles[pc] = {"f": minitpu.lbegin(loop_blocks // 2)}; pc += 1
        pc = emit_block(pc, compute_slot=0)
        pc = emit_block(pc, compute_slot=1)
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1                  # closes the block loop

        # Returns both DMA slots, which the group's trailing prefetch still owns, for the drain to use.
        bundles[pc] = {"f": minitpu.flush_slot(3)}; pc += 1

        if not fuse_groups:
            pc = emit_drain(pc)
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1                  # closes the group loop
        if fuse_groups:
            pc = emit_drain(pc)
        if batched:
            # Undoes this column tile's walk down the activation. Every term is a `saddi` emitted above: the leading
            # block, then per contraction group a rewind of one block and an advance of `loop_blocks` of them.
            walked = _BLOCK_ACTIVATION_BYTES * (1 + groups * loop_blocks)
            pc = _advance(bundles, pc, 0, -walked)
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1                  # closes the column loop
        if batched:
            # Moves on to the next problem. The weight and drain pointers are already there: both only ever walk
            # forward, and one problem's operands end exactly where the next one's begin.
            pc = _advance(bundles, pc, 0, batch_activation_bytes)
            bundles[pc] = {"f": minitpu.loop_end()}; pc += 1              # closes the batch loop
        bundles[pc] = {"f": minitpu.halt()}; pc += 1

        for index in range(pc):
            minitpu.bundle(**bundles.get(index, {}))

    return minitpu.asm(looped_gemm)


_ADD_CHUNK_TILE_SHIFT: int = (_MAX_DESCRIPTOR_ROWS // _BLOCK_TOKENS * _NUM_SUBLANES).bit_length() - 1
"""Shift the broadcasting add advances its bias pointer by, one chunk at a time.

A chunk is `_MAX_DESCRIPTOR_ROWS` rows of `_BLOCK_TOKENS` tokens, so it spans their quotient in feature
tiles, and the bias holds `_NUM_SUBLANES` rows per tile."""


def build_elementwise_add(chunks: int, *, broadcast_right: bool = False) -> MinitpuKernel:
    """Assembles `out = left + right` over DMEM buffers, a descriptor's worth at a time.

    Notes:
        This is the kernel that lets a projection's bias and a block's residual stop
        being numpy. Both are the same operation on two DMEM tensors, and both were
        forcing every intermediate back to PS DDR to be added there.

        One iteration moves `_MAX_DESCRIPTOR_ROWS` rows of each operand in, adds them a
        VREG at a time, and moves the result out. The row count is what one descriptor
        covers, so a larger tensor is more iterations rather than a different kernel.

        With `broadcast_right`, the right operand is one feature row rather than
        `BLOCK_TOKENS` materialised copies. `pack_bias` repeats each feature tile
        across the four sublane rows, so one contiguous `vld` gives every token row
        of a VREG the same tile. Four copies of a vector cost far less than a
        materialised `[tokens, features]` tensor.

    Args:
        chunks: Iterations, each covering `_MAX_DESCRIPTOR_ROWS` VMEM rows.
        broadcast_right: Read one right-hand row repeatedly down the token axis.

    Returns:
        The program, taking the left, right and output pointers as its three arguments.

    Raises:
        ValueError: If the chunk count is not positive.
    """
    if chunks < 1:
        message = f"Unable to add over {chunks} chunks. At least one is needed."
        raise ValueError(message)
    vregs = _MAX_DESCRIPTOR_ROWS // _NUM_SUBLANES

    def elementwise_add() -> None:
        """Adds one descriptor's worth of rows per loop iteration."""
        bundles = {}
        pc = 0
        bundles[pc] = {"s": minitpu.smovi(3, _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(0, 0)}; pc += 1      # sreg0 = left
        bundles[pc] = {"s": minitpu.smov_arg(1, 1)}; pc += 1      # sreg1 = right
        bundles[pc] = {"s": minitpu.smov_arg(2, 2)}; pc += 1      # sreg2 = out
        pc += 1                                                   # scalar writeback

        out_row = _ROW_TX_IN if broadcast_right else _ROW_ADD_OUT
        if broadcast_right:
            # Stages one bias row per feature tile. The 3072-wide FFN bias is 192 VMEM rows, so it takes two
            # descriptors, because the descriptor ABI tops out at 128. It is staged once per launch, not once per token
            # block.
            bias_rows = chunks * _MAX_DESCRIPTOR_ROWS // _BLOCK_TOKENS
            first = min(bias_rows, _MAX_DESCRIPTOR_ROWS)
            bundles[pc] = {"d": minitpu.dma_load(
                1, _ROW_ADD_RIGHT, first, 1, 3)}; pc += 1
            bundles[pc] = {"f": minitpu.flush_slot(2)}; pc += 1
            if bias_rows > first:
                pc = _advance(bundles, pc, 1, first * _ROW_BYTES)
                bundles[pc] = {"d": minitpu.dma_load(
                    1, _ROW_ADD_RIGHT + first, bias_rows - first, 1, 3)}; pc += 1
                bundles[pc] = {"f": minitpu.flush_slot(2)}; pc += 1

        bundles[pc] = {"f": minitpu.lbegin(chunks)}; pc += 1
        bundles[pc] = {"d": minitpu.dma_load(0, _ROW_ADD_LEFT, _MAX_DESCRIPTOR_ROWS, 0, 3)}
        pc += 1
        if not broadcast_right:
            bundles[pc] = {"d": minitpu.dma_load(
                1, _ROW_ADD_RIGHT, _MAX_DESCRIPTOR_ROWS, 1, 3)}
            pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1 if broadcast_right else 3)}; pc += 1

        # Loads one VREG at a time into the SAME two registers each time, so the add has to consume them before the next
        # pair lands. Hoisting all the loads first, which reads more naturally, makes every add see only the last pair,
        # and the board reports a plausible wrong answer rather than a fault.
        VLD_L, ALU_L = 7, 6
        for vreg in range(vregs):
            row = _NUM_SUBLANES * vreg
            bundles[pc] = {"x": minitpu.vld(0, _ROW_ADD_LEFT + row)}
            if broadcast_right:
                feature_in_chunk = vreg // (_BLOCK_TOKENS // _NUM_SUBLANES)
                # Reads a bias staged with each feature tile REPLICATED across the four sublane rows, so the four token
                # rows of a VREG read four copies of the same tile at stride one.
                bundles[pc + 1] = {"x": minitpu.vld(
                    1, _ROW_ADD_BIAS + feature_in_chunk * _NUM_SUBLANES,
                    agu=True, shift=_ADD_CHUNK_TILE_SHIFT, level=0)}
            else:
                bundles[pc + 1] = {"x": minitpu.vld(1, _ROW_ADD_RIGHT + row)}
            add_at = pc + 1 + VLD_L
            bundles[add_at] = {"e": minitpu.vadd(2, 0, 1)}
            store_at = add_at + ALU_L
            bundles[store_at] = {"x": minitpu.vst(2, out_row + row)}
            pc = store_at + 1

        bundles[pc] = {"d": minitpu.dma_store(0, out_row, _MAX_DESCRIPTOR_ROWS, 2, 3)}
        pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
        span = _MAX_DESCRIPTOR_ROWS * _ROW_BYTES
        bundles[pc] = {"s": minitpu.saddi(0, 0, span)}; pc += 1
        if not broadcast_right:
            bundles[pc] = {"s": minitpu.saddi(1, 1, span)}; pc += 1
        bundles[pc] = {"s": minitpu.saddi(2, 2, span), "f": minitpu.loop_end()}; pc += 1

        bundles[pc] = {"f": minitpu.halt()}; pc += 1
        for index in range(pc):
            minitpu.bundle(**bundles.get(index, {}))

    # Packs, because the hand placement pads with empty bundles that carry no wait. Dropping them shrinks the image and
    # leaves the schedule as it was.
    return minitpu.asm(elementwise_add, placement="pack")


def build_tile_transpose(tiles: int, *, load_stride: int, store_stride: int,
                         store_advance: int) -> MinitpuKernel:
    """Assembles a 16x16 tile transpose, one tile per loop iteration.

    Notes:
        This is the operation with no other route. `sublane_stride` permutes rows
        uniformly across every lane and a DMA descriptor moves whole 16-lane rows,
        so exchanging lanes with sublanes needed the staging array `vtxin` and
        `vtxout` drive.

        Four `vtxin` fill the tile with rows, four `vtxout` read it back as
        columns. The tile persists between them the way the MXU's weights persist
        between `vmatload` and `vmatpop`.

    Args:
        tiles: Tiles to transpose in one launch, walked along the row axis.
        load_stride: Rows between consecutive rows OF ONE TILE in the source, in
            VMEM rows. A tile of a row-major [R, C] array is 16 rows at stride
            C / NUM_LANES, not 16 consecutive rows -- this is the first kernel here
            that needed a descriptor stride other than one, and every other kernel
            in this package still walks at stride one.
        store_stride: The same for the destination.
        store_advance: Rows the destination pointer moves per tile.

    Note:
        All three are required, with no defaults, because the two callers disagree
        on every one of them and a default is right for only one. A stack of square
        tiles is stride 1 and advance `_TILE`, and a tile column of a matrix is stride
        `C / NUM_LANES` and advance 1, because tile (i, j) becomes tile (j, i) and
        it is the destination tile's OWN rows that `store_stride` separates. Giving
        this a default silently rewrote the stacked caller's descriptors from
        stride 1 to stride 16, and every simulation still passed: the one shape
        under test named its strides explicitly.

    Returns:
        The program, taking the input and output pointers as its arguments.

    Raises:
        ValueError: If the tile count is not positive.
    """
    if tiles < 1:
        message = f"Unable to transpose {tiles} tiles. At least one is needed."
        raise ValueError(message)

    def tile_transpose() -> None:
        """Transposes one tile per iteration, advancing both pointers."""
        bundles = {}
        pc = 0
        # Spends all four scalar registers here: two pointers and two strides. The strides are immediates because a
        # kernel is built per shape anyway.
        bundles[pc] = {"s": minitpu.smovi(2, load_stride * _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smovi(3, store_stride * _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(0, 0)}; pc += 1      # sreg0 = input
        bundles[pc] = {"s": minitpu.smov_arg(1, 2)}; pc += 1      # sreg1 = output
        pc += 1

        bundles[pc] = {"f": minitpu.lbegin(tiles)}; pc += 1
        bundles[pc] = {"d": minitpu.dma_load(0, _ROW_TX_IN, _TILE, 0, 2)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"x": minitpu.vld(quarter, _ROW_TX_IN + _NUM_SUBLANES * quarter)}
            pc += 1
        pc += 8
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"e": minitpu.vtxin(quarter, quarter)}; pc += 2
        pc += 4
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"e": minitpu.vtxout(8 + quarter, quarter)}; pc += 6
        pc += 6
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"x": minitpu.vst(8 + quarter,
                                            _ROW_TX_OUT + _NUM_SUBLANES * quarter)}
            pc += 1
        pc += 8
        bundles[pc] = {"d": minitpu.dma_store(0, _ROW_TX_OUT, _TILE, 1, 3)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
        # Advances the source by a whole tile of rows at its stride, because this walks DOWN a source tile column. Where
        # the destination moves is the caller's to say.
        bundles[pc] = {"s": minitpu.saddi(0, 0, _TILE * load_stride * _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.saddi(1, 1, store_advance * _ROW_BYTES)}; pc += 1
        pc += 2                    # scalar writeback, before the next descriptor reads it
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1
        bundles[pc] = {"f": minitpu.halt()}; pc += 1
        for index in range(pc):
            minitpu.bundle(**bundles.get(index, {}))

    return minitpu.asm(tile_transpose)


#: Names the widest addend an S-slot immediate carries. `sequencer_scalar_agu` sign-extends SADDI's 16-bit field, so a
#: pointer rewind larger than this takes more than one add.
_MAX_SADDI = 0x7FFF


def _advance(bundles: dict, pc: int, register: int, delta: int) -> int:
    """Emits the adds that move a pointer by `delta` bytes, however large it is.

    A displacement past what one signed 16-bit immediate holds is split across several adds, each spaced from the
    next because the scalar unit takes two bundles to write back and interlocks nothing.

    Args:
        bundles: The program under construction, keyed by bundle index.
        pc: Where to start emitting.
        register: The scalar register to move.
        delta: The signed byte displacement.

    Returns:
        The next free bundle index.
    """
    remaining = delta
    while remaining:
        step = max(-_MAX_SADDI, min(_MAX_SADDI, remaining))
        bundles[pc] = {"s": minitpu.saddi(register, register, step)}
        pc += 2                # the scalar writeback, before anything reads it
        remaining -= step
    return pc


def build_matrix_transpose(down: int, across: int, *, load_stride: int,
                           store_stride: int) -> MinitpuKernel:
    """Assembles a whole [R, C] transpose as ONE launch, over a two-deep loop.

    Notes:
        The tile transpose walks one tile column per launch, which made a GPT-2 key
        matrix 48 launches -- C / NUM_LANES of them, one per column block. Measured
        on .193, that was the dominant cost of a batched prefill: launches scaled
        with B at ~51 per sequence per layer, and 48 of those were this.

        Nothing about it needed to be 48 launches. The tile column was the host's
        loop only because `agu_level` was never exposed and a kernel could not nest,
        so this is what that unlocks: the outer loop walks tile columns, the inner
        walks down one, and the whole matrix is a single image and a single launch.

        The pointer arithmetic is the entire content. Down a column the source moves
        a whole tile of rows at its stride and the destination moves one row, because
        tile (i, j) becomes tile (j, i). Across to the next column the source moves
        back to one row past where the column started -- nearly the whole array, and
        the reason `_advance` exists -- while the destination moves on to the tile
        row that column becomes.

    Args:
        down: Tiles down a column, R / NUM_LANES / ... i.e. R / 16.
        across: Tile columns, C / 16.
        load_stride: VMEM rows between consecutive rows of the source, C / NUM_LANES.
        store_stride: The same for the destination, R / NUM_LANES.

    Returns:
        The program, taking the source and destination pointers as its arguments.

    Raises:
        ValueError: If either tile count is not positive.
    """
    if down < 1 or across < 1:
        message = (
            f"Unable to transpose a matrix of {down} x {across} tiles. Both counts "
            f"must be positive."
        )
        raise ValueError(message)

    def matrix_transpose() -> None:
        """Transposes one tile per inner iteration, one tile column per outer."""
        bundles = {}
        pc = 0
        bundles[pc] = {"s": minitpu.smovi(2, load_stride * _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smovi(3, store_stride * _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(0, 0)}; pc += 1      # sreg0 = source
        bundles[pc] = {"s": minitpu.smov_arg(1, 2)}; pc += 1      # sreg1 = destination
        pc += 1

        bundles[pc] = {"f": minitpu.lbegin(across)}; pc += 1      # level 0: tile columns
        bundles[pc] = {"f": minitpu.lbegin(down)}; pc += 1        # level 1: down one

        bundles[pc] = {"d": minitpu.dma_load(0, _ROW_TX_IN, _TILE, 0, 2)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"x": minitpu.vld(quarter, _ROW_TX_IN + _NUM_SUBLANES * quarter)}
            pc += 1
        pc += 8
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"e": minitpu.vtxin(quarter, quarter)}; pc += 2
        pc += 4
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"e": minitpu.vtxout(8 + quarter, quarter)}; pc += 6
        pc += 6
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"x": minitpu.vst(8 + quarter,
                                            _ROW_TX_OUT + _NUM_SUBLANES * quarter)}
            pc += 1
        pc += 8
        bundles[pc] = {"d": minitpu.dma_store(0, _ROW_TX_OUT, _TILE, 1, 3)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1

        # Steps down the column.
        pc = _advance(bundles, pc, 0, _TILE * load_stride * _ROW_BYTES)
        pc = _advance(bundles, pc, 1, _ROW_BYTES)
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1

        # Across to the next column: back to one row past this column's start, and on to the destination tile row this
        # column becomes.
        pc = _advance(bundles, pc, 0, (1 - down * _TILE * load_stride) * _ROW_BYTES)
        pc = _advance(bundles, pc, 1, (_TILE * store_stride - down) * _ROW_BYTES)
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1

        bundles[pc] = {"f": minitpu.halt()}; pc += 1
        for index in range(pc):
            minitpu.bundle(**bundles.get(index, {}))

    return minitpu.asm(matrix_transpose)


def build_head_transpose(heads: int, head_tiles: int, down: int, depth: int,
                              head_advance: int) -> MinitpuKernel:
    """Assembles K's block-diagonal operand -- transpose and scatter, one launch.

    Notes:
        Attention gives the weight-stationary array a block-diagonal operand so
        every head shares one product and no head can reach another's keys. Head h
        needs its Kᵀ at block (h, h), and building that was the last thing in a
        forward pass to happen on the host: 3.46 MB a layer of download, numpy and
        upload, more than everything else in the pass together.

        It is not arithmetic. In the packed ACTIVATION layout a 16x16 tile of K is
        sixteen CONSECUTIVE VMEM rows, and in the packed WEIGHT layout its
        transpose is sixteen consecutive rows as well -- so the staging tile that
        `vtxin` and `vtxout` already drive does the whole operation and the only
        new thing is where the pointers walk.

        Three loop levels, which is what `agu_level` bought: heads outermost, then
        the four feature tiles of a head, then the tiles down the token axis.

    Args:
        heads: Attention heads, each getting one diagonal block.
        head_tiles: Feature tiles in one head, head_dim / TILE.
        down: Tiles down the token axis, tokens / TILE.
        depth: Rows of the destination weight matrix, which is the model width.
        head_advance: Destination rows to move after a head's tiles, on top of what
            the inner loops already walked. It is a SEMANTIC choice and the two
            callers differ:

            - block-diagonal: `span_tiles * depth`. Head h's block starts one column
              band on AND one row band down, and the row band is exactly what the
              feature-tile loop just walked, so it cancels.
            - grouped: `span_tiles * depth - head_tiles * TILE`. The problems simply
              follow one another with no row band, so the feature-tile loop's walk
              has to be taken back.

            Deriving it from the other counts works for the block-diagonal and is
            silently wrong for the grouped form, which is why it is passed in.

    Returns:
        The program, taking the packed K and the destination as its arguments.

    Raises:
        ValueError: If any count is not positive.
    """
    for name, value in (("heads", heads), ("head_tiles", head_tiles),
                        ("down", down), ("depth", depth)):
        if value < 1:
            message = f"Unable to build a block diagonal with {name} of {value}."
            raise ValueError(message)

    def head_transpose() -> None:
        """One tile per innermost iteration, one head per outermost."""
        bundles = {}
        pc = 0
        # Names both strides as one, because a tile is consecutive rows at each end. They still occupy registers because
        # a descriptor names its stride by register.
        bundles[pc] = {"s": minitpu.smovi(2, _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smovi(3, _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(0, 0)}; pc += 1      # sreg0 = packed K
        bundles[pc] = {"s": minitpu.smov_arg(1, 2)}; pc += 1      # sreg1 = destination
        pc += 1

        bundles[pc] = {"f": minitpu.lbegin(heads)}; pc += 1        # level 0
        bundles[pc] = {"f": minitpu.lbegin(head_tiles)}; pc += 1   # level 1
        bundles[pc] = {"f": minitpu.lbegin(down)}; pc += 1         # level 2

        bundles[pc] = {"d": minitpu.dma_load(0, _ROW_TX_IN, _TILE, 0, 2)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"x": minitpu.vld(quarter, _ROW_TX_IN + _NUM_SUBLANES * quarter)}
            pc += 1
        pc += 8
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"e": minitpu.vtxin(quarter, quarter)}; pc += 2
        pc += 4
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"e": minitpu.vtxout(8 + quarter, quarter)}; pc += 6
        pc += 6
        for quarter in range(_NUM_SUBLANES):
            bundles[pc] = {"x": minitpu.vst(8 + quarter,
                                            _ROW_TX_OUT + _NUM_SUBLANES * quarter)}
            pc += 1
        pc += 8
        bundles[pc] = {"d": minitpu.dma_store(0, _ROW_TX_OUT, _TILE, 1, 3)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1

        # Steps down the token axis, so the source runs on and the destination moves one whole tile-COLUMN of the weight
        # matrix, because these tiles are a column of Kᵀ.
        pc = _advance(bundles, pc, 0, _TILE * _ROW_BYTES)
        pc = _advance(bundles, pc, 1, depth * _ROW_BYTES)
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1

        # Across a head's feature tiles: back up the column, and one tile along it.
        pc = _advance(bundles, pc, 1, (_TILE - down * depth) * _ROW_BYTES)
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1

        # Moves on to the next head's operand, a block down a diagonal or simply the next problem. The source never
        # rewinds: K is walked once, straight through, which is what makes it one launch either way.
        pc = _advance(bundles, pc, 1, head_advance * _ROW_BYTES)
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1

        bundles[pc] = {"f": minitpu.halt()}; pc += 1
        for index in range(pc):
            minitpu.bundle(**bundles.get(index, {}))

    return minitpu.asm(head_transpose)


def build_head_copy(heads: int, head_tiles: int, tokens: int, depth: int,
                         head_advance: int) -> MinitpuKernel:
    """Assembles V's block-diagonal operand -- the same scatter, no transpose.

    Notes:
        V needs no transpose: head h's values go to block (h, h) as they are. In
        the packed layouts that makes it `tokens` contiguous rows landing `tokens`
        contiguous rows away, so this is a DMA load and a DMA store with a loop
        nest around them and no compute at all.

        It is a separate kernel from the transposing one rather than a flag on it,
        because the two differ in the only thing either of them does.

    Args:
        heads: Attention heads.
        head_tiles: Feature tiles in one head, head_dim / TILE.
        tokens: Rows one block carries, which is the padded token count.
        depth: Rows of the destination weight matrix.
        head_advance: Destination rows to skip after a head's tiles. For a
            block-diagonal operand that is one head's block, `span`, which steps
            past the diagonal; for a GROUPED operand -- one independent [span, head]
            matrix per head, laid end to end -- the inner loop has already walked
            the whole of it and this is zero.

    Returns:
        The program, taking the packed V and the destination as its arguments.

    Raises:
        ValueError: If the run exceeds what one descriptor moves.
    """
    if not 0 < tokens <= _MAX_DESCRIPTOR_ROWS:
        message = (
            f"Unable to copy {tokens} rows in one descriptor. The row count field "
            f"is seven bits, so a descriptor reaches {_MAX_DESCRIPTOR_ROWS} rows."
        )
        raise ValueError(message)

    def head_copy() -> None:
        """One contiguous run per inner iteration, one head per outer."""
        bundles = {}
        pc = 0
        bundles[pc] = {"s": minitpu.smovi(2, _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smovi(3, _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(0, 0)}; pc += 1      # sreg0 = packed V
        bundles[pc] = {"s": minitpu.smov_arg(1, 2)}; pc += 1      # sreg1 = destination
        pc += 1

        bundles[pc] = {"f": minitpu.lbegin(heads)}; pc += 1        # level 0
        bundles[pc] = {"f": minitpu.lbegin(head_tiles)}; pc += 1   # level 1

        bundles[pc] = {"d": minitpu.dma_load(0, _ROW_TX_IN, tokens, 0, 2)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
        bundles[pc] = {"d": minitpu.dma_store(0, _ROW_TX_IN, tokens, 1, 3)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1

        pc = _advance(bundles, pc, 0, tokens * _ROW_BYTES)
        pc = _advance(bundles, pc, 1, depth * _ROW_BYTES)
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1

        # Leaves this head's feature tiles to the inner loop, which already walked them. Where the next head's begins is
        # the caller's to say, being one block further down a diagonal or nowhere at all when the problems simply follow
        # one another.
        pc = _advance(bundles, pc, 1, head_advance * _ROW_BYTES)
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1

        bundles[pc] = {"f": minitpu.halt()}; pc += 1
        for index in range(pc):
            minitpu.bundle(**bundles.get(index, {}))

    return minitpu.asm(head_copy)


def build_softmax_rows(rows: int, *, chunks: int = 1) -> MinitpuKernel:
    """Assembles a row-wise softmax over VREG-sized rows.

    Notes:
        One row is one VREG: NUM_LANES x NUM_SUBLANES = 64 BF16 values, laid out as four consecutive
        VMEM rows, so a single stride-1 vld gathers a whole row and vredmax/vredsum reduce exactly
        over it. That is also the head dimension GPT-2 attention wants, which is why the row is the
        unit here rather than a tile.

        The maximum is subtracted before the exponential because the SFU's exp table covers
        non-positive inputs only -- ``sfu.sv`` asserts on a positive one -- so the numerically stable
        form is the only form this machine implements.

    Args:
        rows: Rows one descriptor's worth covers, bounded by what
            `_MAX_DESCRIPTOR_ROWS` holds.
        chunks: How many such descriptors to stream inside one launch. A row is
            `_TILE_VREGS` VMEM rows and a descriptor's row count is seven bits, so
            one descriptor reaches 32 softmax rows, and the 384 score rows of a
            32-token, 12-head layer take twelve of them. Streaming them inside one
            launch spares eleven image stagings, operand copies and result copies.

            Streaming them needs the inner loop to name its own nesting level: the
            row index rides the AGU, and `iv_by_level[0]` is the OUTERMOST loop, so
            wrapping this in a chunk loop without saying `level=1` repoints the
            index at the chunk counter. That returned 187 % relative error on the
            board rather than a fault, which is what a machine with no interlocks
            does with a wrong address.

    Returns:
        The program, taking the scores pointer and the output pointer as its arguments.

    Raises:
        ValueError: If the rows would not fit one DMA descriptor.
    """
    vmem_rows = rows * _TILE_VREGS
    if not 0 < vmem_rows <= _MAX_DESCRIPTOR_ROWS:
        message = (
            f"Unable to softmax {rows} rows in one launch. Each row spans {_TILE_VREGS} VMEM rows "
            f"and one descriptor covers {_MAX_DESCRIPTOR_ROWS}, so at most "
            f"{_MAX_DESCRIPTOR_ROWS // _TILE_VREGS} rows fit."
        )
        raise ValueError(message)

    def softmax_rows_kernel() -> None:
        """Reduces, exponentiates and normalises one row per loop iteration."""
        bundles = {}
        pc = 0
        bundles[pc] = {"s": minitpu.smovi(3, _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(0, 0)}; pc += 1      # sreg0 = scores
        bundles[pc] = {"s": minitpu.smov_arg(1, 2)}; pc += 1      # sreg1 = output
        pc += 1                                                   # scalar writeback latency
        bundles[pc] = {"f": minitpu.lbegin(chunks)}; pc += 1
        bundles[pc] = {"d": minitpu.dma_load(0, _ROW_SCORES, vmem_rows, 0, 3)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1

        bundles[pc] = {"f": minitpu.lbegin(rows)}; pc += 1
        # Shifts the AGU by 2, which advances the row pointer by _TILE_VREGS VMEM rows per iteration, and level names
        # THIS loop. The chunk loop is emitted unconditionally, so this is ALWAYS the inner one, level 1 even at chunks
        # = 1. Making it conditional on chunks > 1 looks right and is not, because a one-iteration outer loop still
        # occupies level 0.
        level = 1
        bundles[pc] = {"x": minitpu.vld(0, _ROW_SCORES, agu=True, shift=2,
                                        level=level)}; pc += 1
        bundles[pc] = {"e": minitpu.vredmax(1, 0)}; pc += 1
        bundles[pc] = {"e": minitpu.vsub(2, 0, 1)}; pc += 1        # <= 0, which is exp's domain
        bundles[pc] = {"e": minitpu.vexp(3, 2)}; pc += 1
        bundles[pc] = {"e": minitpu.vredsum(4, 3)}; pc += 1
        bundles[pc] = {"e": minitpu.vrecip(5, 4)}; pc += 1
        bundles[pc] = {"e": minitpu.vmul(6, 3, 5)}; pc += 1
        bundles[pc] = {"x": minitpu.vst(6, _ROW_SOFTMAX_OUT, agu=True, shift=2,
                                        level=level),
                       "f": minitpu.loop_end()}; pc += 1

        bundles[pc] = {"d": minitpu.dma_store(0, _ROW_SOFTMAX_OUT, vmem_rows, 1, 3)}; pc += 1
        bundles[pc] = {"f": minitpu.flush_slot(1)}; pc += 1
        span = vmem_rows * _ROW_BYTES
        bundles[pc] = {"s": minitpu.saddi(0, 0, span)}; pc += 1
        bundles[pc] = {"s": minitpu.saddi(1, 1, span)}; pc += 1
        pc += 2                    # scalar writeback, before the next iteration's descriptor reads it
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1
        bundles[pc] = {"f": minitpu.halt()}; pc += 1
        for index in range(pc):
            minitpu.bundle(**bundles.get(index, {}))

    return minitpu.asm(softmax_rows_kernel)


def build_softmax_packed(tokens: int, heads: int, *,
                         span: int = _VREG_ELEMENTS) -> MinitpuKernel:
    """Assembles the attention softmax over the scores a grouped GEMM leaves.

    A softmax row is one token and head pair. The scores are tile-major in DMEM, so one row's tiles sit `tokens` rows
    apart there, and one descriptor per token gathers them into consecutive VMEM rows. That is what leaves every load
    and store contiguous.

    Args:
        tokens: Rows in the score array, being the padded token count.
        heads: Attention heads, each owning `span / _TILE` feature tiles.
        span: Keys one softmax row covers, a whole number of VREGs because `vredmax` and `vredsum` reduce over
            exactly one. A wider row is folded with `vmax` and `vadd` first.

    Returns:
        The program, taking the packed scores and the output as its arguments.

    Raises:
        ValueError: If the span is not a whole number of VREGs, the scores or their result overrun the VMEM,
            or the shape exceeds what the instruction encodes.
    """
    if span < _VREG_ELEMENTS or span % _VREG_ELEMENTS:
        message = (
            f"Unable to softmax a row of {span}. `vredmax` and `vredsum` reduce over "
            f"one VREG, so a row is a whole number of {_VREG_ELEMENTS}."
        )
        raise ValueError(message)
    if span > MAX_ATTENTION_ROWS:
        message = (
            f"Unable to softmax a row of {span}. The scores stage between `_ROW_SCORES` and "
            f"`_ROW_SOFTMAX_OUT`, which holds a row of at most {MAX_ATTENTION_ROWS}."
        )
        raise ValueError(message)
    vregs = span // _VREG_ELEMENTS         # VREGs one softmax row occupies
    tiles = span // _TILE                  # feature tiles one head owns

    # Rows per token, as a power of two: the AGU applies a SHIFT to the loop index, not a multiply, so `tiles` is
    # rounded up. The padding is at most `tokens` rows of VMEM and costs nothing to walk.
    rows_per_token = 1 << (tiles - 1).bit_length()
    max_shift = (1 << AGU_SHIFT_W) - 1
    if rows_per_token.bit_length() - 1 > max_shift:
        message = (
            f"Unable to walk {rows_per_token} rows per token. The AGU shift field is "
            f"{AGU_SHIFT_W} bits, so the shift reaches {max_shift} and the stride "
            f"{1 << max_shift}."
        )
        raise ValueError(message)
    staged = tokens * rows_per_token
    if staged > _ROW_SOFTMAX_OUT - _ROW_SCORES:
        message = (
            f"Unable to stage {tokens} tokens of {rows_per_token} rows. The scores own the "
            f"{_ROW_SOFTMAX_OUT - _ROW_SCORES} rows below `_ROW_SOFTMAX_OUT`, but that wants {staged}."
        )
        raise ValueError(message)
    if _ROW_SOFTMAX_OUT + staged > _VMEM_ENTRIES_PER_LANE:
        message = (
            f"Unable to write {staged} rows from `_ROW_SOFTMAX_OUT`. A lane holds {_VMEM_ENTRIES_PER_LANE} "
            f"rows, but that reaches {_ROW_SOFTMAX_OUT + staged}."
        )
        raise ValueError(message)
    if tiles > _MAX_DESCRIPTOR_ROWS:
        message = (
            f"Unable to gather {tiles} feature tiles in one descriptor. The row "
            f"count field reaches {_MAX_DESCRIPTOR_ROWS}."
        )
        raise ValueError(message)

    # Spends the VREGs on the row itself, a running reduction, the broadcast maximum, the exponentials, a second running
    # reduction, the sum and its reciprocal.
    v_in = list(range(vregs))
    v_acc = vregs
    v_max = vregs + 1
    v_exp = [vregs + 2 + i for i in range(vregs)]
    v_total = 2 * vregs + 2
    v_sum = 2 * vregs + 3
    v_recip = 2 * vregs + 4

    # Issues one descriptor per token, covering that token's tiles and landing them `rows_per_token` apart so
    # the AGU reaches token t with a shift.
    #
    #   row = _ROW_SCORES + token * rows_per_token + tile
    descriptors, rows_each = tokens, tiles
    walk_bytes = _ROW_BYTES                    # the next token is one row on
    head_bytes = tiles * tokens * _ROW_BYTES   # one head, in DMEM
    agu_shift = (rows_per_token - 1).bit_length()

    def softmax_packed() -> None:
        """One head per outer iteration, one (token, head) score row per inner."""
        bundles = {}
        pc = 0
        # Names the DMA row stride, since a token's tiles are `tokens` rows apart in DMEM.
        bundles[pc] = {"s": minitpu.smovi(2, tokens * _ROW_BYTES)}; pc += 1
        bundles[pc] = {"s": minitpu.smov_arg(0, 0)}; pc += 1      # sreg0 = packed scores
        bundles[pc] = {"s": minitpu.smov_arg(1, 2)}; pc += 1      # sreg1 = output
        pc += 1
        bundles[pc] = {"f": minitpu.lbegin(heads)}; pc += 1        # level 0

        # Issues descriptors in pairs with a flush between, because only two slots exist. The pointer advance follows
        # the descriptor that reads it.
        for index in range(descriptors):
            bundles[pc] = {"d": minitpu.dma_load(index % 2,
                                                 _ROW_SCORES + index * rows_per_token,
                                                 rows_each, 0, 2)}; pc += 1
            bundles[pc] = {"s": minitpu.saddi(0, 0, walk_bytes)}; pc += 2
            if index % 2 or index == descriptors - 1:
                bundles[pc] = {"f": minitpu.flush_slot(3)}; pc += 1
        # Moves on to the next head. The loads have all been flushed, so sreg0 is free.
        if descriptors * walk_bytes != head_bytes:
            bundles[pc] = {"s": minitpu.saddi(
                0, 0, head_bytes - descriptors * walk_bytes)}; pc += 2

        bundles[pc] = {"f": minitpu.lbegin(tokens)}; pc += 1       # level 1
        # Reads the token as the induction variable, shifted by the rows a token owns. The four sublanes then reach four
        # CONSECUTIVE feature tiles.
        for index, vreg in enumerate(v_in):
            bundles[pc] = {"x": minitpu.vld(vreg,
                                            _ROW_SCORES + index * _NUM_SUBLANES,
                                            agu=True, shift=agu_shift, level=1)}
            pc += 1

        # Takes the maximum across VREGs then within one. `vmax` is elementwise and `vredmax` reduces a VREG, and they
        # order BF16 identically, which is why `bf16_gt` lives in vpu_pkg where both reach it, rather than in either.
        bundles[pc] = {"e": minitpu.vmov(v_acc, v_in[0])}; pc += 1
        for vreg in v_in[1:]:
            bundles[pc] = {"e": minitpu.vmax(v_acc, v_acc, vreg)}; pc += 1
        bundles[pc] = {"e": minitpu.vredmax(v_max, v_acc)}; pc += 1

        for index, vreg in enumerate(v_in):
            bundles[pc] = {"e": minitpu.vsub(v_exp[index], vreg, v_max)}; pc += 1
            bundles[pc] = {"e": minitpu.vexp(v_exp[index], v_exp[index])}; pc += 1

        # Takes the sum the same way, with `vadd` across VREGs and `vredsum` within one.
        bundles[pc] = {"e": minitpu.vmov(v_total, v_exp[0])}; pc += 1
        for vreg in v_exp[1:]:
            bundles[pc] = {"e": minitpu.vadd(v_total, v_total, vreg)}; pc += 1
        bundles[pc] = {"e": minitpu.vredsum(v_sum, v_total)}; pc += 1
        bundles[pc] = {"e": minitpu.vrecip(v_recip, v_sum)}; pc += 1

        for index, vreg in enumerate(v_exp):
            bundles[pc] = {"e": minitpu.vmul(vreg, vreg, v_recip)}; pc += 1
        for index, vreg in enumerate(v_exp):
            bundles[pc] = {"x": minitpu.vst(vreg,
                                            _ROW_SOFTMAX_OUT + index * _NUM_SUBLANES,
                                            agu=True, shift=agu_shift, level=1)}
            pc += 1
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1

        for index in range(descriptors):
            bundles[pc] = {"d": minitpu.dma_store(index % 2,
                                                  _ROW_SOFTMAX_OUT + index * rows_per_token,
                                                  rows_each, 1, 2)}; pc += 1
            bundles[pc] = {"s": minitpu.saddi(1, 1, walk_bytes)}; pc += 2
            if index % 2 or index == descriptors - 1:
                bundles[pc] = {"f": minitpu.flush_slot(3)}; pc += 1
        if descriptors * walk_bytes != head_bytes:
            bundles[pc] = {"s": minitpu.saddi(
                1, 1, head_bytes - descriptors * walk_bytes)}; pc += 2
        bundles[pc] = {"f": minitpu.loop_end()}; pc += 1
        bundles[pc] = {"f": minitpu.halt()}; pc += 1
        for index in range(pc):
            minitpu.bundle(**bundles.get(index, {}))

    return minitpu.asm(softmax_packed)


def _norm_geometry(width: int) -> tuple[int, int]:
    """Returns the VMEM rows and VREGs one hidden row of the given width occupies.

    Args:
        width: Features in the row.

    Returns:
        The rows and VREGs pair.

    Raises:
        ValueError: If the width is not a whole number of VREGs, or a row does not fit in one DMA descriptor.
    """
    if width <= 0 or width % _VREG_ELEMENTS:
        message = (
            f"Unable to normalise a row of {width} features. `vredsum` reduces exactly one VREG, so a row is a "
            f"whole number of {_VREG_ELEMENTS}."
        )
        raise ValueError(message)
    width_rows = width // _NUM_LANES
    if width_rows > _MAX_DESCRIPTOR_ROWS:
        message = (
            f"Unable to stream a row of {width} features: {width_rows} VMEM rows past the "
            f"{_MAX_DESCRIPTOR_ROWS} one descriptor covers. Split the row across descriptors, or widen DMA's row "
            f"field."
        )
        raise ValueError(message)
    return width_rows, width // _VREG_ELEMENTS


def _require_norm_fits(*, name: str, width_rows: int, const_rows: int, out_row: int) -> None:
    """Rejects a width whose staged row and constants would reach the output block.

    Args:
        name: The kernel the regions belong to, which is what the diagnostic names.
        width_rows: VMEM rows one hidden row occupies.
        const_rows: VMEM rows the constants block occupies.
        out_row: The first VMEM row of the output block.

    Raises:
        ValueError: If the input row plus the constants block reaches the output row, or the output block runs past
            the rows a lane holds.
    """
    end = width_rows + const_rows
    if end > out_row or out_row + width_rows > _VMEM_ENTRIES_PER_LANE:
        message = (
            f"Unable to place the {name} regions: the staged row and constants occupy rows [0:{end}) and the "
            f"output block is [{out_row}:{out_row + width_rows}) of {_VMEM_ENTRIES_PER_LANE}. Move the output "
            f"block, or stage the row in pieces."
        )
        raise ValueError(message)


def build_layernorm_rows(rows: int, *, packed: bool = False) -> MinitpuKernel:
    """Assembles LayerNorm over several hidden rows, one per loop iteration.

    Notes:
        gamma, beta and the three broadcast scalars are staged once and stay resident; only the row
        itself is streamed, one DMA in and one out per iteration, with the pointers walked by scalar
        adds the way the GEMM walks its operands. A hidden row is 48 VMEM rows, which is not a power
        of two, so the AGU's `iv << shift` cannot step it -- the scalar add can.

        The body runs to a few hundred bundles, well past `LB_CAP`, so it refetches from IRAM each
        iteration rather than replaying from the loop buffer. That costs throughput, not correctness.

    Args:
        rows: Hidden rows to normalise in one launch.
        packed: Read and write in the layout a GEMM operand uses -- `[tiles, tokens,
            TILE]` -- instead of plain row-major. This is a DMEM LAYOUT choice, and
            it costs nothing but two scalar values: a descriptor computes
            `DRAM_addr(r) = base + r * stride` from SREGs, so setting the stride to
            one token's worth of rows and advancing the base by ONE row per token
            scatters a token's feature tiles exactly where the packed order wants
            them. Row-major is `row = t * WIDTH_ROWS + c`; packed is
            `row = c * tokens + t`, the same rows with the index pair transposed --
            and each row is sixteen contiguous BF16 that stays intact, so this is a
            permutation of rows, not of lanes. Only lane permutation needs hardware.

            Without it a normalised tensor has to be downloaded, re-packed and
            re-uploaded before it can feed a projection.

    Returns:
        The program, taking the input, the constants and the output pointers as its arguments.
    """
    width_rows, width_vregs = _norm_geometry(_MODEL_WIDTH)
    gamma_row = width_rows                     # the row block ends where x ends
    beta_row = gamma_row + width_rows
    zero_row = beta_row + width_rows
    invn_row = zero_row + _NUM_SUBLANES
    eps_row = invn_row + _NUM_SUBLANES
    const_rows = eps_row + _NUM_SUBLANES - width_rows
    _require_norm_fits(name="layernorm", width_rows=width_rows, const_rows=const_rows, out_row=_LN_ROW_OUT)

    def layernorm_rows_kernel() -> None:
        """Streams one hidden row per iteration through mean, variance and the affine."""
        bundles = {}
        pc = 0

        def emit(**slots: int) -> None:
            nonlocal pc
            bundles[pc] = slots
            pc += 1

        # Loads the constants at the row stride before sreg3 becomes the packed one, since they are contiguous
        # whichever layout the rows use. There are only four scalar registers and all four are spoken for.
        emit(s=minitpu.smovi(3, _ROW_BYTES))
        emit(s=minitpu.smov_arg(0, 0))          # sreg0 = input
        emit(s=minitpu.smov_arg(1, 1))          # sreg1 = constants
        emit(s=minitpu.smov_arg(2, 2))          # sreg2 = output
        emit()                                  # scalar writeback boundary
        emit(d=minitpu.dma_load(0, gamma_row, const_rows, 1, 3))
        emit(f=minitpu.flush_slot(1))

        stride_bytes = rows * _ROW_BYTES if packed else _ROW_BYTES
        advance_bytes = _ROW_BYTES if packed else width_rows * _ROW_BYTES
        if packed:
            emit(s=minitpu.smovi(3, stride_bytes))
            emit()                              # scalar writeback boundary

        emit(f=minitpu.lbegin(rows))
        emit(d=minitpu.dma_load(0, _LN_ROW_X, width_rows, 0, 3))
        emit(s=minitpu.saddi(0, 0, advance_bytes))
        emit(f=minitpu.flush_slot(1))

        emit(x=minitpu.vld(1, zero_row))        # running sum
        emit(x=minitpu.vld(5, invn_row))        # 1/width, broadcast
        emit(x=minitpu.vld(10, eps_row))
        for tile in range(width_vregs):
            emit(x=minitpu.vld(2, _LN_ROW_X + _NUM_SUBLANES * tile))
            emit(e=minitpu.vredsum(16 + tile, 2))
        for tile in range(width_vregs):
            emit(e=minitpu.vadd(1, 14, 16 + tile) if tile & 1 else minitpu.vadd(14, 1, 16 + tile))
        # The pair alternates between registers 1 and 14, so which one holds the total follows the parity of the
        # tile count. Naming 1 outright reads the seed back at an odd count.
        sum_reg = 1 if width_vregs % 2 == 0 else 14
        emit(e=minitpu.vmul(4, sum_reg, 5))     # mean, broadcast

        emit(x=minitpu.vld(6, zero_row))
        for tile in range(width_vregs):
            emit(x=minitpu.vld(2, _LN_ROW_X + _NUM_SUBLANES * tile))
            emit(e=minitpu.vsub(7, 2, 4))
            emit(e=minitpu.vmul(8, 7, 7))
            emit(e=minitpu.vredsum(16 + tile, 8))
        for tile in range(width_vregs):
            emit(e=minitpu.vadd(6, 15, 16 + tile) if tile & 1 else minitpu.vadd(15, 6, 16 + tile))
        square_reg = 6 if width_vregs % 2 == 0 else 15
        emit(e=minitpu.vmul(9, square_reg, 5))  # variance
        emit(e=minitpu.vadd(14, 9, 10))
        emit(e=minitpu.vrsqrt(11, 14))

        for tile in range(width_vregs):
            emit(x=minitpu.vld(2, _LN_ROW_X + _NUM_SUBLANES * tile))
            emit(e=minitpu.vsub(7, 2, 4))
            emit(e=minitpu.vmul(8, 7, 11))
            emit(x=minitpu.vld(12, gamma_row + _NUM_SUBLANES * tile))
            emit(e=minitpu.vmul(14, 8, 12))
            emit(x=minitpu.vld(13, beta_row + _NUM_SUBLANES * tile))
            emit(e=minitpu.vadd(8, 14, 13))
            emit(x=minitpu.vst(8, _LN_ROW_OUT + _NUM_SUBLANES * tile))

        emit(d=minitpu.dma_store(0, _LN_ROW_OUT, _WIDTH_ROWS, 2, 3))
        emit(f=minitpu.flush_slot(1))
        emit(s=minitpu.saddi(2, 2, advance_bytes))
        emit(f=minitpu.loop_end())

        emit(f=minitpu.halt())
        for index in range(pc):
            minitpu.bundle(**bundles.get(index, {}))

    # Packs, because this kernel waits on a dependent chain of E-slot reductions rather than on the matrix array.
    # Co-issuing the independent work between those links is what shortens it.
    return minitpu.asm(layernorm_rows_kernel, placement="pack")


def _emit_gemm_step(
    put: Callable[..., None], origin: int, *, activation_row: int, weight_row: int, close_loop: bool
) -> int:
    """Emits one accumulation step of the plain GEMM, returning the bundles it spans.

    Notes:
        Carries single_gemm's verified schedule and differs from it only in naming both steps' VMEM rows
        outright. The AGU term single_gemm walks its steps with indexes the outermost loop, which a step loop
        nested inside a block loop would read wrongly.

    Args:
        put: The bundle merger the calling builder assembles its schedule with.
        origin: The bundle index this step's first staging load occupies.
        activation_row: First VMEM row of this step's activation pair, spanning both token groups.
        weight_row: First VMEM row of this step's weight tile.
        close_loop: Determines whether this step's final accumulate carries the block loop's closing bundle.

    Returns:
        The number of bundles the step spans.
    """
    MATLOAD_AT = 10         # L = 7 after the last weight load, the first slot with V24..V27 released
    PUSH_AT = 29            # far enough past vmatload for the array to hold the weights

    for vreg in range(_TILE_VREGS):
        put(slot=origin + vreg, x=minitpu.vld(24 + vreg, weight_row + _NUM_SUBLANES * vreg))
    for vreg in range(_TILE_VREGS):
        put(slot=origin + _TILE_VREGS + vreg, x=minitpu.vld(12 + vreg, activation_row + _NUM_SUBLANES * vreg))
    put(slot=origin + MATLOAD_AT, m=minitpu.vmatload(24))

    def stage_next(group: int, vreg: int) -> int | None:
        """Loads the next token group's tile under this group's pushes, which read no VMEM."""
        if group + 1 >= _TOKEN_GROUPS:
            return None
        return minitpu.vld(12 + _TILE_VREGS * (group + 1) + vreg,
                           activation_row + _TILE_ROWS * (group + 1) + _NUM_SUBLANES * vreg)

    slot = _emit_pipelined_matrix(put, origin + PUSH_AT, stage_next=stage_next, close_loop=close_loop)
    return slot - origin


def _emit_gelu_gemm_step(
    put: Callable[..., None], origin: int, *, activation_row: int, weight_row: int, close_loop: bool
) -> int:
    """Emits one accumulation step that activates each staged tile, returning the bundles it spans.

    Notes:
        Stages and activates both token groups before the first push, rather than loading the second group
        behind the first group's pushes the way the plain step does. VREG read port A is one physical port whose
        static-priority mux the matrix controller wins (``vpu.sv`` ``effective_raddr_a``), so a vgelu issued
        while a vmatpush streams silently reads the pushed register instead of its own operand. A vmatpush holds
        that port for its whole 5-cycle occupancy, which leaves no gap between pushes at II=6, so every
        activation has to precede the first push.

    Args:
        put: The bundle merger the calling builder assembles its schedule with.
        origin: The bundle index this step's first staging load occupies.
        activation_row: First VMEM row of this step's activation pair, spanning both token groups.
        weight_row: First VMEM row of this step's weight tile.
        close_loop: Determines whether this step's final accumulate carries the block loop's closing bundle.

    Returns:
        The number of bundles the step spans.
    """
    GELU_AT = 8             # L = 7 after the first activation load, and clear of every activation load
    MATLOAD_AT = 18         # L = 7 after the last weight load, and clear of the last vgelu's port A read
    MATLOAD_SLOTS = 19      # slots vmatload holds the array

    tiles = _TOKEN_GROUPS * _TILE_VREGS
    for index in range(tiles):
        group, vreg = divmod(index, _TILE_VREGS)
        put(slot=origin + index,
            x=minitpu.vld(12 + index, activation_row + _TILE_ROWS * group + _NUM_SUBLANES * vreg))
    for vreg in range(_TILE_VREGS):
        put(slot=origin + tiles + vreg, x=minitpu.vld(24 + vreg, weight_row + _NUM_SUBLANES * vreg))
    for index in range(tiles):
        put(slot=origin + GELU_AT + index, e=minitpu.vgelu(12 + index, 12 + index))
    put(slot=origin + MATLOAD_AT, m=minitpu.vmatload(24))

    slot = _emit_pipelined_matrix(put, origin + MATLOAD_AT + MATLOAD_SLOTS,
                                  stage_next=lambda group, vreg: None, close_loop=close_loop)
    return slot - origin


def _emit_pipelined_matrix(
    put: Callable[..., None], slot: int, *, stage_next: Callable[[int, int], int | None], close_loop: bool
) -> int:
    """Pushes every token group as early as the MXU output FIFO allows, draining only when it must.

    Notes:
        One rule generates the schedule: **push as early as the FIFO allows, pop only when you have
        to.** Everything else follows from the two facts it balances.

        The MXU's per-lane output FIFO holds ``asm.mxu_output_fifo_depth()`` entries. One vmatpush
        streams NUM_SUBLANES activation rows and so produces that many; one vmatpop consumes the
        same. A push that would exceed the depth has nowhere to put its results and hardware drops
        them silently, so a pop has to come first. Against that, every pop stalls until the array
        produces the result it is taking -- 82 cycles after its push -- so a pop issued sooner than
        the FIFO forces is a pop that idles the matrix controller for nothing.

        Pushing greedily is therefore both the correct and the fast answer, and it reads the FIFO
        depth rather than assuming one: at 16 entries this alternates pop and push after the first
        four pushes; at 32 it issues eight pushes before draining any; at 64 it would issue sixteen.
        A deeper FIFO needs no change here, which is the point -- the emitter states the constraint
        instead of encoding one solution to it.

    Args:
        put: The bundle merger the calling builder assembles its schedule with.
        slot: The bundle index the first push occupies.
        stage_next: Given a group and VREG, the X-slot word to co-issue on that push, or None. Pushes
            read VREG port A and no VMEM, so a load rides underneath one for free.
        close_loop: Determines whether the last accumulate carries the loop's closing bundle.

    Returns:
        The next free bundle index.
    """
    PUSH_II = 6             # vmatpush command initiation interval
    POP_II = 5              # vmatpop command initiation interval
    ADD_POP_GAP = 2         # keeps the accumulate clear of the next pop's write window

    depth, per_push, per_pop = asm_module.mxu_output_fifo()
    outstanding = 0
    popped = 0             # pops issued, which names the group and VREG each one drains
    slot_ref = [slot]
    pending = [-1]         # a drained tile whose accumulate has not been emitted yet

    def flush_accumulate() -> None:
        """Emits a drained tile's accumulate, deferred so it never splits a run of pushes.

        The adds only have to precede the pops that overwrite V28..V31, so holding them until then
        keeps the push stream contiguous -- the pushes are what the array is waiting on.
        """
        if pending[0] < 0:
            return
        last = pending[0] == _TOKEN_GROUPS - 1
        slot_ref[0] = _emit_accumulate(put, slot_ref[0], pending[0], close_loop=close_loop and last)
        if not last:
            slot_ref[0] += ADD_POP_GAP
        pending[0] = -1

    def drain_one() -> None:
        """Pops the oldest un-drained result, and marks a tile ready once all of it has landed."""
        nonlocal outstanding, popped
        vreg = popped % _TILE_VREGS
        if vreg == 0:
            flush_accumulate()      # before this pop reuses V28..V31
        put(slot=slot_ref[0], m=minitpu.vmatpop(28 + vreg))
        slot_ref[0] += POP_II
        outstanding -= per_pop
        popped += 1
        if vreg == _TILE_VREGS - 1:
            pending[0] = popped // _TILE_VREGS - 1

    for group in range(_TOKEN_GROUPS):
        for vreg in range(_TILE_VREGS):
            while outstanding + per_push > depth:
                drain_one()
            slots = {"m": minitpu.vmatpush(12 + _TILE_VREGS * group + vreg)}
            staged = stage_next(group, vreg)
            if staged is not None:
                slots["x"] = staged
            put(slot=slot_ref[0], **slots)
            slot_ref[0] += PUSH_II
            outstanding += per_push
    while outstanding > 0:
        drain_one()
    flush_accumulate()
    return slot_ref[0]


def _emit_accumulate(put: Callable[..., None], slot: int, group: int, *, close_loop: bool) -> int:
    """Adds one popped result tile into its resident accumulator, returning the next free bundle index.

    Args:
        put: The bundle merger the calling builder assembles its schedule with.
        slot: The bundle index the accumulate starts at.
        group: Which token group is being accumulated, which picks its accumulator tile.
        close_loop: Determines whether the final add carries the loop's closing bundle.

    Returns:
        The next free bundle index.
    """
    for vreg in range(_TILE_VREGS):
        put(slot=slot, e=minitpu.vadd(_TILE_VREGS * group + vreg, _TILE_VREGS * group + vreg, 28 + vreg))
        if close_loop and vreg == _TILE_VREGS - 1:
            put(slot=slot, f=minitpu.loop_end())
        slot += 1
    return slot

