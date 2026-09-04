"""CUDA-style host<->device memcpy over the ABI v8 COPY command.

Public surface (CUDA naming):
  - :class:`MemcpyKind`        - HostToDevice / DeviceToHost.
  - :func:`minitpuMemcpy`      - SYNCHRONOUS COPY, the v1 primitive.
  - :func:`minitpuMemcpyAsync` - post-v1 stub (raises NotImplementedError).

Address alignment (256-bit AXI CDMA, no DRE)
--------------------------------------------
The v1 copy engine is a 256-bit (32-byte) AXI CDMA with the Data Realignment
Engine (DRE) unavailable at that data width (Xilinx PG034: DRE is not supported
for a data-width > 64 bits).  Both COPY operands must therefore be 32-byte
aligned: ``host_phys`` (the host buffer's ``phys_addr``) AND ``dev_addr``.  An
unaligned address makes the CDMA raise ``DMAIntErr``: no data moves and, until a
copy-engine soft reset, every subsequent COPY is poisoned.  The runtime ENFORCES
this up front: :func:`minitpuMemcpy` validates both addresses against
:data:`CDMA_ALIGN` and raises :class:`ValueError` (naming the misaligned operand)
BEFORE any CMD-window write or doorbell, so a misaligned request never reaches
the CDMA.  ``nbytes`` is not required to be aligned (the CDMA handles a partial
final beat) but a zero-length COPY is rejected.

The descriptor layout, opcode, direction, and doorbell bit are all sourced from
``minitpu.abi`` (the SSOT).  This module owns only the *sequence*: poll
``instr_ready`` for room, stage W1..W7, ring the doorbell in W0, poll
``instr_ready`` again for completion (the two-phase poll; see
:func:`_ring_command`).

Backend / host-buffer model
---------------------------
All register traffic goes through an :class:`~runtime.backend.Mmio`
backend.  Selection is internal: the board backend (:class:`UioMmio`) is
auto-detected on first use, and tests inject a fake through the private
:func:`_set_backend` hook.  There is no public backend setter.

The host side of a COPY (the ``host_phys`` operand) is a :class:`HostBuffer`:
any object exposing an integer ``phys_addr`` and optional ``sync_to_device`` /
``sync_from_device`` cache hooks (a :class:`~runtime.backend.UdmabufBuffer`
satisfies this).  A bare int is also accepted as a raw physical address (test /
advanced use).  The device side is an integer device address.
"""

from __future__ import annotations

import atexit
import sys
import time
from typing import Protocol, Union, runtime_checkable

from minitpu.abi import (
    AXI_REG,
    CMD_WINDOW_WORDS,
    COPY_DESCRIPTOR,
    DOORBELL_BIT,
    CmdDir,
    CmdOpc,
    build_cmd_words,
)

from .backend import Mmio, UioMmio
from .address_space import _device_to_copy_address


class MemcpyKind:
    """Direction selector for :func:`minitpuMemcpy` (CUDA ``cudaMemcpyKind``).

    Maps to ``abi.CmdDir`` in the COPY descriptor header; the direction also
    picks which operand is the host buffer vs the device address.
    """

    HostToDevice = "HostToDevice"
    DeviceToHost = "DeviceToHost"


@runtime_checkable
class HostBuffer(Protocol):
    """Host-side buffer usable as a COPY ``host_phys`` operand.

    ``phys_addr`` is the buffer's physical byte address (a udmabuf instance's
    sysfs ``phys_addr``).  The sync hooks are optional cache-coherence
    operations, invoked around the transfer for the relevant direction.
    """

    phys_addr: int


HostOperand = Union[HostBuffer, int]


# ---------------------------------------------------------------------------
# Backend selection (internal; CUDA-style implicit context)
# ---------------------------------------------------------------------------
# The register backend is NOT part of the public API: on a board the UIO/udmabuf
# backend is auto-detected on first use; tests inject a fake through the private
# _set_backend() hook below.  There is no public minitpuSetBackend function.
_BACKEND: Mmio | None = None

# UIO node name that exposes the MINITPU AXI-Lite CMD window used by the auto-detect
# backend path.  The runtime is board-AGNOSTIC: there is NO hardcoded board node
# name here.  A caller binds the name (its board's UIO node) via
# ``open_device(uio_name=...)``, which calls :func:`_set_uio_name`.  Until a name
# is bound, the auto-detect path in :func:`_backend` has nothing to open and
# raises a clear RuntimeError (same off-board failure mode as before).
_UIO_NAME: str | None = None

# Completion-poll parameters (seconds).
_POLL_DELAY = 0.001
#: How long to spin on the register before falling back to sleeping. Sized above
#: the 1.212 ms a DMA-carrying launch takes, so the common case never sleeps.
_POLL_SPIN = 0.002
_POLL_TIMEOUT = 10.0

# COPY address alignment: the v1 copy engine is a 256-bit AXI CDMA and the Data
# Realignment Engine (DRE) is unavailable at that width, so both the host and
# device addresses must be 32-byte aligned (matching byte offset).  Unaligned
# addresses raise DMAIntErr on the CDMA (no data moved, sticky poisoning until a
# copy-engine soft reset); the runtime rejects them up front instead.
CDMA_ALIGN = 32  # bytes, 256-bit AXI CDMA with no DRE (PG034)


def _set_backend(backend: Mmio | None) -> None:
    """Install the process-wide :class:`Mmio` backend (private; tests only).

    Not exported: production callers get the auto-detected board backend.  Pass
    ``None`` to clear it (test teardown) and force re-detection next call.
    """
    global _BACKEND
    _BACKEND = backend


def _set_uio_name(name: str | None) -> None:
    """Bind the UIO node name used by the auto-detect backend path (private).

    The runtime carries no board node name of its own; :func:`open_device`
    supplies the caller's board node name here so any later backend re-detection
    (after :func:`_set_backend(None)`) opens the same node.  Pass ``None`` to
    clear the binding.  Not exported: this is a caller-supplied value, not a
    board constant baked into the runtime.
    """
    global _UIO_NAME
    _UIO_NAME = name


def _backend() -> Mmio:
    """Return the process-wide backend, auto-detecting the bound UIO on first use."""
    global _BACKEND
    if _BACKEND is None:
        if _UIO_NAME is None:
            # No backend installed and no UIO node name bound: the runtime has no
            # board constant to fall back on.  Callers bring up the board via
            # open_device(uio_name=...) (or inject a fake via _set_backend).
            raise RuntimeError(
                "no MMIO backend installed and no UIO node name bound; call "
                "open_device(uio_name=...) to bring up a board (or _set_backend "
                "to inject a fake). The runtime carries no default board node name."
            )
        # Board path: mmap the MINITPU AXI-Lite window over the bound UIO node.
        # Off-board this raises a clear RuntimeError (no /sys/class/uio device).
        _BACKEND = UioMmio.open_by_name(_UIO_NAME)
    return _BACKEND


def _host_phys(operand: HostOperand) -> int:
    """Resolve the physical byte address of a host operand."""
    if isinstance(operand, int):
        return operand
    phys = getattr(operand, "phys_addr", None)
    if phys is None:
        raise TypeError(
            "host operand must be an int physical address or expose "
            f"'phys_addr'; got {type(operand).__name__}"
        )
    return int(phys)


def _issue_command(mmio: Mmio, words) -> None:
    """Write W1..W7, then W0 with the doorbell as the final latching write."""
    assert len(words) == CMD_WINDOW_WORDS
    for i in range(1, CMD_WINDOW_WORDS):
        mmio.write32(AXI_REG[f"cmd_w{i}"], words[i])
    mmio.write32(AXI_REG["cmd_w0"], words[0] | DOORBELL_BIT)


def _format_copy_diagnostics(mmio: Mmio) -> str:
    try:
        status = mmio.read32(AXI_REG.get("copy_status", 0x28))
        cdmasr = mmio.read32(AXI_REG.get("copy_cdmasr", 0x2C))
        st_err = (status >> 0) & 1
        st_oob = (status >> 1) & 1
        st_timeout = (status >> 2) & 1
        st_done = (status >> 3) & 1
        busy = (status >> 4) & 1
        st_reset_to = (status >> 5) & 1
        xfers = (status >> 8) & 0xFF

        idle = (cdmasr >> 1) & 1
        dma_int_err = (cdmasr >> 4) & 1
        dma_slv_err = (cdmasr >> 5) & 1
        dma_dec_err = (cdmasr >> 6) & 1

        return (
            f"copy_status=0x{status:08X} (err={st_err} oob={st_oob} timeout={st_timeout} done={st_done} busy={busy} reset_to={st_reset_to} xfers={xfers}) "
            f"copy_cdmasr=0x{cdmasr:08X} (idle={idle} dma_int_err={dma_int_err} dma_slv_err={dma_slv_err} dma_dec_err={dma_dec_err})"
        )
    except Exception as exc:
        return f"(failed to read status: {exc})"


def _poll_ready(mmio: Mmio, timeout: float | None = None) -> None:
    """Poll ``instr_ready`` until it reads 1 (all sub-FSMs idle: room + retired).

    Spins before it sleeps. `_POLL_DELAY` is a millisecond and the operations
    being waited on are not: a doorbell retires in 0.111 ms and a launch carrying
    DMA in 1.212 ms, so sleeping quantised every wait up to a millisecond. Profiled
    on a board, `time.sleep` was 60 ms of a 182 ms layer -- the single largest
    entry, ahead of every device operation.

    The spin window is bounded so a genuinely stuck device still yields the CPU
    rather than burning it for ten seconds.
    """
    # Read at CALL time, not bound as a default: a default argument freezes the
    # module value at import, so nothing -- a test, or a caller with a slower
    # device -- could ever change it.
    timeout = _POLL_TIMEOUT if timeout is None else timeout
    deadline = time.monotonic() + timeout
    spin_until = time.monotonic() + _POLL_SPIN
    while mmio.read32(AXI_REG["instr_ready"]) != 1:
        now = time.monotonic()
        if now > deadline:
            diag = _format_copy_diagnostics(mmio)
            raise TimeoutError(
                f"timed out after {timeout}s waiting for instr_ready==1 (device not idle) [{diag}]"
            )
        if now > spin_until:
            time.sleep(_POLL_DELAY)


#: The device this process rang a doorbell for and has not yet seen retire, or
#: None. A host that dies between the doorbell and the retire poll leaves the
#: device executing a command nobody is waiting on: `instr_ready` stays 0, and the
#: next process to open the board times out on its OWN first poll and blames
#: itself. Nothing on the device times out, so without this the only recovery is
#: to reprogram the PL -- which has cost real time more than once.
#:
#: This is the host half of D6. It cannot rescue a kernel that will never retire
#: (that needs a device-side watchdog), and it cannot run at all under SIGKILL.
#: What it does is convert a silent wedge into a named one, at the moment it
#: happens, in the process that caused it.
_IN_FLIGHT: Mmio | None = None

#: How long the exit drain waits. Longer than any legal kernel here -- the
#: slowest is a 256-row body launch at a few milliseconds -- and short enough
#: that a genuinely stuck device does not hold up an exiting process.
_DRAIN_TIMEOUT = 2.0


def minitpuDrain(timeout: float = _DRAIN_TIMEOUT) -> bool:
    """Waits for an outstanding command to retire, so the device is left idle.

    Called automatically when the process exits. Safe to call at any time and
    from anywhere: with nothing outstanding it returns immediately.

    Args:
        timeout: Seconds to wait for `instr_ready`.

    Returns:
        True if the device is idle, False if a command is still outstanding --
        in which case the PL needs reprogramming before the board is usable.
    """
    global _IN_FLIGHT
    mmio, _IN_FLIGHT = _IN_FLIGHT, None
    if mmio is None:
        return True
    try:
        _poll_ready(mmio, timeout=timeout)
        return True
    except (TimeoutError, OSError, ValueError):
        return False


def _drain_at_exit() -> None:
    """Drains on the way out, and says so plainly when it cannot."""
    if _IN_FLIGHT is None:
        return
    if not minitpuDrain():
        print(
            "MINITPU: exiting with a command still executing on the device.\n"
            "  The doorbell was rung and the retire poll never completed, so the\n"
            "  device is left busy and the next process will time out opening it.\n"
            "  Reprogram before using this board again:\n"
            "    sudo make -C board_package minitpu",
            file=sys.stderr,
        )


atexit.register(_drain_at_exit)


def _ring_command(mmio: Mmio, words) -> None:
    """Two-phase doorbell handshake against the depth-1 ``instr_ready`` flag.

    ``instr_ready`` is one combinational all-idle flag that means both "device
    has room" (not full) and "last command retired" (charter §4).  So the same
    poll serves both phases:

      1. PRE-RING backpressure: poll ``instr_ready==1`` to confirm the depth-1
         slot is free before touching the CMD window (no silent overwrite).
      2. Stage W1..W7 then ring the doorbell in W0 (final latching write).
      3. POST-RING completion: poll ``instr_ready==1`` for the command to retire.

    Stale-ready safety on the post-ring poll: the phase-3 poll cannot observe a
    stale idle=1 left over from before the ring.  The doorbell write in phase 2
    drops ``instr_ready`` to 0 within 1-2 CMD-FSM cycles of the W0 latch, whereas
    the *next* register access (the first phase-3 read) is a fresh AXI-Lite
    read transaction whose round-trip latency (address + data handshakes, tens
    of cycles) is far greater than that 1-2 cycle doorbell-to-busy latch.  So by
    the time the first completion read returns, busy=0 is already visible: we
    never read the pre-ring idle=1 and mistake it for "already retired".
    """
    global _IN_FLIGHT
    _poll_ready(mmio)  # phase 1: wait for room (device idle / not full)
    _issue_command(mmio, words)  # phase 2: stage W1..W7, ring W0 last
    # Set between the doorbell and the retire poll, and cleared only on the far
    # side of it. An interrupt anywhere in phase 3 therefore leaves it SET, which
    # is exactly the window the exit drain exists for. Two module-global writes,
    # against a launch that costs 132 us.
    _IN_FLIGHT = mmio
    _poll_ready(mmio)  # phase 3: wait for retire (busy already latched, see above)
    _IN_FLIGHT = None


def minitpuMemcpy(
    dst: HostOperand,
    src: HostOperand,
    nbytes: int,
    kind: str,
) -> None:
    """Synchronous host<->device copy via a COPY command (ABI v8, charter §9).

    CUDA-style argument order: ``(dst, src, nbytes, kind)``.  The ``kind``
    selects which operand is the host buffer and which is the device address:

      - HostToDevice: ``src`` is the host buffer, ``dst`` is the device address.
      - DeviceToHost: ``dst`` is the host buffer, ``src`` is the device address.

    Builds the 8-word COPY descriptor from ``abi.COPY_DESCRIPTOR`` (host_phys
    split across W1/W2, dev_addr in W3, len in W4, opc=COPY and dir in W0), then
    runs the two-phase doorbell handshake (:func:`_ring_command`): poll
    ``instr_ready==1`` for room, write W1..W7, ring the doorbell (W0[4]) last,
    poll ``instr_ready==1`` for completion.  Host-buffer cache is flushed before
    an H2D transfer and invalidated after a D2H transfer when the buffer exposes
    the sync hooks.

    Alignment guard: both ``host_phys`` (from the host buffer's ``phys_addr``)
    and ``dev_addr`` must be :data:`CDMA_ALIGN` (32-byte) aligned, because the v1
    copy engine is a 256-bit AXI CDMA with no Data Realignment Engine (PG034).
    A misaligned operand raises :class:`ValueError` (naming which operand is off)
    BEFORE any CMD-window write or doorbell, so it never reaches the CDMA (an
    unaligned address there raises DMAIntErr and poisons subsequent COPYs).  A
    zero-length COPY is also rejected; ``nbytes`` itself need not be aligned.

    Blocks until the command retires; raises TimeoutError otherwise.
    """
    mmio = _backend()

    if kind == MemcpyKind.HostToDevice:
        host, dev, direction = src, dst, CmdDir.H2D
    elif kind == MemcpyKind.DeviceToHost:
        host, dev, direction = dst, src, CmdDir.D2H
    else:
        raise ValueError(
            f"unknown memcpy kind {kind!r}; use MemcpyKind.HostToDevice / MemcpyKind.DeviceToHost"
        )

    host_phys = _host_phys(host)
    device_ptr = int(dev)
    dev_addr = _device_to_copy_address(device_ptr, nbytes)
    if nbytes < 0:
        raise ValueError(f"nbytes must be non-negative, got {nbytes}")
    if nbytes == 0:
        raise ValueError("zero-length COPY: nbytes must be > 0 (nothing to transfer)")

    # Alignment guard (256-bit AXI CDMA, no DRE): reject a misaligned COPY up
    # front so it never reaches the CDMA, where it would raise DMAIntErr (no data
    # moved) and poison every subsequent COPY until a copy-engine soft reset.
    host_off = host_phys % CDMA_ALIGN
    dev_off = dev_addr % CDMA_ALIGN
    if host_off or dev_off:
        raise ValueError(
            f"COPY requires {CDMA_ALIGN}-byte-aligned addresses "
            f"(256-bit AXI CDMA has no DRE, PG034): "
            f"host_phys=0x{host_phys:x} (off {host_off}), "
            f"dev_addr=0x{dev_addr:x} (off {dev_off}). "
            f"Align both to {CDMA_ALIGN} bytes."
        )

    # Flush host cache -> memory before the device reads it (H2D only).
    if direction == CmdDir.H2D and hasattr(host, "sync_to_device"):
        host.sync_to_device()

    words = build_cmd_words(
        COPY_DESCRIPTOR,
        {
            "opc": CmdOpc.COPY,
            "dir_tgt": direction,
            "host_phys_lo": host_phys & 0xFFFF_FFFF,
            "host_phys_hi": (host_phys >> 32) & 0xFFFF_FFFF,
            "dev_addr_lo": dev_addr & 0xFFFF_FFFF,
            "dev_addr_hi": (dev_addr >> 32) & 0xFFFF_FFFF,
            "len": nbytes,
        },
    )
    _ring_command(mmio, words)

    # Invalidate host cache <- memory after the device wrote it (D2H only).
    if direction == CmdDir.D2H and hasattr(host, "sync_from_device"):
        host.sync_from_device()


def minitpuMemcpyAsync(
    dst: HostOperand,
    src: HostOperand,
    nbytes: int,
    kind: str,
    stream=None,
):
    """Asynchronous COPY -- POST-v1, not implemented.

    v1 is intentionally sequential (host-mediated, depth-1); async COPY needs
    the [v2] concurrency model (real command ring + per-command completion +
    multi-slot ``dma_slot_done``).  The intended v2 shape returns a slot handle
    that ``minitpuStreamSynchronize`` polls against ``dma_slot_done``.

    TODO([v2]): implement single-slot async over ``dma_slot_done`` + a
    ``minitpuStreamSynchronize`` wait; until then use :func:`minitpuMemcpy`.
    """
    raise NotImplementedError(
        "minitpuMemcpyAsync is post-v1 ([v2] concurrency, dma_slot_done "
        "multi-slot). v1 is sequential/depth-1: use minitpuMemcpy (synchronous). "
        "TODO([v2]): single-slot async via dma_slot_done + minitpuStreamSynchronize."
    )
