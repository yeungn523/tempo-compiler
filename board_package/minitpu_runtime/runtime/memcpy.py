"""Provides the CUDA-style host to device memcpy built on the ABI COPY command.

The descriptor comes from ``minitpu.abi``, so this module owns only the doorbell sequence: poll for room, stage
the words, ring, then poll for retirement.
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

from .address_space import _device_to_copy_address
from .backend import Mmio, UioMmio

CDMA_ALIGN: int = 32
"""Byte alignment both COPY operands require.

The copy engine is a 256-bit AXI CDMA and the Data Realignment Engine is unavailable above a 64-bit data width
(Xilinx PG034), so an unaligned address raises DMAIntErr on the CDMA: no data moves and every subsequent copy is
poisoned until a copy-engine soft reset. The runtime rejects a misaligned request before any window write.
"""

_BACKEND: Mmio | None = None
"""The process-wide register backend, auto-detected on a board and injected by tests."""

_UIO_NAME: str | None = None
"""UIO node name the auto-detect path opens. The runtime is board-agnostic and carries no default, so a caller
binds its board's node name through ``open_device``."""

_POLL_DELAY: float = 0.001
"""Seconds the completion poll sleeps for once it stops spinning."""

_POLL_SPIN: float = 0.002
"""Seconds the completion poll spins on the register before it starts sleeping. Sized above the 1.212 ms a
DMA-carrying launch takes, so the common case never sleeps."""

_POLL_TIMEOUT: float = 10.0
"""Seconds the completion poll waits before it reports the device is not idle."""

_DRAIN_TIMEOUT: float = 2.0
"""Seconds the exit drain waits. Longer than any legal kernel here, whose slowest is a 256-row body launch at a
few milliseconds, and short enough that a stuck device does not hold up an exiting process."""

_IN_FLIGHT: Mmio | None = None
"""The device this process rang a doorbell for and has not yet seen retire, or None.

A host that dies in that window leaves the device busy with nothing on either side timing out, so the next process
to open the board times out and blames itself. The marker turns that silent wedge into a named one, in the process
that caused it.
"""


class MemcpyKind:
    """Defines the direction selector a copy takes, matching CUDA's memcpy kind.

    The direction maps to the command direction field in the COPY descriptor and also picks which operand is the
    host buffer and which is the device address.
    """

    HostToDevice = "HostToDevice"
    """Copies from the host buffer to the device address."""
    DeviceToHost = "DeviceToHost"
    """Copies from the device address to the host buffer."""


@runtime_checkable
class HostBuffer(Protocol):
    """Defines a host-side buffer usable as the host operand of a copy.

    The physical address is the buffer's physical byte address, being a udmabuf instance's sysfs value. The cache
    sync hooks are optional and are invoked around the transfer for the relevant direction.
    """

    phys_addr: int


HostOperand = Union[HostBuffer, int]
"""A host operand, being either a host buffer or a raw physical address."""


def minitpuMemcpy(dst: HostOperand, src: HostOperand, nbytes: int, kind: str) -> None:
    """Copies bytes between the host and the device with a synchronous COPY command.

    The host buffer's cache is flushed before a host to device transfer and invalidated after a device to host
    transfer whenever the buffer exposes the sync hooks.

    Args:
        dst: The destination, being the device address for a host to device copy and the host buffer otherwise.
        src: The source, being the host buffer for a host to device copy and the device address otherwise.
        nbytes: The number of bytes to transfer.
        kind: The direction, being one of the ``MemcpyKind`` values.

    Raises:
        ValueError: When the kind is unknown, the byte count is negative or zero, or either operand address is
            misaligned.
        TimeoutError: When the command does not retire within the poll timeout.
    """
    mmio = _backend()

    if kind == MemcpyKind.HostToDevice:
        host, dev, direction = src, dst, CmdDir.H2D
    elif kind == MemcpyKind.DeviceToHost:
        host, dev, direction = dst, src, CmdDir.D2H
    else:
        message = (
            f"Unable to copy. The kind must be MemcpyKind.HostToDevice or MemcpyKind.DeviceToHost, but got "
            f"{kind!r}."
        )
        raise ValueError(message)

    host_phys = _host_phys(host)
    device_ptr = int(dev)
    dev_addr = _device_to_copy_address(device_ptr, nbytes)
    if nbytes < 0:
        message = f"Unable to copy. The byte count must be non-negative, but got {nbytes}."
        raise ValueError(message)
    if nbytes == 0:
        message = "Unable to copy. The byte count must be positive, but got a zero-length request."
        raise ValueError(message)

    host_off = host_phys % CDMA_ALIGN
    dev_off = dev_addr % CDMA_ALIGN
    if host_off or dev_off:
        message = (
            f"Unable to copy. Both addresses must be {CDMA_ALIGN}-byte aligned, because the 256-bit AXI CDMA has "
            f"no Data Realignment Engine (PG034), but got host_phys=0x{host_phys:x} off by {host_off} and "
            f"dev_addr=0x{dev_addr:x} off by {dev_off}."
        )
        raise ValueError(message)

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

    if direction == CmdDir.D2H and hasattr(host, "sync_from_device"):
        host.sync_from_device()


def minitpuMemcpyAsync(dst: HostOperand, src: HostOperand, nbytes: int, kind: str,
                       stream: None = None) -> None:
    """Copies bytes between the host and the device.

    Args:
        dst: The destination, being the device address for a host to device copy and the host buffer otherwise.
        src: The source, being the host buffer for a host to device copy and the device address otherwise.
        nbytes: The number of bytes to transfer.
        kind: The direction, being one of the ``MemcpyKind`` values.
        stream: The stream to order the copy on. The device runs one command at a time and exposes no stream, so
            None is the only value it accepts.

    Raises:
        ValueError: When the kind is unknown, the byte count is negative or zero, either operand address is
            misaligned, or a stream is named.
        TimeoutError: When the command does not retire within the poll timeout.
    """
    if stream is not None:
        message = (
            f"Unable to order a copy on a stream. The command window is one deep and the device exposes no "
            f"stream, so the stream must be None, but got {stream!r}."
        )
        raise ValueError(message)
    minitpuMemcpy(dst=dst, src=src, nbytes=nbytes, kind=kind)


def minitpuDrain(timeout: float = _DRAIN_TIMEOUT) -> bool:
    """Waits for an outstanding command to retire, so the device is left idle.

    The process exit hook calls this, and with nothing outstanding it returns immediately.

    Args:
        timeout: Seconds to wait for the device to report idle.

    Returns:
        True when the device is idle, and False when a command is still outstanding, in which case the PL needs
        reprogramming before the board is usable.
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


def _set_backend(backend: Mmio | None) -> None:
    """Installs the process-wide register backend, or clears it when given None to force re-detection."""
    global _BACKEND
    _BACKEND = backend


def _set_uio_name(name: str | None) -> None:
    """Binds the UIO node name the auto-detect backend path opens, or clears it when given None."""
    global _UIO_NAME
    _UIO_NAME = name


def _backend() -> Mmio:
    """Returns the process-wide backend, auto-detecting the bound UIO node on first use.

    Raises:
        RuntimeError: When no backend is installed and no node name is bound, or when the node cannot be opened.
    """
    global _BACKEND
    if _BACKEND is None:
        if _UIO_NAME is None:
            message = (
                "Unable to reach the device. No MMIO backend is installed and no UIO node name is bound, and the "
                "runtime carries no default board node name, so call open_device(uio_name=...) to bring up a "
                "board."
            )
            raise RuntimeError(message)
        _BACKEND = UioMmio.open_by_name(_UIO_NAME)
    return _BACKEND


def _host_phys(operand: HostOperand) -> int:
    """Resolves the physical byte address of a host operand.

    Raises:
        TypeError: When the operand is neither an integer nor an object exposing a physical address.
    """
    if isinstance(operand, int):
        return operand
    phys = getattr(operand, "phys_addr", None)
    if phys is None:
        message = (
            f"Unable to resolve a host operand. The operand must be an integer physical address or expose "
            f"'phys_addr', but got {type(operand).__name__}."
        )
        raise TypeError(message)
    return int(phys)


def _issue_command(mmio: Mmio, words) -> None:
    """Stages every descriptor word, then writes the first word with the doorbell as the latching write."""
    assert len(words) == CMD_WINDOW_WORDS
    for index in range(1, CMD_WINDOW_WORDS):
        mmio.write32(AXI_REG[f"cmd_w{index}"], words[index])
    mmio.write32(AXI_REG["cmd_w0"], words[0] | DOORBELL_BIT)


def _format_copy_diagnostics(mmio: Mmio) -> str:
    """Formats the copy-engine status words as one line of decoded flags, or names the read failure."""
    try:
        status = mmio.read32(AXI_REG.get("copy_status", 0x28))
        cdmasr = mmio.read32(AXI_REG.get("copy_cdmasr", 0x2C))
        status_flags = (
            f"err={(status >> 0) & 1} oob={(status >> 1) & 1} timeout={(status >> 2) & 1} "
            f"done={(status >> 3) & 1} busy={(status >> 4) & 1} reset_to={(status >> 5) & 1} "
            f"xfers={(status >> 8) & 0xFF}"
        )
        cdmasr_flags = (
            f"idle={(cdmasr >> 1) & 1} dma_int_err={(cdmasr >> 4) & 1} dma_slv_err={(cdmasr >> 5) & 1} "
            f"dma_dec_err={(cdmasr >> 6) & 1}"
        )
        return f"copy_status=0x{status:08X} ({status_flags}) copy_cdmasr=0x{cdmasr:08X} ({cdmasr_flags})"
    except Exception as exc:  # noqa: BLE001 - diagnostics must never mask the failure being reported
        return f"(failed to read status: {exc})"


def _poll_ready(mmio: Mmio, timeout: float | None = None) -> None:
    """Polls the ready flag until every sub-FSM reports idle, meaning room and retirement both.

    Args:
        mmio: The register backend to poll through.
        timeout: Seconds to wait, read at call time so a caller with a slower device can change the default.

    Raises:
        TimeoutError: When the flag does not report idle within the timeout.
    """
    timeout = _POLL_TIMEOUT if timeout is None else timeout
    deadline = time.monotonic() + timeout
    spin_until = time.monotonic() + _POLL_SPIN
    while mmio.read32(AXI_REG["instr_ready"]) != 1:
        now = time.monotonic()
        if now > deadline:
            diagnostics = _format_copy_diagnostics(mmio)
            message = (
                f"Unable to reach an idle device. The instr_ready flag did not read 1 within {timeout}s "
                f"[{diagnostics}]."
            )
            raise TimeoutError(message)
        if now > spin_until:
            time.sleep(_POLL_DELAY)


def _ring_command(mmio: Mmio, words) -> None:
    """Runs the two-phase doorbell handshake against the depth-1 ready flag.

    The flag means both that the device has room and that the last command retired, so one poll serves both
    phases. The retire poll cannot observe a stale idle from before the ring, because the doorbell drops the flag
    within one or two cycles while the first retire read is a fresh AXI-Lite transaction costing tens of cycles.
    The in-flight marker is cleared only past the retire poll, so an interrupt inside it leaves the marker set for
    the exit drain.
    """
    global _IN_FLIGHT
    _poll_ready(mmio)
    _issue_command(mmio, words)
    _IN_FLIGHT = mmio
    _poll_ready(mmio)
    _IN_FLIGHT = None


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
