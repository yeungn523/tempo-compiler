"""Callable MINITPU programs: stage an image, transfer arguments, and launch.

The public convention is deliberately small: a callable program receives one or
more C-contiguous ``numpy.uint16`` host arrays.  Every argument except the last
is copied host-to-device; the final argument is copied device-to-host after the
program returns.  The corresponding device pointers are passed to the program
in that same order through the four hardware argument slots.

``uint16`` is the MINITPU's host wire container, matching the BF16-native
element width on the device.  A program-specific caller is
responsible for converting float values to the required representation (BF16 in
the included board example) before it invokes the callable program.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

import numpy as np

from .address_space import minitpuGetAddressSpace
from .device import minitpuFree, minitpuHostAlloc, minitpuHostReset, minitpuMalloc, minitpuSetDevice
from .launch import DMA_ARG_SLOT_COUNT, INSTR_BYTES, minitpuLaunchKernel
from .memcpy import MemcpyKind, minitpuMemcpy

# Mirrored from src/pkg/minitpu_config_pkg.sv (localparam int IRAM_DEPTH).
# hwparams.py was retired: it claimed to generate minitpu_config_pkg.sv via a
# build step that does not exist in this repository, and had drifted from the
# RTL on nearly every constant it declared. Of everything it exported the
# runtime used exactly this one name, so it is hardcoded here rather than
# kept alive as a stale mirror of an entire package. Unifying RTL constants
# with their software mirrors is tracked as CONST_MGMT work.
IRAM_DEPTH = 4096

_ACTIVE_UIO_NAME: str | None = None


class KernelImage(Protocol):
    """Assembler result needed by :func:`minitpuKernel`."""

    name: str

    def to_bytes(self, *, pad_to: int | None = None) -> bytes: ...


@dataclass(frozen=True)
class MinitpuKernel:
    """A callable assembled program bound to a board UIO device."""

    image: KernelImage
    uio_name: str

    def __call__(self, *host_buffers: np.ndarray) -> None:
        minitpuRunKernel(self.image, *host_buffers, uio_name=self.uio_name)


def minitpuKernel(image: KernelImage, *, uio_name: str) -> MinitpuKernel:
    """Bind an assembled image to a board, returning a callable MINITPU program.

    For example, ``program = minitpuKernel(image, uio_name="minitpu")`` makes
    ``program(h_src1, h_src2, h_dest)`` perform all staging, COPY operations,
    argument binding, LAUNCH, and output COPY.  The last host array is the sole
    output; preceding arrays are inputs.
    """
    if not isinstance(uio_name, str) or not uio_name:
        raise ValueError("uio_name must be a non-empty UIO device name")
    return MinitpuKernel(image, uio_name)


class minitpuAssembly:
    """Namespace for declaring callable MINITPU assembly programs.

    The common form is ``@minitpu.asm``::

        @minitpu.asm
        def vector_add():
            minitpu.bundle(f=minitpu.halt())

    ``vector_add(h_src1, h_src2, h_dest)`` then performs all runtime work.
    Inside the decorated function, ``minitpu`` is temporarily the assembler
    builder, exposing ``bundle``, ``dma_load``, ``vadd``, and the other ISA
    constructors. Use ``@minitpu.asm(uio_name="other-minitpu")`` when the board UIO
    device is not named ``minitpu``.
    """

    def asm(self, function=None, *, uio_name: str = "minitpu"):
        """Assemble a function and return its callable MINITPU-program wrapper."""
        if function is None:
            return lambda assembly_function: self.asm(assembly_function, uio_name=uio_name)
        assembler = import_module("asm")
        image = assembler.asm(function, namespace="minitpu")
        return minitpuKernel(image, uio_name=uio_name)


minitpu = minitpuAssembly()


#: The device's element container. BF16-native hardware stores one BF16 value
#: per 16-bit VMEM/VREG element; the retired 32-bit container put a zero in
#: every other byte on the wire (docs/DDR_BW_UTIL.md).
ELEMENT_DTYPE = np.dtype("<u2")


def _host_array(value: np.ndarray, *, output: bool) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("program arguments must be numpy.uint16 arrays")
    if value.dtype != ELEMENT_DTYPE:
        raise TypeError(f"program arguments must have dtype numpy.uint16, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError("program arguments must be C-contiguous")
    if not value.size:
        raise ValueError("program arguments must not be empty")
    if output and not value.flags.writeable:
        raise ValueError("the output program argument must be writeable")
    return value.reshape(-1)


def minitpuRunKernel(image: KernelImage, *host_buffers: np.ndarray, uio_name: str) -> None:
    """Run ``image`` with host input arrays followed by one output array.

    This owns all transient resources: instruction-image staging, pinned host
    regions, device allocations, transfers, launch, and allocator cleanup.
    A fresh host allocation session is used for each call, so repeated calls do
    not exhaust udmabuf merely by retaining internal staging regions.
    """
    if not 2 <= len(host_buffers) <= DMA_ARG_SLOT_COUNT:
        raise ValueError(
            f"a program needs at least one input and one output, and supports at most "
            f"{DMA_ARG_SLOT_COUNT} total arguments; got {len(host_buffers)}"
        )
    host_arrays = tuple(
        _host_array(value, output=index == len(host_buffers) - 1)
        for index, value in enumerate(host_buffers)
    )
    image_bytes = image.to_bytes(pad_to=4)
    address_space = minitpuGetAddressSpace()
    max_image_bytes = min(IRAM_DEPTH * INSTR_BYTES, address_space.reserved)
    if not image_bytes or len(image_bytes) > max_image_bytes:
        raise ValueError(
            f"{image.name} is {len(image_bytes)} bytes; kernel images must be between "
            f"1 and {max_image_bytes} bytes for this board"
        )

    # The kernel image occupies the address-space's reserved low prefix.  It is
    # intentionally not exposed as a host-side allocation or user argument.
    global _ACTIVE_UIO_NAME
    if _ACTIVE_UIO_NAME != uio_name:
        minitpuSetDevice(uio_name)
        _ACTIVE_UIO_NAME = uio_name
    minitpuHostReset()
    device_buffers: list[int] = []
    try:
        host_kernel = minitpuHostAlloc(len(image_bytes))
        host_kernel.view[: len(image_bytes) // 4] = np.frombuffer(image_bytes, dtype="<u4")
        minitpuMemcpy(address_space.kernel_base, host_kernel, len(image_bytes), MemcpyKind.HostToDevice)

        host_regions = [minitpuHostAlloc(array.nbytes) for array in host_arrays]
        for array, region in zip(host_arrays[:-1], host_regions[:-1], strict=True):
            region.view.view(ELEMENT_DTYPE)[: array.size] = array
            device_addr = minitpuMalloc(array.nbytes)
            device_buffers.append(device_addr)
            minitpuMemcpy(device_addr, region, array.nbytes, MemcpyKind.HostToDevice)

        output_array = host_arrays[-1]
        output_region = host_regions[-1]
        output_addr = minitpuMalloc(output_array.nbytes)
        device_buffers.append(output_addr)
        minitpuLaunchKernel(address_space.kernel_base, len(image_bytes), args=device_buffers)
        minitpuMemcpy(output_region, output_addr, output_array.nbytes, MemcpyKind.DeviceToHost)
        output_array[:] = output_region.view.view(ELEMENT_DTYPE)[: output_array.size]
    finally:
        for address in device_buffers:
            minitpuFree(address)
        minitpuHostReset()
