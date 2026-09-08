"""Provides callable MINITPU programs that stage an image, transfer arguments and launch.

A program takes C-contiguous uint16 host arrays, copies all but the last to the device through the hardware
argument slots, and copies the last back after the launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Protocol

import numpy as np

from .address_space import minitpuGetAddressSpace
from .device import minitpuFree, minitpuHostAlloc, minitpuHostReset, minitpuMalloc, minitpuSetDevice
from .launch import DMA_ARG_SLOT_COUNT, INSTR_BYTES, IRAM_DEPTH, minitpuLaunchKernel
from .memcpy import MemcpyKind, minitpuMemcpy

_ELEMENT_DTYPE: np.dtype = np.dtype("<u2")
"""The device's element container, holding one BF16 value per 16-bit VMEM and VREG element."""

_ACTIVE_UIO_NAME: str | None = None
"""UIO node name the device selection currently in force names, so a repeated run skips reselecting it."""


class KernelImage(Protocol):
    """Defines the assembler result a callable program is built from.

    Attributes:
        name: The program name, used to report a malformed image.
    """

    name: str

    def to_bytes(self, *, pad_to: int | None = None) -> bytes: ...


@dataclass(frozen=True)
class MinitpuKernel:
    """Binds an assembled program to a board UIO device, making it callable.

    Attributes:
        image: The assembled program.
        uio_name: The UIO node name of the board it runs on.
    """

    image: KernelImage
    uio_name: str

    def __call__(self, *host_buffers: np.ndarray) -> None:
        """Runs the program against the given host arrays, the last of which is the output."""
        minitpuRunKernel(self.image, *host_buffers, uio_name=self.uio_name)


class minitpuAssembly:
    """Declares callable MINITPU assembly programs through its decorator.

    Inside a decorated function the ``minitpu`` name is temporarily the assembler builder, exposing ``bundle``,
    ``dma_load``, ``vadd`` and the other ISA constructors. The decorated name becomes a callable program that
    performs every runtime step.
    """

    def asm(self, function=None, *, uio_name: str = "minitpu", placement: str | None = None):
        """Assembles a function and returns its callable program wrapper.

        Args:
            function: The function to assemble, or None to return a decorator that takes it.
            uio_name: The UIO node name of the board the program runs on.
            placement: The re-placement pass the kernel is built with, or None to take the one `MINITPU_PACK`
                selects.
        """
        if function is None:
            return lambda assembly_function: self.asm(assembly_function, uio_name=uio_name, placement=placement)
        assembler = import_module("asm")
        image = assembler.asm(function, namespace="minitpu", placement=placement)
        return minitpuKernel(image, uio_name=uio_name)


minitpu = minitpuAssembly()
"""The assembly namespace a program is declared through."""


def minitpuKernel(image: KernelImage, *, uio_name: str) -> MinitpuKernel:
    """Binds an assembled image to a board, returning a callable MINITPU program.

    Calling the result performs all staging, copies, argument binding, launch and output copy. The last host array
    passed to it is the sole output and the preceding arrays are inputs.

    Raises:
        ValueError: When the UIO node name is empty or not a string.
    """
    if not isinstance(uio_name, str) or not uio_name:
        message = f"Unable to bind a kernel. The UIO device name must be a non-empty string, but got {uio_name!r}."
        raise ValueError(message)
    return MinitpuKernel(image, uio_name)


def minitpuRunKernel(image: KernelImage, *host_buffers: np.ndarray, uio_name: str) -> None:
    """Runs an image with host input arrays followed by one output array.

    This owns every transient resource the run needs: instruction-image staging, pinned host regions, device
    allocations, transfers, launch and allocator cleanup. Each call starts a fresh host allocation session, so
    repeated calls do not exhaust the udmabuf by retaining staging regions.

    Args:
        image: The assembled program to run.
        host_buffers: The input arrays followed by the single output array.
        uio_name: The UIO node name of the board the program runs on.

    Raises:
        ValueError: When the argument count falls outside the hardware slots, an argument is not a writeable
            C-contiguous non-empty uint16 array, or the image does not fit the staging prefix.
        TypeError: When an argument is not a uint16 numpy array.
    """
    if not 2 <= len(host_buffers) <= DMA_ARG_SLOT_COUNT:
        message = (
            f"Unable to run {image.name}. A program needs at least one input and one output and supports at most "
            f"{DMA_ARG_SLOT_COUNT} arguments, but got {len(host_buffers)}."
        )
        raise ValueError(message)
    host_arrays = tuple(
        _host_array(value, output=index == len(host_buffers) - 1)
        for index, value in enumerate(host_buffers)
    )
    image_bytes = image.to_bytes(pad_to=4)
    address_space = minitpuGetAddressSpace()
    max_image_bytes = min(IRAM_DEPTH * INSTR_BYTES, address_space.reserved)
    if not image_bytes or len(image_bytes) > max_image_bytes:
        message = (
            f"Unable to run {image.name}. A kernel image must be between 1 and {max_image_bytes} bytes on this "
            f"board, but got {len(image_bytes)} bytes."
        )
        raise ValueError(message)

    global _ACTIVE_UIO_NAME
    if _ACTIVE_UIO_NAME != uio_name:
        minitpuSetDevice(uio_name)
        _ACTIVE_UIO_NAME = uio_name
    minitpuHostReset()
    device_buffers: list[int] = []
    try:
        # Stages the image in the aperture's reserved low prefix, which is neither a host allocation nor a
        # user argument.
        host_kernel = minitpuHostAlloc(len(image_bytes))
        host_kernel.view[: len(image_bytes) // 4] = np.frombuffer(image_bytes, dtype="<u4")
        minitpuMemcpy(
            address_space.kernel_base, host_kernel, len(image_bytes), MemcpyKind.HostToDevice
        )

        host_regions = [minitpuHostAlloc(array.nbytes) for array in host_arrays]
        for array, region in zip(host_arrays[:-1], host_regions[:-1], strict=True):
            region.view.view(_ELEMENT_DTYPE)[: array.size] = array
            device_addr = minitpuMalloc(array.nbytes)
            device_buffers.append(device_addr)
            minitpuMemcpy(device_addr, region, array.nbytes, MemcpyKind.HostToDevice)

        output_array = host_arrays[-1]
        output_region = host_regions[-1]
        output_addr = minitpuMalloc(output_array.nbytes)
        device_buffers.append(output_addr)
        minitpuLaunchKernel(address_space.kernel_base, len(image_bytes), args=device_buffers)
        minitpuMemcpy(output_region, output_addr, output_array.nbytes, MemcpyKind.DeviceToHost)
        output_array[:] = output_region.view.view(_ELEMENT_DTYPE)[: output_array.size]
    finally:
        for address in device_buffers:
            minitpuFree(address)
        minitpuHostReset()


def _host_array(value: np.ndarray, *, output: bool) -> np.ndarray:
    """Validates one program argument and returns it flattened.

    Args:
        value: The candidate host array.
        output: Determines whether the array is the program's output and must therefore be writeable.

    Raises:
        TypeError: When the value is not a uint16 numpy array.
        ValueError: When the array is not C-contiguous, is empty, or is a read-only output.
    """
    if not isinstance(value, np.ndarray):
        message = f"Unable to bind a program argument. Arguments must be numpy arrays, but got {type(value).__name__}."
        raise TypeError(message)
    if value.dtype != _ELEMENT_DTYPE:
        message = f"Unable to bind a program argument. Arguments must have dtype uint16, but got {value.dtype}."
        raise TypeError(message)
    if not value.flags.c_contiguous:
        message = "Unable to bind a program argument. Arguments must be C-contiguous."
        raise ValueError(message)
    if not value.size:
        message = "Unable to bind a program argument. Arguments must not be empty."
        raise ValueError(message)
    if output and not value.flags.writeable:
        message = "Unable to bind a program argument. The output argument must be writeable."
        raise ValueError(message)
    return value.reshape(-1)
