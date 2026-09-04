"""Provides the minitpu host runtime, a CUDA-style host to device API over the ABI v8 command window.

Copies and launches are synchronous and depth-1, and the board hardware abstraction layer is raw Linux UIO for
registers and udmabuf for host DMA buffers.
"""

from .address_space import (
    ZCU102_ADDRESS_SPACE,
    ZCU104_ADDRESS_SPACE,
    DeviceAddressSpace,
    minitpuConfigureAddressSpace,
    minitpuGetAddressSpace,
)
from .backend import HostRegion, Mmio, MockMmio, UdmabufBuffer, UioMmio
from .device import (
    minitpuDeviceReset,
    minitpuFree,
    minitpuHostAlloc,
    minitpuHostReset,
    minitpuHostRewind,
    minitpuInAperture,
    minitpuMalloc,
    minitpuSetDevice,
    open_device,
)
from .launch import minitpuLaunchKernel
from .memcpy import MemcpyKind, minitpuDrain, minitpuMemcpy, minitpuMemcpyAsync
from .perf import PerfDelta, PerfSample, read_perf
from .program import MinitpuKernel, minitpu, minitpuAssembly, minitpuKernel, minitpuRunKernel

__all__ = [
    "DeviceAddressSpace",
    "HostRegion",
    "MemcpyKind",
    "MinitpuKernel",
    "Mmio",
    "MockMmio",
    "PerfDelta",
    "PerfSample",
    "UdmabufBuffer",
    "UioMmio",
    "ZCU102_ADDRESS_SPACE",
    "ZCU104_ADDRESS_SPACE",
    "minitpu",
    "minitpuAssembly",
    "minitpuConfigureAddressSpace",
    "minitpuDeviceReset",
    "minitpuDrain",
    "minitpuFree",
    "minitpuGetAddressSpace",
    "minitpuHostAlloc",
    "minitpuHostReset",
    "minitpuHostRewind",
    "minitpuInAperture",
    "minitpuKernel",
    "minitpuLaunchKernel",
    "minitpuMalloc",
    "minitpuMemcpy",
    "minitpuMemcpyAsync",
    "minitpuRunKernel",
    "minitpuSetDevice",
    "open_device",
    "read_perf",
]
