"""minitpu host runtime -- in-repo, CUDA-style host<->device API (ABI v8).

This is the **v1 host runtime**: a Python package that drives the MINITPU directly
over the ``minitpu.abi`` SSOT (ABI_VERSION=8: encoded CMD window + COPY/LAUNCH
descriptors + doorbell, plus the RO launch_status CSR @0x3C that
``minitpuLaunchKernel`` raises on).  NB: the runtime does NOT verify ABI_VERSION
against the device, and cannot - v8+ exposes no readable abi_version CSR (see
the NB in ``minitpu/abi.py``).  A bitstream that predates launch_status reads 0
at 0x3C, so the launch guard silently passes there.  Matching the bitstream to
this runtime is a provisioning responsibility, not a checked one.

DMA is done by the hardware AXI CDMA, so the host only writes ~8 registers and
polls; Python is not on the throughput path.

Board HAL is raw Linux **UIO + udmabuf** (PYNQ is retired):
  - :class:`UioMmio`       - MINITPU AXI-Lite CMD window over ``/dev/uioN``.
  - :class:`UdmabufBuffer` - contiguous host DMA buffer (``host_phys`` + cache sync).
The register backend is auto-detected on a board; off-board (or in tests) a
:class:`MockMmio` is injected through a private hook.  There is no public
backend setter.

Public surface (CUDA naming):
  - ``minitpuSetDevice(uio_name)`` - select the MINITPU device (``cudaSetDevice`` analog).
  - ``minitpuHostAlloc(nbytes, ...)`` - pinned host DMA alloc (``cudaHostAlloc`` analog).
  - ``minitpuHostReset()`` - drop the cached udmabuf (test isolation / re-run).
  - ``minitpuMemcpy(dst, src, nbytes, kind)`` - SYNCHRONOUS COPY (v1 primitive).
  - ``minitpuMemcpyAsync(...)`` - post-v1 stub (raises NotImplementedError).
  - ``minitpuLaunchKernel(kbin_addr, kbin_size)`` - LAUNCH a resident kernel binary.
  - ``MemcpyKind`` - HostToDevice / DeviceToHost.
  - ``Mmio`` / ``MockMmio`` - backend base + test fake.
  - ``UioMmio`` / ``UdmabufBuffer`` - board register + host-buffer HAL.
  - ``HostRegion`` - board-agnostic phys-addressed sub-view of a udmabuf.
  - ``open_device(uio_name, ...)`` - bring up a board (UIO window + udmabuf).

v1 is intentionally sequential (host-mediated, depth-1); async/streams are [v2].
A C/C++ production runtime remains a FUTURE option (a C consumer, or a measured
control-path bottleneck), not the committed near-term path.
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
from .perf import PerfDelta, PerfSample, read_perf
from .program import MinitpuKernel, minitpuAssembly, minitpuKernel, minitpuRunKernel, minitpu
from .memcpy import (
    MemcpyKind,
    minitpuDrain,
    minitpuMemcpy,
    minitpuMemcpyAsync,
)

__all__ = [
    "DeviceAddressSpace",
    "ZCU102_ADDRESS_SPACE",
    "ZCU104_ADDRESS_SPACE",
    "minitpuConfigureAddressSpace",
    "minitpuGetAddressSpace",
    "Mmio",
    "MockMmio",
    "UioMmio",
    "UdmabufBuffer",
    "HostRegion",
    "open_device",
    "minitpuDrain",
    "minitpuSetDevice",
    "minitpuHostAlloc",
    "minitpuHostReset",
    "minitpuHostRewind",
    "minitpuMalloc",
    "minitpuFree",
    "minitpuDeviceReset",
    "minitpuInAperture",
    "MemcpyKind",
    "minitpuMemcpy",
    "minitpuMemcpyAsync",
    "minitpuLaunchKernel",
    "read_perf",
    "PerfSample",
    "PerfDelta",
    "minitpuKernel",
    "minitpuAssembly",
    "MinitpuKernel",
    "minitpuRunKernel",
    "minitpu",
]
