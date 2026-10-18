#!/usr/bin/env bash
# Provision UIO + udmabuf for the NPU on ZCU104.  Runs ON the board, as root.
#
# After `make minitpu` programs the PL, this script:
#   1. Compiles and applies the device-tree overlay (minitpu_0 → generic-uio)
#   2. Manually binds uio_pdrv_genirq (no auto-match on this kernel)
#   3. Loads u-dma-buf if not already present
#   4. Verifies /dev/uioN (name "minitpu") and /dev/udmabuf0 exist
#
# Prerequisites:
#   - PL must be programmed first (`make minitpu`)
#   - dtc (device-tree compiler) must be installed
#   - The bundled minitpu/u-dma-buf.ko must match the board kernel, unless an
#     in-tree module is available.
#
# Usage:
#   sudo ./minitpu/scripts/provision.sh                          # defaults
#   sudo ./minitpu/scripts/provision.sh --udmabuf-size 0x1000000 # 16 MiB DMA buffer
set -euo pipefail

minitpu_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
package_dir=$(cd "$minitpu_dir/.." && pwd)
OVERLAY_SRC="${OVERLAY_SRC:-$minitpu_dir/overlay.dtso}"
BOARD_SETUP_DIR="${BOARD_SETUP_DIR:-$package_dir/../board_setup}"

# Platform device name = <address>.<node-name> after the overlay is applied.
DEV_NAME="${DEV_NAME:-90000000.minitpu}"
UIO_NAME="${UIO_NAME:-minitpu}"
UDMABUF_SIZE="${UDMABUF_SIZE:-0x1000000}"   # 16 MiB default

[[ $EUID -eq 0 ]] || { echo "ERROR: run as root (sudo)" >&2; exit 1; }

# ---- step 1: device-tree overlay ----
dtbo="$minitpu_dir/overlay.dtbo"
if [[ -f "$OVERLAY_SRC" ]]; then
    if command -v dtc >/dev/null 2>&1; then
        echo "Compiling overlay: $OVERLAY_SRC → $dtbo"
        dtc -@ -I dts -O dtb -o "$dtbo" "$OVERLAY_SRC" 2>/dev/null || true
    elif [[ ! -f "$dtbo" ]]; then
        echo "ERROR: dtc not found and no pre-compiled overlay.dtbo" >&2
        exit 1
    fi
fi

configfs="/sys/kernel/config/device-tree/overlays"
if [[ -d "$configfs" ]]; then
    if [[ ! -d "$configfs/minitpu" ]]; then
        echo "Applying device-tree overlay via configfs..."
        mkdir -p "$configfs/minitpu"
        cat "$dtbo" > "$configfs/minitpu/dtbo"
        # Give the kernel a moment to probe the new node.
        sleep 0.5
    else
        echo "Overlay 'minitpu' already applied in configfs; skipping."
    fi
elif command -v fpgautil >/dev/null 2>&1 && [[ -f "$dtbo" ]]; then
    # fpgautil can apply overlays too (-o flag).
    echo "Applying device-tree overlay via fpgautil..."
    fpgautil -o "$dtbo"
    sleep 0.5
else
    echo "WARNING: cannot apply overlay (no configfs and no fpgautil)." >&2
    echo "  The platform device $DEV_NAME may not exist."             >&2
    echo "  Try: mkdir -p $configfs/minitpu && cat $dtbo > $configfs/minitpu/dtbo" >&2
fi

# ---- step 2: bind uio_pdrv_genirq ----
dev="/sys/bus/platform/devices/$DEV_NAME"
echo "Waiting for platform device $DEV_NAME..."
for _ in $(seq 1 20); do [[ -d "$dev" ]] && break; sleep 0.2; done
if [[ ! -d "$dev" ]]; then
    echo "ERROR: platform device $DEV_NAME not found." >&2
    echo "  Is the PL programmed?  Was the overlay applied?"       >&2
    echo "  Check: ls /sys/bus/platform/devices/ | grep minitpu"   >&2
    exit 1
fi

if [[ -e "$dev/driver" ]]; then
    echo "  $DEV_NAME already bound to $(basename "$(readlink -f "$dev/driver")")"
else
    echo "  Binding $DEV_NAME to uio_pdrv_genirq..."
    echo "uio_pdrv_genirq" > "$dev/driver_override"
    echo "$DEV_NAME" > /sys/bus/platform/drivers_probe
fi

# ---- step 3: u-dma-buf ----
if [[ -d /sys/class/u-dma-buf/udmabuf0 || -c /dev/udmabuf0 || -d /sys/devices/virtual/u-dma-buf/udmabuf0 ]]; then
    echo "  udmabuf0 already present; skipping."
else
    echo "  Loading u-dma-buf (udmabuf0=${UDMABUF_SIZE})..."
    # Try an in-tree module first, then the kernel-matched module bundled with
    # this package. BOARD_SETUP_DIR remains a compatibility fallback for older
    # board deployments that keep the module outside the package.
    if modprobe u-dma-buf udmabuf0="$UDMABUF_SIZE" 2>/dev/null || modprobe u_dma_buf udmabuf0="$UDMABUF_SIZE" 2>/dev/null; then
        true
    elif [[ -f "$minitpu_dir/u-dma-buf.ko" ]]; then
        insmod "$minitpu_dir/u-dma-buf.ko" "udmabuf0=$UDMABUF_SIZE"
    elif [[ -f "$minitpu_dir/u_dma_buf.ko" ]]; then
        insmod "$minitpu_dir/u_dma_buf.ko" "udmabuf0=$UDMABUF_SIZE"
    elif [[ -f "$BOARD_SETUP_DIR/u-dma-buf.ko" ]]; then
        insmod "$BOARD_SETUP_DIR/u-dma-buf.ko" "udmabuf0=$UDMABUF_SIZE"
    elif [[ -f "$BOARD_SETUP_DIR/u_dma_buf.ko" ]]; then
        insmod "$BOARD_SETUP_DIR/u_dma_buf.ko" "udmabuf0=$UDMABUF_SIZE"
    else
        echo "ERROR: u-dma-buf module not found." >&2
        echo "  The package should include minitpu/u-dma-buf.ko for this kernel." >&2
        echo "  Alternatively install it (modprobe u-dma-buf) or set BOARD_SETUP_DIR" >&2
        exit 1
    fi
fi

# ---- step 4: verify ----
echo ""
echo "=== Verify ==="
found=""
for u in /sys/class/uio/uio*; do
    [[ -e "$u/name" ]] || continue
    n=$(cat "$u/name")
    echo "  $(basename "$u"): name=$n"
    [[ "$n" == "$UIO_NAME" ]] && found="$u"
done
if [[ -z "$found" ]]; then
    echo "ERROR: no UIO device named '$UIO_NAME'" >&2
    exit 1
fi

if [[ -d /sys/class/u-dma-buf/udmabuf0 ]]; then
    phys=$(cat /sys/class/u-dma-buf/udmabuf0/phys_addr)
    size=$(cat /sys/class/u-dma-buf/udmabuf0/size)
    echo "  udmabuf0: phys=$phys size=$size"
else
    echo "ERROR: /sys/class/u-dma-buf/udmabuf0 not found" >&2
    exit 1
fi

# `make minitpu` needs root to configure the FPGA, but the programs that launch
# should not need root merely to mmap their two device nodes. Give the user who
# invoked sudo access to this boot's selected nodes; device-node ownership is
# reset at reboot, so this does not create a persistent system-wide policy.
if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
    uio_dev="/dev/$(basename "$found")"
    echo "  Granting $SUDO_USER access to $uio_dev and /dev/udmabuf0"
    chown root:"$SUDO_USER" "$uio_dev" /dev/udmabuf0
    chmod 660 "$uio_dev" /dev/udmabuf0
    for sync_node in /sys/class/u-dma-buf/udmabuf0/sync_for_device \
                     /sys/class/u-dma-buf/udmabuf0/sync_for_cpu \
                     /sys/class/u-dma-buf/udmabuf0/sync_offset \
                     /sys/class/u-dma-buf/udmabuf0/sync_size; do
        [[ -e "$sync_node" ]] || continue
        chown root:"$SUDO_USER" "$sync_node"
        chmod 660 "$sync_node"
    done
fi

echo ""
echo "Provisioning OK."
echo "  UIO '$UIO_NAME' at $(basename "$found") (/dev/$(basename "$found"))"
echo "  udmabuf0 at $phys ($size bytes)"
