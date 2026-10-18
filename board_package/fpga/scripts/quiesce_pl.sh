#!/usr/bin/env bash
# Detach Linux drivers for the boot-time PL image before replacing that image.
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "ERROR: run as root" >&2; exit 1; }

# Remove a previous minitpu overlay first, so its UIO platform device cannot
# survive across reprogramming with a different address map.
overlay=/sys/kernel/config/device-tree/overlays/minitpu
if [[ -d "$overlay" ]]; then
    echo "Removing previous minitpu device-tree overlay..."
    rmdir "$overlay"
fi

echo "Quiescing drivers bound to the boot-time PL image..."
for dev in /sys/bus/platform/devices/*; do
    [[ -e "$dev/of_node" && -e "$dev/driver" ]] || continue
    of_path=$(readlink -f "$dev/of_node" 2>/dev/null || true)
    [[ "$of_path" == */amba_pl/* ]] || continue

    driver=$(readlink -f "$dev/driver")
    name=$(basename "$dev")
    if [[ -w "$driver/unbind" ]]; then
        echo "  unbind $name from $(basename "$driver")"
        echo "$name" > "$driver/unbind"
    else
        # The boot image's AXI interrupt controller is built in and exposes no
        # unbind hook. Its range is 0x8000_0000, so the MINITPU control window is
        # deliberately placed at the non-conflicting 0x9000_0000 aperture.
        echo "  keep $name bound to $(basename "$driver") (no unbind hook)"
    fi
done

sync
echo "Boot-time PL clients quiesced."
