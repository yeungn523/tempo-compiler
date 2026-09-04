#!/usr/bin/env bash
# Load the PL bitstream. Runs ON the ZCU104, as root.
#
# fpga_manager wants the bootgen-wrapped .bit.bin, not the raw .bit; the .bin
# is pre-generated in minitpu/ so the board does not need bootgen.
set -euo pipefail

minitpu_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin="$minitpu_dir/minitpu_zcu104.bit.bin"
name=minitpu_zcu104.bit.bin

[[ -f "$bin" ]] || { echo "ERROR: missing $bin" >&2; exit 1; }
[[ $EUID -eq 0 ]] || { echo "ERROR: run as root" >&2; exit 1; }

# S_AXI_HP2_FPD is AFI_FM4 in the generated ZynqMP psu_init.  The server boot
# image uses a different PL design before this full bitstream is loaded.  Even
# after its Linux drivers are unbound, the hard-PS AFI can retain channel state
# across a fabric-only reconfiguration; the first new CDMA read can then wedge
# the PS memory path.  Hold only HP2 in reset while replacing the fabric and
# release it after the new SmartConnect/CDMA logic is initialized.
#
# CRF_APB.RST_FPD_TOP[11] = AFI_FM4_RESET (0xFD1A0100, mask 0x00000800),
# confirmed by the psu_init generated from this ZCU104 hardware design.
afi_reg=0xFD1A0100
afi_mask=$((0x00000800))
afi_before=""
afi_reset_held=0

restore_afi_hp2() {
    [[ $afi_reset_held -eq 1 ]] || return 0
    current=$(devmem "$afi_reg" 32)
    restored=$(( (current & ~afi_mask) | (afi_before & afi_mask) ))
    devmem "$afi_reg" 32 "$restored" >/dev/null
    readback=$(( $(devmem "$afi_reg" 32) ))
    if (( (readback & afi_mask) != (afi_before & afi_mask) )); then
        echo "ERROR: failed to restore S_AXI_HP2_FPD reset state" >&2
        return 1
    fi
    afi_reset_held=0
    echo "  S_AXI_HP2_FPD reset released."
}

command -v devmem >/dev/null 2>&1 || {
    echo "ERROR: devmem is required to reset S_AXI_HP2_FPD safely" >&2
    exit 1
}
afi_before=$(( $(devmem "$afi_reg" 32) ))
if (( afi_before & afi_mask )); then
    echo "ERROR: S_AXI_HP2_FPD was already held in reset before programming" >&2
    exit 1
fi
devmem "$afi_reg" 32 "$((afi_before | afi_mask))" >/dev/null
afi_asserted=$(( $(devmem "$afi_reg" 32) ))
if (( !(afi_asserted & afi_mask) )); then
    echo "ERROR: failed to assert S_AXI_HP2_FPD reset" >&2
    exit 1
fi
afi_reset_held=1
trap restore_afi_hp2 EXIT INT TERM
echo "  S_AXI_HP2_FPD held in reset during PL replacement."

# Copy to /lib/firmware (fpga_manager loads from there).
# A RAM rootfs loses /lib/firmware across reboots, so always re-copy.
dst="/lib/firmware/$name"
mkdir -p /lib/firmware
# Compare by inode rather than by resolved path: realpath answers "do these
# names spell the same place", cp asks "is this one file", and only the second
# question is the one that matters here.
#
# This does not silence every case. On .193, cp still reports source and
# destination as the same file even though /lib/firmware/minitpu_zcu104.bit.bin
# does not exist before or after the run -- fpgautil appears to stage and remove
# it -- so neither realpath nor stat can see what cp sees. That message is
# cosmetic: fpgautil reports success, fpga_manager reaches "operating", and the
# board passes all ten checks. What the inode test does buy is safety on a host
# where the two really are one file: busybox cp exits 0 when it refuses, so
# `set -e` lets it pass, but GNU cp exits 1 and would abort programming for no
# reason at all.
same_file() {
    local a b
    a=$(stat -c '%d:%i' "$1" 2>/dev/null) || return 1
    b=$(stat -c '%d:%i' "$2" 2>/dev/null) || return 1
    [[ "$a" == "$b" ]]
}
if same_file "$bin" "$dst"; then
    echo "  $dst is the package's own copy; not replacing it."
else
    cp -f "$bin" "$dst"
fi

if command -v fpgautil >/dev/null 2>&1; then
    echo "Programming PL via fpgautil..."
    fpgautil -b "/lib/firmware/$name" -f Full
elif [[ -d /sys/class/fpga_manager/fpga0 ]]; then
    echo "Programming PL via fpga_manager sysfs..."
    echo 0 > /sys/class/fpga_manager/fpga0/flags
    echo "$name" > /sys/class/fpga_manager/fpga0/firmware
else
    echo "ERROR: neither fpgautil nor /sys/class/fpga_manager/fpga0 found" >&2
    exit 1
fi

state=$(cat /sys/class/fpga_manager/fpga0/state 2>/dev/null || echo unknown)
echo "  fpga_manager state: $state"
[[ "$state" == "operating" ]] || { echo "ERROR: PL not operating" >&2; exit 1; }
restore_afi_hp2
trap - EXIT INT TERM
echo "PL programmed: $name"
