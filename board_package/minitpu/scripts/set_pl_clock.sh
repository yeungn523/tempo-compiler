#!/usr/bin/env bash
# Set the PL0 clock the fabric actually runs at.  Runs ON the ZCU104, as root.
#
# The bitstream closes timing at 200 MHz, but the fabric was measured running at
# 99.999 MHz.  The reason is that PL0_REF_CTRL is programmed by the FSBL inside
# BOOT.BIN -- dated May 2025, older than any bitstream here -- and loading a
# bitstream through fpga_manager does not touch it.  The 200 MHz in
# fpga/boards/_ps_common.tcl only reaches hardware through an FSBL built from the
# same .xsa, which is not what boots these boards.
#
# So the clock is a runtime register, and this sets it.  The write is VOLATILE:
# a reboot returns the board to whatever BOOT.BIN's FSBL programs.
#
#   sudo minitpu/scripts/set_pl_clock.sh 187        # then reprogram
#   sudo make -C board_package minitpu
#
# ALWAYS reprogram after changing it.  The fabric is not designed to survive a
# clock change underneath itself; programming holds it in reset and starts it
# cleanly on the new clock.  Then re-run `python3 board_checks.py` -- 16/16 at
# 187 MHz with accuracy identical to 100 MHz is the evidence this is safe, and it
# is evidence for THIS bitstream only.
set -euo pipefail

PL0_REF_CTRL=0xFF5E00C0
IOPLL_CTRL=0xFF5E0020
CRL_WPROT0=0xFF5E001C
PS_REF_MHZ=33.333

#: Timing closed at 200 MHz with WNS +0.049 ns -- 1 % margin, which is not enough
#: to run there across temperature.  Refuse past a deliberate ceiling.
MAX_MHZ=190

[[ $EUID -eq 0 ]] || { echo "ERROR: run as root" >&2; exit 1; }
want=${1:?usage: set_pl_clock.sh <target MHz>}

wprot=$(devmem "$CRL_WPROT0" 32)
[[ $((wprot)) -eq 0 ]] || {
    echo "ERROR: CRL_APB is write protected (WPROT0=$wprot)" >&2; exit 1; }

iopll=$(devmem "$IOPLL_CTRL" 32)
fbdiv=$(( (iopll >> 8) & 0x7F ))
div2=$(( (iopll >> 16) & 0x1 ))
pll=$(awk -v r="$PS_REF_MHZ" -v f="$fbdiv" -v d="$div2" 'BEGIN{printf "%.3f", r*f/(d?2:1)}')

# Refuse the REQUEST past the ceiling, not the achievable rate. The divisor search
# below always lands on the nearest reachable frequency, so checking its output
# would silently hand back 187 MHz to someone who asked for 200 and thought they
# got it.
over=$(awk -v w="$want" -v m="$MAX_MHZ" 'BEGIN{print (w > m) ? 1 : 0}')
[[ $over -eq 0 ]] || {
    echo "ERROR: refusing ${want} MHz. Timing closed at 200 MHz with WNS +0.049 ns," >&2
    echo "       which is 1 % margin -- not enough to run there across temperature." >&2
    echo "       This script stops at ${MAX_MHZ} MHz. Raise MAX_MHZ if you mean it." >&2
    exit 1; }

best=$(awk -v pll="$pll" -v want="$want" 'BEGIN{
    bd=0; berr=1e9
    for (d = 1; d <= 63; d++) { e = (pll/d) - want; if (e < 0) e = -e
        if (e < berr) { berr = e; bd = d } }
    printf "%d", bd }')
got=$(awk -v pll="$pll" -v d="$best" 'BEGIN{printf "%.3f", pll/d}')

current=$(devmem "$PL0_REF_CTRL" 32)
updated=$(printf "0x%08X" $(( (current & ~(0x3F << 8)) | (best << 8) )))

echo "IOPLL FBDIV=${fbdiv} DIV2=${div2} -> ${pll} MHz"
echo "PL0_REF_CTRL ${current} -> ${updated}  (DIVISOR0=${best})"
echo "PL0 ${want} MHz requested, ${got} MHz achievable"
devmem "$PL0_REF_CTRL" 32 "$updated"
echo "written: $(devmem "$PL0_REF_CTRL" 32)"
echo
echo "NOW REPROGRAM:  sudo make -C board_package minitpu"
echo "then verify:    python3 board_package/board_checks.py"
