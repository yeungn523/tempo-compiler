#!/usr/bin/env bash
# Deploy the board package and program the PL, with the copies gated on md5.
#
#   tools/deploy_board.sh <account> <board-id> [--program]
#
# The account is the caller's own login on the board, which the board authorises
# a key for. It is not baked in, because each board carries its own accounts.
#
# The md5 gate is not ceremony: `tar | ssh` once reported success while writing
# zero-byte files, and every board check passed against the stale copies that were
# already there. Nothing is trusted to have arrived until its digest matches.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCOUNT="${1:?usage: deploy_board.sh <account> <board-id> [--program]}"
IP="132.236.59.${2:?usage: deploy_board.sh <account> <board-id> [--program]}"
PROGRAM="${3:-}"
REMOTE=/home/$ACCOUNT/board_package

# Where a repository path lands on the board. Package files keep their tree so
# `minitpu_runtime/runtime/launch.py` stays importable; a tool is a flat script.
remote_of() {
  case "$1" in
    board_package/*) printf '%s\n' "${1#board_package/}" ;;
    *) basename "$1" ;;
  esac
}

mapfile -t FILES < <(cd "$HERE" && ls board_package/*.py board_package/minitpu_runtime/**/*.py 2>/dev/null)
FILES+=("board_package/minitpu/minitpu_zcu104.bit.bin")
# The Makefile is how the board is driven, so it ships under the same gate as the
# sources rather than leaving the board on whatever copy it last received.
FILES+=("board_package/Makefile" "board_package/README.md")

# The scripts that program the board, and the overlay and module they load, go
# over the gate too and keep their subdirectory, so `sudo make minitpu` never runs
# a stale copy.
while IFS= read -r f; do FILES+=("$f"); done < <(
  cd "$HERE" && ls board_package/minitpu/scripts/* board_package/minitpu/overlay.dtso \
                   board_package/minitpu/u-dma-buf.ko 2>/dev/null)

# The compiled images the layer drivers launch. The board carries no
# compiler, so tools/build_gpt2_images.py writes them here and they travel as data.
while IFS= read -r f; do FILES+=("$f"); done < <(
  cd "$HERE" && ls board_package/images/* 2>/dev/null)

# GPT-2's weights and tokenizer, which tools/export_gpt2_weights.py writes here
# on a host. A board that never ran the export carries none.
for f in board_package/gpt2_weights.npz board_package/gpt2_vocab.json board_package/gpt2_merges.txt; do
  [ -f "$HERE/$f" ] && FILES+=("$f")
done

# scp creates no directories, and a board erases the home directory once no
# session holds it, so every directory a copy lands in is made first.
DIRS=()
for f in "${FILES[@]}"; do DIRS+=("$REMOTE/$(dirname "$(remote_of "$f")")"); done
ssh -o BatchMode=yes "$ACCOUNT@$IP" "mkdir -p $(printf '%s\n' "${DIRS[@]}" | sort -u | tr '\n' ' ')"

echo "deploying $((${#FILES[@]})) files to $IP"
for f in "${FILES[@]}"; do
  scp -q -o BatchMode=yes "$HERE/$f" "$ACCOUNT@$IP:$REMOTE/$(remote_of "$f")"
done

echo "verifying"
FAIL=0
for f in "${FILES[@]}"; do
  want=$(md5sum "$HERE/$f" | cut -d' ' -f1)
  got=$(ssh -o BatchMode=yes "$ACCOUNT@$IP" "md5sum $REMOTE/$(remote_of "$f") 2>/dev/null | cut -d' ' -f1")
  if [ "$want" != "$got" ]; then echo "  MISMATCH $(remote_of "$f")"; FAIL=1; fi
done
[ "$FAIL" -eq 0 ] || { echo "deploy FAILED"; exit 1; }
echo "all digests match"

if [ "$PROGRAM" = "--program" ]; then
  # Only programming needs the password. Reading it up front made a key-authenticated
  # deploy fail on a missing board.cred it never uses.
  [ -f "$HERE/board.cred" ] || { echo "--program needs $HERE/board.cred"; exit 1; }
  # The value is delimited in board.cred; strip the wrapper rather than the last
  # character, which is how this was misread as a rotated password for a whole day.
  PW="$(awk 'NR==2 {print $NF}' "$HERE/board.cred" | tr -d '`"'"'")"

  echo "programming the PL"
  printf '%s\n' "$PW" | ssh -o BatchMode=yes "$ACCOUNT@$IP" \
    "cd $REMOTE && sudo -S -p '' ./minitpu/scripts/program_pl.sh" 2>&1 | tail -4
fi
