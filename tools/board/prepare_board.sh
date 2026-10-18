#!/usr/bin/env bash
# Puts everything a board runs back on it: the package, the compiled images, and GPT-2's weights and tokenizer.
#
#   tools/board/prepare_board.sh <account> <board-id> [--program]
#
# A board erases the home directory whenever no session holds it, so this is the one command to run after
# every login. Hold a session open on the board while it runs, or the board erases the copies as they land.
# The export reuses its cached download and the image build is deterministic, so a rerun only costs the copy.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACCOUNT="${1:?usage: prepare_board.sh <account> <board-id> [--program]}"
BOARD="${2:?usage: prepare_board.sh <account> <board-id> [--program]}"
PYTHON="${PYTHON:-python3}"

echo "exporting GPT-2's weights and tokenizer"
"$PYTHON" "$HERE/tools/board/export_gpt2_weights.py"

echo "building the compiled images"
"$PYTHON" "$HERE/tools/board/build_gpt2_images.py" > /dev/null

"$HERE/tools/board/deploy_board.sh" "$ACCOUNT" "$BOARD" "${3:-}"
