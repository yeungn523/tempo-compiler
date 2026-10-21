
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACCOUNT="${1:?usage: prepare_board.sh <account> <board-id> [--program]}"
BOARD="${2:?usage: prepare_board.sh <account> <board-id> [--program]}"
PYTHON="${PYTHON:-python3}"
PROGRAM=""
for flag in "${@:3}"; do
  case "$flag" in
    --program) PROGRAM="--program" ;;
    *) echo "unknown flag $flag, expected --program"; exit 1 ;;
  esac
done

echo "exporting GPT-2's weights and tokenizer"
"$PYTHON" "$HERE/tools/board/export_gpt2_weights.py"

echo "building the compiled images"
"$PYTHON" "$HERE/tools/board/build_images.py" > /dev/null

"$HERE/tools/board/deploy_board.sh" "$ACCOUNT" "$BOARD" "$PROGRAM"
