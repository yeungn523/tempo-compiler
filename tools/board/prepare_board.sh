
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACCOUNT="${1:?usage: prepare_board.sh <account> <board-id> [--program] [--mlir]}"
BOARD="${2:?usage: prepare_board.sh <account> <board-id> [--program] [--mlir]}"
PYTHON="${PYTHON:-python3}"
PROGRAM=""
MLIR=""
for flag in "${@:3}"; do
  case "$flag" in
    --program) PROGRAM="--program" ;;
    --mlir) MLIR="--mlir" ;;
    *) echo "unknown flag $flag, expected --program or --mlir"; exit 1 ;;
  esac
done

echo "exporting GPT-2's weights and tokenizer"
"$PYTHON" "$HERE/tools/board/export_gpt2_weights.py"

echo "building the compiled images"
"$PYTHON" "$HERE/tools/board/build_gpt2_images.py" > /dev/null
if [ -n "$MLIR" ]; then
  echo "rebuilding the images mlir/ covers"
  "$PYTHON" "$HERE/tools/board/build_mlir_images.py" --output "$HERE/board_package/images"
fi

"$HERE/tools/board/deploy_board.sh" "$ACCOUNT" "$BOARD" "$PROGRAM"
