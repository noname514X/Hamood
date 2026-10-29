#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOOKS_DIR="${1:-"$SCRIPT_DIR/books"}"
PORT="${PORT:-8000}"
SERVER_PORT="${SERVER_PORT:-}"
SOURCE_DIR="$SCRIPT_DIR/.jupyter-book-generated/books-directory"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./book_start.sh [BOOKS_DIR]

Generate one aggregate Jupyter Book from BOOKS_DIR and preview it with
jupyter-book start.

Environment:
  PORT         Browser-facing port. Default: 8000
  SERVER_PORT Jupyter Book content-server port. Optional.
EOF
  exit 0
fi

if [[ ! -d "$BOOKS_DIR" ]]; then
  echo "Error: books directory not found: $BOOKS_DIR" >&2
  exit 1
fi

if command -v jupyter-book >/dev/null 2>&1; then
  JB_CMD=(jupyter-book)
elif python3 -c 'import jupyter_book' >/dev/null 2>&1; then
  JB_CMD=(python3 -m jupyter_book)
else
  echo "Error: jupyter-book is not installed." >&2
  echo "Install it with: python3 -m pip install -U jupyter-book" >&2
  exit 1
fi

"$SCRIPT_DIR/book_generate.sh" "$BOOKS_DIR" "$SCRIPT_DIR/_build/books" source

echo "Serving aggregate Jupyter Book source from: $SOURCE_DIR"
echo "Open: http://127.0.0.1:$PORT"
echo "Press Ctrl+C to stop the server."

cd "$SOURCE_DIR"

if [[ -n "$SERVER_PORT" ]]; then
  exec "${JB_CMD[@]}" start --port "$PORT" --server-port "$SERVER_PORT"
else
  exec "${JB_CMD[@]}" start --port "$PORT"
fi
