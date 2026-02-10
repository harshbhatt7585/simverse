#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup_uv.sh                 # install base + dev
#   ./scripts/setup_uv.sh base            # install base only
#   ./scripts/setup_uv.sh dev             # install base + dev
#   ./scripts/setup_uv.sh pettingzoo      # install base + pettingzoo
#   ./scripts/setup_uv.sh all             # install base + dev + pettingzoo

MODE="${1:-dev}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  echo "Usage: ./scripts/setup_uv.sh [base|dev|pettingzoo|all]"
  exit 0
fi

case "$MODE" in
  base)
    uv sync
    ;;
  dev)
    uv sync --extra dev
    ;;
  pettingzoo)
    uv sync --extra pettingzoo
    ;;
  all)
    uv sync --extra dev --extra pettingzoo
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Valid modes: base | dev | pettingzoo | all"
    exit 1
    ;;
esac

echo
echo "Environment is ready."
echo "Activate it with: source .venv/bin/activate"
