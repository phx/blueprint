#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-../.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python"
fi
PYTEST=("$PYTHON" -m pytest)

case "${1:---all}" in
  --all)
    "$PYTHON" validate_all.py
    ;;
  --quick)
    "${PYTEST[@]}" -q --tb=short
    ;;
  --paper)
    "${PYTEST[@]}" --no-cov \
      tests/test_paper_validation_matrix.py \
      tests/test_documentation_traceability.py
    ;;
  --self-replication)
    "${PYTEST[@]}" --no-cov tests/test_self_replication.py
    ;;
  --help|-h)
    printf '%s\n' \
      "Usage: ./verify.sh [--all|--quick|--paper|--self-replication]" \
      "" \
      "  --all               Run full validation with active-core coverage." \
      "  --quick             Run pytest without coverage." \
      "  --paper             Run paper/data/documentation traceability checks." \
      "  --self-replication  Run W[F] -> F' tests only."
    ;;
  *)
    printf 'Unknown option: %s\n' "$1" >&2
    printf 'Run ./verify.sh --help for usage.\n' >&2
    exit 2
    ;;
esac
