#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_DIR="${OUT_DIR:-$ROOT_DIR/artifacts/perf}"
mkdir -p "$OUT_DIR"

PINNED_BASELINE_FILE="${PINNED_BASELINE_FILE:-$ROOT_DIR/artifacts/perf/baseline_single_core_1x8_pinned.md}"
UNPINNED_BASELINE_FILE="${UNPINNED_BASELINE_FILE:-$PINNED_BASELINE_FILE}"

PINNED_ROUNDS="${PINNED_ROUNDS:-9}"
PINNED_DURATION="${PINNED_DURATION:-10s}"
UNPINNED_ROUNDS="${UNPINNED_ROUNDS:-5}"
UNPINNED_DURATION="${UNPINNED_DURATION:-$PINNED_DURATION}"

PINNED_AUTO_PIN_CORES="${PINNED_AUTO_PIN_CORES:-1}"
UNPINNED_AUTO_PIN_CORES="${UNPINNED_AUTO_PIN_CORES:-0}"
INCLUDE_ALIAS_REMAP_SCENARIO="${INCLUDE_ALIAS_REMAP_SCENARIO:-0}"
SKIP_UNPINNED_OBSERVE="${SKIP_UNPINNED_OBSERVE:-0}" # 0 | 1

echo "[dual-gate] pinned hard gate start"
AUTO_PIN_CORES="$PINNED_AUTO_PIN_CORES" \
BASELINE_FILE="$PINNED_BASELINE_FILE" \
ROUNDS="$PINNED_ROUNDS" \
DURATION="$PINNED_DURATION" \
INCLUDE_ALIAS_REMAP_SCENARIO="$INCLUDE_ALIAS_REMAP_SCENARIO" \
scripts/perf_zero_regression_gate.sh

cp "$OUT_DIR/zero_regression_gate_latest.md" \
  "$OUT_DIR/zero_regression_gate_pinned_latest.md"

if [[ "$SKIP_UNPINNED_OBSERVE" != "1" ]]; then
  echo "[dual-gate] unpinned observe gate start (non-blocking)"
  set +e
  AUTO_PIN_CORES="$UNPINNED_AUTO_PIN_CORES" \
  BASELINE_FILE="$UNPINNED_BASELINE_FILE" \
  ROUNDS="$UNPINNED_ROUNDS" \
  DURATION="$UNPINNED_DURATION" \
  INCLUDE_ALIAS_REMAP_SCENARIO="$INCLUDE_ALIAS_REMAP_SCENARIO" \
  scripts/perf_zero_regression_gate.sh
  unpinned_rc=$?
  set -e

  cp "$OUT_DIR/zero_regression_gate_latest.md" \
    "$OUT_DIR/zero_regression_gate_unpinned_latest.md"
else
  unpinned_rc=0
fi

{
  echo "# Dual Perf Gate"
  echo
  echo "- Date: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  echo "- Pinned baseline: \`$PINNED_BASELINE_FILE\`"
  echo "- Unpinned baseline: \`$UNPINNED_BASELINE_FILE\`"
  echo "- Pinned rounds/duration: $PINNED_ROUNDS / $PINNED_DURATION"
  echo "- Unpinned rounds/duration: $UNPINNED_ROUNDS / $UNPINNED_DURATION"
  echo "- Pinned output: \`artifacts/perf/zero_regression_gate_pinned_latest.md\`"
  if [[ "$SKIP_UNPINNED_OBSERVE" == "1" ]]; then
    echo "- Unpinned output: skipped"
  else
    echo "- Unpinned output: \`artifacts/perf/zero_regression_gate_unpinned_latest.md\`"
    if [[ "$unpinned_rc" -eq 0 ]]; then
      echo "- Unpinned verdict: PASS (observe)"
    else
      echo "- Unpinned verdict: FAIL (observe-only, non-blocking)"
    fi
  fi
  echo
  echo "## Contract"
  echo
  echo "- Pinned gate is blocking."
  echo "- Unpinned gate is observational and never blocks merge."
} > "$OUT_DIR/zero_regression_dual_gate_latest.md"

cat "$OUT_DIR/zero_regression_dual_gate_latest.md"
