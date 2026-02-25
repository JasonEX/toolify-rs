#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${1:-nonstream}" # nonstream | stream | all

(
  cd "$ROOT_DIR"
  UPSTREAM_MODE="alias_group" \
  BENCH_MODEL="smart" \
  BENCH_LABEL_PREFIX="alias_remap" \
  scripts/bench_forwarding_wrk.sh "$MODE"
)
