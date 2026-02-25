#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PGO_DIR="${PGO_DIR:-$ROOT_DIR/artifacts/pgo}"
MERGED_PROFDATA="${MERGED_PROFDATA:-$PGO_DIR/merged.profdata}"
MODE="${1:-}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/pgo_build.sh gen      # build release binary with profile generation
  scripts/pgo_build.sh profile  # run representative single-core workload to collect .profraw
  scripts/pgo_build.sh merge    # merge generated *.profraw into merged.profdata
  scripts/pgo_build.sh use      # build release binary using merged.profdata

Environment:
  PGO_DIR            profile output directory (default: artifacts/pgo)
  MERGED_PROFDATA    merged profile path (default: $PGO_DIR/merged.profdata)
  PROFILE_SCENARIOS  space-separated mock scenarios for profile mode (default: "text code full")
  PROFILE_ROUNDS     rounds per scenario in profile mode (default: 1)
  PROFILE_DURATION   benchmark duration per scenario in profile mode (default: 10s)
USAGE
}

find_llvm_profdata() {
  if command -v llvm-profdata >/dev/null 2>&1; then
    command -v llvm-profdata
    return 0
  fi

  local host
  host="$(rustc -vV | awk '/^host: / {print $2}')"
  if [[ -n "$host" ]]; then
    local candidate
    candidate="$(rustc --print sysroot)/lib/rustlib/$host/bin/llvm-profdata"
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  return 1
}

if [[ -z "$MODE" ]]; then
  usage
  exit 1
fi

mkdir -p "$PGO_DIR"

case "$MODE" in
  gen)
    echo "[pgo] generating profile-instrumented release build -> $PGO_DIR"
    (
      cd "$ROOT_DIR"
      # Only instrument toolify itself to keep merged profile focused on proxy hot paths.
      RUSTFLAGS="-Cprofile-generate=$PGO_DIR" cargo build --locked --release
      # Build benchmark mock upstream without instrumentation; it is workload generator, not PGO target.
      cargo build --locked --release --manifest-path "$ROOT_DIR/tools/mock-openai-upstream/Cargo.toml" --target-dir "$ROOT_DIR/target"
    )
    echo "[pgo] run your representative workload now, then execute:"
    echo "      scripts/pgo_build.sh profile   # optional one-command profile run"
    echo "      scripts/pgo_build.sh merge"
    echo "      scripts/pgo_build.sh use"
    ;;
  profile)
    profile_scenarios="${PROFILE_SCENARIOS:-text code full}"
    profile_rounds="${PROFILE_ROUNDS:-1}"
    profile_duration="${PROFILE_DURATION:-10s}"
    profile_upstream_transport="${PROFILE_UPSTREAM_TRANSPORT:-h2c}"
    profile_require_h2="${PROFILE_REQUIRE_UPSTREAM_H2:-1}"
    profile_include_alias_remap="${PROFILE_INCLUDE_ALIAS_REMAP_SCENARIO:-0}"
    profile_pin_proxy="${PIN_PROXY_CORE:-}"
    profile_pin_upstream="${PIN_UPSTREAM_CORE:-}"
    profile_pin_wrk="${PIN_WRK_CORE:-}"

    for scenario in $profile_scenarios; do
      case "$scenario" in
        text|code|full|error) ;;
        *)
          echo "invalid PROFILE_SCENARIOS entry: $scenario (use text|code|full|error)"
          exit 1
          ;;
      esac

      out_file="$ROOT_DIR/artifacts/perf/_pgo_profile_${scenario}_r${profile_rounds}.md"
      echo "[pgo] collecting profile workload scenario=$scenario rounds=$profile_rounds duration=$profile_duration -> $out_file"
      (
        cd "$ROOT_DIR"
        PIN_PROXY_CORE="$profile_pin_proxy" \
        PIN_UPSTREAM_CORE="$profile_pin_upstream" \
        PIN_WRK_CORE="$profile_pin_wrk" \
        ROUNDS="$profile_rounds" \
        DURATION="$profile_duration" \
        WRK_THREADS=1 \
        CONNECTIONS=8 \
        WORKER_THREADS=1 \
        AUTO_BUILD_RELEASE=0 \
        UPSTREAM_TRANSPORT="$profile_upstream_transport" \
        REQUIRE_UPSTREAM_H2="$profile_require_h2" \
        MOCK_SCENARIO="$scenario" \
        INCLUDE_ALIAS_REMAP_SCENARIO="$profile_include_alias_remap" \
        OUT_FILE="$out_file" \
        scripts/perf_recut_baseline.sh
      )
    done
    ;;
  merge)
    llvm_profdata="$(find_llvm_profdata || true)"
    if [[ -z "$llvm_profdata" ]]; then
      echo "llvm-profdata not found in PATH"
      exit 1
    fi
    shopt -s nullglob
    profraw_files=("$PGO_DIR"/*.profraw)
    shopt -u nullglob
    if [[ "${#profraw_files[@]}" -eq 0 ]]; then
      echo "no .profraw files found in $PGO_DIR"
      exit 1
    fi
    echo "[pgo] merging ${#profraw_files[@]} profile files -> $MERGED_PROFDATA"
    "$llvm_profdata" merge -o "$MERGED_PROFDATA" "${profraw_files[@]}"
    ;;
  use)
    if [[ ! -f "$MERGED_PROFDATA" ]]; then
      echo "merged profile not found: $MERGED_PROFDATA"
      exit 1
    fi
    echo "[pgo] building optimized release binary using $MERGED_PROFDATA"
    (
      cd "$ROOT_DIR"
      RUSTFLAGS="-Cprofile-use=$MERGED_PROFDATA -Cllvm-args=-pgo-warn-missing-function" cargo build --locked --release
    )
    ;;
  *)
    echo "unknown mode: $MODE"
    usage
    exit 1
    ;;
esac
