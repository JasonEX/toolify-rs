#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLIFY_BIN="${TOOLIFY_BIN:-$ROOT_DIR/target/release/toolify}"
MOCK_UPSTREAM_BIN="${MOCK_UPSTREAM_BIN:-$ROOT_DIR/target/release/mock_openai_upstream}"
MOCK_UPSTREAM_MANIFEST="${MOCK_UPSTREAM_MANIFEST:-$ROOT_DIR/tools/mock-openai-upstream/Cargo.toml}"

REQS="${REQS:-2000}"
CONCURRENCY="${CONCURRENCY:-16}"
PROXY_PORT="${PROXY_PORT:-18080}"
UPSTREAM_PORT="${UPSTREAM_PORT:-19001}"
WORKER_THREADS="${WORKER_THREADS:-auto}"
MODE="${1:-all}" # all | nonstream | stream
UPSTREAM_TRANSPORT="${UPSTREAM_TRANSPORT:-h2c}" # auto | h2c
MOCK_SCENARIO="${MOCK_SCENARIO:-text}" # text | full | error
REQUIRE_UPSTREAM_H2="${REQUIRE_UPSTREAM_H2:-1}"

case "$UPSTREAM_TRANSPORT" in
  auto|h2c) ;;
  *)
    echo "unknown UPSTREAM_TRANSPORT: $UPSTREAM_TRANSPORT (use auto|h2c)"
    exit 1
    ;;
esac

case "$MOCK_SCENARIO" in
  text|full|error) ;;
  *)
    echo "unknown MOCK_SCENARIO: $MOCK_SCENARIO (use text|full|error)"
    exit 1
    ;;
esac

FORCE_H2C_UPSTREAM=false
if [[ "$UPSTREAM_TRANSPORT" == "h2c" ]]; then
  FORCE_H2C_UPSTREAM=true
fi

if [[ ! -x "$TOOLIFY_BIN" ]]; then
  echo "toolify binary not found: $TOOLIFY_BIN"
  echo "build it first: cargo build --release"
  exit 1
fi

if [[ ! -x "$MOCK_UPSTREAM_BIN" ]]; then
  echo "mock upstream binary not found: $MOCK_UPSTREAM_BIN"
  echo "build it first: cargo build --release --manifest-path \"$MOCK_UPSTREAM_MANIFEST\" --target-dir \"$ROOT_DIR/target\""
  exit 1
fi

TMP_DIR="$(mktemp -d)"
CFG_FILE="$TMP_DIR/config.yaml"
UPSTREAM_LOG="$TMP_DIR/upstream.log"
PROXY_LOG="$TMP_DIR/proxy.log"
CONFIG_BAK="$TMP_DIR/config.yaml.bak"
if [[ -f "$ROOT_DIR/config.yaml" ]]; then
  cp "$ROOT_DIR/config.yaml" "$CONFIG_BAK"
fi

cleanup() {
  pkill -P $$ >/dev/null 2>&1 || true
  if [[ -f "$CONFIG_BAK" ]]; then
    cp "$CONFIG_BAK" "$ROOT_DIR/config.yaml"
  else
    rm -f "$ROOT_DIR/config.yaml"
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

make_config() {
  cat >"$CFG_FILE" <<EOF
server:
  port: $PROXY_PORT
  host: "127.0.0.1"
  timeout: 30
  http_use_env_proxy: false
  http_force_h2c_upstream: $FORCE_H2C_UPSTREAM
EOF
  if [[ "$WORKER_THREADS" != "auto" ]]; then
    cat >>"$CFG_FILE" <<EOF
  runtime_worker_threads: $WORKER_THREADS
  runtime_max_blocking_threads: 8
EOF
  else
    cat >>"$CFG_FILE" <<EOF
  runtime_worker_threads: null
  runtime_max_blocking_threads: null
EOF
  fi
  cat >>"$CFG_FILE" <<EOF
upstream_services:
  - name: "mock-openai"
    provider: "openai"
    base_url: "http://127.0.0.1:$UPSTREAM_PORT/v1"
    api_key: "sk-upstream"
    models:
      - "m1"
    is_default: true
client_authentication:
  allowed_keys:
    - "sk-client"
features:
  enable_function_calling: true
  log_level: "DISABLED"
  convert_developer_to_system: true
  enable_fc_error_retry: false
  fc_error_retry_max_attempts: 3
EOF
}

start_mock() {
  local mode="$1"
  MOCK_MODE="$mode" MOCK_TRANSPORT="$UPSTREAM_TRANSPORT" MOCK_SCENARIO="$MOCK_SCENARIO" \
    UPSTREAM_PORT="$UPSTREAM_PORT" exec "$MOCK_UPSTREAM_BIN" >>"$UPSTREAM_LOG" 2>&1
}

run_load() {
  local stream="$1"
  local label="$2"

  local payload
  if [[ "$stream" == "true" ]]; then
    payload='{"model":"m1","messages":[{"role":"user","content":"hi"}],"stream":true}'
  else
    payload='{"model":"m1","messages":[{"role":"user","content":"hi"}]}'
  fi

  local start_ms end_ms wall_ms
  start_ms="$(date +%s%3N)"
  seq 1 "$REQS" | xargs -P"$CONCURRENCY" -I{} \
    curl -s -o /dev/null --noproxy '*' --max-time 2 \
      -H "Authorization: Bearer sk-client" \
      -H "Content-Type: application/json" \
      -d "$payload" \
      "http://127.0.0.1:$PROXY_PORT/v1/chat/completions"
  end_ms="$(date +%s%3N)"
  wall_ms="$((end_ms - start_ms))"
  awk -v reqs="$REQS" -v ms="$wall_ms" -v label="$label" \
    'BEGIN { printf "%s wall_ms=%d rps=%.2f\n", label, ms, (reqs * 1000.0 / ms) }'
}

sample_pid_stats() {
  local pid="$1"
  local sample_file="$2"
  while kill -0 "$pid" 2>/dev/null; do
    local ts rss stat_line ut st
    ts="$(date +%s%3N)"
    rss="$(awk "/VmRSS:/{print \$2+0}" "/proc/$pid/status" 2>/dev/null || echo 0)"
    stat_line="$(cat "/proc/$pid/stat" 2>/dev/null || true)"
    if [[ -n "$stat_line" ]]; then
      ut="$(awk '{print $14}' <<<"$stat_line")"
      st="$(awk '{print $15}' <<<"$stat_line")"
      echo "$ts $rss $ut $st" >>"$sample_file"
    fi
    sleep 0.05
  done
}

assert_upstream_h2() {
  local label="$1"
  if [[ "$REQUIRE_UPSTREAM_H2" != "1" ]]; then
    return 0
  fi
  local stats_json h1 h2 stats_url
  stats_url="http://127.0.0.1:$UPSTREAM_PORT/_mock/stats"
  if [[ "$UPSTREAM_TRANSPORT" == "h2c" ]]; then
    stats_json="$(curl -sS --noproxy '*' --http2-prior-knowledge --max-time 2 "$stats_url" || true)"
  else
    stats_json="$(curl -sS --noproxy '*' --max-time 2 "$stats_url" || true)"
  fi
  h1="$(printf '%s' "$stats_json" | sed -n 's/.*"h1":[[:space:]]*\([0-9][0-9]*\).*/\1/p')"
  h2="$(printf '%s' "$stats_json" | sed -n 's/.*"h2":[[:space:]]*\([0-9][0-9]*\).*/\1/p')"
  if [[ -z "$h1" || -z "$h2" ]]; then
    echo "$label failed to parse upstream stats JSON: $stats_json"
    sed -n '1,120p' "$UPSTREAM_LOG"
    exit 1
  fi
  if (( h2 <= 0 || h1 != 0 )); then
    echo "$label upstream protocol assertion failed: expected pure h2 traffic, got h2=$h2 h1=$h1"
    sed -n '1,120p' "$UPSTREAM_LOG"
    exit 1
  fi
  echo "$label upstream_proto_h2_ok h2=$h2 h1=$h1"
}

run_case() {
  local stream="$1"
  local label="$2"
  local sample_file="$TMP_DIR/${label}_sample.txt"

  : >"$UPSTREAM_LOG"
  : >"$PROXY_LOG"
  : >"$sample_file"

  if [[ "$stream" == "true" ]]; then
    start_mock stream &
  else
    start_mock nonstream &
  fi
  local upstream_pid=$!

  sleep 0.3
  if ! kill -0 "$upstream_pid" 2>/dev/null; then
    echo "failed to start mock upstream"
    exit 1
  fi

  (cd "$ROOT_DIR" && cp "$CFG_FILE" config.yaml)
  (cd "$ROOT_DIR" && "$TOOLIFY_BIN") >"$PROXY_LOG" 2>&1 &
  local proxy_pid=$!

  sleep 0.8
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "failed to start toolify"
    sed -n '1,80p' "$PROXY_LOG"
    exit 1
  fi

  sample_pid_stats "$proxy_pid" "$sample_file" &
  local sampler_pid=$!

  run_load "$stream" "$label"
  assert_upstream_h2 "$label"

  kill -TERM "$proxy_pid" >/dev/null 2>&1 || true
  kill -TERM "$upstream_pid" >/dev/null 2>&1 || true
  wait "$proxy_pid" 2>/dev/null || true
  wait "$upstream_pid" 2>/dev/null || true
  wait "$sampler_pid" 2>/dev/null || true

  if [[ -s "$sample_file" ]]; then
    local peak_rss cpu_pct wall_s ticks cpu_ticks first_ticks last_ticks clk
    peak_rss="$(awk 'max<$2{max=$2} END{print max+0}' "$sample_file")"
    first_ticks="$(awk 'NR==1{print $3+$4}' "$sample_file")"
    last_ticks="$(awk 'END{print $3+$4}' "$sample_file")"
    cpu_ticks="$((last_ticks - first_ticks))"
    clk="$(getconf CLK_TCK)"
    wall_s="$(awk 'NR==1{s=$1} END{e=$1; printf "%.6f", (e-s)/1000.0}' "$sample_file")"
    cpu_pct="$(awk -v ticks="$cpu_ticks" -v clk="$clk" -v wall="$wall_s" \
      'BEGIN { if (wall > 0) printf "%.2f", (ticks/clk) * 100.0 / wall; else print "0.00" }')"
    echo "$label cpu_pct=$cpu_pct peak_rss_kb=$peak_rss"
  fi
}

make_config

case "$MODE" in
  nonstream)
    run_case false "forward_nonstream"
    ;;
  stream)
    run_case true "forward_stream"
    ;;
  all)
    run_case false "forward_nonstream"
    run_case true "forward_stream"
    ;;
  *)
    echo "unknown mode: $MODE (use all|nonstream|stream)"
    exit 1
    ;;
esac
