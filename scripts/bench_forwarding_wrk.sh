#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLIFY_BIN="${TOOLIFY_BIN:-$ROOT_DIR/target/release/toolify}"
MOCK_UPSTREAM_BIN="${MOCK_UPSTREAM_BIN:-$ROOT_DIR/target/release/mock_openai_upstream}"
MOCK_UPSTREAM_MANIFEST="${MOCK_UPSTREAM_MANIFEST:-$ROOT_DIR/tools/mock-openai-upstream/Cargo.toml}"

DURATION="${DURATION:-20s}"
WRK_THREADS="${WRK_THREADS:-2}"
CONNECTIONS="${CONNECTIONS:-16}"
WRK_TIMEOUT="${WRK_TIMEOUT:-2s}"
PROXY_PORT="${PROXY_PORT:-18080}"
UPSTREAM_PORT="${UPSTREAM_PORT:-19001}"
LOCK_FILE="${LOCK_FILE:-/tmp/toolify_forwarding_wrk.lock}"
GLOBAL_LOCK_FILE="${GLOBAL_LOCK_FILE:-/tmp/toolify_bench_global.lock}"
AUTO_BUILD_RELEASE="${AUTO_BUILD_RELEASE:-0}" # 0 | 1
WORKER_THREADS="${WORKER_THREADS:-auto}"
RUNTIME_THREAD_STACK_SIZE_KB="${RUNTIME_THREAD_STACK_SIZE_KB:-512}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-10}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-200}"
WARMUP_CONNECT_TIMEOUT_S="${WARMUP_CONNECT_TIMEOUT_S:-2}"
WARMUP_MAX_TIME_S="${WARMUP_MAX_TIME_S:-2}"
PIN_PROXY_CORE="${PIN_PROXY_CORE:-}"
PIN_UPSTREAM_CORE="${PIN_UPSTREAM_CORE:-}"
PIN_WRK_CORE="${PIN_WRK_CORE:-}"
AUTO_PIN_CORES="${AUTO_PIN_CORES:-0}" # 0 | 1
MODE="${1:-all}" # all | nonstream | stream
UPSTREAM_TRANSPORT="${UPSTREAM_TRANSPORT:-h2c}" # auto | h2c
MOCK_SCENARIO="${MOCK_SCENARIO:-text}" # text | code | full | error
REQUIRE_UPSTREAM_H2="${REQUIRE_UPSTREAM_H2:-1}"
UPSTREAM_MODE="${UPSTREAM_MODE:-single}" # single | alias_group
BENCH_MODEL="${BENCH_MODEL:-m1}"
BENCH_LABEL_PREFIX="${BENCH_LABEL_PREFIX:-forward}"

auto_pin_cores_if_unset() {
  if ! command -v taskset >/dev/null 2>&1; then
    return
  fi

  local cpu_count
  cpu_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 0)"
  if [[ "$cpu_count" -lt 2 ]]; then
    return
  fi

  if [[ -z "$PIN_PROXY_CORE" ]]; then
    PIN_PROXY_CORE=0
  fi
  if [[ -z "$PIN_UPSTREAM_CORE" ]]; then
    PIN_UPSTREAM_CORE=1
  fi
  if [[ -z "$PIN_WRK_CORE" ]]; then
    if [[ "$cpu_count" -ge 3 ]]; then
      PIN_WRK_CORE=2
    else
      PIN_WRK_CORE=0
    fi
  fi
}

if [[ "$AUTO_PIN_CORES" == "1" ]]; then
  auto_pin_cores_if_unset
fi
echo "[bench_forwarding_wrk] auto_pin_cores=$AUTO_PIN_CORES pin_proxy_core=${PIN_PROXY_CORE:-none} pin_upstream_core=${PIN_UPSTREAM_CORE:-none} pin_wrk_core=${PIN_WRK_CORE:-none}"

case "$UPSTREAM_TRANSPORT" in
  auto|h2c) ;;
  *)
    echo "unknown UPSTREAM_TRANSPORT: $UPSTREAM_TRANSPORT (use auto|h2c)"
    exit 1
    ;;
esac

case "$MOCK_SCENARIO" in
  text|code|full|error) ;;
  *)
    echo "unknown MOCK_SCENARIO: $MOCK_SCENARIO (use text|code|full|error)"
    exit 1
    ;;
esac

case "$UPSTREAM_MODE" in
  single|alias_group) ;;
  *)
    echo "unknown UPSTREAM_MODE: $UPSTREAM_MODE (use single|alias_group)"
    exit 1
    ;;
esac

FORCE_H2C_UPSTREAM=false
if [[ "$UPSTREAM_TRANSPORT" == "h2c" ]]; then
  FORCE_H2C_UPSTREAM=true
fi

if [[ "$AUTO_BUILD_RELEASE" == "1" ]]; then
  cargo build --release >/dev/null
  cargo build --release --manifest-path "$MOCK_UPSTREAM_MANIFEST" --target-dir "$ROOT_DIR/target" >/dev/null
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

if ! command -v wrk >/dev/null 2>&1; then
  echo "wrk not found in PATH"
  exit 1
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "another bench_forwarding_wrk run is active (lock: $LOCK_FILE)"
    exit 2
  fi
  exec 10>"$GLOBAL_LOCK_FILE"
  if ! flock -n 10; then
    echo "another toolify benchmark run is active (global lock: $GLOBAL_LOCK_FILE)"
    exit 2
  fi
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

run_maybe_pinned() {
  local core="$1"
  shift
  if [[ -n "$core" ]] && command -v taskset >/dev/null 2>&1; then
    taskset -c "$core" "$@"
  else
    "$@"
  fi
}

run_maybe_pinned_exec() {
  local core="$1"
  shift
  if [[ -n "$core" ]] && command -v taskset >/dev/null 2>&1; then
    exec taskset -c "$core" "$@"
  else
    exec "$@"
  fi
}

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
  runtime_thread_stack_size_kb: $RUNTIME_THREAD_STACK_SIZE_KB
EOF
  else
    cat >>"$CFG_FILE" <<EOF
  runtime_worker_threads: null
  runtime_max_blocking_threads: null
  runtime_thread_stack_size_kb: null
EOF
  fi
  if [[ "$UPSTREAM_MODE" == "alias_group" ]]; then
    cat >>"$CFG_FILE" <<EOF
upstream_services:
  - name: "mock-openai-a"
    provider: "openai"
    base_url: "http://127.0.0.1:$UPSTREAM_PORT/v1"
    api_key: "sk-upstream"
    models:
      - "smart:m1"
    is_default: true
  - name: "mock-openai-b"
    provider: "openai"
    base_url: "http://127.0.0.1:$UPSTREAM_PORT/v1"
    api_key: "sk-upstream"
    models:
      - "smart:m2"
    is_default: false
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
  else
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
  fi
}

start_mock() {
  local mode="$1"
  MOCK_MODE="$mode" MOCK_TRANSPORT="$UPSTREAM_TRANSPORT" MOCK_SCENARIO="$MOCK_SCENARIO" \
    UPSTREAM_PORT="$UPSTREAM_PORT" run_maybe_pinned_exec "$PIN_UPSTREAM_CORE" "$MOCK_UPSTREAM_BIN" >>"$UPSTREAM_LOG" 2>&1
}

read_pid_total_ticks() {
  local pid="$1"
  if [[ ! -r "/proc/$pid/stat" ]]; then
    echo 0
    return
  fi
  awk '{print $14+$15}' "/proc/$pid/stat" 2>/dev/null || echo 0
}

read_pid_vm_hwm_kb() {
  local pid="$1"
  if [[ ! -r "/proc/$pid/status" ]]; then
    echo 0
    return
  fi
  awk '/VmHWM:/{print $2+0; found=1} END{if(!found) print 0}' "/proc/$pid/status" 2>/dev/null
}

resolve_stats_pid() {
  local pid="$1"
  local depth=0

  while (( depth < 8 )); do
    if [[ ! -r "/proc/$pid/comm" ]]; then
      break
    fi
    local comm
    comm="$(cat "/proc/$pid/comm" 2>/dev/null || true)"
    if [[ "$comm" == "toolify" ]]; then
      break
    fi

    case "$comm" in
      bash|sh|taskset)
        local -a children
        mapfile -t children < <(pgrep -P "$pid" || true)
        if (( ${#children[@]} == 0 )); then
          break
        fi

        local selected=""
        local child child_comm

        # Prefer the real benchmark target process when visible.
        for child in "${children[@]}"; do
          child_comm="$(cat "/proc/$child/comm" 2>/dev/null || true)"
          if [[ "$child_comm" == "toolify" ]]; then
            selected="$child"
            break
          fi
        done

        # Otherwise pick the first non-wrapper child.
        if [[ -z "$selected" ]]; then
          for child in "${children[@]}"; do
            child_comm="$(cat "/proc/$child/comm" 2>/dev/null || true)"
            case "$child_comm" in
              bash|sh|taskset) ;;
              *)
                selected="$child"
                break
                ;;
            esac
          done
        fi

        if [[ -z "$selected" ]]; then
          selected="${children[0]}"
        fi
        pid="$selected"
        ;;
      *)
        break
        ;;
    esac
    depth=$((depth + 1))
  done

  echo "$pid"
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if (echo >/dev/tcp/"$host"/"$port") >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.05
  done
  return 1
}

warmup_requests() {
  local payload="$1"
  local target_url="$2"

  if (( WARMUP_REQUESTS <= 0 )); then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi

  for _ in $(seq 1 "$WARMUP_REQUESTS"); do
    curl -sS -o /dev/null \
      --noproxy '*' \
      --connect-timeout "$WARMUP_CONNECT_TIMEOUT_S" \
      --max-time "$WARMUP_MAX_TIME_S" \
      -X POST "$target_url" \
      -H "Authorization: Bearer sk-client" \
      -H "Content-Type: application/json" \
      --data "$payload" || true
  done
}

wait_for_http_ready() {
  local payload="$1"
  local target_url="$2"
  local timeout_s="${3:-10}"
  local deadline=$((SECONDS + timeout_s))

  while (( SECONDS < deadline )); do
    local code
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
      --noproxy '*' \
      --connect-timeout 1 \
      --max-time 1 \
      -X POST "$target_url" \
      -H "Authorization: Bearer sk-client" \
      -H "Content-Type: application/json" \
      --data "$payload" || true)"
    if [[ "$code" == "200" || "$code" == "4"* ]]; then
      return 0
    fi
    sleep 0.05
  done
  return 1
}

run_wrk() {
  local payload="$1"
  local target_url="$2"
  local lua_file="$TMP_DIR/wrk.lua"
  cat >"$lua_file" <<EOF
wrk.method = "POST"
wrk.headers["Authorization"] = "Bearer sk-client"
wrk.headers["Content-Type"] = "application/json"
wrk.body = '$payload'
EOF
  run_maybe_pinned "$PIN_WRK_CORE" \
    wrk -t"$WRK_THREADS" -c"$CONNECTIONS" -d"$DURATION" --timeout "$WRK_TIMEOUT" --latency -s "$lua_file" "$target_url"
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
  local wrk_out="$TMP_DIR/${label}_wrk.out"

  : >"$UPSTREAM_LOG"
  : >"$PROXY_LOG"

  if [[ "$stream" == "true" ]]; then
    start_mock stream &
  else
    start_mock nonstream &
  fi
  local upstream_pid=$!

  if ! kill -0 "$upstream_pid" 2>/dev/null; then
    echo "failed to start mock upstream"
    exit 1
  fi
  if ! wait_for_port "127.0.0.1" "$UPSTREAM_PORT" "$STARTUP_TIMEOUT_S"; then
    echo "mock upstream did not become ready in ${STARTUP_TIMEOUT_S}s"
    sed -n '1,80p' "$UPSTREAM_LOG"
    exit 1
  fi

  (cd "$ROOT_DIR" && cp "$CFG_FILE" config.yaml)
  (cd "$ROOT_DIR" && run_maybe_pinned_exec "$PIN_PROXY_CORE" "$TOOLIFY_BIN") >"$PROXY_LOG" 2>&1 &
  local proxy_pid=$!

  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "failed to start toolify"
    sed -n '1,80p' "$PROXY_LOG"
    exit 1
  fi
  if ! wait_for_port "127.0.0.1" "$PROXY_PORT" "$STARTUP_TIMEOUT_S"; then
    echo "toolify did not become ready in ${STARTUP_TIMEOUT_S}s"
    sed -n '1,120p' "$PROXY_LOG"
    exit 1
  fi

  local payload
  if [[ "$stream" == "true" ]]; then
    payload="{\"model\":\"$BENCH_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true}"
  else
    payload="{\"model\":\"$BENCH_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
  fi
  local target_url="http://127.0.0.1:$PROXY_PORT/v1/chat/completions"

  if ! wait_for_http_ready "$payload" "$target_url" "$STARTUP_TIMEOUT_S"; then
    echo "toolify HTTP endpoint did not become ready in ${STARTUP_TIMEOUT_S}s"
    sed -n '1,120p' "$PROXY_LOG"
    exit 1
  fi

  warmup_requests "$payload" "$target_url"

  local stats_pid
  stats_pid="$(resolve_stats_pid "$proxy_pid")"
  local start_ms end_ms start_ticks end_ticks
  start_ms="$(date +%s%3N)"
  start_ticks="$(read_pid_total_ticks "$stats_pid")"

  run_wrk "$payload" "$target_url" | tee "$wrk_out"
  assert_upstream_h2 "$label"

  end_ms="$(date +%s%3N)"
  end_ticks="$(read_pid_total_ticks "$stats_pid")"
  local peak_rss
  peak_rss="$(read_pid_vm_hwm_kb "$stats_pid")"

  kill -TERM "$proxy_pid" >/dev/null 2>&1 || true
  kill -TERM "$upstream_pid" >/dev/null 2>&1 || true
  wait "$proxy_pid" 2>/dev/null || true
  wait "$upstream_pid" 2>/dev/null || true

  local req_sec total_req p99 latency_avg
  req_sec="$(awk '/Requests\/sec:/ {print $2}' "$wrk_out" | tail -n1)"
  total_req="$(awk '/requests in/ {print $1}' "$wrk_out" | tail -n1)"
  p99="$(awk '/^[[:space:]]*99%[[:space:]]+/ {print $2}' "$wrk_out" | tail -n1)"
  latency_avg="$(awk '/^[[:space:]]*Latency[[:space:]]+/ {print $2}' "$wrk_out" | head -n1)"
  echo "$label wrk_requests=${total_req:-0} wrk_rps=${req_sec:-0} wrk_p99=${p99:-n/a} wrk_latency_avg=${latency_avg:-n/a}"

  local cpu_pct wall_s cpu_ticks clk
  if (( end_ticks >= start_ticks )); then
    cpu_ticks="$((end_ticks - start_ticks))"
  else
    cpu_ticks=0
  fi
  clk="$(getconf CLK_TCK)"
  wall_s="$(awk -v s="$start_ms" -v e="$end_ms" \
    'BEGIN { if (e > s) printf "%.6f", (e-s)/1000.0; else print "0.000001" }')"
  cpu_pct="$(awk -v ticks="$cpu_ticks" -v clk="$clk" -v wall="$wall_s" \
    'BEGIN { if (wall > 0) printf "%.2f", (ticks/clk) * 100.0 / wall; else print "0.00" }')"
  echo "$label cpu_pct=$cpu_pct peak_rss_kb=$peak_rss"
}

make_config

case "$MODE" in
  nonstream)
    run_case false "${BENCH_LABEL_PREFIX}_nonstream_wrk"
    ;;
  stream)
    run_case true "${BENCH_LABEL_PREFIX}_stream_wrk"
    ;;
  all)
    run_case false "${BENCH_LABEL_PREFIX}_nonstream_wrk"
    run_case true "${BENCH_LABEL_PREFIX}_stream_wrk"
    ;;
  *)
    echo "unknown mode: $MODE"
    exit 1
    ;;
esac
