#!/usr/bin/env bash
# Single-card (1x RTX 2080 Ti 22GB) driver for the fork's launcher.
# Reuses the repo's own smoke test and tools/profile_request.py, so numbers stay
# on the same measuring lane as the published profile tables.
#
# usage: run_single_card.sh dry-run|start|stop|status|logs|perf [N]
# knobs: MODE=normal|fast|safe  MAX_NUM_SEQS=1  KV_OFFLOAD=1  KV_CPU_BYTES=4294967296
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$ROOT/.tooling/models/Qwen3.8-9B-Distill-FP8"}
PROFILE=${PROFILE:-local/1x2080Ti/qwen9b/w8a16/nomtp-fp8kv-1x32K-text-only.env}
MODE=${MODE:-normal}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
PORT=${PORT:-8100}
SERVED_NAME=${SERVED_NAME:-qwen9b-awq-singlecard}
LOG_DIR=${LOG_DIR:-"$ROOT/run-logs"}

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.4}
export PATH="$CUDA_HOME/bin:$PATH"
export PROFILE_DIR=${PROFILE_DIR:-"$ROOT/profiles"}
export PROFILE MODEL_DIR MODE MAX_NUM_SEQS PORT SERVED_NAME LOG_DIR
export GPU_DEVICES=${GPU_DEVICES:-0}
export NON_INTERACTIVE=1 ASSUME_YES=1
export ENFORCE_EAGER=${ENFORCE_EAGER:-0}
if [[ "${KV_OFFLOAD:-0}" == "1" ]]; then
  export KV_DISK_CACHE_DIR=${KV_DISK_CACHE_DIR:-"$ROOT/.tooling/kvdisk"}
  export KV_DISK_CPU_BYTES=${KV_DISK_CPU_BYTES:-4294967296}
fi

pid_files() { ls -t "$LOG_DIR"/vllm-*.pid 2>/dev/null; }
live_pid() {   # only pids that really are our runtime
  local f=$1 p
  p=$(cat "$f" 2>/dev/null || true)
  [[ -n "$p" ]] || return 1
  tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "vllm.entrypoints.openai.api_server" || return 1
  printf '%s' "$p"
}

# Direct launch using the launcher's OWN emitted command line and environment,
# bypassing only its checkpoint-mmap preflight. Needed on hosts where
# CommitLimit - Committed_AS < checkpoint size and sudo is not available
# non-interactively (the launcher then refuses to start by design).
launch_direct() {
  local tag=${1:-direct} dry cmd log
  dry=$(DRY_RUN=1 "$ROOT/launcher.sh" 2>&1)
  cmd=$(printf '%s\n' "$dry" | awk '/^Command:/{getline; sub(/^[ \t]+/,""); print; exit}')
  if [ -z "$cmd" ]; then
    echo "ERROR: 未能从 launcher 取得命令行。DRY_RUN 输出尾部：" >&2
    printf '%s\n' "$dry" | tail -15 >&2
    return 1
  fi
  # 照搬 launcher 在 DRY_RUN 中打印的环境（SM75 相关开关）
  while IFS= read -r kv; do
    case "$kv" in
      VLLM_*|CUDA_VISIBLE_DEVICES=*) export "$kv" ;;
    esac
  done < <(printf '%s\n' "$dry" \
    | sed -n '/^Environment:/,/^Command:/p' \
    | sed -n 's/^[ \t]\+\([A-Za-z_][A-Za-z0-9_]*=.*\)$/\1/p')
  export FLASHQLA_ROOT=${FLASHQLA_ROOT:-"$ROOT/.deps/FlashQLA-SM70-SM75"}
  export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"$FLASHQLA_ROOT/.torch_extensions_vllm_flashqla_legacy"}
  export PYTHONPATH="$ROOT${FLASHQLA_ROOT:+:$FLASHQLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# 已有服务在跑时必须先停掉：否则新进程绑不上端口，而健康检查会打到旧进程
for p in $(pgrep -f 'vllm.entrypoints.openai.api_serve[r]' 2>/dev/null); do kill -TERM "$p" 2>/dev/null; done
  # V1 架构的 EngineCore 子进程不匹配上面的模式，会留着显存不放
  for p in $(pgrep -f 'EngineCor[e]' 2>/dev/null); do kill -TERM "$p" 2>/dev/null; done
  for i in $(seq 1 30); do
    pgrep -f 'vllm.entrypoints.openai.api_serve[r]|EngineCor[e]' >/dev/null 2>&1 || break
    sleep 1
  done
  for p in $(pgrep -f 'vllm.entrypoints.openai.api_serve[r]|EngineCor[e]' 2>/dev/null); do kill -KILL "$p" 2>/dev/null; done
  # 显存必须真正归还，否则新服务会因 "free memory < util×total" 启动失败
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ "${used:-99999}" -lt 1024 ] && break
    sleep 2
  done
  if [ "${used:-99999}" -ge 1024 ]; then
    echo "ERROR: 启动前 GPU 仍有 ${used} MiB 被占用（可能是 ComfyUI 等其它服务）。" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader >&2
    echo "请先停掉占显存的服务，或把 GPU_UTIL 降到可用显存以内。" >&2
    return 1
  fi
  rm -f "$LOG_DIR"/vllm-*.pid
  mkdir -p "$LOG_DIR"
  log="$LOG_DIR/vllm-$tag-$(date +%H%M%S).log"
  echo "  direct launch → $log"
  eval "$cmd" >"$log" 2>&1 &
  echo $! > "$LOG_DIR/vllm-$tag.pid"
  for i in $(seq 1 420); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "  health: ok (${i}s)"; nvidia-smi --query-gpu=memory.used --format=csv,noheader; return 0
    fi
    if ! kill -0 "$(cat "$LOG_DIR/vllm-$tag.pid")" 2>/dev/null; then
      echo "  启动失败，日志尾部："; tail -25 "$log"; return 1
    fi
    sleep 1
  done
  echo "  超时未就绪，日志尾部："; tail -25 "$log"; return 1
}
case "${1:-}" in
  dry-run)
    "$ROOT/launcher.sh" --print-config
    ;;
  start)
    "$ROOT/launcher.sh"
    ;;
start-direct)
    launch_direct direct
    ;;
  stop)
    stopped=0
    for f in $(pid_files); do
      if p=$(live_pid "$f"); then
        echo "stopping pid $p ($(basename "$f"))"
        kill -TERM "$p" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
        kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null || true
        stopped=1
      fi
      rm -f "$f"
    done
    (( stopped )) && echo "stopped" || echo "no running service found"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    ;;
  status)
    found=0
    for f in $(pid_files); do
      if p=$(live_pid "$f"); then echo "running: pid $p ($(basename "$f"))"; found=1; fi
    done
    (( found )) || echo "not running"
    curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && echo "health: ok" || echo "health: no response"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
    ;;
  logs)
    ls -t "$LOG_DIR"/vllm-*.log 2>/dev/null | head -1 | xargs -r tail -40
    ;;
  perf)
    # launcher-equivalent lane: 3x 4K/128 sequential, then 1x 32K/512
    n=${2:-1}
    stamp=$(date +%Y%m%d-%H%M%S)
    pids=()
    for i in $(seq 1 "$n"); do
      for run in 1 2 3; do
        "$ROOT/.venv/bin/python" "$ROOT/tools/profile_request.py" \
          --model-dir "$MODEL_DIR" --served-name "$SERVED_NAME" \
          --base-url "http://127.0.0.1:$PORT/v1" --endpoint completions \
          --prompt-tokens 4096 --gen-tokens 128 \
          --label "singlecard-$i-$run" --prompt-variant "$stamp-$i-$run" \
--out /dev/null --ignore-eos --pure-filler --allowed-token-text " the" --read-timeout "${READ_TIMEOUT:-600}" &
        pids+=($!)
      done
    done
    wait "${pids[@]}"
    "$ROOT/.venv/bin/python" "$ROOT/tools/profile_request.py" \
      --model-dir "$MODEL_DIR" --served-name "$SERVED_NAME" \
      --base-url "http://127.0.0.1:$PORT/v1" --endpoint completions \
      --prompt-tokens 32768 --gen-tokens 512 \
      --label "singlecard-long" --prompt-variant "$stamp-long" \
      --out /dev/null --ignore-eos --pure-filler --allowed-token-text " the"
    ;;
  *)
    echo "usage: $0 dry-run|start|stop|status|logs|perf [N]" >&2
    exit 2
    ;;
esac