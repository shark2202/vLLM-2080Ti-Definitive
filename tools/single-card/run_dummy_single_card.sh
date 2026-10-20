#!/usr/bin/env bash
# Zero-weight mechanism validation: same route parameters as the single-card
# profile, but the model is instantiated with --load-format dummy (random
# weights, no checkpoint download). VRAM footprint and code paths are the real
# ones; throughput numbers are indicative only.
#
# usage: run_dummy_single_card.sh [MAX_MODEL_LEN] [MAX_NUM_SEQS] [offload_bytes|0]
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=$ROOT/.venv/bin/python
MODEL_DIR=${MODEL_DIR:-"$ROOT/.tooling/models/qwen9b-config-only"}
MAX_MODEL_LEN=${1:-32768}
MAX_NUM_SEQS=${2:-1}
OFFLOAD_BYTES=${3:-0}
PORT=${PORT:-8100}
LOG=$ROOT/run-logs/dummy-$MAX_MODEL_LEN-seq$MAX_NUM_SEQS-$(date +%H%M%S).log
mkdir -p "$ROOT/run-logs"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.4}
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=7.5
# 与 fork launcher 的 set_sm75_runtime_env 保持一致：GDN 的 SM75 legacy 后端需要这三项
export FLASHQLA_ROOT=${FLASHQLA_ROOT:-"$ROOT/.deps/FlashQLA-SM70-SM75"}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"$FLASHQLA_ROOT/.torch_extensions_vllm_flashqla_legacy"}
export PYTHONPATH="$ROOT${FLASHQLA_ROOT:+:$FLASHQLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  --model "$MODEL_DIR"
  --load-format dummy
  --served-model-name qwen9b-dummy
  --quantization compressed-tensors
--kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
--mamba-cache-mode "${MAMBA_CACHE_MODE:-align}"
  --generation-config auto
  --compilation-config "${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[1],\"max_cudagraph_capture_size\":1}}"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization 0.90
  --max-num-seqs "$MAX_NUM_SEQS"
  --enable-prefix-caching
  --language-model-only
  --gdn-prefill-backend flashqla_legacy
  --no-enable-log-requests
  --port "$PORT"
)
if (( MAX_NUM_SEQS > 1 )); then
  ARGS+=(--prefill-batch-barrier)
fi
if (( OFFLOAD_BYTES > 0 )); then
  ARGS+=(--kv-transfer-config "$(python3 - "$ROOT/.tooling/kvdisk" "$OFFLOAD_BYTES" <<'PY'
import json,os,sys
root,cpu=sys.argv[1],int(sys.argv[2])
print(json.dumps({"kv_connector":"OffloadingConnector","kv_role":"kv_both",
 "kv_connector_extra_config":{"cpu_bytes_to_use":cpu,"spec_name":"TieringOffloadingSpec",
 "secondary_tiers":[{"type":"fs","root_dir":root}]}},separators=(",",":")))
PY
)")
fi

echo "=== dummy 单卡启动 ==="
echo "  MAX_MODEL_LEN=$MAX_MODEL_LEN  MAX_NUM_SEQS=$MAX_NUM_SEQS  offload=$OFFLOAD_BYTES bytes"
echo "  log: $LOG"
"$PY" -m vllm.entrypoints.openai.api_server "${ARGS[@]}" >"$LOG" 2>&1 &
echo $! > "$ROOT/run-logs/dummy.pid"
for i in $(seq 1 300); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "  health: ok (${i}s)"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    echo "  运行中；用 .tooling/run_single_card.sh status / perf 交互，或 kill $(cat "$ROOT/run-logs/dummy.pid")"
    exit 0
  fi
  if ! kill -0 "$(cat "$ROOT/run-logs/dummy.pid")" 2>/dev/null; then
    echo "  启动失败，日志末尾："; tail -25 "$LOG"; exit 1
  fi
  sleep 1
done
echo "  超时未就绪，日志末尾："; tail -25 "$LOG"; exit 1