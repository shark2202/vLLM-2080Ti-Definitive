#!/usr/bin/env bash
# A/B experiment: the fork's launcher pins cudagraph_capture_sizes to [1] for this
# profile; vLLM's own auto-derivation used [1,2]. This script rebuilds the same
# command line with a wider capture set to measure what the conservative policy
# costs in concurrency throughput (real weights, 32K, 4 sequences).
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$ROOT/.tooling/models/Qwen3.8-9B-Distill-FP8"}
PORT=${PORT:-8100}
LOG=$ROOT/run-logs/ab-cudagraph-$(date +%H%M%S).log
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.4}
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHQLA_ROOT=${FLASHQLA_ROOT:-"$ROOT/.deps/FlashQLA-SM70-SM75"}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"$FLASHQLA_ROOT/.torch_extensions_vllm_flashqla_legacy"}
export PYTHONPATH="$ROOT${FLASHQLA_ROOT:+:$FLASHQLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# 与 launcher 原样命令一致的旗标，唯一差别是 capture sizes 放宽到 [1,2,4]
"$ROOT/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port "$PORT" --model "$MODEL_DIR" --served-model-name ab-cudagraph \
  --dtype half --tensor-parallel-size 1 --generation-config auto \
  --gpu-memory-utilization 0.90 --max-model-len 32768 --enable-chunked-prefill \
  --max-num-seqs 4 --max-num-batched-tokens 2048 --prefill-batch-barrier \
  --quantization compressed-tensors --kv-cache-dtype fp8 --mamba-cache-mode align \
  --no-enable-prefix-caching --enable-prompt-tokens-details --language-model-only \
  --skip-mm-profiling --disable-log-stats \
  --additional-config '{"gdn_prefill_backend":"flashqla_legacy"}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4}' \
  >"$LOG" 2>&1 &
echo $! > "$ROOT/run-logs/ab-cudagraph.pid"
echo "  ab log → $LOG"
for i in $(seq 1 420); do
  curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "  health: ok (${i}s)"; break; }
  kill -0 "$(cat "$ROOT/run-logs/ab-cudagraph.pid")" 2>/dev/null || { echo "  启动失败："; tail -20 "$LOG"; exit 1; }
  sleep 1
done
grep -o "'cudagraph_capture_sizes': \[[0-9, ]*\]" "$LOG" | head -1
SERVED_NAME=ab-cudagraph MODEL_DIR="$MODEL_DIR" bash "$ROOT/.tooling/concurrency_sweep.sh" 1 2 4
kill -TERM "$(cat "$ROOT/run-logs/ab-cudagraph.pid")" 2>/dev/null
rm -f "$ROOT/run-logs/ab-cudagraph.pid"