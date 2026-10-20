#!/usr/bin/env bash
# Real-weights single-card validation ladder, driven through the fork's own
# launcher + local route profiles. Produces one transparent log per step.
#
# usage: validate_single_card.sh baseline|long|offload|all
#   baseline : 32K route, MODE=normal, reference lane (4K/128 x3 + 32K/512)
#   long     : 131K route, reference lane
#   offload  : 131K route + tiered KV offload (4 GiB CPU staging + SSD tier)
#              with a repeated prompt to observe prefix reuse
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=${MODEL_DIR:-"$ROOT/.tooling/models/Qwen3.8-9B-Distill-FP8"}
SERVED_NAME=${SERVED_NAME:-qwen9b-singlecard}
P=local/1x2080Ti/qwen9b/w8a16
export MODEL_DIR SERVED_NAME

step_header() { echo; echo "######## $1 ########"; date '+%H:%M:%S'; }

start_server() {  # profile mode seqs offload_bytes [disable_prefix_cache]
  # start-direct：用 launcher 的原样命令行执行，但绕开它的 checkpoint-mmap 预检
  # （本机 CommitLimit−Committed_AS ≈2.6GiB < 11.1GiB 检查点，且 sudo 需要密码）
  # 参考口径按仓库 README：Prefix Cache 仅在测试时禁用
  PROFILE=$1 MODE=$2 MAX_NUM_SEQS=$3 KV_OFFLOAD=${4:-0} DISABLE_PREFIX_CACHING=${5:-1} \
    bash "$ROOT/.tooling/run_single_card.sh" start-direct 2>&1 | tail -3
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
}

stop_server() { bash "$ROOT/.tooling/run_single_card.sh" stop 2>&1 | tail -2; }

reference_lane() { bash "$ROOT/.tooling/run_single_card.sh" perf 1; }

baseline() {
  step_header "baseline: 32K / MODE=normal / 单请求参考口径"
  start_server "$P/nomtp-fp8kv-1x32K-text-only.env" normal 1
  reference_lane
quality_sanity
  stop_server
  step_header "baseline: 32K / MAX_NUM_SEQS=4 / 并发扫描"
  start_server "$P/nomtp-fp8kv-1x32K-text-only.env" normal 4
  bash "$ROOT/.tooling/concurrency_sweep.sh" 1 2 4
  stop_server
}

long_ctx() {
  step_header "long: 131K / MODE=normal / 单请求"
  start_server "$P/nomtp-fp8kv-1x131K-text-only.env" normal 1
  reference_lane
  stop_server
}

offload() {
  step_header "offload: 131K / 分层 KV offload (4 GiB CPU + SSD tier)"
  start_server "$P/nomtp-fp8kv-1x131K-text-only.env" normal 1 4294967296 0   # offload 需要前缀缓存
  echo "--- 同一 prompt 连续两次（观察前缀复用 / 分层命中）---"
  for run in 1 2; do
    "$ROOT/.venv/bin/python" "$ROOT/tools/profile_request.py" \
      --model-dir "$MODEL_DIR" --served-name "$SERVED_NAME" \
      --base-url "http://127.0.0.1:8100/v1" --endpoint completions \
      --prompt-tokens 32768 --gen-tokens 128 --label "offload-$run" \
      --prompt-variant "offload-fixed" \
      --out /dev/null --ignore-eos --pure-filler --allowed-token-text " the" 2>&1 \
      | "$ROOT/.venv/bin/python" -c "
import json,sys
d=json.load(sys.stdin)
print(f\"  run{$run}: prefill={d.get('prefill_tok_s')} tok/s ttft={d.get('ttft_s')}s elapsed={d.get('elapsed_s')}s decode={d.get('decode_tok_s')} tok/s\")
"
  done
  echo "--- 引擎侧分层状态（日志）---"
  grep -iE 'offload|tier|cpu tier|secondary' "$(ls -t "$ROOT"/run-logs/vllm-*.log | head -1)" | tail -6
  stop_server
}
quality_sanity() {
  step_header "quality sanity: 贪婪解码 + 8K 长上下文"
  bash "$ROOT/.tooling/sanity_check.sh" 8100 "$SERVED_NAME" || echo "  ⚠ 质量抽查未全过，吞吐数字仅供参考"
}

fast_mode() {
  step_header "fast: 32K / MODE=fast / 并发扫描（对比 normal 的保守图策略）"
  start_server "$P/nomtp-fp8kv-1x32K-text-only.env" fast 4
  bash "$ROOT/.tooling/concurrency_sweep.sh" 1 2 4
  stop_server
}
case "${1:-all}" in
  baseline) baseline ;;
  long)     long_ctx ;;
sanity)   quality_sanity ;;
fast)     fast_mode ;;
  offload)  offload ;;
  all)      baseline; fast_mode; long_ctx; offload ;;
  *) echo "usage: $0 baseline|long|offload|all" >&2; exit 2 ;;
esac
echo; echo "######## 完成 ########"