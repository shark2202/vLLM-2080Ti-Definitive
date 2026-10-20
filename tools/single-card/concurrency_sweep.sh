#!/usr/bin/env bash
# Concurrency sweep on the running single-card server.
# Reuses the repo's own tools/profile_request.py so the per-request numbers stay
# on the same lane as the published profile tables; aggregate = sum of the
# per-request decode rates measured over the same wall clock.
#
# usage: concurrency_sweep.sh <N> [N ...]     (e.g. 1 2 4 8)
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=$ROOT/.venv/bin/python
PORT=${PORT:-8100}
SERVED_NAME=${SERVED_NAME:-qwen9b-dummy}
MODEL_DIR=${MODEL_DIR:-"$ROOT/.tooling/models/qwen9b-config-only"}
PROMPT_TOKENS=${PROMPT_TOKENS:-4096}
GEN_TOKENS=${GEN_TOKENS:-128}
OUT=$ROOT/run-logs/concurrency-$(date +%H%M%S)
mkdir -p "$OUT"

printf '%-4s %-14s %-16s %-12s %-14s %s\n' "N" "per-req dec" "aggregate dec" "wall(s)" "prefill" "preemptions"
for n in "$@"; do
  stamp=$(date +%s%N)
  pids=()
  for i in $(seq 1 "$n"); do
    "$PY" "$ROOT/tools/profile_request.py" \
      --model-dir "$MODEL_DIR" --served-name "$SERVED_NAME" \
      --base-url "http://127.0.0.1:$PORT/v1" --endpoint completions \
      --prompt-tokens "$PROMPT_TOKENS" --gen-tokens "$GEN_TOKENS" \
      --label "sweep-$n-$i" --prompt-variant "sweep-$stamp-$i" \
      --out /dev/null --ignore-eos --pure-filler --allowed-token-text " the" --read-timeout "${READ_TIMEOUT:-600}" \
      > "$OUT/n$n-$i.json" 2>&1 &
    pids+=($!)
  done
  pre=$(curl -sS --max-time 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | awk '/^vllm:num_preemptions_total/{print $2; exit}')
  wait "${pids[@]}"
  post=$(curl -sS --max-time 5 "http://127.0.0.1:$PORT/metrics" 2>/dev/null | awk '/^vllm:num_preemptions_total/{print $2; exit}')
  wall=$(( ($(date +%s%N) - stamp) / 1000000 ))

  "$PY" - "$OUT" "$n" "$wall" "${pre:-0}" "${post:-0}" <<'PY'
import json, sys, pathlib
out, n, wall_ms, pre, post = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
dec, pre_tok = [], []
for f in sorted(pathlib.Path(out).glob(f"n{n}-*.json")):
    try:
        d = json.loads(f.read_text())
    except Exception:
        d = {}
    if d.get("decode_tok_s"):
        dec.append(d["decode_tok_s"])
    if d.get("prefill_tok_s"):
        pre_tok.append(d["prefill_tok_s"])
agg = sum(dec)
per = agg / len(dec) if dec else 0.0
pref = sum(pre_tok) / len(pre_tok) if pre_tok else 0.0
print(f"{n:<4} {per:<14.2f} {agg:<16.2f} {wall_ms/1000:<12.2f} {pref:<14.1f} {float(post)-float(pre):.0f}")
PY
done
echo "原始 JSON: $OUT"