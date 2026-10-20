#!/usr/bin/env bash
# Quality sanity check on a running single-card server: does the model produce
# sensible text (the FP8 numeric path end to end), and does it still answer a
# long-context question? Not a benchmark — a pass/fail gate before trusting
# any throughput number.
#
# usage: sanity_check.sh [port] [served_name]
set -u
PORT=${1:-8100}
NAME=${2:-qwen9b-singlecard}
pass=0; fail=0

ask() {  # <expected-substring> <prompt> [max_tokens]
  local want=$1 prompt=$2 max=${3:-64}
  local out
  out=$(curl -sS --max-time 120 "http://127.0.0.1:$PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 - "$NAME" "$prompt" "$max" <<'PY'
import json,sys
name,prompt,mx=sys.argv[1],sys.argv[2],int(sys.argv[3])
print(json.dumps({"model":name,"prompt":prompt,"max_tokens":mx,"temperature":0}))
PY
)" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception as e: print(f'<bad response: {e}>'); raise SystemExit
print(d['choices'][0]['text'].strip().replace('\n',' ')[:160])
")
  printf '  Q: %-52s → %s\n' "${prompt:0:50}" "$out"
  if printf '%s' "$out" | grep -qi -- "$want"; then pass=$((pass+1)); echo "     ✓ 命中 '$want'"
  else fail=$((fail+1)); echo "     ✗ 未命中 '$want'"; fi
}

echo "=== 质量抽查（贪婪解码，$NAME @ $PORT）==="
ask "Paris"        "The capital of France is"
ask "2"           "What is 1 + 1? Answer with a single digit:"
ask "water"       "H2O is commonly known as"
echo
echo "=== 长上下文抽查（约 8K 填空 + 尾部提问）==="
LONG=$(python3 - <<'PY'
import random
random.seed(7)
words=['lorem','ipsum','dolor','sit','amet','consectetur','adipiscing','elit','sed','do','eiusmod','tempor']
body=' '.join(random.choice(words) for _ in range(8000))
print(body + "\n\nIgnore the filler above. The capital of France is")
PY
)
out=$(curl -sS --max-time 240 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d "$(python3 - "$NAME" "$LONG" <<'PY'
import json,sys
name,prompt=sys.argv[1],sys.argv[2]
print(json.dumps({"model":name,"prompt":prompt,"max_tokens":16,"temperature":0}))
PY
)" | python3 -c "
import json,sys
d=json.load(sys.stdin); print(d['choices'][0]['text'].strip().replace('\n',' ')[:120])
")
printf '  → %s\n' "$out"
if printf '%s' "$out" | grep -qi 'Paris'; then pass=$((pass+1)); echo "     ✓ 长上下文可读"; else fail=$((fail+1)); echo "     ✗ 长上下文输出异常"; fi

echo
echo "sanity: pass=$pass fail=$fail"
(( fail == 0 )) || exit 1