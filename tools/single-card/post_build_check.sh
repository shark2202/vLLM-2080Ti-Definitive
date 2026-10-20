#!/usr/bin/env bash
# Post-build verification for the single-card SM75 route. No model weights needed.
# usage: post_build_check.sh
set -u
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PY=$ROOT/.venv/bin/python
fail=0

echo "=== 1) runtime identity ==="
"$PY" - <<'PY' || fail=1
import torch, vllm
print("vllm        :", getattr(vllm, "__version__", "?"))
print("torch       :", torch.__version__, "cuda runtime", torch.version.cuda)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      :", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
import importlib
for mod in ("flashinfer", "triton", "flash_qla"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:<12}: {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        print(f"{mod:<12}: MISSING ({exc})")
PY

echo "=== 2) fork 的 SM75 单元测试 ==="
for t in tests/tools/test_patch_torch_inductor_e8m0.py \
         tests/kernels/mamba/test_flashqla_qkv_dtype.py \
         tests/tools/test_flashqla_sm75_lock_recovery.py; do
  printf '  %-52s ' "$t"
  if "$PY" -m pytest -q "$ROOT/$t" >/tmp/pb_$(basename "$t").log 2>&1; then echo PASS
  else echo "FAIL (/tmp/pb_$(basename "$t").log)"; fail=1; fi
done

echo "=== 3) 单卡 GPU 内核冒烟（不需要模型权重）==="
for t in tests/kernels/test_cache_kernels.py tests/v1/attention/test_flashinfer_sm75_spec_graph.py; do
  printf '  %-52s ' "$t"
  if "$PY" -m pytest -q "$ROOT/$t" >/tmp/pb_$(basename "$t").log 2>&1; then echo PASS
  else echo "SKIP/FAIL (/tmp/pb_$(basename "$t").log)"; fi
done

echo "=== 4) 结论 ==="
(( fail )) && { echo "POST-BUILD CHECK: FAILURES ABOVE"; exit 1; }
echo "POST-BUILD CHECK: OK"