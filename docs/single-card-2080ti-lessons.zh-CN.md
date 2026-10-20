# vLLM 2080 Ti 单卡落地：经验教训与实测记录

日期：2026-10-20 · 主机：Ubuntu 22.04.5 / kernel 6.8 / 单张 RTX 2080 Ti 22,528 MiB（驱动 580.178.04）
对象：`vLLM-2080Ti-Definitive`（fork，上游基线 `b23433088b`，fork 版本 `v0.2.2-post3`）
本文位置：仓库内 `.tooling/`（已加入 `.git/info/exclude`，不污染 tracked 文件）

---

## 0. 一句话结论

**这个 fork 的 SM75 栈在你这张单卡上真实跑通了**（真权重、质量 4/4、prefill ~3,020 tok/s、decode 47.2 tok/s、并发 1/2/4 全程 0 preemption）。
**但它与 ComfyUI 争用同一块 GPU**：`GPU_UTIL=0.90` 在 ComfyUI 占用 4.56 GiB 时会拒绝启动（vLLM 拒绝超分显存——这正是"不崩溃"的设计）。共存时上限约 `GPU_UTIL=0.78` → 约 130K 上下文；要 229K/262K 必须独占 GPU。

---

## 1. 主机与网络现实（决定了全部时间成本）

### 1.1 主机
| 项 | 实测 |
|---|---|
| OS / kernel / glibc | Ubuntu 22.04.5 / 6.8.0-138 / glibc 2.35 |
| 编译器 | **只有 gcc/g++ 11 与 gcc 12；没有 `g++-12`** |
| CUDA | `PATH` 上 `/usr/bin/nvcc` 是 **11.5**；可用的是 `/usr/local/cuda-13.4/bin/nvcc`（13.4.92） |
| Python | 系统 3.10.12（项目要 3.12 → 用 `uv python install`） |
| 内存 | 31 GB RAM + 2 GB swap；`CommitLimit` 18.46 GB，`Committed_AS` ~15.8 GB → **commit 余量仅 ~2.6 GiB** |
| commit 大户 | ComfyUI 5.4 GB、ToDesk 1.3 GB、awesun 0.8 GB、code-server ×5 ≈3.5 GB、桌面组件 ~2 GB |

### 1.2 网络（代理是最大的坑）
| 路径 | 结果 |
|---|---|
| Mac Clash 代理 `http://192.168.8.184:7890` | ~300 KB/s，且对 **hf-mirror / github / jsdelivr / pypi.org / download.pytorch.org** 会 TLS 断（`SSL routines::unexpected eof`） |
| **hf-mirror 直连（绕过代理）** | 单流 1.09 MB/s，多连接可持续 ~2 MB/s、突发 9.5 MB/s |
| aliyun PyPI / pytorch-wheels 直连 | 可用 ✓ |
| **ghfast.top / gh-proxy.com 直连** | **git 可用** ✓（build.sh 的 `BUILD_GIT_MIRROR_PREFIX` / `FLASHQLA_REPO` 可指向它） |
| ModelScope 直连 | 可用，但**没有 9B 的 qwen3.5 量化权重**（只有官方 27B FP8=30.9GB，单卡装不下） |
| `github.com` 的 `git clone` | 走代理时 `gnutls_handshake() failed`，强制 `http.version=HTTP/1.1` 也不行 |

**操作准则**：能直连就不要走代理；下载大文件用**多连接 + 短尝试 + 速度下限**（见 3.4/3.5）。

---

## 2. 有效做法（可直接复用的命令）

```bash
ROOT=/home/a/ai-work/vLLM-2080Ti-Definitive

# 2.1 隔离工具链（零全局改动）
python3 -m venv $ROOT/.tooling/bootstrap && $ROOT/.tooling/bootstrap/bin/pip install uv
export UV_CACHE_DIR=$ROOT/.tooling/uv-cache UV_PYTHON_INSTALL_DIR=$ROOT/.tooling/python

# 2.2 构建（build.sh 在纯镜像网络 + 非常规主机上需这些改造）
source /home/a/ai-work/.proxy_env   # 仅给 aliyun/本地用；不要用它访问 github/hf
unset http_proxy https_proxy all_proxy          # 走直连
export CUDA_HOME=/usr/local/cuda-13.4 PATH="/usr/local/cuda-13.4/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=7.5 VLLM_TARGET_DEVICE=cuda VLLM_MAIN_CUDA_VERSION=cu130
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple UV_HTTP_TIMEOUT=300
export VLLM_CUTLASS_SRC_DIR=$ROOT/.deps/cutlass-src
export TRITON_KERNELS_SRC_DIR=$ROOT/.deps/triton-kernels-tar/triton-3.5.1/python/triton_kernels/triton_kernels
export FLASHQLA_REPO=https://ghfast.top/https://github.com/weicj/FlashQLA-SM70-SM75.git
export BUILD_GIT_MIRROR_PREFIX=https://ghfast.top/
export ALLOW_HOST_MISMATCH=1 ASSUME_YES=1 MAX_JOBS=6
unset CC CXX                                    # ← 不要乱设 CXX（见 3.1）
uv pip install --python .venv/bin/python --no-build-isolation -e .   # 去掉 --torch-backend（见 3.2）
SKIP_VLLM_BUILD=1 bash build.sh                 # 收尾：inductor 补丁 + FlashQLA 扩展 + 运行时校验

# 2.3 启动（launcher 的 mmap 预检在本机不可用，见 3.8）
bash .tooling/run_single_card.sh start-direct   # 用 launcher 的 DRY_RUN 命令行原样执行

# 2.4 验证
bash .tooling/sanity_check.sh 8100 <served-name>     # 质量闸门
bash .tooling/validate_single_card.sh baseline       # 参考口径 + 并发
bash .tooling/concurrency_sweep.sh 1 2 4             # 并发扫描（复用 profile_request.py 同口径）
```

---

## 3. 我踩过的坑（最有价值的部分）

### 3.1 强制指定不存在的编译器 → CMake 直接失败
`CXX=/usr/bin/g++-12`，但本机只有 `gcc-12`（**没有 `g++-12`**）。
教训：设置 `CC/CXX` 前先 `test -x`；不确定就用系统默认（gcc-11 对 CUDA 13.4 是合法宿主编译器）。

### 3.2 `--torch-backend=cu130` 会让 uv 去访问 `download.pytorch.org`
即使 torch 已装好，uv 也会**查询**该索引页；而该域名在本机代理下必断 → 构建失败。
教训：预装 torch 家族（`torch/triton/torchvision/torchaudio/torchcodec`）后**直接 `uv pip install -e .`**（不加 `--torch-backend`），再 `SKIP_VLLM_BUILD=1 bash build.sh` 跑收尾。

### 3.3 CMake FetchContent 需要 `git clone github.com`（cutlass / triton_kernels）
本机 git 走代理不稳。解法：**预置源码树**
- cutlass：`git clone --depth 1 -b v4.7.1 https://github.com/nvidia/cutlass.git .deps/cutlass-src`（或 tarball）
- triton_kernels：**不需要 clone 整个 triton**——用 HTTP range 读 zip 中央目录，从 aliyun 上的 `vllm-0.29.0` wheel 里抽出 40 个 `.py`（226 KB）即可，放到 `.deps/triton-kernels-tar/triton-3.5.1/python/triton_kernels/triton_kernels/`
教训：`triton_kernels` 在**运行时可选**（缺失只打日志），但**构建期硬依赖**。

### 3.4 `curl --append -r A-B -o f` 根本不追加，而是截断覆写 ★
实测：先写 1MB，再 `--append -r 1000000-1999999 -o 同文件` → 文件仍是 1MB（数据错位无法续传）。
后果：我的下载器"负增长"（每次尝试都从头覆写）。
正确做法：**每次尝试写独立临时文件 → 校验 HTTP 206 且 `0 < size ≤ 剩余` → `cat >> part`**；再用 `--max-filesize $span` 让"忽略 Range 的 200 巨包"在传输前就被拒绝。

### 3.5 僵死连接与超时
- 去掉速度下限后，僵死 TCP 会白占 `--max-time 240`；恢复 `--speed-limit 4096 --speed-time 30` 后可快速丢弃。
- `UV_HTTP_TIMEOUT=1800` 会把一次坏 TLS 变成 **13 分钟挂起**；改成 300s 快速失败重试。

### 3.6 进程清理的三个坑（我在这里栽了 4 次）
1. **`pkill -f <pattern>` / `pgrep -f <pattern>` 匹配到我自己的命令行** → 任务自杀（98ms/111ms 退出，退出码 143/137）。
   解法：括号技巧（`dl_range[d].sh`）+ **杀进程命令里绝不出现脚本字面名** + 排除 `$$`/`$PPID` + 杀完验证残留数。
2. **同一条消息里"杀旧任务"与"启动新任务"并发执行** → 新任务被杀。
   解法：**破坏性步骤与启动步骤分成两条消息/两次调用**。
3. **只杀父进程不杀子进程**：
   - `curl` 子进程继续用 `-o` 截断我刚"转正"的分片 → 数据错位；
   - **vLLM V1 的 `EngineCore` 子进程**不匹配 `vllm.entrypoints.openai.api_server` 模式，被留下后**占着 ~4.5 GiB 显存不放**，导致后续启动 `free memory < util×total` 失败。
   解法：两者都要杀；杀完**等显存降到 <1 GiB**，否则明确报错而不是硬启动。

### 3.7 口径与命名
- fork 的 README 明确："Prefix Cache **仅在测试时禁用**" → 参考口径必须关前缀缓存。
- 文件名约定 `<N>x<CTX>K` 的 **K 是十进制千**：`131072` → `1x131K`（写 `1x128K` 会被 validator 拒）。
- vLLM 参数改名：`--disable-log-requests` → **`--no-enable-log-requests`**。
- 32K profile 跑不了 32K/512 长样本（prompt+变体 > `max_model_len` → HTTP 400），长样本要用 ≥131K 的 profile。
- 我自己的质量闸门因 served-name 不匹配拿到 404（`KeyError 'choices'`）→ 应从服务日志里读实际 `served_model_name`。

### 3.8 launcher 的 checkpoint mmap 预检在本机不可用
```
CommitLimit − Committed_AS ≈ 2.6 GiB  <  检查点 11.09 GiB
ERROR: checkpoint mmap needs vm.overcommit_memory=1, but non-interactive
       launcher cannot prompt for sudo.
```
占用 commit 的是**用户自己的进程**（ComfyUI/ToDesk/code-server），sudo 需密码 → 不改系统参数。
解法：`start-direct` = 用 `DRY_RUN=1` 取出 launcher 的**原样命令行**再执行（配置完全一致，仅跳过预检）。
注：实测真权重 `Loading weights took 4.10s`，**预检担心的 mmap 失败并未发生**。

### 3.9 SM75 专属运行时环境（缺一即启动失败）
```
ImportError: FLASHQLA_ROOT is required for the SM75 legacy backend
```
必须导出（照抄 launcher 的 `set_sm75_runtime_env`）：
```
FLASHQLA_ROOT=<repo>/.deps/FlashQLA-SM70-SM75
TORCH_EXTENSIONS_DIR=$FLASHQLA_ROOT/.torch_extensions_vllm_flashqla_legacy
PYTHONPATH=<repo>:$FLASHQLA_ROOT
```
另需 `--gdn-prefill-backend flashqla_legacy`（或 `--additional-config '{"gdn_prefill_backend":"flashqla_legacy"}'`）。

### 3.10 `--load-format dummy` 是个好用的"零权重验证"手段
只需 `config.json` + tokenizer（几 MB）即可实例化**真实架构**（同形状、同量化位宽、同显存足迹），用于验证内核路径/KV 记账/并发/offload。
注意：checkpoint 指纹会**递归扫描 MODEL_DIR**，所以下载的临时分片目录不要放在 MODEL_DIR 里。

---

## 4. fork 自身的问题清单

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| 1 | `tests/tools/test_flashqla_sm75_lock_recovery.py` | 引用 `_recover_orphan_extension_locks()`，**全仓库不存在**（实现是 `_clear_stale_extension_lock()`） | 3 个用例必红，测试与实现脱节 |
| 2 | `build.sh::install_torch_from_mirror` | 只覆盖 `torch`+`triton`；`torchvision`/`torchaudio`/`torchcodec` 仍打向硬编码 `download.pytorch.org` | 纯国内镜像网络下构建必失败 |
| 3 | `tools/validate_profiles.sh` | 27 白名单把 `profiles/local/` 也算进去 | 用户按文档放本地路线 → 官方自检 `expected exactly 27, found 28` |
| 4 | `launcher.sh::check_checkpoint_mmap_policy` | 非交互无 sudo 时**硬失败无旁路** | 如上；建议加显式 override 开关 |
| 5 | launcher 的 compilation-config | `MODE=normal` 把 `cudagraph_capture_sizes` 钉为 `[1]`（`MODE=fast` 也只快 14%） | 疑似让并发吞吐腰斩（A/B 待完成） |

---

## 5. 实测数据（本机、单卡）

### 5.1 设备能力矩阵（同架构、同内核；带 ⚠️ 者为 dummy 权重）
| KV 精度 | MAX_MODEL_LEN | GPU KV tokens | 并发余量 | 显存 | 权重 |
|---|---:|---:|---:|---:|---|
| FP8 | 32K | 246,622 | 7.53× | 18,136 MiB | dummy |
| FP8 | 131K | 501,275 | 3.82× | ~19.8 GiB | dummy（+offload） |
| FP8 | 229K | 292,864 | 1.28× | 18,136 MiB | dummy |
| FP8 | 262K | 294,012 | 1.12× | 18,136 MiB | dummy |
| TQ4NC | 229K | 456,808 | 1.99× | 17,286 MiB | dummy |
| TQ4NC | 262K | 444,079 | 1.69× | 17,154 MiB | dummy |
| TQ4NC+offload | 262K | 915,547（含 CPU 层） | 3.49× | 20,202 MiB | dummy |

### 5.2 真权重（9B FP8，`compressed-tensors`，SHA256 已校验）
| 项 | 数值 |
|---|---|
| 加载 | 4.10 s（11.91 GB） |
| 质量闸门 | 4/4（Paris / 2 / Water / 8K 长上下文取回 Paris） |
| 4K/128 参考口径 | prefill **~3,020 tok/s**、decode **47.2 tok/s**、TTFT 1.36 s |
| 并发 1/2/4（normal） | 47.29 / 40.08 / **74.80** tok/s 聚合，**0 preemption** |
| 并发 1/2/4（fast） | 47.28 / 42.53 / **85.25** tok/s 聚合，**0 preemption** |
| 关键内核 | Marlin weight-only FP8（SM75 回退）、FlashQLA legacy SM75 GDN prefill、Triton GDN decode、mamba align（attention block 1056） |

### 5.3 与 llama.cpp/KVMem 基线（同机、单卡）
| | vLLM 9B FP8（实测） | llama.cpp 27B Q2（已跑通） |
|---|---:|---:|
| 单流 decode | 47.2 tok/s | 45–48 tok/s |
| 单流 prefill (4K) | **~3,020 tok/s** | 361–380 tok/s（~8×） |
| 并发聚合 | 74.8–85.3 tok/s @4 路 | 无同类能力 |
| 单卡上下文 | 229K（FP8）/ 262K（TQ4NC） | 256K logical（KV 外置） |
| 模型规模 | 9B FP8 | **27B Q2** |

结论：**长上下文/大模型单选 llama.cpp；多客户端/长文档 prefill 选 vLLM**。二者端口不冲突（8080 vs 8100），但**显存互斥**。

### 5.4 共存预算（ComfyUI 占 4.56 GiB 时）
| GPU_UTIL | vLLM 预算 | KV 余量 | FP8 上下文上限 |
|---|---:|---:|---:|
| 0.90（独占） | 19.33 GiB | 4.61 GiB | ~302K |
| 0.80 | 17.18 GiB | 2.46 GiB | ~161K |
| **0.78（共存上限）** | 16.75 GiB | 2.03 GiB | ~133K |
| 0.70 | 15.04 GiB | 0.32 GiB | ~21K |

---

## 6. 未完成 / 待观察（不要当成结论）

1. **真权重的 131K/229K/262K 长上下文**：仅 dummy 验证过；真权重尝试被上述显存争用挡住（编排缺陷已修）。
2. **真权重的分层 offload**：dummy 下已验证（CPU mmap 4.29 GB + fs 层建立；同 prompt 第二次 TTFT 12.53 s → **0.244 s，51×**），真权重待跑。
3. **cudagraph capture sizes A/B**：dummy 下"自动 [1,2]"时 8 路聚合 259 tok/s，而 launcher 的 `[1]` 下 4 路只有 74.8–85.3 → 疑似 ~2× 差距，待真权重 A/B 确认。
4. **长跑服务上的 prefill 递减异常**：在"先服务过长上下文/混合流量"的服务上复现两次（prefill 2948→806→453，TTFT 每次 +3.7 s），**全新服务不复现**；已排除降频（63°C/1845 MHz/97–100%）与 prefix cache（开关都出现）。需要专门复现才有定论。
5. `MODE=fast` 的实际收益在真权重下只有 +14%（不是换档就有质变）。

---

## 7. 一页速查：下次要做什么

```bash
# 1) 想独占 GPU 跑满 262K
先停 ComfyUI → GPU_UTIL=0.90 → 单卡 262K（FP8 余量 1.12×；TQ4NC 余量 1.69×）

# 2) 与 ComfyUI 共存
GPU_UTIL=0.78 → 约 133K 上下文（vLLM 会拒绝更高 util，这是保护而非故障）

# 3) 参考口径（与仓库 README 同口径）
关 prefix cache；4K/128 ×3 取中位数；长样本用 ≥131K 的 profile

# 4) 并发结论只在同一 capture-size 策略下可比
launcher 默认 capture_sizes=[1]；要对比"理论上限"需显式放宽（见 .tooling/ab_cudagraph.sh）
```
---

## 8. 追加：GGUF 清单（local-ai-zone）的核对结果与更好的替代路线

数据源：`https://local-ai-zone.github.io/gguf_models.json`（**38,814 条** GGUF 条目，字段含
`quantFormat` / `fileSize` / `minRamGB` / `huggingFaceLink`）。该文件单流下载会被代理打断，
用 `.tooling/dl_ranged.sh` 多连接分块可稳定拉全（33.97 MB / 8 块）。

### 8.1 vLLM 能不能用它？——不能，而且阻塞点是**架构**不是量化类型
实测 `vllm-gguf-plugin`（aliyun PyPI 镜像上有 `0.0.5`，cp310-abi3 可装）：
- 支持的 GGML 类型**很广**：`IQ1_S/IQ1_M/IQ2_XXS…IQ4_XS/Q2_K/Q3_K/Q4_K_M/Q8_0` 等都在（所以"K-quant 不被支持"这个说法是过时的）；
- 但代码里**完全没有** `qwen3_5` / `hybrid` / `gdn` / `linear_attention` / `mamba` 的处理 →
  Qwen3.8 的混合 GDN 线性注意力不被支持。
⇒ **任何 Qwen3.8 GGUF（即使 5.71 GB 的 IQ1_S 27B）都无法被 vLLM 加载。**

### 8.2 27B 在单卡的最终结论（三条路都不通）
| 路线 | 结果 |
|---|---|
| vLLM 支持的 safetensors 量化 | 最小 18.60 GB（RedHatAI INT4）→ 权重后 KV 余量 ≈ 0 → **不可服务** |
| GGUF + 官方插件 | 架构不支持（见 8.1） |
| 权重卸载 `--cpu-offload-gb` | 能塞进显存，但被卸载权重每 token 走 PCIe → 估 decode 5–11 tok/s |
⇒ 27B 的最优解仍是 llama.cpp（13 GB Q2 mix + KVMem 256K，45–48 tok/s）。

### 8.3 清单带来的**更好路线**：同家族小模型（safetensors，vLLM 原生支持）
清单里出现 `Qwen3.8 2b/4b Distill`，其 HF safetensors 版本（tags 均含 `qwen3_5`，与已验证的
9B/27B 同架构，走同一套 SM75 GDN/FlashQLA 路径）：

| 模型 | 体积 | 单卡 KV 余量（util 0.90，扣 2 GiB 激活） | FP8 KV 容量量级 |
|---|---:|---:|---:|
| `empero-ai/Qwen3.8-2B-Distill` | 4.57 GB | ~13 GiB | **~800K tokens** → 256K × 3 并发 |
| `empero-ai/Qwen3.8-4B-Distill` | 9.34 GB | ~8.6 GiB | **~550K tokens** → 256K × 2 并发 |

（实际 KV/token 取决于层数与 full-attention 层数，需按 config.json 核算；9B 实测为 16 KiB/token。）
⇒ 若目标是"单卡多会话 + 长上下文"，**2B/4B 才是这个设备的甜点区**，而不是 27B。

### 8.4 对 llama.cpp 侧的价值
清单含 732 条 Qwen3.8-27B 条目，最小者为 `IQ1_S 5.71 GB`、`IQ2_XXS 8.37 GB`、`IQ2_XS 8.75 GB`
（当前用的是 13.0 GB 的 Q2-LynnStyle 混合配方）。若需要腾出 4.6–7 GB 显存（换更长 KV / 更多
草稿 / 同卡双模型），这些是可选项，代价是质量下降，需实测对比。

---

## 9. 追加：CPU 车道与 CPU+GPU 融合（同机实测的工具链核对）

### 9.1 纯 CPU 引擎 `qwen3.8-27b-in-c`（本机适配结论）
本机 `make strict` **编译通过**（含其自带 HTTP/JSON 与工具调用测试），依赖仅 `AVX2 + 内存`，
与 CUDA/torch 零耦合；权重由 ModelScope 分发（本机已验证可**直连**），带固定 revision 与 SHA256。

| 项 | 结论 |
|---|---|
| 工具链适配 | ✅ 本机 gcc 11.4 与其参考环境一致；`avx2/fma` 具备；单 NUMA 节点（无需绑核） |
| 预期速度 | IQ1_M 6.73GB → **约 1.5–2.5 tok/s**（估计）；Q4_K_M 16.46GB → 约 1.2–1.8 tok/s（参考机 i5-1340P 为 2.52 / 1.88） |
| 内存账 | Q4_K_M 峰值 RSS 14.49GiB；本机 31GB 中约 15.8GB 被 ComfyUI/code-server/ToDesk 提交 → 可用 ~27GB，两者可跑 |
| 定位 | **第三车道**：GPU 被占满时的独立兜底 / 批处理 / 无 GPU 机器移植；**不是** 27B 的主力（同机 llama.cpp GPU 为 45–48 tok/s，差 20–30×） |
| 落地注意 | 端口避开 8080(llama.cpp)/8100(vLLM)/8188(ComfyUI) → 用 `--server 8190`；MTP 默认关闭（其 README 明确"参考负载下验证成本高于收益"） |
| 同门更优选择 | `qwen3.8-flash-next-in-c`（131B-A6B 稀疏 / 51B PLE，8GB 下限，参考机 batch ~10 tok/s）——对这种"CPU 跑超大稀疏模型"的场景比 27B dense 版更有价值 |

### 9.2 三层 CPU+GPU 融合（工具支持已在本机二进制核对）
| 层 | 机制 | 本机支持 | 适用性 |
|---|---|---|---|
| ① KV 层 | 算力在 GPU，KV 主体/老块放内存或 SSD | KVMem（已跑通 256K logical）；vLLM `TieringOffloadingSpec`（实测前缀复用 TTFT 51×） | ✅ 性价比最高 |
| ② **权重层·MoE 专家卸载** | **专家权重放内存，attention + KV 留显存** | llama.cpp 二进制实测支持 `-cmoe/--cpu-moe`、`--n-cpu-moe`、`-ot/--override-tensor`、`--n-gpu-layers`、`--flash-attn`、`--cache-type-k`、`--tensor-split` | ✅ **解锁 >22GB 模型**，且速度损失小（每 token 只激活少数专家） |
| ③ 计算层 | CPU 也参与算子计算 | 无（需 ktransformers 一类；SM75 支持待验证） | 收益低于 ② |

反例：**dense 模型做权重卸载不划算** —— 18.6GB 的 27B INT4 卸载 3GB，被卸载权重每 token 都要过 PCIe
→ 估 decode **5–11 tok/s**（PCIe 带宽主导），远低于同机 llama.cpp 的 45–48 tok/s。

### 9.3 融合/全显存能解锁的具体目标（GGUF 体积取自 local-ai-zone 索引实测）
组合预算：**显存 21.2 GiB（ComfyUI 关闭）+ 内存 27 GiB ≈ 48 GiB**

| 目标 | GGUF 体积 | 放置 | 预期 decode |
|---|---:|---|---|
| Qwen3.6-35B-A3B IQ3_S / Q3_K_M | 16.8–17.3 GB | **全显存** | **100–300 tok/s**（仅 3B 激活；理论上限 ~360） |
| Qwen3.6-35B-A3B Q4_K_M | 22.2 GB | 显存吃紧 → `--n-cpu-moe N` 挪部分专家 | 略降，仍远超 27B dense |
| Qwen3.8-35B-A3B Distill IQ2_M | 12.3 GB | 全显存且可留大 KV | 同档 |
| Qwen3.8-Flash-Next 131B-A6B IQ2 档 | ~32–42 GB | **必须融合**（attention/KV 显存 + 专家内存） | 5–20 tok/s（受内存带宽限） |
⇒ 结论：本卡"用尽"的形态是**稀疏 MoE**（35B-A3B 类），而不是 dense 27B 的更高精度。

### 9.4 命令清单
```bash
# A. 全显存（ComfyUI 关闭）
llama-server -m Qwen3.6-35B-A3B-Q3_K_M.gguf -ngl 99 -fa -c 65536 --cache-type-k q8_0
# B. 部分融合（ComfyUI 占用显存 / 模型更大）
llama-server -m <model.gguf> -ngl 99 --n-cpu-moe 12 -fa -c 32768
# C. 显式张量级放置
llama-server -m <model.gguf> -ngl 99 -ot '\.ffn_(up|down|gate)_exps\.=CPU' -fa
```

### 9.5 vLLM 侧对比与待验证项
- vLLM 的权重卸载 `--cpu-offload-gb`（`uva`/`prefetch`）是**按参数**、非专家感知 → 对 dense 代价极大，对 MoE 不如 llama.cpp 成熟；其分层 KV offload 则已验证有效。
- **待验证（5 分钟可做）**：`vllm-gguf-plugin 0.0.5` 注册了 `qwen3_moe`/`qwen2_moe`。若 `Qwen3.6-35B-A3B` 的 GGUF 架构字段是标准 `qwen3moe`（非混合 GDN），则 vLLM+插件有可能直接加载 10–12GB 的 IQ2 版 35B-A3B，叠加分层 KV offload → 单卡得到"35B 级 + vLLM 并发"。只需 range 读 GGUF 头部确认 `general.architecture`。
