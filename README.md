<div align="center">

```text
  _____      _____ _  _    ___  _ _____ ___ _  _ 
 / _ \ \    / / __| \| |  | _ \/_\_   _/ __| || |
| (_) \ \/\/ /| _|| .` |  |  _/ _ \| || (__| __ |
 \__\_\\_/\_/ |___|_|\_|  |_|/_/ \_\_| \___|_||_|
```

**Speculative decoding bug fixes and optimizations for vLLM 0.19.0 + Qwen 3.5-27B.**

*8 robust fixes to supercharge speculative decoding.*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![CUDA](https://img.shields.io/badge/CUDA-13.0-green?style=for-the-badge&logo=nvidia)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)

</div>

---

## 📑 Table of Contents
- [⚡ Results](#-results)
- [🔧 Patches](#-patches)
- [📦 GH200 Quick Start](#-gh200-quick-start)
- [🚀 Usage](#-usage)
- [🧠 Recurrent-Rollback (Patch 7)](#-recurrent-rollback-patch-7)
- [💻 Hardware](#-hardware)
- [🤝 Compatibility & License](#-compatibility--license)

---

## ⚡ Results

Baseline: **186 tok/s** (stock vLLM, MTP spec=7, batch=1, GH200)

| Optimization | tok/s | Change | Status | Patches |
|---|---:|---:|---|---|
| Stock MTP spec=7 (baseline) | 186 | — | Production | None |
| Stock MTP spec=7, batch=8 | 1,030 | +454% agg | Production | None |
| Tree speculation | 27 | -85% | Working, low acceptance | 1, 6 |
| DeltaNet self-speculative (modal\_mtp) | 3.2 | -98% | Working, no CUDA graphs | 2, 4, 5 |
| Standalone DeltaNet draft model | 5 | -97% | 0% acceptance | 3 |
| Sibling MTP heads (weight swap) | 139 | -25% | Swap overhead | None |
| Adaptive MTP chain length | 186 | 0% | All positions profitable | None |
| DeltaNet weight transplant | 174 | -6% | Within noise | None |
| Partial-layer verification (layer 60) | — | -3-7% | Not worth deploying | None |
| Cascade MTP (depth-trained) | 47 | -75% | Training data mismatch | None |

> [!WARNING]
> **No optimization beat baseline.** The 11 patches fix real bugs in experimental vLLM features, but none of those features currently outperform stock MTP on this hardware.

---

## 🔧 Patches

| # | Name | File | Bugs | What it fixes |
|---|---|---|---:|---|
| 1 | `eagle` | `v1/spec_decode/eagle.py` | 5 | Tree speculation crashes on multimodal M-RoPE models |
| 2 | `qwen3_next` | `model_executor/models/qwen3_next.py` | 1 | Tensor shape error in modal\_mtp compiled forward |
| 3 | `speculative` | `config/speculative.py` | 1 | Config forces MTP extraction on standalone draft models |
| 4 | `gdn` | `model_executor/layers/mamba/gdn_linear_attn.py` | 1 | DeltaNet state corruption during draft forwards |
| 5 | `qwen3_5` | `model_executor/models/qwen3_5.py` | 1 | Missing shadow state methods for modal\_mtp |
| 6 | `gpu_model_runner` | `v1/worker/gpu_model_runner.py` | 1 | CUDA graph segfault with tree attention |
| 7 | `rollback` | `gdn_linear_attn.py` + `qwen3_5.py` | 0 | O(1) GDN state rollback for MTP spec decode |

---

## 📦 GH200 Quick Start

> [!TIP]
> Full agent-executable install guide: **[docs/08-GH200-AGENT-INSTALL.md](docs/08-GH200-AGENT-INSTALL.md)**

Sets up vLLM in a venv, downloads the model, applies all patches (eagle + qwen3_next + recurrent-rollback), and launches with MTP=5. Every step has a verification command. No decisions required.

---

## 🚀 Usage

```bash
git clone https://github.com/quivent/vllm-qwen-patches.git
cd vllm-qwen-patches
chmod +x apply.sh

./apply.sh check          # show vLLM version and patch state
./apply.sh eagle          # apply one patch
./apply.sh all            # apply safe patches (1 + 2)
./apply.sh rollback       # apply recurrent-rollback (patch 7)
./apply.sh revert         # restore ALL files to stock from pip wheel
./apply.sh revert eagle   # restore one file to stock
```

> [!NOTE]
> Revert extracts clean files from the pip wheel, not `.bak` files. Also clears torch compile cache.

---

## 🧠 Recurrent-Rollback (Patch 7)

Qwen3.5-27B has 48 DeltaNet (GDN) layers whose recurrence state is non-invertible:

```text
S_{t+1} = g_t * S_t + beta_t * k_t * (v_t - k_t^T @ S_t)
```

The `k_t^T @ S_t` retrieval makes the update state-dependent. When MTP speculative decoding rejects at position K, you cannot algebraically undo the state updates to recover `S_K` from `S_N`. The standard approach is to checkpoint the full state before verification and recompute the forward pass for accepted tokens on rejection -- this costs ~8.7ms per step at 51% rejection rate.

The recurrent-rollback patch saves `.clone()` snapshots of both ssm_state and conv_state at each speculative position during the verification forward pass. On rejection, it restores the correct state with a single `.copy_()` per layer -- O(1) instead of O(K) recomputation.

<details>
<summary><b>View Rollback Cost & API Details</b></summary>

**Memory cost**: 48 layers x 6 positions x ~3.1 MB/checkpoint = ~893 MB. Checkpoints are only allocated during verification passes, not during normal generation.

**Timing** (GH200, measured):
- Rollback: 0.85 ms (48 layers, one `.copy_()` each)
- Checkpoint save: 4.8 ms total (48 layers x 5 positions)
- Eliminated recomputation: ~8.7 ms expected per step

**Net savings**: ~3.1 ms per verification step.

API:
```python
model.setup_rollback_manager(max_positions=6)  # once at init
model.begin_verification()                      # before verify forward
logits = model.forward(draft_tokens)            # auto-saves checkpoints
model.rollback_gdn_state(K - 1, state_index=slot_id)  # on rejection
model.end_verification()                        # release memory
```

Based on the [recurrent-rollback technique](https://github.com/quivent/recurrent-rollback) (originally implemented for MLX). The PyTorch version uses explicit `.clone()` instead of MLX's zero-cost immutable array references.

</details>

---

## 💻 Hardware

| Metric | Value |
|---|---|
| GPU | NVIDIA GH200 480GB |
| HBM3e bandwidth | 4.8 TB/s |
| Bandwidth utilization (batch=1) | 13% |
| MTP acceptance per position | 87 / 68 / 54 / 39 / 28 / 21 / 16% |

---

## 🤝 Compatibility & License

- vLLM 0.19.0
- Qwen 3.5-27B (all quantizations)
- Python 3.10+
- CUDA 13.0.

**License**: Apache-2.0
