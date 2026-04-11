# vllm-qwen-patches

Speculative decoding bug fixes and optimizations for vLLM 0.19.0 + Qwen 3.5-27B.

## Results

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

**No optimization beat baseline.** The 11 patches fix real bugs in experimental vLLM features, but none of those features currently outperform stock MTP on this hardware.

## Patches

| # | Name | File | Bugs | What it fixes |
|---|---|---|---:|---|
| 1 | `eagle` | `v1/spec_decode/eagle.py` | 5 | Tree speculation crashes on multimodal M-RoPE models |
| 2 | `qwen3_next` | `model_executor/models/qwen3_next.py` | 1 | Tensor shape error in modal\_mtp compiled forward |
| 3 | `speculative` | `config/speculative.py` | 1 | Config forces MTP extraction on standalone draft models |
| 4 | `gdn` | `model_executor/layers/mamba/gdn_linear_attn.py` | 1 | DeltaNet state corruption during draft forwards |
| 5 | `qwen3_5` | `model_executor/models/qwen3_5.py` | 1 | Missing shadow state methods for modal\_mtp |
| 6 | `gpu_model_runner` | `v1/worker/gpu_model_runner.py` | 1 | CUDA graph segfault with tree attention |

## Usage

```bash
git clone https://github.com/quivent/vllm-qwen-patches.git
cd vllm-qwen-patches
chmod +x apply.sh

./apply.sh check          # show vLLM version and patch state
./apply.sh eagle          # apply one patch
./apply.sh all            # apply safe patches (1 + 2)
./apply.sh revert         # restore ALL files to stock from pip wheel
./apply.sh revert eagle   # restore one file to stock
```

Revert extracts clean files from the pip wheel, not `.bak` files. Also clears torch compile cache.

## Hardware

| Metric | Value |
|---|---|
| GPU | NVIDIA GH200 480GB |
| HBM3e bandwidth | 4.8 TB/s |
| Bandwidth utilization (batch=1) | 13% |
| MTP acceptance per position | 87 / 68 / 54 / 39 / 28 / 21 / 16% |

## Compatibility

vLLM 0.19.0, Qwen 3.5-27B (all quantizations), Python 3.10+, CUDA 13.0.

## License

Apache-2.0
