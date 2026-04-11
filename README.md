# vllm-qwen-patches

Bug fixes for vLLM 0.19.0 speculative decoding with Qwen 3.5-27B on NVIDIA GH200.

11 bugs found across 6 files. Each patch is independent.

## Setup

```bash
git clone https://github.com/quivent/vllm-qwen-patches
cd vllm-qwen-patches
chmod +x apply.sh
./apply.sh check
```

## Patches

### 1. `eagle` — Tree speculation for multimodal MTP models

**File:** `vllm/v1/spec_decode/eagle.py`
**Bugs:** 5

| Fix | Description |
|-----|-------------|
| positions device | `self.positions.device` crashes on M-RoPE models |
| positions write | `self.positions[:n] = ...` crashes same way |
| positions read | `positions=self.positions[:n]` crashes same way |
| tuple unpack | Unconditional tuple unpack breaks MTP (single tensor return) |
| MRoPE 1D | Positions assumed 1D but M-RoPE gives `(3, batch)` |

```bash
./apply.sh eagle                    # apply
./apply.sh revert eagle             # rollback
```

**Test:**
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/model --speculative-model "[mtp]" \
    --speculative-token-tree "[(0,),(1,),(2,)]" \
    --attention-backend TREE_ATTN --enforce-eager \
    --port 8002 &
curl -s localhost:8002/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"model","messages":[{"role":"user","content":"test"}],"max_tokens":10}'
```

### 2. `qwen3_next` — Tensor shape fix for modal_mtp

**File:** `vllm/model_executor/models/qwen3_next.py`
**Bugs:** 1

Fixes IndexError when modal_mtp toggles between `input_ids` and `inputs_embeds` in the compiled forward pass.

```bash
./apply.sh qwen3_next               # apply
./apply.sh revert qwen3_next        # rollback
```

### 3. `speculative` — Standalone draft model support

**File:** `vllm/config/speculative.py`
**Bugs:** 1

Guards `hf_config_override()` so standalone Qwen 3.5 draft models aren't forced into MTP extraction mode.

```bash
./apply.sh speculative              # apply
./apply.sh revert speculative       # rollback
```

**Test:**
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /path/to/main \
    --speculative-config '{"method":"draft_model","model":"/path/to/draft","num_speculative_tokens":5}' \
    --port 8002 &
```

### 4. `gdn` — DeltaNet shadow state for modal_mtp

**File:** `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
**Bugs:** 1

GDN layers check `_draft_kv_cache` attribute before writing to real cache. Draft forwards write to a shadow cache instead.

```bash
./apply.sh gdn                      # apply
./apply.sh revert gdn               # rollback
```

### 5. `qwen3_5` — Shadow state setup/clear

**File:** `vllm/model_executor/models/qwen3_5.py`
**Bugs:** 1

Adds `setup_draft_shadow_state()` and `clear_draft_shadow_state()` to Qwen3_5Model. Required by modal_mtp with the gdn patch.

```bash
./apply.sh qwen3_5                  # apply
./apply.sh revert qwen3_5           # rollback
```

### 6. `gpu_model_runner` — CUDA graph + tree attention fix

**File:** `vllm/v1/worker/gpu_model_runner.py`
**Bugs:** 1

Downgrades CUDA graph mode from PIECEWISE to NONE when TREE_ATTN backend is used with speculative decoding. Prevents segfault during graph replay with tree-shaped attention metadata.

```bash
./apply.sh gpu_model_runner          # apply
./apply.sh revert gpu_model_runner   # rollback
```

## Apply/Revert

```bash
# Apply individual patch
./apply.sh eagle

# Apply safe patches (eagle + qwen3_next only)
./apply.sh all

# Check patch state
./apply.sh check

# Revert ONE patch (extracts clean file from pip wheel)
./apply.sh revert eagle

# Revert ALL patches to stock (nuclear option, extracts from wheel)
./apply.sh revert

# The revert command also:
#   - Removes non-stock files (modal_mtp.py, etc.)
#   - Clears torch compile cache
```

## Safety

- `./apply.sh revert` extracts clean files directly from the installed pip wheel — it does NOT rely on `.bak` files
- Each patch only touches ONE file
- Standard MTP speculative decoding (`qwen3_5_mtp` method) works without any patches
- Patches are only needed for tree speculation, modal_mtp, or standalone draft models

## Tested on

- vLLM 0.19.0 (`vllm-0.19.0-cp38-abi3-manylinux_2_31_aarch64.whl`)
- NVIDIA GH200 480GB
- Qwen 3.5-27B W4A16
- CUDA 13.0, Python 3.10

## License

Apache-2.0
