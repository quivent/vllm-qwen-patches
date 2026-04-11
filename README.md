# vllm-qwen-patches

Bug fixes for vLLM 0.19.0 speculative decoding with Qwen 3.5-27B.

8 bugs found across 3 files. Each patch is independent and can be applied separately.

## Quick start

```bash
git clone https://github.com/quivent/vllm-qwen-patches
cd vllm-qwen-patches
chmod +x apply.sh

# Check your vLLM installation
./apply.sh check

# Apply the safe patches (eagle + qwen3_next)
./apply.sh all

# Or apply individually
./apply.sh eagle
./apply.sh qwen3_next

# Revert everything
./apply.sh revert
```

## Patches

### `eagle` — 5 fixes for tree speculation with MTP on multimodal models

**File:** `vllm/v1/spec_decode/eagle.py`

Enables `propose_tree()` on Qwen3_5ForConditionalGeneration (multimodal variant with M-RoPE).

| # | Bug | Impact |
|---|---|---|
| 1 | `self.positions.device` crashes — attribute doesn't exist on M-RoPE models | Server crash on tree spec config |
| 2 | `self.positions[:n] = ...` same crash | Server crash during drafting |
| 3 | `positions=self.positions[:n]` same crash | Server crash during drafting |
| 4 | Unconditional tuple unpack of model output | Crash with MTP models (return single tensor, not tuple) |
| 5 | Positions assumed 1D but M-RoPE gives `(3, batch)` | Tensor shape mismatch in tree position math |

**Apply:** `./apply.sh eagle`

**Safe to apply:** Yes. Only affects tree speculation code path. Standard MTP chain is unaffected.

### `qwen3_next` — tensor shape fix for modal_mtp

**File:** `vllm/model_executor/models/qwen3_next.py`

Fixes IndexError when modal_mtp toggles between `input_ids` and `inputs_embeds` in the compiled forward pass.

**Apply:** `./apply.sh qwen3_next`

**Safe to apply:** Yes. Only affects the modal_mtp draft code path.

### `speculative` — allow standalone draft models for Qwen 3.5

**File:** `vllm/config/speculative.py`

Guards `hf_config_override()` so standalone Qwen 3.5 draft models aren't forced into MTP extraction mode. Without this, `--speculative-config '{"method": "draft_model", "model": "/path/to/draft"}'` fails for any model with `model_type == "qwen3_5"`.

**Apply:** `./apply.sh speculative`

**Safe to apply:** Use with caution. This changes MTP detection logic. Test your specific config after applying.

### `modal_mtp` — DeltaNet self-speculative proposer

**File:** `vllm/v1/spec_decode/modal_mtp.py` (new file)

Self-speculative decoding using the main model's DeltaNet linear attention layers for drafting. 3 bugs fixed from the original implementation.

**Apply:** `./apply.sh modal_mtp`

**Known issue:** DeltaNet recurrent state is corrupted by draft forwards without snapshot/restore (40GB memory cost). Draft acceptance rate drops to 0% after the first request. Not production-ready — research only.

## Which patches does your friend need?

**Running Qwen 3.5 with standard MTP speculation?**
No patches needed. Stock vLLM 0.19.0 works fine for the default `qwen3_5_mtp` method.

**Want tree speculation (branching draft candidates)?**
Apply `eagle`. Then use `--attention-backend TREE_ATTN --enforce-eager` with a `speculative_token_tree` config.

**Want to use a separate draft model (not MTP head)?**
Apply `eagle` + `speculative`.

**Want DeltaNet self-speculative (experimental)?**
Apply all four: `eagle` + `qwen3_next` + `speculative` + `modal_mtp`.

## Verify

After patching, verify syntax:
```bash
python3 -c "
import py_compile
py_compile.compile('$(python3 -c \"import vllm; print(vllm.__path__[0])\")/v1/spec_decode/eagle.py', doraise=True)
print('OK')
"
```

## Tested on

- vLLM 0.19.0 (`vllm-0.19.0-cp38-abi3-manylinux_2_31_aarch64.whl`)
- NVIDIA GH200 480GB
- Qwen 3.5-27B W4A16 (Huihui-abliterated)
- CUDA 13.0, Python 3.10

## License

Apache-2.0
