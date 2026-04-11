# vllm-qwen-patches

Bug fixes for vLLM 0.19.0 speculative decoding with Qwen 3.5-27B.

8 bugs found. All in the speculative decoding path for MTP (Multi-Token Prediction) and modal_mtp (DeltaNet self-speculative) modes on multimodal Qwen3.5 models.

## Patches

### `eagle.patch` — 5 fixes for `vllm/v1/spec_decode/eagle.py`

Enables tree speculation (`propose_tree()`) on Qwen3_5ForConditionalGeneration (multimodal, M-RoPE).

| # | Bug | Fix |
|---|---|---|
| 1 | `self.positions.device` crashes on M-RoPE models (no `self.positions` attribute) | Use `positions` parameter |
| 2 | `self.positions[:n] = ...` crashes same way | Use `self._set_positions()` |
| 3 | `positions=self.positions[:n]` crashes same way | Use `self._get_positions()` |
| 4 | Unconditional tuple unpack of model output breaks MTP | Check `model_returns_tuple()` |
| 5 | Positions assumed 1D but M-RoPE gives `(3, batch)` | Extract `positions_1d = positions[0]`, expand back for `_set_positions` |

```bash
cd /path/to/vllm
patch -p1 < eagle.patch
```

### `qwen3_next.patch` — fix for `vllm/model_executor/models/qwen3_next.py`

Fixes tensor shape issue in compiled forward pass when modal_mtp toggles between `input_ids` and `inputs_embeds`.

```bash
cd /path/to/vllm
patch -p1 < qwen3_next.patch
```

### `modal_mtp.py` — new file: `vllm/v1/spec_decode/modal_mtp.py`

DeltaNet self-speculative decoding proposer. Uses the main model's DeltaNet (linear attention) layers for drafting, skipping full attention layers. 3 bugs fixed:

1. AOT compiled graph mismatch — pass `input_ids=None, inputs_embeds=tensor` with `skip_compiled=True`
2. Missing GDN attention metadata — build proper `GDNAttentionMetadata` per layer
3. DeltaNet state corruption — **known issue**, draft forwards write to recurrent state cache

**Status:** Functional but not production-ready. DeltaNet state snapshot/restore costs 40GB. Without it, draft acceptance is 0%.

## Affected models

- Qwen3_5ForConditionalGeneration (multimodal variant with M-RoPE)
- Qwen3.5-27B, Qwen3.5-27B-AWQ, Qwen3.5-27B-GPTQ

## vLLM version

Tested against vLLM 0.19.0 (`vllm-0.19.0-cp38-abi3-manylinux_2_31_aarch64.whl`).

## Apply all patches

```bash
VLLM=$(python3 -c "import vllm; print(vllm.__path__[0])")
cd $(dirname $VLLM)
patch -p1 < eagle.patch
patch -p1 < qwen3_next.patch
cp modal_mtp.py $VLLM/v1/spec_decode/modal_mtp.py
```

## License

Apache-2.0
