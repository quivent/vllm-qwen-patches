# Next Steps — What To Work On

## Priority 1: MTP Retraining (unlocks GDN metacognition)

### The Problem
The MTP head was trained against FP16 model outputs. It predicts what the model will say next. If we change the hidden states (e.g., by running a GDN inhibition cycle), MTP's predictions no longer match, and acceptance drops from 50% to 34%.

### The Solution
Retrain the MTP head against the MODIFIED hidden states (post-GDN-cycle). This is distillation:
1. Run the quantized model on text, with the GDN cycle active
2. Collect (hidden_states_after_cycle, actual_next_token) pairs
3. Train the MTP head to predict correctly from the cycled states
4. Save in compressed-tensors format (critical — BF16 doesn't work)

### What Exists
- Training script: `/home/ubuntu/mtp_distill_train.py`
- Training data collection: worked, 200K samples in 6 minutes
- Training: worked, 3 epochs in 106 seconds
- **Blocker**: saving retrained weights back into compressed-tensors checkpoint. The format stores weights as packed INT4 with scales. Our retrained BF16 weights aren't compatible. Need to either:
  - Re-quantize the retrained MTP head to match the format
  - Or modify llm-compressor to accept mixed-precision (INT4 body + BF16 MTP)

### Expected Impact
- MTP acceptance: 50% -> 60-65%
- That's ~20% more tokens accepted per step
- Net throughput: ~140 tok/s -> ~165-175 tok/s
- PLUS the GDN cycle would now actually modify hidden_states, improving output quality

---

## Priority 2: GDN Metacognition (after MTP retraining)

### The Architecture
```
Prompt arrives
    -> Full 64-layer prefill (normal)
    -> GDN inhibition cycle: run 48 GDN layers on the last hidden state (~2.4ms)
       This modifies hidden_states. The model "thinks" before speaking.
    -> Full 64-layer decode with MTP (retrained to work with cycled states)
```

### What Was Proven
- GDN-only thinking runs 1.7x faster than full model (HuggingFace test)
- Two-phase generation (think + respond) was 31% faster total (HuggingFace test)
- Side-effect-only GDN cycle at production speed: zero overhead (5090 test)
- But: output doesn't change without modifying hidden_states
- And: modifying hidden_states breaks MTP (needs retraining, see Priority 1)

### Implementation
The GDN cycle code is simple (already written and tested):
```python
# After norm, before return in the model's forward():
_s = torch.empty_like(hidden_states)
for layer in self.layers:
    if layer.layer_type == 'linear_attention':
        layer.linear_attn(hidden_states=hidden_states, output=_s)
# Use _s as the new hidden_states (once MTP is retrained)
hidden_states = _s
```

---

## Priority 3: Port to MLX

### Key Insights for MLX
1. **GDN inhibition is free on MLX** — immutable arrays mean state snapshots cost zero
2. **MTP must be calibrated against quantized weights** — don't train on FP16 and deploy on Q4
3. **The recurrent-rollback technique** (quivent/recurrent-rollback) already works on MLX at 42.7 tok/s
4. **Q4_K_M GGUF at 13 GB** is smaller than any GPTQ/AWQ (18-19 GB) because GGUF quantizes embeddings too
5. **Attention-skip thinking**: monkey-patching layers is trivial on MLX (no CUDA graphs to break)

---

## Priority 4: Fix FP8 KV Cache on NixOS

### The Problem
FP8 KV cache would double KV capacity (26K -> ~52K tokens). Blocked by flashinfer's JIT kernel compilation failing on NixOS due to:
- Dynamic linked binaries (ptxas, ninja) need glibc wrappers
- CUDA headers not in the JIT build path
- We fixed ptxas and ninja but flashinfer's FP8 kernels need additional CUDA include paths

### The Fix
Add CUDA include paths to the flashinfer JIT environment:
```bash
export CPATH=/nix/store/ci651krm2gbzk660hbwarqihhmzv9zly-cuda-merged-12.8/include
```
This is already in the systemd service env but the flashinfer JIT subprocess might not inherit it.

---

## Priority 5: Proper AWQ Quantization

### Why
Properly AWQ-quantized model with MTP could be smaller than the current cyankiwi model and have better MTP acceptance.

### How
Use llm-compressor (patched for transformers 5.x, PR submitted) with `Qwen3_5ForConditionalGeneration` (NOT `ForCausalLM` — the latter drops MTP weights). The compressed-tensors format quantizes MTP together with the model.

### What Exists
- llm-compressor patches: quivent/llmcompressor-transformers5
- AutoAWQ patches: quivent/autoawq-qwen35
- PR: vllm-project/llm-compressor#2608
