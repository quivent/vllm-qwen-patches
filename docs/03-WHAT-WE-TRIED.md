# What We Tried — Complete History

## Optimizations That Worked

### 1. MTP=5 speculative decoding
- **What**: Model guesses 5 tokens ahead, verifies in batch
- **Impact**: ~2.5x effective throughput
- **Status**: DEPLOYED

### 2. Performance mode: interactivity
- **What**: vLLM scheduling optimized for low-latency single requests
- **Impact**: +28% batch=1 throughput
- **Status**: DEPLOYED

### 3. Prefix caching
- **What**: Reuse KV cache for repeated prompt prefixes
- **Impact**: Free speedup on repeated system prompts
- **Status**: DEPLOYED

### 4. INT8 embeddings
- **What**: Quantize 248K x 5120 embedding table from FP16 to INT8 on GPU
- **Impact**: +18% KV cache (22K -> 26K tokens), zero throughput penalty
- **Status**: DEPLOYED

### 5. Thinking/reasoning disabled
- **What**: Modified chat template to skip the `<think>` block
- **Impact**: No wasted tokens on verbose self-narration
- **Status**: DEPLOYED

### 6. Vision encoder removed
- **What**: Stripped 333 vision tensors (0.92 GB) from the model
- **Impact**: More VRAM for KV cache
- **Status**: DEPLOYED (using textonly model variant)

## Optimizations That Were Explored But Not Deployed

### 7. GDN inhibition cycle (metacognition)
- **What**: After processing the prompt, run one extra pass through the 48 GDN recurrence kernels before generating
- **Why not deployed**: The cycle updates GDN state but doesn't change the hidden_states that produce logits. To be useful, it needs to modify hidden_states, but that breaks MTP (trained on un-cycled states). Needs MTP retraining.
- **Result**: Side-effect-only version showed no measurable impact. Output-modifying version broke MTP (51% -> 34%).
- **Path forward**: Retrain MTP head against cycled hidden states, then deploy both together.

### 8. Recurrent-rollback for MTP verification
- **What**: Save GDN state checkpoints during MTP verify, restore on rejection instead of recomputing
- **Impact**: Saves ~3.7ms per MTP rejection
- **Status**: Patch written and tested (all tests pass). Not integrated into vLLM's spec decode loop — needs deeper scheduler integration.

### 9. CPU embedding offload
- **What**: Move embedding table to CPU, transfer 10KB per token
- **Impact**: +41% KV cache (22K -> 31K), but -53% batch=4 throughput
- **Status**: NOT DEPLOYED — throughput penalty too severe for batch workloads

### 10. FP8 KV cache
- **What**: Store attention keys/values at 8-bit instead of 16-bit
- **Impact**: Would double KV cache capacity
- **Status**: BLOCKED by NixOS — flashinfer JIT kernel compilation fails (dynamic linking issues)

### 11. Flashinfer attention backend
- **What**: Alternative attention implementation
- **Status**: CRASHES with GDN/DeltaNet layers. Incompatible.

### 12. MTP head retraining (distillation)
- **What**: Train MTP head against quantized model's actual outputs
- **Impact**: Should improve acceptance from 50% to 60-65%
- **Status**: Training script works, but saving retrained weights in compressed-tensors format breaks (BF16 weights in INT4 checkpoint = 0% acceptance). Needs proper format-aware saving.

### 13. AWQ quantization (our own)
- **What**: Quantized Huihui-Qwen3.5-27B-abliterated with AutoAWQ
- **Impact**: MTP acceptance dropped to 31% (vs 51% GPTQ)
- **Root cause**: AutoAWQ's activation-aware scaling doesn't preserve MTP head correlations as well as GPTQ's Hessian-optimal rounding
- **Status**: Model published on HuggingFace but NOT recommended for MTP workloads

### 14. LoRA fine-tune (direct/concise style)
- **What**: Trained model to be less verbose and more direct
- **Impact**: 31-81% shorter responses in testing
- **Status**: Merged model saved at `/home/ubuntu/models/Qwen3.5-27B-direct/`. Not quantized or deployed.

### 15. DeltaNet self-speculative decoding
- **What**: Use only the 48 GDN layers as a fast draft model
- **Status**: NOT VIABLE — standalone GDN-only model gets 0% acceptance. The 16 attention layers are essential for prediction quality.

## Benchmark History

| Config | s256 tok/s | b4 tok/s | MTP % |
|---|---:|---:|---:|
| Stock MTP=7 (no optimizations) | ~30 | ~98 | — |
| MTP=5 + interactivity | 149 | 410 | 50% |
| + prefix caching | 143 | 416 | 51% |
| + INT8 embeddings | 140 | 453 | 53% |
| MTP=7 baseline | 132 | 299 | 43% |
| MTP=3 | 131 | 418 | 65% |
| AWQ (our quant) | 77 | 313 | 31% |
