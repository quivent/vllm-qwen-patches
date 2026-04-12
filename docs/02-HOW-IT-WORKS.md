# How It Works — Simple Explanation

## The Model: Qwen3.5-27B

A large language model with 27 billion parameters. It has a special architecture:

- **64 layers** total
- **48 GDN layers** (fast, uses recurrence — like short-term memory)
- **16 attention layers** (slow, precise — like careful reasoning)
- Pattern: GDN, GDN, GDN, Attention, GDN, GDN, GDN, Attention, ... repeating 16 times
- **1 MTP head** (predicts future tokens to speed up generation)

## Why It's Fast

### 1. Quantization (4-bit weights)
The model's weights are compressed from 16-bit to 4-bit. This makes the model 4x smaller in memory (19 GB instead of 55 GB), so it fits on a 32 GB GPU. The quality loss is minimal.

### 2. MTP Speculative Decoding
Instead of generating one token at a time, the model guesses 5 tokens ahead using its MTP head, then verifies them all at once. On average, ~50% of guesses are correct, so we get ~2.5 tokens per step instead of 1.

### 3. CUDA Graphs
The GPU captures the computation pattern once and replays it, avoiding overhead from Python and CUDA setup on each token.

### 4. torch.compile
The model's forward pass is compiled into optimized GPU kernels.

### 5. Marlin Kernel
A specialized GPU kernel for 4-bit matrix multiplication that's much faster than generic implementations.

### 6. Prefix Caching
When multiple requests share the same system prompt, the computation for that prompt is cached and reused.

### 7. INT8 Embeddings
The embedding table (which maps token IDs to vectors) is stored at 8-bit instead of 16-bit, saving 1.27 GB of GPU memory that becomes available for KV cache (= longer conversations).

## Key Concepts

### KV Cache
When the model processes text, it stores intermediate results (keys and values from attention) so it doesn't have to recompute them. More KV cache = longer conversations or more concurrent users.

### MTP (Multi-Token Prediction)
A small neural network head that predicts what the model will say next. It's trained alongside the model. When its predictions are correct, we skip expensive computation.

### GDN (Gated Delta Net)
A type of layer that maintains a compressed "memory" (256x256 matrix per head) that gets updated with each token. Unlike attention (which looks back at all previous tokens), GDN uses a fixed-size state that accumulates information. This is O(1) per token regardless of context length.

## The Numbers

| What | Size |
|---|---|
| Original model (FP16) | 55.6 GB |
| Quantized model (4-bit) | 19.1 GB |
| Embeddings (FP16 -> INT8) | 2.54 GB -> 1.27 GB |
| KV cache available | 8.13 GB = 26,112 tokens |
| GPU total | 32 GB |
