# Metacognition Architecture for Qwen3.5-27B

## Latest: GDN Recurrence-Only Refinement (2.4ms, ~0% overhead)

The simplest and cheapest approach: after each response, run ONE "reflection token"
through the 48 GDN recurrence kernels only (no projections, no MLP, no attention).
Cost: 48 layers × 0.05ms = **2.4ms total**. The GDN state absorbs the model's own
output representation, enriching it for the next turn.

This is human-like inhibition: think (fast GDN association) → speak (full model).
The GDN layers ARE the heuristic — O(1) recurrent state, pattern matching from
compressed memory. The attention layers are the deliberation.

### Validated results (HuggingFace, GH200)
- GDN-only thinking runs at **1.7x** the speed of full model (14.1 vs 8.3 tok/s)
- Total two-phase (think 30 tok + respond 120 tok) = **0.69x** baseline time (faster, not slower)
- The thinking phase produces meaningful proto-thoughts that prime the GDN state

### Implementation options (cheapest to most complex)
1. **Recurrence-only (2.4ms)**: Just run the delta rule kernels on the last hidden state. No projections. Essentially free.
2. **GDN-only decode (0.75x per token)**: Generate N thinking tokens with attention layers skipped. The MLP and projections still run. Proven 1.7x faster than full model.
3. **Full Generate→Refine→Speak (1.3-1.5x)**: Three-phase with GDN-only prefill refinement loops. Most powerful but highest overhead.

### Key insight: looping
Each GDN refinement loop with different input genuinely enriches the state — it's
not converging to a fixed point when the input changes each iteration. Multiple
loops = the model has "thought about it, thought about thinking about it."
Cost scales linearly (~2.4ms per loop) but state enrichment compounds.

### Next steps
- Implement option 1 as a one-line addition in vLLM's model forward (after final norm)
- Needs careful gating: only during decode, not prefill/verify, don't modify logits
- Train GDN layers (LoRA) to use the enriched state effectively

---

## Full Architecture: Generate -> Refine -> Speak

A three-phase inference architecture that enables language models to "think about their own output" using the Gated Delta Net (GDN) recurrence state as a metacognitive substrate.

**Phase 1 -- Generate:** Full 64-layer model produces a draft response via normal autoregressive decoding. Nothing changes.

**Phase 2 -- Refine:** The draft response tokens are fed through ONLY the 48 GDN layers in prefill mode. The 16 full-attention layers are skipped entirely. This loop can iterate multiple times. Each pass enriches the GDN recurrence state with the model's "understanding of its own output."

**Phase 3 -- Speak:** Full 64-layer model generates the real response from the same prompt. The GDN layers now operate with enriched recurrence state. The attention layers provide fresh precision. The user sees only this output.

---

## Why GDN Layers Only

### The Recurrence State as Memory

Each GDN layer maintains a recurrence state of shape `[batch, 48 heads, 128, 128]` per layer. This is a compressed key-value associative memory updated by the gated delta rule:

```
state = state * decay_gate + key^T @ (value - state @ key) * beta
```

Key properties:
- **O(1) per token** in decode mode (constant state size regardless of sequence length)
- **Accumulative**: each token's information is folded into the state
- **Associative**: the state maps keys to values -- a learned associative memory
- **Total state**: 48 layers x 48 heads x 128 x 128 x 2 bytes (fp16) = **56.25 MB** (tiny)

### Why Skip Attention Layers

Attention layers:
- Need a KV cache that grows linearly with sequence length
- Operate on explicit token-to-token relationships
- Don't maintain any persistent state beyond the KV cache
- Would require re-computing or extending the KV cache for refinement tokens, creating confusion about what the model is "attending to"

GDN layers:
- Maintain fixed-size recurrence state (no growth)
- Naturally accumulate information across tokens
- The state IS the memory -- running more tokens through it enriches it
- Prefill mode processes all tokens in parallel (chunked delta rule)
- No KV cache needed -- the state is self-contained

### The Refinement Mechanism

After Phase 1, the GDN state contains: `context + what I generated`

After one refinement loop: `context + what I generated + what I processed from my own output`

After two loops: `context + what I generated + what I processed + what I re-processed`

Each loop deepens the GDN state's representation. The attention layers in Phase 3 then operate on hidden states that are informed by this enriched state, producing more considered outputs.

---

## Cost Model

### Baseline Numbers (Qwen3.5-27B)

| Component | Count | Details |
|-----------|-------|---------|
| GDN layers | 48 | ~75% of total layer params |
| Attention layers | 16 | ~25% of total layer params |
| Hidden size | 5120 | |
| Intermediate size | 17408 | MLP bottleneck |
| GDN key heads | 16 | dim 128 |
| GDN value heads | 48 | dim 128 (GQA: 16 key heads expanded to 48) |
| Attention layer indices | 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63 | every 4th layer |

### Per-Token Cost Breakdown

Each decoder layer (GDN or attention) has:
- **Token mixer** (GDN or attention): projections + core operation
- **MLP**: gate_proj + up_proj + down_proj (17408 intermediate)

The MLP dominates cost in both layer types. The token mixer cost differs:
- **Attention**: Q/K/V projections (5120 -> heads*128), attention computation, output projection
- **GDN**: QKV projection (5120 -> key_dim*2 + value_dim), conv1d, delta rule, gate/beta projections, output projection

In practice, **GDN and attention layers cost roughly the same per token** (within ~10%).

### Refinement Cost

Per refinement loop processing N response tokens:
- Process N tokens through 48 GDN layers (skip 16 attention layers)
- Done in **prefill mode** (all N tokens simultaneously, chunked matmuls)
- Cost = 48/64 of a full-model prefill = **0.75x prefill_cost(N)**

Prefill is much faster per-token than decode because:
- Batched matrix multiplications across all tokens simultaneously
- The GDN chunked delta rule is highly parallelized (chunk_size=64)
- No sequential token-by-token dependency

### Total Overhead

For a 200-token response with 2 refinement loops:

| Phase | Cost (relative to baseline generation) |
|-------|---------------------------------------|
| Generate (draft) | 1.0x |
| Refine (2 loops x 200 tokens, 48 layers, prefill) | ~0.3-0.5x |
| Speak (final generation) | ~1.0x |
| **Total (refine + speak only)** | **~1.3-1.5x** |
| **Total (all three phases)** | **~2.3-2.5x** |

If the draft can be made cheaper (greedy, shorter) or eliminated (use the prompt itself as refinement input), overhead drops to 1.3-1.5x.

---

## Implementation Plan for vLLM

### Overview

The metacognition loop integrates into vLLM's scheduling and execution pipeline. The refinement phase is a **prefill operation** that only touches GDN layers.

### Files to Modify

#### 1. `vllm/engine/llm_engine.py` -- Request lifecycle

Add a new request state: `REFINING`. After a request completes generation (draft), it enters the refinement phase before the speak phase.

```
WAITING -> RUNNING (generate draft) -> REFINING (GDN-only prefill) -> RUNNING (speak) -> FINISHED
```

#### 2. `vllm/worker/model_runner.py` -- Execution

Add `execute_refinement()` method that:
- Takes the generated token IDs
- Runs them through only GDN layers in prefill mode
- Updates the GDN recurrence state in the existing cache
- Does NOT touch the attention KV cache

Requires a modified forward pass that skips attention layers. The cleanest approach is a `refinement_mode` flag on the model runner that filters layers during execution.

#### 3. `vllm/model_executor/models/qwen3_5.py` -- Model definition

Add a `forward_refinement()` method to the Qwen3.5 model class:
- Accepts `refinement_mode=True`
- Skips all `full_attention` layers
- Only processes through GDN layers + their MLPs
- Uses the existing cache object (updates GDN recurrent state in-place)

Key code path:
```python
for i, layer in enumerate(self.layers):
    if layer.layer_type == 'full_attention' and refinement_mode:
        continue  # skip attention layers
    hidden_states = layer(hidden_states, ...)
```

#### 4. `vllm/core/scheduler.py` -- Scheduling

Refinement needs to be scheduled as a prefill operation (processes multiple tokens at once):
- Refinement requests batched with other prefills when possible
- Memory budget: refinement does not grow KV cache (GDN state is fixed-size)
- Priority: refinement requests should have high priority (user is waiting)

#### 5. `vllm/sequence.py` -- Sequence state

Add fields:
- `refinement_loops_remaining: int` -- refinement passes left
- `draft_token_ids: List[int]` -- generated draft for refinement input

#### 6. `vllm/sampling_params.py` -- API

Add parameters:
- `metacognition_loops: int = 0` -- number of refinement loops (0 = disabled)
- `metacognition_draft_tokens: int = 0` -- max tokens for draft (0 = same as max_tokens)

### Cache Management

During refinement, only GDN recurrence states are updated. The attention KV cache from the draft phase must be discarded before the speak phase:

1. The speak phase re-processes the prompt through attention layers
2. The draft's attention KV cache would confuse the speak phase

But the GDN recurrence state is preserved across all phases.

In vLLM's block manager:
- GDN state blocks: pinned (never freed between phases)
- Attention KV blocks: freed after draft phase
- New attention KV blocks: allocated for speak phase

---

## Training Plan: Teaching GDN Layers to Use Refinement

### The Problem

The model was trained with GDN state accumulated from a single forward pass. It has never seen GDN state enriched by processing its own output. Refinement works at inference because GDN state naturally accumulates, but it would work much better with training.

### Phase 1: Data Collection

1. Generate responses to diverse prompts using the base model
2. For each (prompt, response) pair, run refinement to produce enriched GDN states
3. Generate "speak" responses from the enriched states
4. Score all responses (base vs. refined) using a reward model
5. Identify cases where refinement helps vs. hurts

### Phase 2: LoRA Fine-Tuning on GDN Layers

Train LoRA adapters on the 48 GDN layers to improve their use of refinement input.

**Training objective**: After refinement, GDN layers should produce hidden states that lead to better final responses.

**LoRA targets** (per GDN layer):
- `in_proj_qkv` -- how the layer reads input
- `in_proj_z` -- gating projection
- `in_proj_b` -- beta (write strength) projection
- `in_proj_a` -- alpha/decay projection
- `out_proj` -- output projection

**NOT targeted** (frozen):
- All attention layer weights
- All MLP weights
- Embedding and LM head

**Training procedure**:
1. Forward pass through full model (generates draft)
2. Refinement pass through GDN layers (with LoRA)
3. Speak pass through full model (GDN layers have LoRA)
4. Loss on speak output vs. preferred response
5. Backprop through speak + refinement (only LoRA params update)

**LoRA config**:
- Rank: 16-64
- Alpha: 32-128
- Target modules: GDN projections only
- Trainable parameters: ~50-200M (vs 27B total)

### Phase 3: Reinforcement Learning (DPO)

After LoRA fine-tuning:
- Pairs: (prompt, base_response, refined_response)
- Reward: prefer refined response when genuinely better
- Penalty: penalize refinement when it degrades quality

### Phase 4: Adaptive Refinement

Train the model to predict whether refinement helps:
- Small classifier head on GDN state after Phase 1
- Predicts: should we refine? How many loops?
- Saves compute on easy prompts, invests on hard ones

---

## Key Metrics

1. **Response quality**: MMLU, HumanEval, MT-Bench with/without refinement
2. **Diversity**: token-level and semantic difference between base and refined responses
3. **GDN state delta**: L2 norm of state change during refinement (should be significant but bounded)
4. **Latency overhead**: wall-clock time for refinement relative to generation
5. **Consistency**: does refinement reliably help or is it noisy?

---

## Open Questions

1. **Optimal loop count**: Diminishing returns likely after 1-3 loops
2. **What to refine on**: Full response? Reasoning steps only? Just the conclusion?
3. **Conv state handling**: The 1D conv state in GDN layers is also updated during refinement -- should we reset it between loops?
4. **MLP contribution**: During refinement, MLPs also process hidden states -- are they helping or adding noise? Skip MLPs too for even cheaper refinement?
5. **State initialization for speak**: Average pre/post-refinement GDN states, or use post-refinement directly?
