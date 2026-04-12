#!/usr/bin/env python3
"""
Metacognition Prototype: Generate -> Refine (GDN-only) -> Speak

Architecture:
  1. GENERATE: Full 64-layer model generates a draft response (normal generation).
  2. REFINE:   Feed the draft tokens through ONLY the 48 GDN (linear_attention) layers
               in prefill mode, skipping all 16 full_attention layers. The GDN recurrence
               state accumulates the model's understanding of its own output. Repeat N times.
  3. SPEAK:    Full 64-layer model generates a new response from the same prompt, but now
               the GDN state is enriched by the refinement passes. The user sees only this.

Usage:
    source /tmp/quant-env/bin/activate
    python metacognition_prototype.py
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

MODEL_PATH = "/home/ubuntu/models/Qwen3.5-27B"

# ---------------------------------------------------------------------------
# Test prompts
# ---------------------------------------------------------------------------
TEST_PROMPTS = [
    "What is the capital of Australia and why was it chosen over Sydney or Melbourne?",
    "Explain the difference between nuclear fission and fusion in three sentences.",
    "Write a haiku about the feeling of debugging code at 3am.",
]

NUM_REFINEMENT_LOOPS = 2
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9


def get_gdn_layer_indices(model):
    """Return sorted list of layer indices that are GDN (linear_attention) layers."""
    text_model = model.model  # Qwen3_5TextModel
    indices = []
    for i, layer in enumerate(text_model.layers):
        if layer.layer_type == "linear_attention":
            indices.append(i)
    return indices


def get_attention_layer_indices(model):
    """Return sorted list of layer indices that are full_attention layers."""
    text_model = model.model
    indices = []
    for i, layer in enumerate(text_model.layers):
        if layer.layer_type == "full_attention":
            indices.append(i)
    return indices


def extract_recurrent_states(cache, gdn_indices):
    """Extract a dict of {layer_idx: recurrent_state_tensor} from the cache."""
    states = {}
    for idx in gdn_indices:
        layer_cache = cache.layers[idx]
        if hasattr(layer_cache, 'recurrent_states') and layer_cache.recurrent_states is not None:
            states[idx] = layer_cache.recurrent_states.clone()
    return states


def compute_state_delta(states_before, states_after):
    """Compute per-layer and total L2 norm of recurrent state change."""
    per_layer = {}
    total_sq = 0.0
    for idx in states_before:
        if idx in states_after:
            diff = states_after[idx].float() - states_before[idx].float()
            norm = diff.norm().item()
            per_layer[idx] = norm
            total_sq += norm ** 2
    return per_layer, total_sq ** 0.5


def sample_token(logits, temperature, top_p):
    """Sample a single token from logits using temperature + top-p."""
    if temperature > 0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_token = sorted_indices.gather(-1, torch.multinomial(sorted_probs, 1))
    else:
        next_token = logits.argmax(dim=-1, keepdim=True)
    return next_token.squeeze(-1)


@torch.no_grad()
def generate_autoregressively(model, tokenizer, input_ids, cache, max_new_tokens,
                               temperature, top_p):
    """Standard autoregressive generation from input_ids with a given cache."""
    # First forward: process prompt (or continue from cache)
    outputs = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
    )
    logits = outputs.logits[:, -1, :]
    generated_ids = []

    for step in range(max_new_tokens):
        next_token = sample_token(logits, temperature, top_p)
        generated_ids.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
        outputs = model(
            input_ids=next_token.unsqueeze(0),
            past_key_values=cache,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]

    return generated_ids


@torch.no_grad()
def run_refinement_pass(model, token_ids, cache, device):
    """
    Run one refinement pass: feed token_ids through ONLY the 48 GDN layers.

    This updates the recurrent state in `cache` in-place for all GDN layers.
    Attention layers are skipped entirely.

    The key insight: each GDN DecoderLayer applies:
      1. input_layernorm
      2. linear_attn (GDN) -- updates recurrent state
      3. residual add
      4. post_attention_layernorm
      5. MLP
      6. residual add

    We run this full block for GDN layers, skip it entirely for attention layers.
    The hidden_states flow through GDN layers sequentially, accumulating residuals
    only from GDN layers. This is NOT the same as a full forward pass (attention
    layers contribute nothing), but it lets the GDN state absorb the token content.

    Args:
        model: Qwen3_5ForCausalLM
        token_ids: 1D tensor of token IDs to feed through GDN layers
        cache: DynamicCache with existing state
        device: torch device

    Returns:
        hidden_states: final hidden states (for diagnostics)
        time_taken: wall-clock time for this pass
    """
    t0 = time.time()
    text_model = model.model  # Qwen3_5TextModel

    # Embed the tokens
    input_ids = token_ids.unsqueeze(0).to(device)  # [1, seq_len]
    inputs_embeds = text_model.embed_tokens(input_ids)
    batch_size, seq_len, _ = inputs_embeds.shape

    # Position IDs: continue from where the cache's attention layers left off.
    # This keeps positions consistent even though attention layers are skipped.
    past_seen_tokens = cache.get_seq_length()
    position_ids = torch.arange(seq_len, device=device) + past_seen_tokens
    position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)

    text_position_ids = position_ids[0]
    rope_position_ids = position_ids[1:]

    # Compute RoPE embeddings (needed by decoder layer signature, even if
    # GDN layers don't use rotary embeddings — they use conv1d + delta rule)
    position_embeddings = text_model.rotary_emb(inputs_embeds, rope_position_ids)

    # Linear attention mask: None because we have previous state
    linear_attn_mask = None

    hidden_states = inputs_embeds

    for i, decoder_layer in enumerate(text_model.layers):
        if decoder_layer.layer_type == "full_attention":
            continue  # SKIP attention layers during refinement

        # GDN layer: full forward (norm + GDN + residual + norm + MLP + residual)
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=linear_attn_mask,
            position_ids=text_position_ids,
            past_key_values=cache,
            use_cache=True,
        )

    hidden_states = text_model.norm(hidden_states)
    elapsed = time.time() - t0
    return hidden_states, elapsed


@torch.no_grad()
def generate_with_enriched_state(model, tokenizer, input_ids, enriched_cache,
                                  max_new_tokens, temperature, top_p):
    """
    Generate a response using the full model, but with GDN state from the
    enriched cache.

    Strategy:
      1. Save the enriched GDN recurrent + conv states
      2. Create a fresh cache
      3. Run the prompt through the full model (populates attention KV + GDN state)
      4. Overwrite GDN states with the enriched versions
      5. Generate autoregressively
    """
    t0 = time.time()
    device = input_ids.device

    # Identify layer types from config
    layer_types = model.config.layer_types
    gdn_indices = [i for i, lt in enumerate(layer_types) if lt == "linear_attention"]

    # 1. Save enriched GDN states
    saved_gdn = {}
    for idx in gdn_indices:
        lc = enriched_cache.layers[idx]
        if hasattr(lc, 'recurrent_states') and lc.recurrent_states is not None:
            saved_gdn[idx] = {
                'recurrent_states': lc.recurrent_states.clone(),
                'conv_states': lc.conv_states.clone() if lc.conv_states is not None else None,
                'has_previous_state': lc.has_previous_state,
            }

    # 2. Fresh cache + prompt forward
    fresh_cache = DynamicCache(config=model.config)
    outputs = model(
        input_ids=input_ids,
        past_key_values=fresh_cache,
        use_cache=True,
    )

    # 3. Overwrite GDN states with enriched versions
    for idx, sd in saved_gdn.items():
        lc = fresh_cache.layers[idx]
        if sd['recurrent_states'] is not None and lc.recurrent_states is not None:
            lc.recurrent_states.copy_(sd['recurrent_states'])
        if sd['conv_states'] is not None and lc.conv_states is not None:
            lc.conv_states.copy_(sd['conv_states'])
        lc.has_previous_state = sd['has_previous_state']

    # 4. Autoregressive generation
    logits = outputs.logits[:, -1, :]
    generated_ids = []

    for step in range(max_new_tokens):
        next_token = sample_token(logits, temperature, top_p)
        generated_ids.append(next_token.item())
        if next_token.item() == tokenizer.eos_token_id:
            break
        outputs = model(
            input_ids=next_token.unsqueeze(0),
            past_key_values=fresh_cache,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]

    elapsed = time.time() - t0
    return generated_ids, elapsed


def main():
    print("=" * 80)
    print("METACOGNITION PROTOTYPE: Generate -> Refine (GDN-only) -> Speak")
    print("=" * 80)

    # -------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------
    print(f"\nLoading model from {MODEL_PATH}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded in {time.time() - t0:.1f}s on {device}")

    gdn_indices = get_gdn_layer_indices(model)
    attn_indices = get_attention_layer_indices(model)
    print(f"\nArchitecture: {len(model.model.layers)} layers total")
    print(f"  GDN layers: {len(gdn_indices)} (indices: {gdn_indices[:4]}...{gdn_indices[-2:]})")
    print(f"  Attention layers: {len(attn_indices)} (indices: {attn_indices})")

    config = model.config
    nv = config.linear_num_value_heads
    kd = config.linear_key_head_dim
    vd = config.linear_value_head_dim
    state_mb = nv * kd * vd * 2 / 1024 / 1024  # fp16
    print(f"  GDN state per layer: [{nv} heads, {kd}, {vd}] = {state_mb:.1f} MB")
    print(f"  Total GDN state: {state_mb * len(gdn_indices):.1f} MB across {len(gdn_indices)} layers")

    # -------------------------------------------------------------------
    # Run experiments
    # -------------------------------------------------------------------
    for prompt_idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'='*80}")
        print(f"PROMPT {prompt_idx + 1}: {prompt}")
        print(f"{'='*80}")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[1]
        print(f"Prompt tokens: {prompt_len}")

        # ---------------------------------------------------------------
        # STEP 1: GENERATE (baseline draft)
        # ---------------------------------------------------------------
        print("\n--- STEP 1: GENERATE (baseline) ---")
        torch.manual_seed(42 + prompt_idx)
        t1 = time.time()

        cache_baseline = DynamicCache(config=model.config)
        baseline_ids = generate_autoregressively(
            model, tokenizer, input_ids, cache_baseline,
            MAX_NEW_TOKENS, TEMPERATURE, TOP_P
        )
        baseline_time = time.time() - t1
        baseline_text = tokenizer.decode(baseline_ids, skip_special_tokens=True)
        print(f"  {len(baseline_ids)} tokens in {baseline_time:.2f}s "
              f"({len(baseline_ids)/baseline_time:.1f} tok/s)")
        print(f"  >>> {baseline_text[:400]}")

        # ---------------------------------------------------------------
        # STEP 2: REFINE (GDN-only passes on draft tokens)
        # ---------------------------------------------------------------
        print(f"\n--- STEP 2: REFINE ({NUM_REFINEMENT_LOOPS} loops, GDN-only) ---")

        # Fresh forward pass on prompt to populate cache for refinement path
        torch.manual_seed(42 + prompt_idx)
        cache_refine = DynamicCache(config=model.config)
        _ = model(
            input_ids=input_ids,
            past_key_values=cache_refine,
            use_cache=True,
        )

        # Snapshot GDN state before refinement
        states_before_refine = extract_recurrent_states(cache_refine, gdn_indices)

        # The tokens we refine on = the baseline draft
        refine_tokens = torch.tensor(baseline_ids, dtype=torch.long, device=device)

        total_refine_time = 0.0
        for loop_i in range(NUM_REFINEMENT_LOOPS):
            hidden, loop_time = run_refinement_pass(
                model, refine_tokens, cache_refine, device
            )
            total_refine_time += loop_time

            states_after_loop = extract_recurrent_states(cache_refine, gdn_indices)
            per_layer, total_delta = compute_state_delta(
                states_before_refine, states_after_loop
            )

            # Sample per-layer deltas
            sample = sorted(per_layer.keys())[:3] + sorted(per_layer.keys())[-2:]
            delta_str = ", ".join(f"L{k}={per_layer[k]:.4f}" for k in sample)

            print(f"  Loop {loop_i+1}: {loop_time:.3f}s | "
                  f"L2 delta={total_delta:.4f} | "
                  f"hidden norm={hidden.float().norm():.2f} | "
                  f"samples: {delta_str}")

            # Update baseline for next loop's delta measurement
            states_before_refine = states_after_loop

        print(f"  Total refinement time: {total_refine_time:.3f}s for "
              f"{len(refine_tokens)} tokens x {NUM_REFINEMENT_LOOPS} loops")

        # ---------------------------------------------------------------
        # STEP 3: SPEAK (generate with enriched GDN state)
        # ---------------------------------------------------------------
        print(f"\n--- STEP 3: SPEAK (enriched GDN state) ---")
        torch.manual_seed(42 + prompt_idx)  # Same seed as baseline

        refined_ids, speak_time = generate_with_enriched_state(
            model, tokenizer, input_ids, cache_refine,
            MAX_NEW_TOKENS, TEMPERATURE, TOP_P
        )
        refined_text = tokenizer.decode(refined_ids, skip_special_tokens=True)
        print(f"  {len(refined_ids)} tokens in {speak_time:.2f}s "
              f"({len(refined_ids)/speak_time:.1f} tok/s)")
        print(f"  >>> {refined_text[:400]}")

        # ---------------------------------------------------------------
        # COMPARISON
        # ---------------------------------------------------------------
        print(f"\n--- COMPARISON ---")
        min_len = min(len(baseline_ids), len(refined_ids))
        if min_len > 0:
            matching = sum(1 for a, b in zip(baseline_ids[:min_len], refined_ids[:min_len]) if a == b)
            overlap_pct = matching / min_len * 100
        else:
            matching, overlap_pct = 0, 0.0

        identical = baseline_ids == refined_ids
        print(f"  Baseline: {len(baseline_ids)} tokens")
        print(f"  Refined:  {len(refined_ids)} tokens")
        print(f"  Token overlap (first {min_len}): {matching}/{min_len} = {overlap_pct:.1f}%")
        print(f"  Identical: {identical}")

        # Timing
        overhead_refine = total_refine_time / baseline_time if baseline_time > 0 else 0
        total_metacog = total_refine_time + speak_time
        overhead_total = total_metacog / baseline_time if baseline_time > 0 else 0
        print(f"\n  Timing:")
        print(f"    Baseline generation:     {baseline_time:.2f}s")
        print(f"    Refinement ({NUM_REFINEMENT_LOOPS} loops):    {total_refine_time:.3f}s "
              f"({overhead_refine:.2f}x baseline)")
        print(f"    Speak generation:        {speak_time:.2f}s")
        print(f"    Total (refine + speak):  {total_metacog:.2f}s "
              f"({overhead_total:.2f}x baseline)")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
