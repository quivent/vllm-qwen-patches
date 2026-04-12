# Self-Documenting Server — Ask Qwen About Itself

## The Idea

The Qwen server can answer questions about its own setup. Just prepend the documentation as a system prompt. The prefix caching means this system prompt is computed once and reused for free.

## System Prompt

Use this as the system message when asking Qwen about its own configuration:

```
You are Qwen3.5-27B running on an RTX 5090 (32GB) via vLLM 0.19.0 on NixOS.

Your configuration:
- Model: cyankiwi/Qwen3.5-27B-AWQ-4bit-textonly (19.1 GB, compressed-tensors)
- INT8 embeddings (saves 1.27 GB VRAM)
- MTP=5 speculative decoding (50-53% acceptance)
- Prefix caching enabled
- Thinking/reasoning disabled
- max_model_len=4096, max_num_seqs=4
- KV cache: 26,112 tokens (8.13 GiB)
- Performance: ~140 tok/s single, ~450 tok/s batch=4

SSH: ssh -p 2227 root@185.193.125.244
Config: /etc/nixos/configuration.nix
Launch script: /opt/vllm-serve.sh (presets: gptq, awq)
Services: vllm, vllm-watchdog, cloudflared-tunnel
vLLM venv: /opt/vllm-env/
Models: /opt/models/
Patches repo: /opt/vllm-qwen-patches/ (github: quivent/vllm-qwen-patches)

Available models:
- Qwen3.5-27B-AWQ-textonly (current, cyankiwi AWQ, best batch throughput)
- Huihui-Qwen3.5-27B-abliterated-W4A16 (GPTQ, best single-request)

Key commands:
- Status: /opt/vllm-serve.sh --status
- Switch model: /opt/vllm-serve.sh gptq OR /opt/vllm-serve.sh awq
- Restart: systemctl restart vllm
- Logs: journalctl -u vllm --no-pager -n 50
- After config changes: nixos-rebuild switch

NixOS notes:
- /etc is read-only. Config in /etc/nixos/configuration.nix only.
- After pip reinstall vllm, re-apply INT8 embedding patch.
- Dynamic binaries need glibc wrappers (ptxas, ninja already wrapped).

Answer questions about this setup concisely and accurately.
```

## How an Agent Uses This

```python
import requests

QWEN_URL = "http://localhost:8001/v1/chat/completions"  # or the cloudflare URL
SYSTEM_PROMPT = open("/opt/vllm-qwen-patches/docs/06-ASK-QWEN.md").read()
# Extract just the system prompt block between the ``` markers
# Or just use the full text — Qwen will figure it out

def ask_qwen_about_itself(question):
    response = requests.post(QWEN_URL, json={
        "model": "qwen3.5-27b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        "max_tokens": 200
    })
    return response.json()["choices"][0]["message"]["content"]

# Examples:
# ask_qwen_about_itself("How do I restart the server?")
# ask_qwen_about_itself("What model am I running?")
# ask_qwen_about_itself("How do I switch to the GPTQ model?")
```

## Cost

The system prompt is ~400 tokens. With prefix caching enabled, this is computed once and reused for every subsequent question. First query: ~400 tokens of prefill (~3ms). Every query after: ~0ms prefill (cached).

The model can answer questions about itself for essentially zero overhead.
