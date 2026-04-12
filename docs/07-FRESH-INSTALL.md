# Fresh Install — vLLM + Qwen3.5-27B from Zero

For a standard Linux machine (Ubuntu, Lambda Stack, etc.) — NOT NixOS.
NixOS needs the extra workarounds in `05-NIXOS-GUIDE.md`.

## Requirements

- NVIDIA GPU with 24+ GB VRAM (RTX 3090/4090/5090, A5000, A6000, etc.)
- CUDA 12.x drivers installed
- Python 3.10+
- ~25 GB disk for model + vLLM

## Step 1: Install vLLM (2 minutes)

```bash
python3 -m venv /opt/vllm-env
source /opt/vllm-env/bin/activate
pip install vllm==0.19.0
```

## Step 2: Download model (5-10 minutes)

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('cyankiwi/Qwen3.5-27B-AWQ-4bit', local_dir='/opt/models/Qwen3.5-27B-AWQ')
"
```

## Step 3: Strip vision encoder (saves 0.92 GB VRAM)

```bash
python3 -c "
import os, json, shutil
from safetensors import safe_open
from safetensors.torch import save_file

src = '/opt/models/Qwen3.5-27B-AWQ'
dst = '/opt/models/Qwen3.5-27B-AWQ-textonly'
os.makedirs(dst, exist_ok=True)

all_tensors = {}
for f in sorted(os.listdir(src)):
    if not f.endswith('.safetensors'): continue
    with safe_open(os.path.join(src, f), framework='pt') as sf:
        for k in sf.keys():
            if 'visual' not in k:
                all_tensors[k] = sf.get_tensor(k)

save_file(all_tensors, os.path.join(dst, 'model.safetensors'))

total = sum(t.numel() * t.element_size() for t in all_tensors.values())
index = {'metadata': {'total_size': total}, 'weight_map': {k: 'model.safetensors' for k in all_tensors}}
with open(os.path.join(dst, 'model.safetensors.index.json'), 'w') as f:
    json.dump(index, f, indent=2)

config = json.load(open(os.path.join(src, 'config.json')))
if 'vision_config' in config: del config['vision_config']
with open(os.path.join(dst, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

for fn in os.listdir(src):
    if fn.endswith('.safetensors') or fn == 'config.json': continue
    s = os.path.join(src, fn)
    if os.path.isfile(s): shutil.copy2(s, dst)

print(f'Done: {total/1e9:.1f} GB')
"
```

## Step 4: Disable thinking in chat template

```bash
python3 -c "
path = '/opt/models/Qwen3.5-27B-AWQ-textonly/chat_template.jinja'
t = open(path).read()
if 'set enable_thinking = false' not in t:
    t = '{%- if enable_thinking is not defined %}{%- set enable_thinking = false %}{%- endif %}\n' + t
    open(path, 'w').write(t)
    print('Thinking disabled')
"
```

## Step 5: Apply INT8 embedding patch (saves 1.27 GB VRAM)

```bash
python3 -c "
import torch

# Patch VocabParallelEmbedding
f = '$(python3 -c \"import vllm; print(vllm.__path__[0])\")/model_executor/layers/vocab_parallel_embedding.py'
c = open(f).read()

if 'quantize_to_int8' not in c:
    old = '    def forward_native(self, input_):'
    new = '''    def quantize_to_int8(self):
        if hasattr(self, \"weight\") and self.weight.dtype != torch.int8:
            dev = self.weight.device
            w = self.weight.data.cpu().float()
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
            w_int8 = (w / scale).round().clamp(-127, 127).to(torch.int8)
            scale_f16 = scale.to(torch.float16)
            del self.weight
            torch.cuda.empty_cache()
            self.weight = torch.nn.Parameter(w_int8.to(dev), requires_grad=False)
            self._int8_scale = torch.nn.Parameter(scale_f16.to(dev), requires_grad=False)
            self._embed_int8 = True

    def forward_native(self, input_):'''
    c = c.replace(old, new)

    old2 = '        # Get the embeddings.\n        output_parallel = self.quant_method.embedding(self, masked_input.long())'
    new2 = '''        if getattr(self, \"_embed_int8\", False):
            idx = masked_input.long()
            raw = torch.nn.functional.embedding(idx, self.weight.data)
            scale = torch.nn.functional.embedding(idx, self._int8_scale)
            output_parallel = raw.to(torch.float16) * scale
        else:
            # Get the embeddings.
            output_parallel = self.quant_method.embedding(self, masked_input.long())'''
    c = c.replace(old2, new2)
    open(f, 'w').write(c)

# Patch qwen3_5.py to call quantize_to_int8 after loading
f2 = '$(python3 -c \"import vllm; print(vllm.__path__[0])\")/model_executor/models/qwen3_5.py'
c2 = open(f2).read()
if 'quantize_to_int8' not in c2:
    c2 = c2.replace(
        'return loader.load_weights(weights)',
        'result = loader.load_weights(weights)\n        if hasattr(self.model, \"embed_tokens\"):\n            self.model.embed_tokens.quantize_to_int8()\n        return result',
        1
    )
    open(f2, 'w').write(c2)

print('INT8 embedding patch applied')
"
```

## Step 6: Launch

```bash
source /opt/vllm-env/bin/activate

python3 -m vllm.entrypoints.openai.api_server \
    --model /opt/models/Qwen3.5-27B-AWQ-textonly \
    --served-model-name qwen3.5-27b \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.98 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' \
    --performance-mode interactivity \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

## Step 7: Test

```bash
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.5-27b","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

## Expected Performance (varies by GPU)

| GPU | Single tok/s | Batch=4 tok/s | KV cache |
|---|---:|---:|---:|
| RTX 5090 (32 GB) | ~140 | ~450 | ~33K tokens |
| RTX 4090 (24 GB) | ~80-100 | ~250 | ~15K tokens |
| A6000 (48 GB) | ~100-120 | ~350 | ~80K tokens |
| GH200 (96 GB) | ~190 | ~500 | ~200K tokens |

## Optional: Make it a systemd service

```bash
cat > /etc/systemd/system/vllm.service <<EOF
[Unit]
Description=vLLM Qwen3.5-27B
After=network.target

[Service]
Type=simple
ExecStart=/opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /opt/models/Qwen3.5-27B-AWQ-textonly \
    --served-model-name qwen3.5-27b \
    --host 0.0.0.0 --port 8001 --dtype float16 \
    --max-model-len 4096 --max-num-seqs 4 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.98 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' \
    --performance-mode interactivity \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "video": 0}'
Restart=on-failure
RestartSec=10
Environment="PYTORCH_ALLOC_CONF=expandable_segments:True"

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now vllm
```

## Troubleshooting

| Problem | Fix |
|---|---|
| OOM on startup | Lower `--gpu-memory-utilization` to 0.95 |
| Tokenizer error | Copy tokenizer_config.json from the original (non-textonly) model |
| Slow first request | Normal — CUDA graphs + torch.compile warming up |
| `qwen3_5` not recognized | Need vLLM >= 0.19.0 |
| INT8 patch OOM during quantize | The patch quantizes on CPU to avoid this. If still OOM, lower gpu-memory-utilization |
