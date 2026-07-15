# PAS Container Images

## Workflow Images

| Container | Image | Purpose |
|---|---|---|
| Setup | `nvcr.io/nvidia/base/ubuntu:22.04_20240212` | Copy scripts, configs, write `.env` |
| Augmentation | `nvcr.io/nvidia/paidf-augmentation:1.0.0` | Image-edit augmentation with MCQ verification |
| Auto-labeling | `nvcr.io/nvidia/paidf-auto-labeling:1.0.0` | Person-attribute captioning via question bank |

## Model Serving Images

These back the in-cluster NIM endpoints (deployed via the NIM operator, not as
tasks inside the OSMO workflow — see `references/nim/README.md`):

| Model | Image | Notes |
|---|---|---|
| Qwen Image Edit 2511 | `vllm/vllm-omni:v0.20.0` | Requires 80 GB VRAM; launch as `vllm serve Qwen/Qwen-Image-Edit-2511 --omni` (the `--omni` flag is required for `/v1/chat/completions` image editing). Do NOT use the Triton NIM `nvcr.io/nim/qwen/qwen-image-edit`, which lacks `/v1/chat/completions`. |
| Gemma 4 31B IT | `vllm/vllm-openai:latest` | Used for VLM verification + captioning |
| Llama 3.1 70B FP8 | `vllm/vllm-openai:latest` | Used for MCQ generation |

## Public Repositories

- Augmentation: https://github.com/NVIDIA/paidf-augmentation
- Auto-labeling: https://github.com/NVIDIA/paidf-auto-labeling

## Build from Source

```bash
# Augmentation
git clone https://github.com/NVIDIA/paidf-augmentation.git
docker build -t paidf-augmentation:latest -f docker/Dockerfile .

# Auto-labeling
git clone https://github.com/NVIDIA/paidf-auto-labeling.git
cd paidf-auto-labeling
docker build -t paidf-auto-labeling:latest -f docker/Dockerfile .
```
