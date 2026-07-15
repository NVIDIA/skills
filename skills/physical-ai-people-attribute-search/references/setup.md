# PAS Setup Guide

## Prerequisites

1. **OSMO CLI**: Install and log in to OSMO. Register a DATA credential
   profile matching your `storage_url`.

2. **Model Endpoints**: PAS requires three OpenAI-compatible model endpoints:

   | Endpoint | Model | Purpose |
   |---|---|---|
   | Image Edit | `Qwen/Qwen-Image-Edit-2511` | Clothing/appearance image editing |
   | VLM | Gemma 4 31B IT (or `Qwen/Qwen3-VL-30B-A3B-Instruct`) | MCQ verification + person-attribute captioning |
   | LLM | `nvidia/Llama-3.1-70B-Instruct-FP8` (or `Qwen/Qwen2.5-14B-Instruct`) | MCQ question generation |

   Endpoints can be served locally via vLLM, through in-cluster NIMs, or via
   hosted services (e.g., build.nvidia.com).

3. **HF Token**: Required for OSMO `hf_token` credential.

4. **Dataset**: Person-crop images organized as:
   ```
   dataset_root/
   ├── person_0001/
   │   ├── view_a.jpg
   │   ├── view_b.jpg
   │   └── view_c.jpg
   ├── person_0002/
   │   ├── view_a.jpg
   │   └── view_b.jpg
   └── ...
   ```

   Upload to OSMO storage:
   ```bash
   osmo data upload /path/to/dataset_root <storage_url>/datasets/<dataset_name>/
   ```

## Local Model Deployment (optional)

If you prefer to host models locally rather than use external endpoints:

### Qwen Image Edit 2511
```bash
export QWEN_EDIT_PORT=8002
docker run -d --rm --name qwen-image-edit \
  --gpus all --ipc=host --shm-size=32g \
  -p ${QWEN_EDIT_PORT}:${QWEN_EDIT_PORT} \
  -e HF_TOKEN=${HF_TOKEN} \
  --entrypoint vllm \
  vllm/vllm-omni:v0.20.0 \
  serve Qwen/Qwen-Image-Edit-2511 \
  --omni \
  --port ${QWEN_EDIT_PORT} \
  --host 0.0.0.0
```

The `--omni` flag is required: vLLM-Omni only serves the Qwen-Image-Edit
diffusion path (the OpenAI-compatible `/v1/chat/completions` image-edit response
the PAS worker consumes) when launched as `vllm serve <model> --omni`.

### Gemma 4 VLM
```bash
export GEMMA_PORT=8003
docker run -d --rm --name gemma-4-31b-it \
  --gpus all --ipc=host --shm-size=32g \
  -p ${GEMMA_PORT}:${GEMMA_PORT} \
  -e HF_TOKEN=${HF_TOKEN} \
  --entrypoint vllm \
  vllm/vllm-openai:latest \
  serve google/gemma-4-31B-it \
  --served-model-name gemma-4-31b-it \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --dtype bfloat16 \
  --enforce-eager \
  --generation-config vllm \
  --limit-mm-per-prompt '{"image": 1, "audio": 0}' \
  --host 0.0.0.0 \
  --port ${GEMMA_PORT}
```

### Llama 3.1 LLM
```bash
export LLAMA_PORT=8004
docker run -d --rm --name llama-3-1-70b-fp8 \
  --gpus all --ipc=host --shm-size=32g \
  -p ${LLAMA_PORT}:${LLAMA_PORT} \
  -e HF_TOKEN=${HF_TOKEN} \
  --entrypoint vllm \
  vllm/vllm-openai:latest \
  serve nvidia/Llama-3.1-70B-Instruct-FP8 \
  --trust-remote-code \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --host 0.0.0.0 \
  --port ${LLAMA_PORT}
```

## Hardware Requirements

| Component | Model | Minimum | Recommended |
|---|---|---|---|
| Image augmentation | Qwen Image Edit 2511 | 80 GB VRAM (Ampere+) | 1x H100 |
| MCQ generation LLM | Llama 3.1 70B FP8 | 2x A100 | 1x H100 |
| VLM verification | Gemma 4 31B | 58.3 GB VRAM | 1x H100 |
| Auto-labeling VLM | Gemma 4 31B | 58.3 GB VRAM | 1x H100 |

GPU requirements apply to the endpoint provider, not the PAS worker pods.
Worker pods only need CPU and storage when using remote endpoints.
