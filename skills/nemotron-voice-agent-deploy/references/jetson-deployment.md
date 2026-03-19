# Jetson Thor Deployment

Edge deployment using vLLM (not NIM) + Nemotron Speech ASR and TTS.

> vLLM used because Jetson Thor NIMs not yet available. Swap to NIMs when released.

> **Required:** `ENABLE_SPECULATIVE_SPEECH=false` (already set in `config/env.jetson.example`). Multilingual mode not supported on Jetson. WebSocket transport only.

## Prerequisites

- JetPack 7.0 (Ubuntu 22.04/24.04)
- NGC CLI configured ([install](https://org.ngc.nvidia.com/setup/installers/cli))
- HuggingFace token (for LLM download)

## Supported Models

**LLM (vLLM):**

| Model | Params | Quant |
|-------|--------|-------|
| `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` | 8B | W4A16 |
| `nvidia/Nemotron-Mini-4B-Instruct` | 4B | FP16 |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4` | 9B | NVFP4 |

**ASR:** `parakeet-1.1b-en-US-asr-streaming`
**TTS:** `magpie_tts_ensemble-Magpie-Multilingual` (Aria voice)

## Deployment

### 1. Deploy Nemotron Speech ASR and TTS models

```bash
ngc config set  # Configure NGC CLI with your API key

ngc registry resource download-version nvidia/riva/riva_quickstart_arm64:2.24.0

cd riva_quickstart_arm64_v2.24.0
bash riva_init.sh   # 30-60 min first run
bash riva_start.sh
docker ps | grep riva  # Verify
```

### 2. Deploy Voice Agent

```bash
cd nemotron-voice-agent
git submodule update --init
cp config/env.jetson.example .env
```

Export your NVIDIA API Key and HuggingFace token as environment variables:

```bash
export NVIDIA_API_KEY=your-nvidia-api-key    # Replace with your NVIDIA API key
export HF_TOKEN=your-huggingface-token       # Replace with your HuggingFace token
```

Edit `.env` and set the following:
```
NVIDIA_LLM_MODEL=RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16
```


```bash
docker compose -f docker-compose.jetson.yml up -d
docker logs -f llm-nvidia-jetson  # Monitor LLM startup
```

> **Note:** Deployment may take 30-60 minutes on first run.

### 3. Access

- Local: `http://localhost:8081`
- Remote: `http://<JETSON_IP>:8081`
- Port forwarding: `ssh -L 8081:localhost:8081 user@jetson-thor`

## Configuration

**Switch LLM:**
```bash
# Edit NVIDIA_LLM_MODEL in .env
docker compose -f docker-compose.jetson.yml down
docker compose -f docker-compose.jetson.yml up -d
```

**GPU memory:** Adjust `GPU_MEMORY_UTILIZATION` (default 0.15 = 15% of ~122GB) for LLM deployment.

## Stop Services

```bash
docker compose -f docker-compose.jetson.yml down
bash riva_stop.sh
```
