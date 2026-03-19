# Workstation Deployment

Local deployment with self-hosted NIM containers on x86 workstations/servers.

## Prerequisites

- 2 NVIDIA GPUs (24GB+ VRAM) - tested on H100, A100 and RTX 6000
- Docker with NVIDIA Container Toolkit
- NGC API key from https://build.nvidia.com

## Services

| Service | Image | GPU | Port |
|---------|-------|-----|------|
| tts-service | `magpie-tts-multilingual:1.6.0` | 0 | 50151 |
| asr-service | `parakeet-1-1b-ctc-en-us:1.4.0` | 0 | 50152 |
| nvidia-llm | `nemotron-3-nano:1.5.1-variant` | 1 | 18000 |
| python-app | custom | - | 7860 |
| ui-app | custom | - | 9000 |

## Deployment

**First ask:** *"Do you want English-only or Multilingual mode?"*

```bash
cd nemotron-voice-agent
git submodule update --init
cp config/env.example .env
```

Export your NVIDIA API key as an environment variable:

```bash
export NVIDIA_API_KEY=your-api-key
```

Then choose ONE of:

### Option A: English-only (Default)

No additional changes needed. Default configuration uses:
- `NVIDIA_LLM_MODEL=nvidia/nemotron-3-nano`
- `ASR_DOCKER_IMAGE=nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.4.0`

### Option B: Multilingual

Add to `.env`:
```bash
ENABLE_MULTILINGUAL=true
ASR_DOCKER_IMAGE=nvcr.io/nim/nvidia/parakeet-1-1b-rnnt-multilingual:1.4.0
ASR_MODEL_NAME=parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming
ASR_NIM_TAGS=mode=str
NVIDIA_LLM_IMAGE=nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.15.4
NVIDIA_LLM_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
SYSTEM_PROMPT_SELECTOR=llama-3.3-nemotron-super-49b-v1.5/multilingual_voice_assistant
```

### Start Services

```bash
docker login nvcr.io  # Use NVIDIA_API_KEY as password
docker compose up -d
docker compose logs -f  # Monitor startup
```

> **Note:** Deployment may take 30-60 minutes on first run.

**Access:** http://localhost:9000 | Remote: `ssh -L 9000:localhost:9000 user@host`

## GPU Configuration

Default: GPU 0 (ASR+TTS), GPU 1 (LLM)

To change GPU assignment, edit `docker-compose.yml` device_ids.

## Configuration Options

**Transport Mode:** Set `TRANSPORT=WEBSOCKET` in `.env` for WebSocket instead of WebRTC (default).

**Speculative Speech:** `ENABLE_SPECULATIVE_SPEECH=true` in `.env` (enabled by default)

**LLM Options:**

| Model | Image |
|-------|-------|
| `nvidia/nemotron-3-nano` (default) | `nemotron-3-nano:1.5.1-variant` |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | `llama-3.3-nemotron-super-49b-v1.5:1.15.4` |
| `nvidia/nvidia-nemotron-nano-9b-v2` | `nvidia-nemotron-nano-9b-v2:1.12.2` |
| `meta/llama-3.1-8b-instruct` | `llama-3.1-8b-instruct:1.15.4` |

## Production

**TURN Server** (WebRTC NAT traversal):
```bash
TURN_SERVER_URL=turn:your-server:3478
TURN_USERNAME=username
TURN_PASSWORD=password
```

## Stop Services

```bash
docker compose down
```
