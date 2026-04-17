---
name: nemotron-voice-agent-deploy
description: Deploy Nemotron Voice Agent on Workstation (x86), Jetson Thor, or Cloud NIMs. Real-time speech-to-speech using NVIDIA ASR, TTS, LLM with WebRTC/WebSocket transport.
---

# Nemotron Voice Agent Deployment

Real-time conversational AI voice agent using NVIDIA NIMs (ASR, TTS, LLM) with WebRTC (default) or WebSocket transport.

## Deployment Flow

**Always verify hardware first, even if user mentions a specific platform.**

### STEP 1: Hardware Detection

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
```

| Result | Action |
|--------|--------|
| Command fails / No output | → **Cloud NIMs** |
| GPU detected | → **STEP 2: Platform Detection** |

---

### Cloud NIMs (No GPU)

```bash
cd nemotron-voice-agent
git submodule update --init
cp config/env.example .env
```

Export your NVIDIA API key:
```bash
export NVIDIA_API_KEY=your-api-key  # Get from https://build.nvidia.com
```

Then edit `.env`:
```bash
NVIDIA_LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b  # Cloud model name
```

**If user requests WebSocket transport**, also add to `.env`:
```bash
TRANSPORT=WEBSOCKET
```

```bash
docker compose up --build --no-deps -d python-app ui-app
# WebRTC: http://localhost:9000
# WebSocket: http://localhost:7860/static/index.html
```

> **Note:** Deployment may take 30-60 minutes on first run.

**If user requests Multilingual mode**, also add to `.env`:
```bash
ENABLE_MULTILINGUAL=true
ASR_CLOUD_FUNCTION_ID=71203149-d3b7-4460-8231-1be2543a1fca
ASR_MODEL_NAME=parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming
```

**Remote Access:** `ssh -L 9000:localhost:9000 user@host` or `http://<HOST_IP>:9000`

---

### STEP 2: Platform Detection (if GPU detected)

```bash
uname -m  # x86_64 → Workstation, aarch64 → Jetson
cat /etc/nv_tegra_release 2>/dev/null && echo "Jetson"
```

| Platform | Reference | Requirements |
|----------|-----------|--------------|
| Workstation (x86_64) | [workstation-deployment.md](references/workstation-deployment.md) | 2x GPU (24GB+ VRAM), NIM containers |
| Jetson Thor (aarch64) | [jetson-deployment.md](references/jetson-deployment.md) | JetPack 7.0, Nemotron Speech ASR and TTS, vLLM |

> **Note:** Multilingual mode available on Workstation with WebRTC transport only.
