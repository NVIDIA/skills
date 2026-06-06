# VSS Alerts Profile — Reference

Profile: `alerts` | Blueprint: `bp_developer_alerts` | Modes: `2d_cv` (verification) and `2d_vlm` (real-time)

Real-time alert generation and verification on RTSP / live video. VLM is **always served by RT-VLM** (port 8018), same as LVS — there is no standalone Cosmos NIM in the alerts compose graph. The `COMPOSE_PROFILES` for alerts is `${BP_PROFILE}_${MODE},${BP_PROFILE}_${MODE}_${HARDWARE_PROFILE},llm_${LLM_MODE}_${LLM_NAME_SLUG}` — note the absence of `vlm_*_<slug>`. Setting `VLM_NAME_SLUG=none` for alerts is intentional and required.

## Two modes

| Mode | CLI | `MODE` env | `NEXT_PUBLIC_APP_SUBTITLE` | How it works | VLM load |
|---|---|---|---|---|---|
| **verification** | `--mode verification` | `2d_cv` | `Vision (Alerts - CV)` | DeepStream perception (RT-CV with Grounding DINO) generates alerts upstream; behavior analytics filters them; alert-bridge invokes VLM **only** to verify alert clips. | Lower — VLM runs per alert |
| **real-time** | `--mode real-time` | `2d_vlm` | `Vision (Alerts - VLM)` | VLM continuously inspects live video at periodic intervals; broad coverage without upstream CV dependency. RT-CV not deployed. | Higher — VLM runs continuously |

Switch modes by editing `MODE` in `dev-profile-alerts/generated.env` (`MODE=2d_cv` or `MODE=2d_vlm`) and re-resolving the compose. Update `NEXT_PUBLIC_APP_SUBTITLE` in the same file so the UI label matches the deployed mode.

## What's different from `base` and `lvs`

- **VLM is RT-VLM with the integrated Cosmos Reason 2 checkpoint.** Default `RTVI_VLM_MODEL_PATH=ngc:nim/nvidia/cosmos-reason2-8b:hf-1208`, `RTVI_VLM_MODEL_TO_USE=cosmos-reason2`. No standalone `cosmos-reason2-8b` NIM service is started.
- **`VLM_NAME` must match RT-VLM's `/v1/models` basename.** Set `VLM_NAME=nim_nvidia_cosmos-reason2-8b_hf-1208` for the default Cosmos2 path; alert-bridge / agent get HTTP 400 "No such model" otherwise (see `vllm_compatible_model.py::get_model_info` for the lookup logic).
- **`VLM_NAME_SLUG=none`** for alerts — `COMPOSE_PROFILES` does not include a `vlm_local_*_<slug>` segment. The only LLM/VLM compose profile that matters for alerts is `llm_${LLM_MODE}_${LLM_NAME_SLUG}`.
- **`VLM_PORT=8018`** by default (RT-VLM). Set to `30082` when `VLM_MODE=remote` (RT-VLM not started; agent points at the remote endpoint).
- **Alert-bridge** (port 9080) is the bridge between RT-VLM events / behavior analytics and the agent's realtime alerting API. Verification mode reads from RT-CV → behavior analytics → alert-bridge → VLM verification. Real-time mode reads from RT-VLM → alert-bridge directly.

## What gets deployed

Container names below are the actual `container_name:` keys from `deploy/docker/services/**/compose.yml`. LLM/VLM NIM containers are named after the selected model (default shown; varies with `LLM_NAME_SLUG`).

| Service | Container | Port | Purpose | Mode |
|---|---|---|---|---|
| RT-CV (DeepStream perception) | `vss-rtvi-cv` | — (host net) | Object detection (Grounding DINO via `MODEL_NAME_2D=GDINO`) | **2d_cv only** |
| Behavior analytics | `vss-behavior-analytics` | — | Rule-based alerts from RT-CV metadata | **2d_cv only** |
| RT-VLM | `vss-rtvi-vlm` | 8018 | VLM runner (Cosmos Reason 2 by default) | both |
| Alert-bridge | `vss-alert-bridge` | 9080 | Realtime alerting API; drives `POST/DELETE /api/v1/realtime` on the agent | both |
| LLM NIM (default) | `nvidia-nemotron-nano-9b-v2` | 30081 | Same options as `base` (Nano 9B v2 default). Container name = `${LLM_NAME_SLUG}`. | both |
| nvstreamer-alerts | `vss-vios-nvstreamer` | 31000 | Plays back dataset video to simulate live cameras | both |
| VST Ingress | `vss-vios-ingress` | 30888 | Video storage + ingest | both |
| VSS Agent | `vss-agent` | 8000 | Orchestrates alert verification and incident reports | both |
| VSS Agent UI | `vss-agent-ui` | 3000 | Alerts tab | both |
| Video-Analytics MCP | `vss-va-mcp` | 9901 | Analytics API for the agent | both |
| Elasticsearch + Kibana | `elasticsearch`, `kibana` | 9200, 5601 | Alert/event storage | both |
| Kafka | `kafka` | 9092 | Message bus | both |
| Phoenix | `phoenix` | 6006 | Observability | both |

## Default models

| Role | Model | `VLM_NAME` / slug | Served by |
|---|---|---|---|
| LLM | `nvidia/nvidia-nemotron-nano-9b-v2` | `nvidia-nemotron-nano-9b-v2` | NIM (port 30081) |
| VLM | Cosmos Reason 2 8B (integrated) | **`nim_nvidia_cosmos-reason2-8b_hf-1208`** / slug **`none`** | RT-VLM (port 8018), `MODEL_PATH=ngc:nim/nvidia/cosmos-reason2-8b:hf-1208` |
| Perception (2d_cv only) | Grounding DINO | (`MODEL_NAME_2D=GDINO`, `MODEL_TYPE=cnn`) | RT-CV (DeepStream) |

LLM alternates: same as `base` — `NVIDIA-Nemotron-Nano-9B-v2-FP8`, `nemotron-3-nano`, `llama-3.3-nemotron-super-49b-v1.5`, `gpt-oss-20b`.

VLM alternates: see [VLM serving paths](#vlm-serving-paths) below.

## Default GPU layout

Reference defaults from `dev-profile-alerts/.env`:

```bash
RT_CV_DEVICE_ID=0           # perception (2d_cv only)
RT_VLM_DEVICE_ID=1          # RT-VLM container
LLM_DEVICE_ID=1             # LLM NIM
VLM_DEVICE_ID=1
LLM_MODE=local_shared       # LLM compose service: shared-GPU variant
VLM_MODE=local_shared       # signals the agent that RT-VLM lives on the LLM's GPU
RESERVED_DEVICE_IDS='0'     # GPU 0 reserved for RT-CV; not picked by shared-LLM compose
```

The default is a **2-GPU layout**: RT-CV alone on GPU 0, LLM + RT-VLM shared on GPU 1. `VLM_MODE=local_shared` here just tells the agent's runtime config that the VLM endpoint is co-located — it doesn't gate any compose service on its own. To move RT-VLM to its own GPU 2, change `RT_VLM_DEVICE_ID` and `VLM_DEVICE_ID` to `2`, set `VLM_MODE=local`, and bump `RTVI_VLLM_GPU_MEMORY_UTILIZATION` per [Sizing](#sizing). To use a remote VLM, see [Path B](#path-b--remote-vlm-user-supplied).

Real-time mode (`MODE=2d_vlm`) doesn't deploy RT-CV, so GPU 0 is free in that mode — often given to RT-VLM for more KV-cache headroom.

## RT-VLM serving paths (alerts-specific)

The alerts profile reuses the LVS VLM serving model — see
[`lvs-profile.md` § VLM serving paths](lvs-profile.md#vlm-serving-paths) for the full
integrated / remote / BYO matrix and `VLM_NAME` slug rules. The
alerts-specific differences are noted below.

### Path A — Integrated (alerts profile)

Default for alerts. RT-VLM serves the VLM locally. Same integrated checkpoint
set as LVS:

| VLM | `RTVI_VLM_MODEL_PATH` | `RTVI_VLM_MODEL_TO_USE` |
|---|---|---|
| Cosmos Reason 2 8B (default) | `ngc:nim/nvidia/cosmos-reason2-8b:hf-1208` | `cosmos-reason2` |
| Cosmos Reason 1 7B | `ngc:nim/nvidia/cosmos-reason1-7b:hf-<tag>` | `cosmos-reason` |
| Nemotron Nano V3 Omni 30B | `git:https://huggingface.co/nvidia/Nemotron-Nano-V3-Omni-GA0420-FP8` | `vllm-compatible` (see [`lvs-profile.md` § Nemotron Omni](lvs-profile.md#path-a--integrated-rt-vlm-loads-the-checkpoint-itself)) |

Switching VLMs is a `dev-profile-alerts/generated.env` edit; **`VLM_NAME_SLUG` stays `none`** and `VLM_NAME` must match the model basename returned by RT-VLM's `/v1/models` (otherwise alert-bridge fails with HTTP 400). For Cosmos Reason 1 the `VLM_NAME` becomes `nim_nvidia_cosmos-reason1-7b_hf-<tag>`.

### Path B — Remote VLM (user supplied)

Use when:

1. **The user supplied a remote VLM endpoint URL**, or
2. **The local GPU(s) can't fit the requested VLM alongside the LLM + RT-CV** per the sizing math (and the user agreed to go remote — same two-trigger rule as [`base.md` § When to use remote LLM/VLM](base.md#when-to-use-remote-llmvlm)).

The full set of env vars the skill writes to `dev-profile-alerts/generated.env`:

```bash
VLM_MODE=remote                                          # agent reads this; switches its VLM tool to remote-endpoint mode
VLM_PORT=30082                                           # was 8018; rtvi-vlm is not started in remote mode
VLM_BASE_URL=<remote-endpoint>                           # no trailing /v1
VLM_NAME=<model-name-served-there>                       # not nim_…; use the model id the remote actually advertises
RTVI_VLM_ENDPOINT=<remote-endpoint>/v1                   # WITH /v1
RTVI_VLM_MODEL_TO_USE=openai-compat
RTVI_VLM_MODEL_PATH=none
NVIDIA_API_KEY=<key if required>
```

> **`/v1` quirk** (same as LVS): `VLM_BASE_URL` no `/v1`; `RTVI_VLM_ENDPOINT` yes `/v1`. Always write both consistently.

### Path C — BYO local VLM (model not in the integrated set)

For VLMs RT-VLM can't load directly (e.g. Qwen3-VL or a third-party HF model): stand it up as a separate NIM/vLLM service per [`base.md` § Swapping a different LLM/VLM](base.md#swapping-a-different-llmvlm), then point RT-VLM at the local URL via Path B's env vars (`VLM_BASE_URL=http://${HOST_IP}:30082`, etc.). Keep `VLM_MODE=remote` so RT-VLM doesn't try to load the model itself.

## Sizing

For LLM weight cost, VLM weight cost, and the general formula, see [`base.md` § Sizing math](base.md#sizing-math) — it applies unchanged. The alerts profile adds **RT-VLM** and **RT-CV** as co-residents that the LLM has to share with.

### Values the skill writes directly

The skill writes these env vars to `dev-profile-alerts/generated.env` itself; there's no auto-tuning, so the agent has to apply the right values per the chosen layout. The numbers below are the upstream `dev-profile.sh:1200-1234` defaults — used as a precedent / starting point, not because the script ever runs.

**`RTVI_VLLM_GPU_MEMORY_UTILIZATION`:**

| Layout | Hardware | Value |
|---|---|---|
| RT-VLM shares GPU with LLM (`VLM_MODE=local_shared`) | any | **0.35** |
| RT-VLM on its own GPU (`VLM_MODE=local`) | DGX-SPARK or L40S | **0.8** |
| RT-VLM on its own GPU (`VLM_MODE=local`) | RTX 4500 (32 GB) — `HARDWARE_PROFILE=RTX4500` | **0.8** with `RTVI_VLM_MAX_MODEL_LEN=20480` for the smaller VRAM target (see [§ RTX 4500](#rtx-4500-32-gb)) |
| RT-VLM on its own GPU (`VLM_MODE=local`) | H100 / RTXPRO6000BW | leave blank → RT-VLM's hardcoded 0.7 fallback applies |
| RT-VLM on its own GPU on edge | OTHER / IGX-THOR / AGX-THOR | leave blank |

**`RT_VLM_DEVICE_ID`:**

| Layout | Value |
|---|---|
| RT-VLM shares GPU with LLM | same as `LLM_DEVICE_ID` (GPU 1 default) |
| RT-VLM on its own GPU | the dedicated GPU index (e.g. 2) |
| `VLM_MODE=remote` | irrelevant; RT-VLM not started |
| Edge (`IGX-THOR` / `AGX-THOR`) | `0` (unified memory) |

Edge platforms also need `VLM_AS_VERIFIER_CONFIG_FILE_PREFIX=EDGE-LOCAL-VLM-` so `vlm-as-verifier` picks up the matching config under `dev-profile-alerts/vlm-as-verifier/configs/EDGE-LOCAL-VLM-config.yml`.

### Default shared-mode budget (H100 80 GB, MODE=2d_cv)

```text
GPU 0 (RT-CV alone):
  Grounding DINO + DeepStream pipeline ≤ NUM_SENSORS streams (default 1).
  Plenty of headroom on a dedicated H100.

GPU 1 (LLM + RT-VLM shared):
  RT-VLM at 0.35 × 80 GB = 28 GB                           # set by dev-profile.sh
  LLM    at NIM_KVCACHE_PERCENT × 80 GB                    # tune in nim/<llm-slug>/hw-H100-shared.env
  framework/CUDA buffer ~ 15% of GPU = 12 GB
  → leaves 80 - 28 - 12 = 40 GB for the LLM
  → NIM_KVCACHE_PERCENT ≈ 0.50 (40 / 80)
```

So on H100 with the default Cosmos2 + Nano 9B pair:
- `RTVI_VLLM_GPU_MEMORY_UTILIZATION=0.35` (already set by the script for shared mode).
- `NIM_KVCACHE_PERCENT=0.50` in `nim/nvidia-nemotron-nano-9b-v2/hw-H100-shared.env`.

For real-time mode (`MODE=2d_vlm`), GPU 0 is free (no RT-CV). Two layouts make sense:

1. **Keep shared on GPU 1** — simpler. Same numbers as above; GPU 0 is unused.
2. **Move RT-VLM to GPU 0** for more KV-cache headroom (`local` mode, `VLM_DEVICE_ID=0`, `RTVI_VLLM_GPU_MEMORY_UTILIZATION=0.7` → ~56 GB for VLM). LLM gets all of GPU 1. Useful when VLM throughput dominates.

### Per-GPU `NIM_KVCACHE_PERCENT` quick-reference (shared mode, alerts)

With RT-VLM at 0.35 and a 15% framework reservation, the LLM gets:

| GPU | VRAM | RT-VLM (0.35) | Framework (0.15) | LLM gets | `NIM_KVCACHE_PERCENT` |
|---|---|---|---|---|---|
| H100 / A100-80 | 80 GB | 28 GB | 12 GB | 40 GB | **0.50** |
| H200 | 141 GB | 49 GB | 21 GB | 71 GB | **0.50** |
| RTX PRO 6000 (Blackwell) | 96 GB | 34 GB | 14 GB | 48 GB | **0.50** |
| L40S | 48 GB | 17 GB | 7 GB | 24 GB | **0.50** (tight — Nano 9B at 23.4 GB barely fits) |
| RTX 4500 (Blackwell) — `HARDWARE_PROFILE=RTX4500` | 32 GB | 11 GB | 5 GB | 16 GB | shared mode won't fit Nano 9B (23.4 GB) — use `LLM_MODE=remote` and run RT-VLM only (see [§ RTX 4500](#rtx-4500-32-gb)) |

### RTX 4500 (32 GB)

32 GB VRAM is too little to host RT-VLM **and** a local Nano 9B LLM together. The supported layout is the default `hf-1208` RT-VLM checkpoint with the LLM remote. Full env block for `dev-profile-alerts/generated.env`:

```env
HARDWARE_PROFILE=RTX4500
LLM_MODE=remote
VLM_MODE=local
# RT-VLM sizing: cap context + lift utilization to fit on 32 GB.
RTVI_VLM_MAX_MODEL_LEN=20480
RTVI_VLLM_GPU_MEMORY_UTILIZATION=0.8
# Keep the default source-backed Cosmos Reason 2 checkpoint.
# VLM_NAME must match the basename rtvi-vlm advertises at /v1/models, or
# alert-bridge / agent get HTTP 400 "No such model" (see § VLM serving paths).
VLM_NAME=nim_nvidia_cosmos-reason2-8b_hf-1208
RTVI_VLM_MODEL_PATH=ngc:nim/nvidia/cosmos-reason2-8b:hf-1208
```

The `hf-1208` Cosmos Reason 2 build is the source-backed default. The model length cap and utilization setting are the documented sizing knobs for this 32 GB target.

Formula: `NIM_KVCACHE_PERCENT = 1 - 0.35 - 0.15 = 0.50`. Same fraction across GPUs because the script's RT-VLM util is fixed at 0.35 in shared mode.

### Hard rules

- **L40S is the floor for shared mode.** 24 GB for the LLM barely fits Nano 9B FP16 (23.4 GB total). Switch to FP8 (`nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8`, 11.7 GB total) for any breathing room, or move to dedicated mode (LLM on its own GPU, RT-VLM on its own GPU at 0.8 util).
- **DGX-Spark / IGX-Thor / AGX-Thor — Cosmos Reason 2 must serve via RT-VLM, not a standalone NIM.** Thor (`AGX-THOR` / `IGX-THOR`) cannot host the standalone `cosmos-reason2-8b` NIM service; the alerts compose graph routes through RT-VLM only, and the source `.env` already pairs `VLM_NAME=nim_nvidia_cosmos-reason2-8b_hf-1208` with `RTVI_VLM_MODEL_PATH=ngc:nim/nvidia/cosmos-reason2-8b:hf-1208` so RT-VLM loads the checkpoint in-process. Don't introduce a remote-VLM override or a different VLM name on Thor — `VLM_AS_VERIFIER_CONFIG_FILE_PREFIX=EDGE-LOCAL-VLM-` and `RT_VLM_DEVICE_ID=0` (unified memory) are also part of the Thor shape. For the LLM side, follow `edge.md`: DGX Spark uses the standalone DGX Spark Nano 9B NIM, while AGX/IGX Thor still uses the Edge 4B fallback.
- **Don't co-deploy a standalone Cosmos NIM with alerts.** `COMPOSE_PROFILES` for alerts has no `vlm_*_<slug>` segment by design. Verify by checking `resolved.yml` doesn't have `cosmos-reason2-8b` / `cosmos-reason2-8b-shared-gpu` services alongside `rtvi-vlm`.
- **`VLM_NAME` mismatch ⇒ HTTP 400.** dev-profile.sh sets `VLM_NAME=nim_nvidia_cosmos-reason2-8b_hf-1208` for the default Cosmos2 path. If you change `RTVI_VLM_MODEL_PATH` you must update `VLM_NAME` to match the new model basename — otherwise alert-bridge / agent get "No such model" from `/v1/models`.
- **`VLM_NAME_SLUG=none` is required.** The alerts compose graph has no `vlm_local_*_<slug>` profiles. Setting a real slug doesn't bring up a VLM service — it just makes the COMPOSE_PROFILES reference dead.
- **`/v1` suffix mismatch.** `VLM_BASE_URL` no `/v1`; `RTVI_VLM_ENDPOINT` yes `/v1`. dev-profile.sh writes them consistently in remote mode; if you edit by hand, mirror that.

## Use cases

- PPE compliance verification (hard hats, safety vests)
- Restricted area / asset presence detection
- Custom object detection scenarios (driven by behavior analytics rules in 2d_cv mode)
- Continuous live-stream incident detection (in 2d_vlm mode)

## Endpoints (after deploy)

| Service | URL |
|---|---|
| Agent UI | `http://<HOST_IP>:3000/` (Alerts tab) |
| Agent REST API | `http://<HOST_IP>:8000/` |
| Alert-bridge realtime API | `http://<HOST_IP>:9080/api/v1/realtime` |
| RT-VLM | `http://<HOST_IP>:8018/v1/` (or remote if `VLM_MODE=remote`) |
| Video-Analytics MCP | `http://<HOST_IP>:9901/` |
| Kibana | `http://<HOST_IP>:5601/` |
| nvstreamer | `http://<HOST_IP>:31000/` |
| Phoenix | `http://<HOST_IP>:6006/` |

## Readiness checks (per mode)

The two modes deploy **different service sets**, so the readiness checks differ. Run the generic compose-ps gate from [`readiness.md`](readiness.md) first, then the per-mode checks below. Follow [`SKILL.md`](../SKILL.md) Step 5b — `up -d` returning is not readiness.

> **Expected container count differs by mode.** `2d_vlm` does **not** start `vss-rtvi-cv` or `vss-behavior-analytics` (both are `bp_developer_alerts_2d_cv`-only — see [What gets deployed](#what-gets-deployed)), so `docker compose -f resolved.yml config --services` yields a smaller set than `2d_cv`. A smaller container count in real-time mode is correct, not a partial deploy — the Gate 0 check in [`SKILL.md`](../SKILL.md) Step 5b derives `expected` from the resolved compose, so it self-adjusts.

**Both modes — these must be reachable:**

```bash
curl -sf http://${HOST_IP}:8000/health            && echo "agent ok"      # VSS Agent
curl -sf http://${HOST_IP}:8018/v1/models         && echo "rt-vlm ok"     # RT-VLM (skip if VLM_MODE=remote; probe the remote /v1/models instead)
curl -sf http://${HOST_IP}:9901/                  && echo "va-mcp ok"     # Video-Analytics MCP
curl -sf http://${HOST_IP}:3000/                  && echo "ui ok"         # Agent UI
```

**`MODE=2d_cv` (verification) — also check the perception path:**

```bash
docker ps --format '{{.Names}}' | grep -qx vss-rtvi-cv            && echo "rt-cv up"
docker ps --format '{{.Names}}' | grep -qx vss-behavior-analytics && echo "behavior-analytics up"
curl -sf http://${HOST_IP}:9000/v1/health                        && echo "rt-cv health ok"
```

RT-CV builds TensorRT engines on first start (3–5 min) — `:9000/v1/health` won't answer until that finishes. See [Stage perception models](#stage-perception-models-rtdetr-its--gdino); if the ONNX files weren't staged, RT-CV never becomes healthy.

**`MODE=2d_vlm` (real-time) — RT-CV / behavior-analytics are intentionally absent:**

```bash
# Confirm the 2d_cv-only services are NOT running (their absence is expected):
docker ps --format '{{.Names}}' | grep -qx vss-rtvi-cv && echo "UNEXPECTED: rt-cv running in 2d_vlm" || echo "rt-cv absent (correct)"
# Confirm RT-VLM is processing the live stream and emitting to the alert-bridge:
docker logs vss-rtvi-vlm  2>&1 | tail -20   # expect continuous VLM inference over the nvstreamer feed
docker logs vss-alert-bridge 2>&1 | tail -20 # expect realtime session active, no HTTP 400 "No such model"
```

In real-time mode the readiness signal is **RT-VLM continuously inspecting the live feed**, not a per-alert verification trigger — there is no RT-CV health endpoint to gate on. Confirm `MODE=2d_vlm` in `resolved.yml` and that `vss-vios-nvstreamer` (`:31000`) is publishing streams.

## Env file location

```
deploy/docker/developer-profiles/dev-profile-alerts/.env            # source defaults (read-only)
deploy/docker/developer-profiles/dev-profile-alerts/generated.env   # skill's working copy (apply overrides here)
```

## Stage perception models (RTDETR-ITS + GDINO)

**MUST run before `docker compose -f resolved.yml up -d` for verification mode (`MODE=2d_cv`).** The alerts compose has no init container that downloads the perception detector models — `dev-profile.sh` stages them via NGC CLI, and since this skill doesn't run that script, the agent stages them directly.

Real-time mode (`MODE=2d_vlm`) doesn't deploy RT-CV and skips this entirely.

Symptom if skipped (verification mode): RT-CV starts but its TensorRT engine build fails because the ONNX detector files are missing under `${VSS_DATA_DIR}/models/`.

```bash
# Source: deploy/docker/scripts/dev-profile.sh (alerts profile, model staging block).
# Requires NGC_CLI_API_KEY exported and ngc CLI on PATH (see references/ngc.md).

DATA="$VSS_DATA_DIR"                                     # e.g. <repo>/data
APPS="$VSS_APPS_DIR"                              # e.g. <repo>/deploy/docker

# Profile-specific dirs
mkdir -p \
    "$DATA/data_log/vss_video_analytics_api" \
    "$DATA/videos/dev-profile-alerts" \
    "$APPS/engines/gdino" \
    "$APPS/engines/rtdetr-its"
chmod -R 777 "$APPS/engines"

# DESTRUCTIVE: dev-profile.sh wipes $DATA/models before re-staging. If the host
# also runs other profiles whose models live under $DATA/models, gate this on
# whether you really want a clean slate.
rm -rf "$DATA/models"
mkdir -p "$DATA/models/rtdetr-its" "$DATA/models/gdino"

# 1. RTDETR-ITS (TrafficcamNet)
NGC_CLI_API_KEY="${NGC_CLI_API_KEY}" ngc registry model \
    download-version \
    nvidia/tao/trafficcamnet_transformer_lite:deployable_resnet50_v2.0
mv trafficcamnet_transformer_lite_vdeployable_resnet50_v2.0/resnet50_trafficcamnet_rtdetr.fp16.onnx \
    "$DATA/models/rtdetr-its/model_epoch_035.fp16.onnx"
rm -rf trafficcamnet_transformer_lite_vdeployable_resnet50_v2.0

# 2. Mask Grounding DINO
NGC_CLI_API_KEY="${NGC_CLI_API_KEY}" ngc registry model \
    download-version \
    nvidia/tao/mask_grounding_dino:mask_grounding_dino_swin_tiny_commercial_deployable_v2.1_wo_mask_arm
mv mask_grounding_dino_vmask_grounding_dino_swin_tiny_commercial_deployable_v2.1_wo_mask_arm/mgdino_mask_head_pruned_dynamic_batch.onnx \
    "$DATA/models/gdino/mgdino_mask_head_pruned_dynamic_batch.onnx"
rm -rf mask_grounding_dino_vmask_grounding_dino_swin_tiny_commercial_deployable_v2.1_wo_mask_arm

chmod -R 777 "$DATA/models"
```

**Verify** before deploying:

```bash
ls -l "$VSS_DATA_DIR/models/rtdetr-its/model_epoch_035.fp16.onnx" \
      "$VSS_DATA_DIR/models/gdino/mgdino_mask_head_pruned_dynamic_batch.onnx"
# expected: both files present, mode 777
```

After RT-CV starts, it builds TensorRT engines from these ONNX files (3–5 min on first start), cached under `$VSS_APPS_DIR/engines/`.

## First-run note

RT-VLM downloads `cosmos-reason2-8b:hf-1208` from NGC on first start (~10–20 min depending on bandwidth). For verification mode (`MODE=2d_cv`), RT-CV builds TensorRT engines from the ONNX models staged in [Stage perception models](#stage-perception-models-rtdetr-its--gdino) above. Real-time mode skips RT-CV entirely.

## Debugging

- **`docker logs vss-rtvi-vlm`** — confirms model load and `Maximum concurrency for X tokens per GPU: Y x` line. OOM → lower `RTVI_VLLM_GPU_MEMORY_UTILIZATION` by 0.05 or drop `RTVI_VLM_MAX_MODEL_LEN` / `RTVI_VLLM_MAX_NUM_SEQS`.
- **`docker logs alert-bridge`** — if it logs HTTP 400 "No such model: …", check `VLM_NAME` matches RT-VLM's `/v1/models` basename. `curl http://${HOST_IP}:8018/v1/models | jq` confirms what's actually advertised.
- **2d_cv: alerts never fire** — check `vss-behavior-analytics` is consuming RT-CV metadata: `docker logs vss-behavior-analytics`. RT-CV side: `curl http://${HOST_IP}:9000/v1/health`.
- **2d_vlm: VLM not running over live streams** — confirm `MODE=2d_vlm` (not `2d_cv`) in `resolved.yml` and that nvstreamer-alerts is publishing streams.
- **OOM on shared GPU 1** — drop `NIM_KVCACHE_PERCENT` for the LLM by 0.05; if RT-VLM is the OOM, raise its `RTVI_VLLM_GPU_MEMORY_UTILIZATION` ceiling and re-tune the LLM down (the 0.35/0.50 split assumes Nano 9B FP16; larger LLMs need different ratios).
- **Edge: `vlm-as-verifier` config not loaded** — verify `VLM_AS_VERIFIER_CONFIG_FILE_PREFIX=EDGE-LOCAL-VLM-` is set in `generated.env` and the matching `EDGE-LOCAL-VLM-config.yml` exists under `dev-profile-alerts/vlm-as-verifier/configs/`.
