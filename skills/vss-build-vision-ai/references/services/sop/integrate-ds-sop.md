# Integration Reference: DS-SOP

## Overview

DS-SOP is a DeepStream-based Standard-Operating-Procedure monitoring microservice. It ingests a video stream — **canonically a Basler/Pylon industrial camera** at the work-cell (also an RTSP source or a file) — runs a **DDM-Net temporal action-detection model** to segment the stream into action chunks, then runs a **Cosmos-Reason VLM (in-process vLLM)** over each chunk to label it against a configured SOP action set, and a **SOP step-checker** that flags missing / mis-ordered / cycle-complete steps. It publishes per-chunk SOP records (JSON) to Kafka for ELK/Kibana, **and — for live `rtsp://`/camera inputs — re-emits an annotated RTSP output** (`rtsp://<host>:8554/ds-out/<stream-name>`, when `ENABLE_RTSP_OUTPUT=true`) **that VIOS records for the VST UI** (on a recorder-enabled composition; the on-demand file path binds no `:8554` stream — see Outputs).

Use this service when the workflow requires **SOP compliance monitoring of a procedural task** (e.g. assembly / installation steps) on a live camera or stored video — structured, deterministic "did the operator perform step N, in order" events, as opposed to free-form dense captions (RT-VLM's job). DS-SOP occupies the same perception/inference slot as RT-VLM or RT-CV, but bundles a CV action model + a VLM in one DeepStream container **and (unlike RT-VLM) produces an annotated RTSP output VIOS records**. The image is `ds-sop:1.0.0`, built per `build-ds-sop.md` (this bundle).

## Required Peer Services

**Prose — peer microservices:**

- **VIOS** (`video-storage`, `rtsp-ingestion`, `sensor-management`) — **required for the live topology**. The canonical SOP flow is **camera → DS-SOP → VIOS**: DS-SOP ingests the camera **directly** (Basler/Pylon, or an RTSP/file source — **not** proxied through VIOS), and **re-streams its annotated result to VIOS**, which records it for the VST UI. Wire it by registering **DS-SOP's output** as a VIOS sensor: `POST /vst/api/v1/sensor/add` with `sensorUrl = rtsp://<host>:${RTSP_PORT:-8554}/ds-out/<stream-name>` (`<stream-name>` = the **source stream's name** — the input video/camera id, NOT a VIOS sensorId), then **publish a `camera_streaming` event to Redis `vst.event`** (the recorder SDR does not provision the stream otherwise — it is never published automatically in the VST microservices split) and **start recording** (`POST /vst/api/v1/record/<stream_id>/start`, with retry; `always_recording` may auto-start it as a fallback). The blueprint's `add-ds-sop-to-vst.py --record` helper performs the whole sequence (sensor/add → camera_streaming → record/start). (The VST API base is `/vst/api/v1/...`; `/api/v1/...` 404s in vst nginx mode. Optionally the raw source camera is also registered for a side-by-side raw view.) **This DS-SOP→VIOS registration is a deploy-time API step** — build-vision-ai composes the services but never wires video flow (same as RT-VLM); build-vision-ai's deploy flow (or the operator) performs it (see `deploy-ds-sop.md`). VIOS in the live topology pulls in the full SDRC stack (see VIOS `integrate-vios-service.md` § Known Integration Constraints — `sensor-ms` calls the SDRC Envoy listener on `localhost:10000` for every sensor-add).
- **Kafka** (`kafka-ingestion`) — **required**. DS-SOP publishes SOP chunk records to the topic named by `DEFAULT_TOPIC` (see Environment Variables). Provided by the ELK peer (`services/elk.md`).
- **ELK** (`caption-storage`, `kafka-ingestion`, `search`, `dashboard`) — **required for storage/search**. Logstash consumes `mdx-vlm-captions` and indexes into Elasticsearch; Kibana visualizes via the SOP dashboard. **Caveat:** build-vision-ai's default ELK decodes this topic as PROTOBUF (RT-VLM), but DS-SOP emits JSON — a dedicated JSON Logstash pipeline must be added (shipped at `sop-vlm-captions-json-logstash.conf`). See § Known Integration Constraints → "ELK indexing".

> DS-SOP's build-vision-ai composition (self-named key `ds-sop` + delta) is in § build-vision-ai composition below and `../sop.md § Delta composition`; peers (VIOS / Kafka / ELK) are in `../sop.md § Required peers`.

## Integration Interfaces

**Inputs:**

- **Video source** — DS-SOP ingests the source **directly** (the API accepts a Basler camera, an RTSP URL, or a file). It does **not** consume video via a VIOS proxy — VIOS is downstream (it records DS-SOP's annotated output, see Outputs):
  1. **Basler/Pylon industrial camera (primary / canonical)** — the SOP work-cell setup; DS-SOP reads the camera directly. For testing without hardware, **camera emulation** replays a sample video as a fake Basler camera: `PYLON_CAMEMU=1` + the in-image `configs/Emulation_0815-0000.pfs` + `CAMERA_EMULATION_DIR`.
  2. **RTSP URL** — any `rtsp://...` source as `video_url` on `/v1/chat/completions` (a real IP camera, or a local `rtsp_server.py` relay of a sample video).
  3. **File / on-demand** — a file path or base64 `video_url` for offline evaluation (deterministic; not realtime-bound). On this route DS-SOP sets `sensor_id` to the uploaded file path (`ds_sop_process.py`), so ES docs / `get_sop_report` / Mode C carry a `/tmp/...`-style sensor id — functional but not human-meaningful; a named `sensor_id` (the stream name) requires the camera/RTSP route.
- **REST** — OpenAI-compatible API server on `:${API_SERVER_PORT:-8300}` (`GET /v1/ready` → `200`; `GET /v1/models` → `ds_sop_model`; `GET /v1/metadata` → version + model info; `POST /v1/chat/completions` with a `video_url`). This is what `VLM_BASE_URL` points at when a VSS Agent is layered on top.
- **Action config** — `${ACTION_CONFIG_PATH}` (JSON, the ordered SOP action set) and `${VLM_PROMPT_PATH}` (VLM prompt template), bind-mounted from the host.

**Outputs:**

- **Kafka** — per-chunk SOP records published to the topic named by **`DEFAULT_TOPIC`** (read by `nvds_action_detector/messager.py`; code default is already `mdx-vlm-captions`). Payload (when `SOP_MESSAGING_SCHEMA=JSON`, the code default) is the flat `chunk_info` dict dumped whole: `{chunk_idx, start_time, end_time, first_timestamp, pipeline_chunk_end_timestamp, pipeline_vlm_ready_timestamp, response, sensor_id, req_id, cv_execute_time, vlm_execute_time, cv_boundary_score, checker_result:{missing_detected, misordered_detected, cycle_completed,...}, ...}`. Kafka key is `request_id` (UUID fallback). **`req_id`** is the unique per-chunk id (e.g. `0001-<uuid>`) — the JSON Logstash pipeline uses it as the ES `document_id` (idempotent upsert; without it all chunks collapse to one ES doc). **`pipeline_chunk_end_timestamp`** is a **wall-clock epoch (seconds)** used to build `@timestamp` — reliable for live AND file/on-demand sources. Do **not** use `first_timestamp + start_time/end_time` for `@timestamp`: those are relative stream seconds (0-based on the file/on-demand path) → `@timestamp` ~1970 → the Kibana recent-time filter shows nothing. The consuming pipeline `sop-vlm-captions-json-logstash.conf` relies on `req_id` + `pipeline_chunk_end_timestamp`.
- **Annotated RTSP output** — with **`ENABLE_RTSP_OUTPUT=true`** (set it for the canonical flow), DS-SOP re-streams the source with SOP overlays at `rtsp://<host>:${RTSP_PORT:-8554}/ds-out/<stream-name>` (H.264; `SW_ENCODER=true` for the SW fallback). **VIOS registers this stream** (`POST /vst/api/v1/sensor/add`) and, **on a recorder-enabled composition**, records it so the VST UI shows the annotated video — the **DS-SOP → VIOS** half of the canonical flow. Current developer profiles ship no `recorder-ms`, so persistent recording is **pending** (`record/start` returns 500 by design — see `deploy-ds-sop.md` § Known Deployment Issues). The in-pipeline server binds/tears down **per LIVE `/v1/chat/completions` request** — `:8554/ds-out` is bound only when the input is an `rtsp://` URL or a Basler camera (`is_live_stream=True` in `api_server.py`); the **on-demand file path binds nothing** (no `:8554` stream, so the DS-SOP → VIOS leg does not apply). It can also hang the pipeline on no-NVENC GPUs, so for continuous, full-length recording feed a **looping source** — Basler camera-emulation (`PYLON_CAMEMU`), or a standalone `rtsp_server.py` relay on `:8554/ds-out` that auto-restarts on EOS (the blueprint's approach). A one-shot/EOS source only yields a short fragment. (Implemented in the source's `api_server.py` / `ds_sop_process.py` / `ds_3d_action_pipeline.py` via `RTSPStreamingServer`.)

## Environment Variables

| Variable | Required | Default (code/compose) | Notes |
|---|---|---|---|
| `DS_SOP_IMAGE` | yes | `ds-sop:1.0.0` | Locally built (see `deploy-ds-sop.md` / `build-ds-sop.md`). |
| `API_SERVER_PORT` | yes | `8300` | REST API server (host network). |
| `DEFAULT_TOPIC` | yes | `mdx-vlm-captions` | Code default is already `mdx-vlm-captions` — keep it; it must match the ELK topic. (No `DS_SOP_KAFKA_TOPIC` exists in the source — `messager.py` reads `DEFAULT_TOPIC`.) |
| `SOP_MESSAGING_SCHEMA` | yes | `JSON` | Code default is already `JSON` (flat-field for the VSS-3.x ELK pipeline + Kibana dashboard). |
| `ENABLE_MESSAGING` | **yes** | `false` | **Must set `1`** to publish to Kafka at all (compose default is `false`). |
| `ENABLE_RTSP_OUTPUT` | **yes (for DS-SOP→VIOS)** | `false` | **Set `true`** so DS-SOP re-streams the annotated output for VIOS (recording needs a recorder-enabled composition — see `deploy-ds-sop.md`). Off → no `:8554` stream (Kafka records only). |
| `RTSP_PORT` | conditional | `8554` | Port of the annotated `/ds-out/<stream-name>` output (used when `ENABLE_RTSP_OUTPUT=true`). |
| `SW_ENCODER` | conditional | `true` | Software H.264 encode fallback for the RTSP output (set `true` if no NVENC available). |
| `KAFKA_BROKER` | yes | `localhost:9092` | DS-SOP is host-networked -> use Kafka's host-published EXTERNAL listener (`:9092`); bridge-side peers (logstash) use `kafka:29092`. |
| `MODEL_ROOT_DIR` | yes | `/opt/models` | Host model root, bind-mounted 1:1. |
| `VLLM_MODEL_PATH` | yes | staged VLM path | Point at where the VLM is staged. `download_assets.sh` verifies `/opt/models/vlm/checkpoint`; NGC `sop-data:1.0` lays it at `/opt/models/cosmos-reason1.1-7b/checkpoint`. (Compose default is HF id `nvidia/cosmos-reason1-7b`.) |
| `DDM_MODEL_PATH` | yes | `/opt/models/gbed_models/ddm/checkpoint.pth.tar` | DDM-Net weights. |
| `VLLM_GPU_MEMORY_UTILIZATION` | conditional | `0.3` | **Set `0.6` on ≤48 GB GPUs** — `0.3` is H100-80GB-tuned and OOMs the KV cache after the ~15.6 GB model load. (On ≥80 GB Blackwell/H100, `0.3` is fine.) |
| `ACTION_CONFIG_PATH` | yes | `/opt/sop/configs/actions.json` | Host SOP action set (staging path; bind-mounted 1:1). |
| `VLM_PROMPT_PATH` | yes | `/opt/sop/configs/vlm_prompts.txt` | Host VLM prompt (staging path; bind-mounted 1:1). |
| `NVIDIA_VISIBLE_DEVICES` | yes | `0` | GPU id. |

## Network Requirements

- `network_mode: host` for the `ds-sop` service (camera access + binds API `:8300` and the annotated RTSP out `:8554`; reaches Kafka via its **host-published EXTERNAL listener** `:9092` — bridge-side peers use `kafka:29092`). The `sop-kibana-init` one-shot joins the default bridge network instead.
- `privileged: true`, `ipc: host`, `shm_size: 16gb` (DeepStream + vLLM).
- Ports used on the host: `8300` (REST API), `8554` (annotated RTSP out → VIOS records, when enabled), random UDP (internal udpsink→RTSP loop). DS-SOP **replaces** RT-VLM in the perception slot, so do not co-deploy them in one profile.

## Known Integration Constraints

- **Publish env.** The messager (`nvds_action_detector/messager.py`) reads `DEFAULT_TOPIC` (code default `mdx-vlm-captions`) and `SOP_MESSAGING_SCHEMA` (code default `JSON`) — both already correct for ELK, so keep them. The one required override is **`ENABLE_MESSAGING=1`** (compose default `false`); without it nothing is published and ES stays empty.
- **ELK indexing — the #1 deployment gotcha (schema mismatch).** build-vision-ai composes ELK from `integrate-elk.md`, whose VSS-3.x Logstash pipeline (`mdx-lvs-logstash.conf`) decodes `mdx-vlm-captions` as **NvSchema `nv.VisionLLM` PROTOBUF only** — tuned for RT-VLM. DS-SOP publishes **flat JSON**; that JSON is NOT decodable by the protobuf codec, so Logstash logs `Google::Protobuf::ParseError` and **0 docs reach Elasticsearch**. **Remediation (REQUIRED):** add a dedicated JSON Logstash pipeline that consumes `mdx-vlm-captions` with `codec => json` and indexes to `mdx-vlm-captions-*` — ships at `sop-vlm-captions-json-logstash.conf`; register it as its **own pipeline-id** (do NOT merge into `mdx-lvs`). build-vision-ai materializes this at **build time** — a build-local Logstash overlay + `patches/logstash.yml` (a compose patch alone can't edit `.conf` content, so it is staged as an overlay) — see `deploy-ds-sop.md` § Known Deployment Issues.
- **DS-SOP → VIOS wiring is deploy-time, not composed.** build-vision-ai composes DS-SOP + VIOS into one profile but does **not** wire the video flow (it never does — even RT-VLM's stream registration is a post-boot API call). So registering DS-SOP's `:8554/ds-out` output as a VIOS sensor (`POST /vst/api/v1/sensor/add`) is a **deploy-time step for the live rtsp/camera topology** — build-vision-ai's deploy flow (or the operator) performs it; the on-demand file path binds no `:8554` and needs no VIOS wiring. (The SOP JSON Logstash pipeline, by contrast, is materialized at build time.) See `deploy-ds-sop.md`.
- **Single perception per profile.** DS-SOP and RT-VLM both target `mdx-vlm-captions` and both want the GPU — select exactly one per profile.
- **Models + configs are host-staged, not in the image** — `/opt/models/...` (DDM + cosmos-reason) and `/opt/sop/configs/...` must exist before bring-up (see `deploy-ds-sop.md`).

## Scope notes

- **Source:** built from `microservices/sop-inference-bp/` in `NVIDIA/sop-monitoring-blueprints` (branch `main`), which ships the `:8554` annotated RTSP output the DS-SOP→VIOS flow uses. See `build-ds-sop.md`.
- **Report generation — available as a separate capability.** SOP compliance reports are composable via the **`sop-report-generation`** capability (`SOP Report Tools`, see `../sop.md` § Patch specifics) — it adds the SOP `get_sop_*` tools to `vss-va-mcp`, and the `vss-generate-video-report` skill (Mode C — `references/report-types/sop-compliance.md` inside that skill's directory) renders the report from them (no report agent, no web UI, no report LLM). This DS-SOP entry delivers SOP **detection → Kafka → ELK/Kibana + annotated stream → VIOS/VST**; compose `sop-detection` + `sop-report-generation` together for the full blueprint (detection **and** on-demand compliance reports).

## Example Compose Snippet

The `ds-sop` service block build-vision-ai emits as `patches/ds-sop.yml` (a genuinely new service, per `composition.md § Artifact contract`). Derived from the source's `deploy/compose.yaml` (`nvds-action-sop` service), renamed to the `ds-sop` service-key and parameterized:

```yaml
services:
  ds-sop:
    image: ${DS_SOP_IMAGE:-ds-sop:1.0.0}
    runtime: nvidia
    network_mode: host
    privileged: true
    ipc: host
    shm_size: '16gb'
    ulimits:
      memlock: -1
      stack: 67108864
    profiles: ["ds-sop"]             # self-named profile (profile-first); activated via COMPOSE_PROFILES
    devices:
      - "/dev/snd:/dev/snd"
    depends_on:
      kafka:
        condition: service_started
    working_dir: ${WORK_DIR_PATH:-/opt/nvidia/nvds_sop}
    entrypoint: ${ENTRYPOINT:-./start_server.sh}
    volumes:
      - "${MODEL_ROOT_DIR:-/opt/models}:${MODEL_ROOT_DIR:-/opt/models}"
      - "${HOST_CACHE:-$HOME/.cache/ds_sop}:/opt/nvidia/nvds_sop/.cache"
      - "${ACTION_CONFIG_PATH:-/opt/sop/configs/actions.json}:${ACTION_CONFIG_PATH:-/opt/sop/configs/actions.json}"
      - "${VLM_PROMPT_PATH:-/opt/sop/configs/vlm_prompts.txt}:${VLM_PROMPT_PATH:-/opt/sop/configs/vlm_prompts.txt}"
    environment:
      NVIDIA_VISIBLE_DEVICES: "${NVIDIA_VISIBLE_DEVICES:-0}"
      API_SERVER_PORT: ${API_SERVER_PORT:-8300}
      DEFAULT_TOPIC: ${DEFAULT_TOPIC:-mdx-vlm-captions}
      SOP_MESSAGING_SCHEMA: ${SOP_MESSAGING_SCHEMA:-JSON}
      ENABLE_MESSAGING: "${ENABLE_MESSAGING:-1}"
      ENABLE_RTSP_OUTPUT: "${ENABLE_RTSP_OUTPUT:-true}"     # DS-SOP→VIOS: re-stream annotated output
      RTSP_PORT: "${RTSP_PORT:-8554}"
      SW_ENCODER: "${SW_ENCODER:-true}"
      KAFKA_BROKER: "${KAFKA_BROKER:-localhost:9092}"
      MODEL_ROOT_DIR: "${MODEL_ROOT_DIR:-/opt/models}"
      VLLM_MODEL_PATH: "${VLLM_MODEL_PATH:-/opt/models/cosmos-reason1.1-7b/checkpoint}"
      DDM_MODEL_PATH: "${DDM_MODEL_PATH:-/opt/models/gbed_models/ddm/checkpoint.pth.tar}"
      VLLM_GPU_MEMORY_UTILIZATION: "${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
      ACTION_CONFIG_PATH: "${ACTION_CONFIG_PATH:-/opt/sop/configs/actions.json}"
      VLM_PROMPT_PATH: "${VLM_PROMPT_PATH:-/opt/sop/configs/vlm_prompts.txt}"

  # One-shot Kibana bootstrap: imports the SOP data view (mdx-vlm-captions*) + "SOP Dashboard"
  # saved objects, so Kibana can actually show the ES docs. Same pattern as the stack's other
  # init one-shots (elasticsearch-init, kafka-topic-init); runs once and exits 0.
  sop-kibana-init:
    image: curlimages/curl:8.7.1
    # joins the stack's default bridge network (kibana resolves by service name)
    restart: "no"
    profiles: ["sop-kibana-init"]    # self-named profile
    depends_on:
      kibana:
        condition: service_healthy
    entrypoint: ["/bin/sh", "-c"]
    command:
      - |
        curl -fsSL --retry 5 --retry-delay 5 --max-time 60 -o /tmp/sop-kibana-objects.ndjson \
          "${SOP_KIBANA_OBJECTS_URL:-https://raw.githubusercontent.com/NVIDIA/sop-monitoring-blueprints/0dd472f/agentic/vss-sop-skills/vss-sop-build/references/deployments/sop/sop-app/kibana-dashboard/sop-kibana-objects.ndjson}"
        # The stack's Kibana is served under base path /kibana (bare :5601 404s) — same URL the
        # upstream vss-kibana-init uses; fall back to a bare base path for non-standard setups.
        # $$ escapes the dollar so the container shell reads the KB variable (Compose would otherwise interpolate it).
        KB="${KIBANA_URL:-http://kibana:5601}"
        curl -fsS --retry 5 --retry-delay 5 -X POST \
          "$$KB/kibana/api/saved_objects/_import?overwrite=true" \
          -H "kbn-xsrf: true" --form file=@/tmp/sop-kibana-objects.ndjson \
        || curl -fsS -X POST \
          "$$KB/api/saved_objects/_import?overwrite=true" \
          -H "kbn-xsrf: true" --form file=@/tmp/sop-kibana-objects.ndjson
```
> `sop-kibana-init` fetches `sop-kibana-objects.ndjson` from the blueprint repo at run time (internet required, like the NGC pulls; override the URL via `SOP_KIBANA_OBJECTS_URL`, or import manually — see `deploy-ds-sop.md`). **Registering DS-SOP's `:8554/ds-out` output as a VIOS sensor is a deploy-time step for live rtsp/camera inputs** (the compose only exposes it; the on-demand file path binds no `:8554`; see `deploy-ds-sop.md`).

## build-vision-ai composition

DS-SOP is a **new service** (self-named profile key `ds-sop`) plus a one-shot
`sop-kibana-init`. build-vision-ai composes it as a **delta** (see `../sop.md § Delta
composition`): it emits `patches/ds-sop.yml` from § Example Compose Snippet above
(reusing the `ds-sop` / `sop-kibana-init` service keys), adds both keys to the
delta's `COMPOSE_PROFILES`, and **removes** the RT-VLM perception key — DS-SOP takes
the perception slot (both target `mdx-vlm-captions` and want the GPU; never select both).

The `override.env` knobs (`DEFAULT_TOPIC=mdx-vlm-captions`, `SOP_MESSAGING_SCHEMA=JSON`,
`ENABLE_MESSAGING=1`, `ENABLE_RTSP_OUTPUT=true`, `RTSP_PORT=8554`, `SW_ENCODER=true`,
`VLLM_GPU_MEMORY_UTILIZATION=0.6` on ≤48 GB GPUs — `0.3` is fine on ≥80 GB —
`DS_SOP_IMAGE=ds-sop:1.0.0`) are in `../sop.md`. The **DS-SOP → VIOS recording** (a live rtsp/camera step — the
on-demand file path binds no `:8554`) is the deploy-time
wiring the Compose patch cannot cover (the **SOP JSON Logstash pipeline** is materialized at build
time — build-local overlay + `patches/logstash.yml`) — see `deploy-ds-sop.md § Known Deployment Issues`.
