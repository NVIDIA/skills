# Base Profile Reference

Profile: `base` | Blueprint: `bp_developer_base` | Mode: `2d`

Video upload, Q&A, and report generation with HITL (Human-in-the-Loop) feedback.

## Services Deployed

| Service | Container | Port | Purpose |
|---|---|---|---|
| VSS Agent | mdx-vss-agent-1 | 8000 | Orchestrates tool calls and model inference |
| VSS UI | mdx-vss-ui-1 | 3000 | Web UI — chat, video upload, views |
| VST | mdx-vst-1 | 30888 | Video Storage Tool — ingest, record, playback |
| VST MCP | mdx-vst-mcp-1 | 8001 | VST management API |
| LLM NIM | mdx-nim-llm-1 | 30081 | Nemotron LLM for reasoning |
| VLM NIM | mdx-nim-vlm-1 | 30082 | Cosmos Reason VLM for vision |
| Elasticsearch | mdx-elasticsearch-1 | 9200 | Analytics data store |
| Kafka | mdx-kafka-1 | 9092 | Message broker |
| Redis | mdx-redis-1 | 6379 | Cache |
| Phoenix | mdx-phoenix-1 | 6006 | Observability / telemetry |

## Default Models

| Role | Model | Slug | Type |
|---|---|---|---|
| LLM | `nvidia/nvidia-nemotron-nano-9b-v2` | `nvidia-nemotron-nano-9b-v2` | nim |
| VLM | `nvidia/cosmos-reason2-8b` | `cosmos-reason2-8b` | nim |

**Alternate LLMs:** `nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8`, `nvidia/nemotron-3-nano`, `nvidia/llama-3.3-nemotron-super-49b-v1.5`, `openai/gpt-oss-20b`

**Alternate VLMs:** `nvidia/cosmos-reason1-7b`, `Qwen/Qwen3-VL-8B-Instruct`

## GPU Layout

| LLM/VLM Mode | LLM_DEVICE_ID | VLM_DEVICE_ID | Description |
|---|---|---|---|
| `local_shared` (default) | 0 | 0 | Both models share one GPU |
| `local` | 0 | 1 | Dedicated GPU per model |
| `remote` | — | — | No local GPU needed for inference |

## Env Overrides — Common Scenarios

### Minimal deploy (auto-detect hardware)

```json
{
  "HARDWARE_PROFILE": "<detected>",
  "MDX_SAMPLE_APPS_DIR": "<repo>/deployments",
  "MDX_DATA_DIR": "<repo>/data",
  "HOST_IP": "<detected>",
  "NGC_CLI_API_KEY": "<from env>"
}
```

> **Note on base URLs**: `LLM_BASE_URL` / `VLM_BASE_URL` must NOT end in `/v1`.
> The agent config appends `/v1` automatically. If the user gives you a URL
> with `/v1`, strip it before writing to the env.

### Remote LLM + local VLM

```json
{
  "HARDWARE_PROFILE": "<detected>",
  "MDX_SAMPLE_APPS_DIR": "<repo>/deployments",
  "MDX_DATA_DIR": "<repo>/data",
  "HOST_IP": "<detected>",
  "NGC_CLI_API_KEY": "<from env>",
  "LLM_MODE": "remote",
  "LLM_BASE_URL": "https://integrate.api.nvidia.com",
  "NVIDIA_API_KEY": "<key>"
}
```

### Remote LLM + remote VLM (`remote-all` — no local GPU for inference)

Fire this recipe when the user says *"deploy in remote-all mode"*,
*"both LLM and VLM are remote"*, or supplies two endpoint URLs (one per
role). Both mode vars MUST flip to `remote`; leaving either at `local`
silently breaks `COMPOSE_PROFILES`.

```json
{
  "HARDWARE_PROFILE": "<detected>",
  "MDX_SAMPLE_APPS_DIR": "<repo>/deployments",
  "MDX_DATA_DIR": "<repo>/data",
  "HOST_IP": "<detected>",
  "LLM_MODE": "remote",
  "LLM_BASE_URL": "<llm-endpoint-from-user>",
  "LLM_NAME":     "<llm-model-from-user>",
  "VLM_MODE": "remote",
  "VLM_BASE_URL": "<vlm-endpoint-from-user>",
  "VLM_NAME":     "<vlm-model-from-user>",
  "NVIDIA_API_KEY": "<key if endpoints require auth>"
}
```

If the user didn't provide endpoint URLs/models, **ask them** — don't
guess. For NVIDIA's public API: `https://integrate.api.nvidia.com` (strip
any trailing `/v1`). For launchpad-style internal endpoints, use the
exact URL they gave you.

Post-write sanity check:
```bash
grep -E '^(LLM_MODE|VLM_MODE|LLM_BASE_URL|VLM_BASE_URL|LLM_NAME|VLM_NAME)=' \
  deployments/developer-workflow/dev-profile-base/.env
```
Expect six lines, all non-empty; `LLM_MODE=remote` and `VLM_MODE=remote`
must both appear. If either is `local`, you didn't overwrite the
template placeholder — re-run the `sed` with the correct value.

### Dedicated GPUs (2-GPU system)

```json
{
  "HARDWARE_PROFILE": "<detected>",
  "MDX_SAMPLE_APPS_DIR": "<repo>/deployments",
  "MDX_DATA_DIR": "<repo>/data",
  "HOST_IP": "<detected>",
  "NGC_CLI_API_KEY": "<from env>",
  "LLM_MODE": "local",
  "VLM_MODE": "local",
  "LLM_DEVICE_ID": "0",
  "VLM_DEVICE_ID": "1"
}
```

### Different LLM model

```json
{
  "LLM_NAME": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
  "LLM_NAME_SLUG": "llama-3.3-nemotron-super-49b-v1.5"
}
```

## COMPOSE_PROFILES (computed — do not set directly)

The `.env` file computes this from other variables:

```
COMPOSE_PROFILES=${BP_PROFILE}_${MODE},${BP_PROFILE}_${MODE}_${HARDWARE_PROFILE},${BP_PROFILE}_${MODE}_${PROXY_MODE},llm_${LLM_MODE}_${LLM_NAME_SLUG},vlm_${VLM_MODE}_${VLM_NAME_SLUG}
```

Example resolved value:
```
bp_developer_base_2d,bp_developer_base_2d_DGX-SPARK,bp_developer_base_2d_no_proxy,llm_local_shared_nvidia-nemotron-nano-9b-v2,vlm_local_shared_cosmos-reason2-8b
```

The agent sets the upstream variables — `COMPOSE_PROFILES` is derived automatically.

## Endpoints (after deploy)

| Service | URL |
|---|---|
| Agent UI | `http://<HOST_IP>:3000/` |
| Agent REST API | `http://<HOST_IP>:8000/` |
| Swagger UI | `http://<HOST_IP>:8000/docs` |
| Reports | `http://<HOST_IP>:8000/static/agent_report_<DATE>.md` |
| Phoenix telemetry | `http://<HOST_IP>:6006/` |

## Env File Location

```
<repo>/deployments/developer-workflow/dev-profile-base/.env
```

## Debugging

After a base deploy is up, confirm the full pipeline (VST upload → VLM →
agent report) by driving a real query through the agent — e.g. ask it over
the REST API or UI to describe a video you've uploaded to VST. If the
agent returns a non-empty answer, the upload → ingest → inference → reply
path is healthy.

Common failure modes and what they mean for base:

| Symptom | Likely cause |
|---|---|
| `POST /api/v1/videos` HTTP 500 | Agent not finished starting — poll `/health` longer |
| VST `sensor/streams` stays empty | VST container unhealthy — check `docker logs vst-ingress-dev` |
| VST returns empty `sensor/streams` but VST container is healthy | `centralizedb-dev` (postgres) can't read PGDATA because `$MDX_DATA_DIR` was `chown`ed to ubuntu. See [SKILL.md § Step 1b](../SKILL.md#step-1b--prepare-the-data-directory) — use `chmod -R 777`, not `chown`. Fix: `sudo rm -rf $MDX_DATA_DIR/data_log/vst/postgres && redeploy` (postgres re-initializes on start) |
| WebSocket query returns `error_message` | LLM or VLM NIM not healthy — `docker logs nvidia-nemotron-nano-9b-v2-shared-gpu` / `cosmos-reason2-8b-shared-gpu` |
| HITL prompt never arrives | `vss-agent` misconfigured HITL config — check `config.yml` |
| Empty report | VLM unreachable from inside `vss-agent` container — check `VLM_BASE_URL` in resolved compose env |

## Known Issues

- `cosmos-reason2-8b` NIM cannot restart after stop/crash — must redeploy full stack
- Reports are in-memory by default — lost on container restart (mount a volume to persist)
- `VLM_NIM_KVCACHE_PERCENT` defaults to `0.7` — may need tuning on memory-constrained GPUs
