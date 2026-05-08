---
name: deploy
description: Deploy, debug, or tear down any VSS profile using a compose-centric workflow — config (dry-run) with env overrides, review resolved compose, then compose up. Use this skill when the user says "deploy vss", "deploy `profile`", "debug deploy", "verify deployment", or "why is my vss deploy broken".
license: Apache-2.0
metadata:
  version: "3.1.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint deployment"
---

# VSS Deploy

Deploy any VSS profile using a compose-centric workflow: build env overrides, generate resolved compose (dry-run), review, then deploy. Replaces direct `dev-profile.sh` execution with validated, auditable steps.

## Profile Routing

| User says | Profile | Reference |
|---|---|---|
| "deploy vss" / "deploy base" | `base` | `references/base.md` |
| "deploy alerts" / "alert verification" / "real-time alerts" | `alerts` | `references/alerts.md` |
| "deploy for incident report" | `alerts` | `references/alerts.md` |
| "deploy lvs" / "video summarization" | `lvs` | `references/lvs.md` |
| "deploy search" / "video search" | `search` | `references/search.md` |

**Edge hardware routing** (DGX Spark, AGX/IGX Thor): see [`references/edge.md`](references/edge.md)
for the 4B-LLM recipe (`config_edge.yml` + standalone vLLM on port 30081). Edge
platforms share a single unified-memory GPU between LLM and VLM, so the
Nemotron Edge 4B is the default and the Nemotron Nano 9B v2 FP8 is an option
when memory allows.

## When to Use

- Deploy VSS / start VSS / bring up a profile
- Deploy a specific profile (base, alerts, lvs, search)
- Do a dry-run / preview what will be deployed
- Change deployment config (hardware, LLM mode, GPU assignment)
- Tear down a running deployment
- **Debug or verify** an existing deployment (see [Debugging a Deployment](#debugging-a-deployment))

## How it works

Run docker compose commands directly on the host:

```bash
# 1. Apply env overrides to the profile .env file
# 2. docker compose --env-file .env config > resolved.yml   (dry-run)
# 3. Review resolved.yml
# 4. docker compose -f resolved.yml up -d
```

## Before Deploying

1. **Repo path** — find `video-search-and-summarization/` on disk. Check `TOOLS.md` if available.
2. **NGC CLI & API key** — see [`references/ngc.md`](references/ngc.md). Check `$NGC_CLI_API_KEY` is set.
3. **System prerequisites (GPU VRAM, driver, Docker, NVIDIA Container Toolkit)** — canonical reference is the [**VSS prerequisites page**](https://docs.nvidia.com/vss/3.1.0/prerequisites.html). That page lists supported hardware, per-profile GPU requirements, and the minimum driver/CUDA version per NIM. Read it and pick the LLM/VLM placement that fits the host — don't guess thresholds from this skill.

### Pre-flight Check

Run before every deploy. Do not proceed if any check fails.

```bash
# 1. GPU visible
nvidia-smi --query-gpu=index,name --format=csv,noheader

# 2. NVIDIA runtime in Docker
docker info 2>/dev/null | grep -i "runtimes"

# 3. NVIDIA runtime works end-to-end
docker run --rm --gpus all ubuntu:22.04 nvidia-smi 2>&1 | head -5
```

If check 2 or 3 fails, see [`references/prerequisites.md`](references/prerequisites.md).

## Deployment Flow

Always follow this sequence. Never skip the dry-run.

### Step 0 — Tear down any existing deployment

If a deployment already exists, tear it down first. Full procedure (resolved.yml-driven path, container-name catch-all patterns covering dev-profile compose files, why leftovers cause /sensor/list 502s) lives in [`references/teardown.md`](references/teardown.md).
# If a resolved.yml from a prior deploy exists, prefer it — it
# knows about all compose-profile services that were brought up.
if [ -f "$REPO/deployments/resolved.yml" ]; then
  docker compose -f "$REPO/deployments/resolved.yml" down --remove-orphans
fi

# Catch-all: remove every VSS-stack container the dev-profile compose
# files bring up. Without this, leftovers from a prior deploy linger
# (especially the *-smc set, which the alerts compose profile shares
# with the *-dev set on host networking and port 30000) and either:
#   - bind ports the new deploy needs → second sensor-ms fails to bind
#     → /sensor/list returns 502 (issue #151), or
#   - pass the new deploy's container-name health checks while serving
#     stale data from the prior deploy's DB.
# The patterns below cover everything declared in
# deployments/vst/{2d,3d,smc,developer,ps}/, deployments/foundational/,
# deployments/agents/, deployments/proxy/, and the dev-profile-*
# compose files.
docker ps -a --format '{{.Names}}' \
  | grep -E '^(vss-|mdx-|perception-|rtvi-|alert-|nvstreamer-|sensor-ms-|vst-ingress-|vst-mcp-|vst-file-proxy|centralizedb-|storage-ms-|streamprocessing-ms-|sdr-(http|streamprocessing)-|envoy-(http|streamprocessing)-|rtspserver-ms-|recorder-ms-|replaystream-ms-|livestream-ms-|metropolis-vss-ui|phoenix)' \
  | xargs -r docker rm -f
```

If this is the host's first deploy, the `docker compose down`
line is a no-op (exit 0 with no containers to stop) — safe to run
unconditionally.

### Step 1 — Gather context

Discover what's available on the host and cross-reference with the
[VSS prerequisites page](https://docs.nvidia.com/vss/3.1.0/prerequisites.html)
to choose a deployment shape that fits.

| Value | How to determine |
|---|---|
| **Profile** | Match user intent to routing table above. Default: `base` |
| **Repo path** | Find `video-search-and-summarization/` on disk |
| **Hardware** | `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` → look up per-GPU VRAM against the prerequisites page |
| **LLM/VLM placement** | Pick `local_shared`, `local`, or `remote` per LLM/VLM based on available GPUs + `$LLM_REMOTE_URL` / `$VLM_REMOTE_URL` / `$NGC_CLI_API_KEY`. If no combination on this host satisfies the prerequisites, stop and report the blocker instead of silently picking another shape. |
| **API keys** | `NGC_CLI_API_KEY` for local NIMs, `NVIDIA_API_KEY` for remote |
| **Host IP** | `hostname -I \| awk '{print $1}'` |

**Hardware profile mapping:**

| GPU name contains | HARDWARE_PROFILE | Recommended LLM path |
|---|---|---|
| H100 | `H100` | Nano 9B v2 (NIM) |
| L40S | `L40S` | Nano 9B v2 (NIM) |
| RTX 6000 Ada, RTX PRO 6000 | `RTXPRO6000BW` | Nano 9B v2 (NIM) |
| GB10 (DGX Spark) | `DGX-SPARK` | **Edge 4B** (vLLM) — see [`references/edge.md`](references/edge.md) |
| IGX | `IGX-THOR` | **Edge 4B** (vLLM) — see [`references/edge.md`](references/edge.md) |
| AGX | `AGX-THOR` | **Edge 4B** (vLLM) — see [`references/edge.md`](references/edge.md) |
| Other | `OTHER` | — |

**Minimum GPU count per (profile × mode × platform).** Canonical source
is the [VSS prerequisites page](https://docs.nvidia.com/vss/3.1.0/prerequisites.html);
reproduced here so the skill can fail fast when the host is too small:

| Profile | Mode | H100 / RTX PRO 6000 (Blackwell) | L40S | DGX-Spark / IGX-Thor / AGX-Thor |
|---|---|---|---|---|
| `base` | shared (`local_shared` LLM + VLM) | **1** | — (48 GB/GPU too small) | **1** (Edge 4B + VLM, unified memory) |
| `base` | dedicated (`local` LLM + VLM) | **2** | **2** | — |
| `base` | `remote-llm` | **1** (VLM local) | **1** (VLM local) | **1** (remote LLM only) |
| `base` | `remote-vlm` | **1** (LLM local) | **1** (LLM local) | — |
| `base` | `remote-all` | **0** | **0** | **0** |
| `lvs` | shared | **1** | — | - |
| `lvs` | dedicated | **2** | **2** | — |
| `lvs` | `remote-llm/vlm` | 1 | 1 | - |
| `lvs` | `remote-all` | 0 | 0 | - |
| `alerts` (verification / CV) | shared | **2**  | — | — |
| `alerts` (verification / CV) | dedicated | **3** | **3**  | — |
| `alerts` (verification / CV) | `remote-all` | 1 | 1 | 1 |
| `alerts` (verification / CV) | `remote-llm/vlm` | 2 | 2 | 1 |
| `alerts` (real-time / VLM) | shared | **2** | — | — |
| `alerts` (real-time / VLM) | dedicated | **3** | **3**  | — |
| `alerts` (real-time / VLM) | `remote-llm` | 2 | 2 | 1 |
| `search` | shared | **2** | — | - |
| `search` | dedicated | **3** | **3**  | — |
| `search` | `remote-*` | **2**  | **2** | - |

A few hard rules encoded in the table:

- **L40S can't do `shared`.** 48 GB is not enough VRAM for LLM + VLM
  on a single GPU. Fall back to `dedicated` or a `remote-*` mode.
- **L40S needs +1 GPU for alerts / search vs H100** because the
  shared-on-one-GPU trick doesn't work — RT-CV / Embed1 must take
  their own GPU, and LLM+VLM still need a second.
- **DGX-Spark / Thor are early-access for most profiles.** Only
  `base` + `lvs` are expected to fully land locally; `alerts` /
  `search` currently require a remote LLM. See
  [`references/edge.md`](references/edge.md).

If the host's (GPU count × VRAM) combination doesn't appear above,
**stop and report the blocker** — don't silently pick a different
mode.

> **Edge shared mode requires Edge 4B + `HF_TOKEN`.** On DGX Spark and AGX/IGX
> Thor, both LLM and VLM must fit in unified memory, AND the standard
> `nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2:1` image has a broken arm64
> manifest. You must run `NVIDIA-Nemotron-Edge-4B-v2.1-EA-020126_FP8` as a
> standalone vLLM container on port 30081 with the agent pointed at it via
> `--use-remote-llm`. Full recipe and the mandatory `HF_TOKEN` verification
> step are in [`references/edge.md`](references/edge.md).

### Step 1b — Prepare the data directory

The data directory layout (asset paths, ownership, mount points, profile-specific subdirs) is documented in [`references/data-directory.md`](references/data-directory.md). Read that file before deploying for the first time on a host or when changing profiles.
# Profile-specific subdirs:
#   alerts → mkdir -p "$DATA/data_log/vss_video_analytics_api" "$DATA/videos/dev-profile-alerts" "$DATA/models/rtdetr-its" "$DATA/models/gdino"
#   search → mkdir -p "$DATA/models"
chmod -R 777 "$DATA/data_log" "$DATA/agent_eval"
# If you created $DATA/models above, also: chmod -R 777 "$DATA/models"
```

> **FORBIDDEN: `chown -R ubuntu:ubuntu $MDX_DATA_DIR` (or any recursive chown).**
>
> This is "good housekeeping" to a shell-admin instinct but is **the** deploy-
> breaking command in this stack. You will observe a "healthy" deploy
> (containers Up, endpoints 200) while the video pipeline is silently broken.
> Use `chmod -R 777` on the specific subdirs above — nothing else.

**Known per-container uid gotchas** (each uses a bind mount under `$DATA`):

| Container | Image | Runs as | Mount path | Symptom if permissions wrong |
|---|---|---|---|---|
| `centralizedb-dev` | postgres:17.6-alpine | uid **70** | `$DATA/data_log/vst/postgres/db` | Can't read own PGDATA → VST `sensor_details` query fails → uploaded videos never appear in `/vst/api/v1/sensor/streams` → warehouse E2E check returns empty |
| `mdx-redis` | redis:8.2.2-alpine | uid **999** | `$DATA/data_log/redis/log`, `/redis/data` | "Can't open the log file: Permission denied" → redis dies → `envoy-streamprocessing` dies (needs Redis Lua script) → stream pipeline broken |
| `elasticsearch` | elasticsearch | uid **1000** | `$DATA/data_log/elastic/{data,logs}` | "AccessDeniedException" on startup → ES refuses to start |
| `vst` / `sensor-ms-dev` | vst | uid **1000** | `$DATA/data_log/vst/*` (videos, clips) | 403 on ingest or stream write |

`chmod -R 777 $DATA/data_log` covers all of these. Do NOT chown them to
individual uids — containers that init their own dirs on first start (like
postgres) will then re-chown to their uid and a later chown back to ubuntu
breaks them.

**If postgres is already broken** (common when redeploying without a clean
`data-dir`):
```bash
sudo rm -rf "$DATA/data_log/vst/postgres"  # postgres re-initializes on next start
docker restart centralizedb-dev
```

### Step 1c — If deploying on Brev, set up secure-link env vars

Brev-specific env vars (`BREV_ENV_ID`, secure-link patterns) are documented in [`references/brev.md`](references/brev.md).
### Step 2 — Build env_overrides

Produce an `env_overrides` dict from the user request and the gathered context: choose remote/local LLM/VLM, set credentials, point at endpoints, set platform-specific flags. The full mapping (every override key, when it applies, defaults, profile-specific differences) lives in [`references/env-overrides.md`](references/env-overrides.md).
### Step 3 — Config / dry-run

**Env file location:** `<repo>/deployments/developer-workflow/dev-profile-<profile>/.env`

> **This is the authoritative `.env`.** Every verifier, healthcheck, and
> post-deploy tool reads from this path. When you apply env overrides
> (from Step 2 or from the user's prompt), write them **directly to this
> file** — not to `generated.env`.
>
> `generated.env` is a scratchpad that `dev-profile.sh` produces during
> its own internal flow; it is NOT read by the verifier and is wiped on
> the next invocation. An agent that uses `dev-profile.sh` as a one-shot
> deploy but leaves the base `.env` untouched will silently fail env
> checks even when the stack comes up cleanly. If you used
> `dev-profile.sh` and see `generated.env` on disk, copy its key/value
> lines back into the base `.env`, or re-apply your `sed` commands
> against the base `.env` after the fact. The base `.env` is the source
> of truth.

```bash
REPO=/path/to/video-search-and-summarization
PROFILE=base
ENV_FILE=$REPO/deployments/developer-workflow/dev-profile-$PROFILE/.env

# Read current .env, apply overrides, write back
# (read lines, update matching keys, append new keys, write)

# Resolve compose
cd $REPO/deployments
docker compose --env-file $ENV_FILE config > resolved.yml
```

The resolved YAML is saved to `<repo>/deployments/resolved.yml`.

### Step 3b — Verify resolved.yml has no unexpanded ${...} tokens

Unexpanded `${VAR}` tokens in `resolved.yml` mean compose did not see those env values. Diagnostic procedure and common culprits live in [`references/troubleshooting.md`](references/troubleshooting.md).
### Step 4 — Review

Show the user a summary of what will be deployed:

- Profile name and hardware
- LLM/VLM models and mode (local/remote/local_shared)
- Services that will start
- GPU device assignment
- Key endpoints (UI port, agent port)

Ask: **"Looks good — deploy now?"** and wait for confirmation before Step 5.

**Exception — autonomous mode.** If the user's request already asks
you to run autonomously (e.g. "deploy X autonomously", "run without
confirmation", "non-interactive"), skip the confirmation prompt and
proceed straight to Step 5. This path exists so automated eval /
CI invocations don't hang waiting for a human reply they'll never
get. In all other cases, a human must approve.

### Step 5 — Deploy

```bash
cd $REPO/deployments
docker compose -f resolved.yml up -d
```

> **Do NOT use `--force-recreate` on retries.** It destroys already-warm
> NIM containers, forcing another 3–5 min torch.compile + CUDA-graph capture
> per NIM. If the previous `up -d` partially failed, fix the root cause
> (usually perms or an env typo) and just re-run `up -d` — Docker will
> re-create only the containers whose config changed or that are down.

Deploy takes ~10-20 min on first run (image pulls + model downloads). Monitor:

```bash
# Container status
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# Logs for a specific service
docker compose -f $REPO/deployments/resolved.yml logs --tail 50 <service>
```

Deploy is complete when all `mdx-*` containers show `Up` status.

### Step 6 — Report endpoints

| Profile | Agent UI | REST API | Other |
|---|---|---|---|
| base | `:3000` | `:8000` (Swagger at `/docs`) | — |
| alerts | `:3000` | `:8000` | VIOS dashboard `:30888/vst/` |
| lvs | `:3000` | `:8000` | — |
| search | `:3000` | `:8000` | — |

Use workflow skills after deployment:
- **alerts** / **incident-report** → alert management and incident queries
- **video-search** → semantic video search
- **video-summarization** → long video summarization
- **vios** → camera/stream management via VIOS
- **video-analytics** → Elasticsearch queries

## Tear Down

```bash
cd $REPO/deployments
docker compose -f resolved.yml down
```

## Debugging a Deployment

Use this workflow when the user asks to "debug the deploy", "verify it's working",
"why is the agent not responding", or similar. The goal is to confirm the full
video-ingestion-to-agent-answer path, not just that containers are "Up".

Each profile reference doc (e.g. [`references/base.md`](references/base.md)) has a
**Debugging** section listing the exact commands to run for that profile.

### Quick checks (all profiles)

```bash
# 1. All expected containers Up
docker ps --format 'table {{.Names}}\t{{.Status}}'

# 2. Agent API + UI responding
curl -sf http://localhost:8000/docs >/dev/null && echo "agent OK"
curl -sf http://localhost:3000/ >/dev/null && echo "ui OK"

# 3. VLM NIM responding (base/lvs profiles)
curl -sf http://localhost:30082/v1/models | python3 -m json.tool

# 4. LLM NIM responding
curl -sf http://localhost:30081/v1/models | python3 -m json.tool
```

### End-to-end video sanity check

After the quick checks above pass, drive a real query through the agent — e.g.
ask it over the REST API or UI to describe a video you've uploaded to VST.
If the agent returns a non-empty answer, the upload → ingest → inference →
reply path is healthy. If it fails, `docker logs vss-agent` shows which stage
tripped.

## Troubleshooting

- `unknown or invalid runtime name: nvidia` → NVIDIA Container Toolkit not installed or Docker not restarted. See [`references/prerequisites.md`](references/prerequisites.md).
- NGC auth error → re-export `NGC_CLI_API_KEY` or follow [`references/ngc.md`](references/ngc.md).
- GPU not detected → run `sudo modprobe nvidia && sudo modprobe nvidia_uvm`, then retry.
- `docker compose up` fails with "no resolved.yml" → run the dry-run (`docker compose config > resolved.yml`, Step 3) first.
- cosmos-reason2-8b crash → must redeploy the full stack (known issue: NIM cannot restart alone).
