---
name: vss-deploy-profile
description: Use to select, configure, deploy, verify, debug, or tear down a VSS profile (base, search, lvs, warehouse, edge). Not for standalone microservices — use the vss-deploy-* skill.
license: Apache-2.0
metadata:
  version: "3.2.0"
  github-url: "https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization"
  tags: "nvidia blueprint deployment"
---
# VSS Deploy

## Purpose

Deploy any VSS profile (`base`, `search`, `lvs`, `warehouse`, `alerts`, `edge`) using a compose-centric workflow: build env overrides, generate resolved compose (dry-run), review, then deploy. This SKILL.md covers the cross-profile concerns (**profile routing**, **prerequisites**, **NGC**, **GPU setup**, and the deploy/teardown flow). Profile-specific service lists, sizing, env recipes, endpoints, and debugging live in per-profile reference docs — load the one that matches the user's intent.

Helper scripts normalize `docker compose config` output and probe selected
remote model endpoints before env mutation. All other deployment work goes
through `compose` / `dev-profile.sh`.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/normalize_resolved_yml.py` | Strip optional `depends_on` entries for services filtered out of `resolved.yml` before deploy. | Path to `resolved.yml` |
| `scripts/probe_remote_models.sh` | Probe an OpenAI-compatible remote LLM/VLM endpoint and verify the selected model id. | Base URL, optional expected model id |

## Profile Routing

Match the user's request to a profile, then load that profile's reference for sizing, services, env recipes, and debugging.

| User says | Profile | Reference |
|---|---|---|
| "deploy vss" / "deploy base" | `base` | [`references/base.md`](references/base.md) |
| "deploy alerts" / "alert verification" / "real-time alerts" / "deploy for incident report" | `alerts` | [`references/alerts.md`](references/alerts.md) |
| "deploy lvs" / "video summarization" | `lvs` | [`references/lvs-profile.md`](references/lvs-profile.md) |
| "deploy search" / "video search" | `search` | [`references/search.md`](references/search.md) |
| "deploy warehouse" / "warehouse blueprint" / "vss warehouse" | `warehouse` | [`references/warehouse.md`](references/warehouse.md) |
| "debug warehouse" / "warehouse not working" / "warehouse FPS low" / "warehouse BEV out of sync" | `warehouse` (debug) | [`references/warehouse-debug.md`](references/warehouse-debug.md) |

**Edge hardware routing** (DGX Spark, AGX/IGX Thor): see [`references/edge.md`](references/edge.md). DGX Spark uses the Spark Nano 9B standalone local LLM on port `30081`; AGX/IGX Thor uses the Edge 4B standalone vLLM fallback.

**Each profile's reference owns its sizing table.** Don't pick a deployment shape from this file — open the profile reference and check minimum GPU count for the host's hardware against the (mode × platform) matrix there.


## Instructions

The deployment flow is always: copy `.env` to `generated.env`, apply overrides, dry-run compose into `resolved.yml`, review, normalize, deploy, then wait for readiness.

```bash
# 1. cp dev-profile-<profile>/.env dev-profile-<profile>/generated.env  (clean copy)
# 2. Apply env overrides to generated.env  (source .env stays untouched)
# 3. docker compose --env-file generated.env config > resolved.yml      (dry-run)
# 4. Review resolved.yml
# 5. docker compose --env-file generated.env -f resolved.yml up -d
```

The source `.env` is treated as **read-only defaults** committed to the repo. The skill's per-deploy working copy is `generated.env` — same pattern `dev-profile.sh` uses internally. This keeps the checked-in `.env` clean across iterations.

## Prerequisites

1. **Repo path** — auto-detect `video-search-and-summarization/` before
   asking the user. Use the detected path as `$REPO` for all subsequent
   commands.
2. **Credential gates** — see [`references/credentials.md`](references/credentials.md): `NGC_CLI_API_KEY` for local/local_shared NIM pulls, `NVIDIA_API_KEY` for remote NIM endpoints, and `HF_TOKEN` for edge recipes that use gated HF models.
3. **System prerequisites (GPU driver, Docker, NVIDIA Container Toolkit, kernel sysctls)** — full checks in [`references/prerequisites.md`](references/prerequisites.md). Canonical hardware/driver matrix is the [VSS prerequisites page](https://docs.nvidia.com/vss/3.2.0/prerequisites.html).

```bash
REPO="${REPO:-}"
if [ -z "$REPO" ]; then
  git_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  candidates=()
  [ -n "$git_root" ] && candidates+=("$git_root")
  candidates+=(
    "$PWD"
    "$PWD/.."
    "$PWD/../.."
    "$HOME/video-search-and-summarization"
    "$HOME/VSS/vss-oss/video-search-and-summarization"
    "$HOME/VSS/video-search-and-summarization"
  )

  for candidate in "${candidates[@]}"; do
    candidate="$(cd "$candidate" 2>/dev/null && pwd -P || true)"
    if [ -n "$candidate" ] \
      && [ -f "$candidate/deploy/docker/compose.yml" ] \
      && [ -x "$candidate/deploy/docker/scripts/dev-profile.sh" ] \
      && [ -d "$candidate/skills/vss-deploy-profile" ]; then
      REPO="$candidate"
      break
    fi
  done
fi

if [ -z "$REPO" ]; then
  echo "Could not auto-detect video-search-and-summarization; ask the user for the checkout path."
else
  echo "REPO=$REPO"
fi
```

### Pre-flight check

Run before every deploy. The full system checklist and remediation steps live
in [`references/prerequisites.md`](references/prerequisites.md#preflight).
For DGX Spark / IGX Thor / AGX Thor, also run the cache-cleaner check in
[`references/edge.md`](references/edge.md#cache-cleaner-every-edge-deploy).

**Detect sudo mode first.** Several pre-flight remediations and the
edge cache-cleaner installer call `sudo`. If the host requires a
sudo password, those steps will silently no-op under `sudo -n` and
leave the deploy in a half-prepared state.

```bash
if sudo -n true 2>/dev/null; then
  echo "passwordless sudo — pre-flight will auto-install missing pieces"
else
  echo "sudo requires password — pre-flight will NOT auto-install; hand commands to the user"
fi
```

When sudo needs a password, the skill **must not** run privileged
installers itself. Surface the copy-pasteable command block from
`references/prerequisites.md` to the user with a *"run this once and
confirm"* handoff, then resume after the user replies.

Minimum smoke test (must succeed):

```bash
nvidia-smi --query-gpu=index,name --format=csv,noheader
docker info 2>/dev/null | grep -qi runtimes \
  && docker run --rm --gpus all ubuntu:22.04 nvidia-smi >/dev/null 2>&1 \
  && echo "nvidia runtime OK"
```

If the smoke test fails, do not proceed; open
[`references/prerequisites.md`](references/prerequisites.md#preflight)
for the remediation tree.

## Model Selection

- `$LLM_REMOTE_URL` / `$VLM_REMOTE_URL` if the user asks for remote
- `$NGC_CLI_API_KEY` (local NIMs) or `$NVIDIA_API_KEY` (remote)

**Endpoint intent gate.** Do not infer remote placement from stray host
environment variables alone. `LLM_ENDPOINT_URL`, `VLM_ENDPOINT_URL`,
`LLM_BASE_URL`, or `VLM_BASE_URL` may be leftovers from an earlier run.
Use remote LLM/VLM only when:

1. the user explicitly requested a remote endpoint or supplied one,
2. local sizing cannot satisfy the selected models and the user agrees to
   remote placement, or
3. an edge recipe requires a standalone local service that VSS treats as
   `remote` (for example DGX Spark Nano 9B on `localhost:30081`).

If any endpoint env var is already set but the user did not ask for remote,
surface it in Step 1 and ask whether to use it or ignore it. Never silently
deploy remote because an env var happened to exist.

If no combination on this host satisfies the profile's sizing requirements, **stop and report the blocker** — don't silently pick another shape.

> **Edge shared mode is platform-specific.** On DGX Spark, run `nvcr.io/nim/nvidia/nvidia-nemotron-nano-9b-v2-dgx-spark:1.0.0-variant` as a standalone local NIM on port `30081` and point the agent at it with `LLM_MODE=remote`. On AGX/IGX Thor, keep using the Edge 4B standalone vLLM fallback with `HF_TOKEN`. Full recipes are in [`references/edge.md`](references/edge.md).

## Deployment Flow

Always follow this sequence. Never skip the dry-run.

### Step 0 — Tear down any existing deployment + clear data volumes

If a deployment already exists, tear it down AND clear stale data volumes before redeploying. 

Full procedure lives in [`references/teardown.md`](references/teardown.md).

### Step 0a — Credentials gate (run before any env mutation)

Validate every credential and selected remote endpoint the chosen profile
needs **before** Step 1c copies `.env` to `generated.env`. A 401 here is a
30-second failure; the same 401 inside a NIM cold-start is a 10–20 min
failure. Run the discovery and probe flow in
[`references/credentials.md`](references/credentials.md), including
`scripts/probe_remote_models.sh` for any LLM/VLM endpoint you plan to write
into `generated.env`. Map the result against the chosen mode: missing
or invalid required credentials/endpoints are blockers, optional credentials
are not.

### Step 1 — Gather context

Before building env overrides, confirm:

| Value | How to determine |
|---|---|
| **Profile** | Match user intent to the routing table above. Default: `base` |
| **Repo path** | Use the `$REPO` value auto-detected in prerequisites. If auto-detect failed, ask the user for the checkout path before continuing. |
| **Hardware** | `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader` |
| **LLM/VLM placement** | Explicitly decide local / local_shared / remote. Cross-reference available GPUs against the chosen profile's **Minimum GPU count** table. If endpoint env vars are present but the user did not request remote, ask whether to use or ignore them. |
| **API keys** | `NGC_CLI_API_KEY` for local NIMs, `NVIDIA_API_KEY` for remote |
| **`HOST_IP`** | `hostname -I \| awk '{print $1}'` — the host's primary internal IP |
| **`EXTERNAL_IP`** | Browser-reachable host/IP. On Brev, use the secure-link domain (see [`references/brev.md`](references/brev.md)). |
| **`HAPROXY_PORT`** | Browser-facing ingress port. Default `7777`; ensure it is free. |

Before `docker compose up`, verify `EXTERNAL_IP`, `HAPROXY_PORT`, `VSS_PUBLIC_HOST`, and `VSS_PUBLIC_PORT` are populated with browser-reachable values. Otherwise the stack may appear healthy while UI/API/VST links 404 or loop through Cloudflare Access.

### Step 1b — Prepare the data directory

Layout (asset paths, ownership, mount points, profile-specific subdirs) is documented in [`references/data-directory.md`](references/data-directory.md). Read that file before deploying for the first time on a host or when changing profiles.

> **FORBIDDEN: `chown -R ubuntu:ubuntu $VSS_DATA_DIR` (or any recursive chown).**
>
> This is "good housekeeping" to a shell-admin instinct but is **the** deploy-breaking command in this stack. You will observe a "healthy" deploy (containers Up, endpoints 200) while the video pipeline is silently broken. Use `chmod -R 777` on the specific subdirs documented in `data-directory.md` — nothing else.

### Step 1c — Initialize `generated.env`

The skill's per-deploy working copy. Always start from a fresh copy of the source `.env` — never mutate the source.

```bash
PROFILE=base
ENV_SRC=$REPO/deploy/docker/developer-profiles/dev-profile-$PROFILE/.env
ENV_GEN=$REPO/deploy/docker/developer-profiles/dev-profile-$PROFILE/generated.env

cp "$ENV_SRC" "$ENV_GEN"
```

All subsequent writes (Brev `EXTERNAL_IP`, the env_overrides dict from Step 2) go to `$ENV_GEN`. `$ENV_SRC` is read-only from here on.

### Step 1d — If deploying on Brev, set `EXTERNAL_IP` to the secure-link domain

Read `BREV_ENV_ID` from `/etc/environment` and write `EXTERNAL_IP` into `generated.env` (NOT `.env`). Full secure-link behavior and troubleshooting are in [`references/brev.md`](references/brev.md).

```bash
brev_env_id=$(awk -F= '/^BREV_ENV_ID=/ {gsub(/"/, "", $2); print $2; exit}' /etc/environment)
sed -i "s|^EXTERNAL_IP=.*|EXTERNAL_IP=7777-${brev_env_id}.brevlab.com|" "$ENV_GEN"
```

### Step 2 — Build env_overrides

Produce an `env_overrides` dict from the user request and the gathered
context: explicitly choose remote/local LLM/VLM, set credentials, point at
endpoints, set platform-specific flags. Do not let existing shell env vars
silently pick placement; write the selected `LLM_MODE` / `VLM_MODE` and
matching endpoint/model fields into `generated.env`. The full mapping (every
override key, when it applies, defaults, profile-specific differences) lives
in [`references/env-overrides.md`](references/env-overrides.md). Each profile
reference has worked examples for that profile's common scenarios.

### Step 3 — Apply overrides + dry-run

**Working env file:** `<repo>/deploy/docker/developer-profiles/dev-profile-<profile>/generated.env` (created in Step 1c).

> **Two env files, distinct roles.**
> - `.env` — **read-only defaults**, checked in. Don't mutate it from the skill.
> - `generated.env` — **the skill's per-deploy working copy**. All overrides (the dict from Step 2, plus the Brev `EXTERNAL_IP` from Step 1d) land here. `--env-file` always points at this file. Post-deploy verifiers should also read from `generated.env` for the actually-deployed values — see [Debugging a Deployment](#debugging-a-deployment).
>
> `generated.env` matches the convention `dev-profile.sh` uses internally — it's a per-invocation scratchpad regenerated by `cp .env generated.env` each run.

```bash
# (Step 1c already ran: cp $ENV_SRC $ENV_GEN)

# Apply the env_overrides dict from Step 2 to generated.env
# (read lines, update matching keys, append new keys, write)
# Example:
#   sed -i "s|^LLM_MODE=.*|LLM_MODE=remote|" "$ENV_GEN"
#   sed -i "s|^LLM_BASE_URL=.*|LLM_BASE_URL=http://localhost:30081|" "$ENV_GEN"

# Resolve compose
cd $REPO/deploy/docker
docker compose --env-file $ENV_GEN config > resolved.yml
```

The resolved YAML is saved to `<repo>/deploy/docker/resolved.yml`.

### Step 3b — Verify resolved.yml has no unexpanded ${...} tokens

Unexpanded `${VAR}` tokens in `resolved.yml` mean compose did not see those env values. Diagnostic procedure and common culprits live in [`references/troubleshooting.md`](references/troubleshooting.md).

### Step 3c — Strip dangling optional `depends_on` from resolved.yml

**MUST run after Step 3, before Step 5.** Skipping this aborts the deploy:

Normalize - drop optional dependencies for services filtered out from resolved.yml

```bash
# From the repo root
uv run skills/vss-deploy-profile/scripts/normalize_resolved_yml.py "$REPO/deploy/docker/resolved.yml"
```
If `uv` isn't on the host, install it once with `curl -LsSf https://astral.sh/uv/install.sh | sh` (no root needed).
**Re-validate** before `up -d`:

```bash
docker compose -f "$REPO/deploy/docker/resolved.yml" config --quiet && echo "resolved.yml OK"
```

If validation still fails after the normalizer runs, capture the error and inspect — that's a different bug (a dependency that's not optional, or another schema violation), not the dangling-depends_on case.

### Step 4 — Review

Show the user a summary of what will be deployed:

- Profile name and hardware
- LLM/VLM models and mode (local/remote/local_shared)
- Services that will start
- GPU device assignment
- Key endpoints (UI port, agent port)

Ask: **"Looks good — deploy now?"** and wait for confirmation before Step 5.

**Exception — autonomous mode.** If the user's request already asks you to run autonomously (e.g. "deploy X autonomously", "run without confirmation", "non-interactive"), skip the confirmation prompt and proceed straight to Step 5. This path exists so automated eval / CI invocations don't hang waiting for a human reply they'll never get. In all other cases, a human must approve.

### Step 5 — Deploy

```bash
cd $REPO/deploy/docker
docker compose --env-file $ENV_GEN -f resolved.yml up -d
```

> **`--env-file` is mandatory.** Without the same `generated.env` used in Step 3, `COMPOSE_PROFILES` may be unset and `up -d` can exit 0 with zero selected services.

> **Avoid broad `--force-recreate` on ordinary retries.** It destroys
> already-warm NIM containers, forcing another 3–5 min torch.compile +
> CUDA-graph capture per NIM. If the previous `up -d` partially failed, fix
> the root cause (usually perms or an env typo) and just re-run `up -d` —
> Docker will re-create only containers whose config changed or that are
> down. Use targeted `--force-recreate --no-deps <service...>` only when a
> profile reference documents it as the recovery path for a specific stale
> service/config issue.

`docker compose up -d` only creates containers; it does not wait for internal services to finish warming. Never declare deploy success until the readiness gates pass.

### Step 5b — Wait until the stack is actually healthy

**Gate 0 — container count must be > 0.** Refuse to proceed past `up -d` until compose started the expected services:

```bash
expected=$(docker compose --env-file "$ENV_GEN" -f resolved.yml config --services | wc -l)
actual=$(docker compose -f resolved.yml ps -q | wc -l)
[ "$expected" -gt 0 ] && [ "$actual" -gt 0 ] && [ "$actual" -ge "$expected" ] \
  || { echo "FAIL: expected $expected services, got $actual — re-check Step 5 --env-file"; exit 1; }
```

Cold deploys can take 10–20 min. The full readiness procedure lives in [`references/readiness.md`](references/readiness.md), and each profile reference lists the required endpoints. **Never declare deploy done after `up -d`; only after every documented endpoint succeeds.**

## Tear Down

```bash
cd $REPO/deploy/docker
docker compose -f resolved.yml down
```

For switching profiles or recovering from a partial deploy, follow the full procedure in [`references/teardown.md`](references/teardown.md).

## Debugging a Deployment

Use this workflow when the user asks to "debug the deploy", "verify it's working", "why is the agent not responding", or similar. The goal is to confirm the full video-ingestion-to-agent-answer path, not just that containers are "Up".

Each profile reference has a **Debugging** section listing the exact commands and failure-mode table for that profile.

### Quick checks (all profiles)

```bash
# 1. All expected containers Up
docker ps --format 'table {{.Names}}\t{{.Status}}'

# 2. Agent API + UI responding
curl -sf http://localhost:8000/health >/dev/null && echo "agent OK"
curl -sf http://localhost:3000/ >/dev/null && echo "ui OK"

if [ -n "${ENV_GEN:-}" ] && [ -f "$ENV_GEN" ]; then
  LLM_MODE="${LLM_MODE:-$(awk -F= '$1=="LLM_MODE"{print $2}' "$ENV_GEN" | tail -1)}"
  VLM_MODE="${VLM_MODE:-$(awk -F= '$1=="VLM_MODE"{print $2}' "$ENV_GEN" | tail -1)}"
  LLM_BASE_URL="${LLM_BASE_URL:-$(awk -F= '$1=="LLM_BASE_URL"{print $2}' "$ENV_GEN" | tail -1)}"
  VLM_BASE_URL="${VLM_BASE_URL:-$(awk -F= '$1=="VLM_BASE_URL"{print $2}' "$ENV_GEN" | tail -1)}"
  LLM_NAME="${LLM_NAME:-$(awk -F= '$1=="LLM_NAME"{print $2}' "$ENV_GEN" | tail -1)}"
  VLM_NAME="${VLM_NAME:-$(awk -F= '$1=="VLM_NAME"{print $2}' "$ENV_GEN" | tail -1)}"
fi

# 3. VLM NIM responding (base/lvs profiles)
# Skip localhost:30082 when VLM_MODE=remote; HTTP 000/connection refused is expected.
# Probe the selected VLM_BASE_URL /v1/models endpoint instead.
if [ "${VLM_MODE:-}" = "remote" ]; then
  echo "VLM_MODE=remote — skip localhost:30082; probing ${VLM_BASE_URL:-<remote-vlm-base-url>}/v1/models"
  REMOTE_API_KEY="${NVIDIA_API_KEY:-}" \
    "$REPO/skills/vss-deploy-profile/scripts/probe_remote_models.sh" "$VLM_BASE_URL" "${VLM_NAME:-}"
else
  curl -sf http://localhost:30082/v1/models | python3 -m json.tool
fi

# 4. LLM NIM responding
# Skip localhost:30081 when LLM_MODE=remote; HTTP 000/connection refused is expected.
# Probe the selected LLM_BASE_URL /v1/models endpoint instead.
if [ "${LLM_MODE:-}" = "remote" ]; then
  echo "LLM_MODE=remote — skip localhost:30081; probing ${LLM_BASE_URL:-<remote-llm-base-url>}/v1/models"
  REMOTE_API_KEY="${NVIDIA_API_KEY:-}" \
    "$REPO/skills/vss-deploy-profile/scripts/probe_remote_models.sh" "$LLM_BASE_URL" "${LLM_NAME:-}"
else
  curl -sf http://localhost:30081/v1/models | python3 -m json.tool
fi
```

### End-to-end video sanity check

After the quick checks above pass, drive a real query through the agent — e.g. ask it over the REST API or UI to describe a video you've uploaded to VST. If the agent returns a non-empty answer, the upload → ingest → inference → reply path is healthy. If it fails, `docker logs vss-agent` shows which stage tripped.

## Examples

- Base profile, remote models: route to `base`, copy `dev-profile-base/.env` to `generated.env`, set `LLM_MODE=remote` / `VLM_MODE=remote`, dry-run, normalize, deploy, then verify `/docs` and UI.
- Search profile on RTX: route to `search`, follow [`references/search.md`](references/search.md) for sizing and endpoints, seed videos, then run the search-profile readiness checks.
- Edge target: route through [`references/edge.md`](references/edge.md), then use the same `generated.env` → dry-run → normalize → deploy flow.

## Limitations

- This skill deploys compose-based VSS profiles only; standalone microservice deployment belongs to the matching `vss-deploy-*` skill.
- Hardware sizing, model placement, and profile-specific readiness are owned by profile references; do not infer them from memory.
- Privileged host remediation requires user approval when passwordless sudo is unavailable.

## Troubleshooting

Start with [`references/agent-failure-modes.md`](references/agent-failure-modes.md) for cross-profile failures such as NIM cold-start timeouts, OOM, remote endpoint 5xx responses, missing `NGC_CLI_API_KEY` / `HF_TOKEN`, unexpanded values in `resolved.yml` etc.
