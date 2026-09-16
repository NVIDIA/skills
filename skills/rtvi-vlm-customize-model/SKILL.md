---
name: rtvi-vlm-customize-model
description: How to swap the VLM in the VSS Alerts Blueprint — covers RTVI-VLM microservice deployment methods, all three VLM consumers (rtvi-vlm, vlm-as-verifier, vss-agent), and health checks.
metadata:
  version: "1.0.0"
  team: accelerated-microservices
  author: "NVIDIA CORPORATION <info@nvidia.com>"
  tags:
    - vlm
    - vss
    - rtvi
  languages:
    - en
  domain: computer-vision
---

# VLM Customization — VSS Alerts Blueprint

## When to use

Use this skill when the user wants to:

- repoint the VSS Alerts Blueprint to a different VLM endpoint,
- run `rtvi-vlm` standalone with either an OpenAI-compatible endpoint or the in-container vLLM path,
- fix the assumption that changing one VLM config automatically updates `rtvi-vlm`, `vlm-as-verifier`, and `vss-agent`.

Do **not** use this skill for CV detector swaps inside `vss-rt-cv`; use `rtvi-cv-customize-model` for those.

## Instructions

- Treat `rtvi-vlm`, `vlm-as-verifier`, and `vss-agent` as three separate VLM consumers. Do not imply that changing only `RTVI_VLM_*` automatically repoints the verifier or the agent UI.
- Treat all paths in this skill as relative to a checkout of the public [VSS Blueprint repository](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization). Blueprint deployment files live under `deploy/docker/...`; customer-accessible RT-VLM source and standalone deployment files live under `services/rtvi/rt-vlm/...`. None of these paths resolve inside the DeepStream repository.
- For blueprint-side OpenAI-compatible routing, cover all three surfaces: `RTVI_VLM_*` env vars, `${VSS_PROFILE_DIR}/vlm-as-verifier/configs/config.yml`, and the `vss-agent` `VLM_MODEL_TYPE` / `VLM_NAME` / `VLM_BASE_URL` settings, then mention force-recreate plus log checks.
- For standalone `vllm-compatible`, set `VLM_MODEL_TO_USE=vllm-compatible`, point `MODEL_PATH` at the weights source, mention `HF_TOKEN` only when the model actually needs auth. Do not treat a log line as proof of a working deployment: v3.2.1 logs `Warmup VlmProcess-0 done` even when warm-up raised an exception. Require the absence of an `Error during warmup` line, readiness, and one successful `/v1/chat/completions` response.
- Never paste credentials into chat or logs; capture with a silent prompt (`read -rsp`), `chmod 600` the env file, and never commit `generated.env`.

## Examples

- "Point the VSS Alerts Blueprint at a host-side NIM on `http://host.docker.internal:30082` for all RTVI-VLM calls."
- "Run `rtvi-vlm` standalone with Qwen3-VL-8B-Instruct served inside the container."
- "I changed only `RTVI_VLM_ENDPOINT` in `generated.env`. That should repoint `vlm-as-verifier` and `vss-agent` too, right?"

## Source locations

This skill is documentation-only — no VSS sources ship here. Use a VSS **v3.2.1 or compatible** checkout and run every command from its root. Clone/LFS, Alerts profile paths, RT-VLM tree, pinned images, and `VSS_*` vars: [references/vss-source-layout.md](references/vss-source-layout.md).

## Which modes this applies to

| Mode flag | Workflow | VLM role |
|-----------|----------|----------|
| `--mode real-time` (`2d_vlm`) | Real-time VLM alerts | **Primary** — every alert is VLM-driven |
| `--mode verification` (`2d_cv`) | CV + VLM verification | **Verifier** — VLM confirms each CV incident |

VLM customization applies to both modes. Three services each make their own VLM calls with separate configuration:

| Service | When used | Config location |
|---------|-----------|----------------|
| **rtvi-vlm** | Real-time alert generation; `rtvi_vlm_alert` tool calls from agent UI | `RTVI_VLM_*` vars in `${VSS_PROFILE_DIR}/.env` / `generated.env` |
| **vlm-as-verifier** (`alert-bridge`) | Post-processing: confirms each `mdx-incidents` event (`2d_cv` only) | `${VSS_PROFILE_DIR}/vlm-as-verifier/configs/config.yml` |
| **vss-agent** | Interactive agent UI queries | `VLM_MODEL_TYPE`, `VLM_BASE_URL`, `VLM_NAME` in `${VSS_PROFILE_DIR}/.env` |

All three can point at the same model endpoint — they don't have to.

`alert-bridge` and `vss-agent` are closed-source; their pinned images (and the
`rtvi-vlm` image tag) are listed in
[references/vss-source-layout.md](references/vss-source-layout.md).

---

## RTVI-VLM: Two Deployment Methods

The implementation is selected by `VLM_MODEL_TO_USE` (standalone) or
`RTVI_VLM_MODEL_TO_USE` (VSS blueprint).

### Method A: OpenAI-Compatible Endpoint (`openai-compat`)

The RTVI container makes HTTP calls to an external inference server; it does
not load the model itself. This covers local or remote NIM, external vLLM, and
OpenAI.

### Method B: vLLM Inside the RTVI Container (`vllm-compatible`)

The container downloads and serves the model using its bundled vLLM engine. No
external inference server is needed.

**Hard constraint:** The model architecture must be supported by the vLLM
version shipped in the selected RTVI image. Check the supported-model guidance
in [`services/rtvi/rt-vlm/README.md`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/v3.2.1/services/rtvi/rt-vlm/README.md). If the model requires a newer vLLM, use Method A with an external container instead.

---

## Configuring RTVI-VLM via VSS Blueprint

The VSS blueprint `.env` / `generated.env` maps most `RTVI_VLM_*` variables to container-native names. The model or deployment identifier is the important exception: v3.2.1 maps `VLM_NAME` to `VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME`. `RTVI_VLM_PORT` is also exceptional: it is required by Compose to publish the service and must be explicitly supplied when `generated.env` does not contain it.

### Required port configuration — all Blueprint workflows

Add `RTVI_VLM_PORT` and update the two hardcoded `8018` lines in the Alerts
profile `.env`, then rerun `dev-profile.sh`. Edit `.env`, not `generated.env` —
the generator overwrites it.

```bash
# ${VSS_PROFILE_DIR}/.env
RTVI_VLM_PORT=8018                                        # add this line
RTVI_VLM_BASE_URL=http://${HOST_IP}:${RTVI_VLM_PORT}     # update: was http://${HOST_IP}:8018
RTVI_VLM_ENDPOINT=http://${HOST_IP}:${RTVI_VLM_PORT}/v1  # update: was http://${HOST_IP}:8018/v1
```

Keep the value at `8018` unless that port is unavailable. Variable ownership,
the `VLM_BASE_URL` generator overwrite, and the patch required for any other
port are in
**[references/port-and-url-wiring.md](references/port-and-url-wiring.md)**.

### Method A — OpenAI-compatible endpoint

In the v3.2.1 blueprint Compose file, these host variables map as follows:

| Blueprint variable | Container variable |
|---|---|
| `RTVI_VLM_MODEL_TO_USE` | `VLM_MODEL_TO_USE` |
| `RTVI_VLM_ENDPOINT` | `VIA_VLM_ENDPOINT` |
| `RTVI_VLM_API_KEY` | `VIA_VLM_API_KEY` |
| `VLM_NAME` | `VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME` |
| `RTVI_VLM_MODEL_PATH` | `MODEL_PATH` |

After applying the shared port configuration above, configure the active
profile environment:

```bash
# ${VSS_PROFILE_DIR}/.env
RTVI_VLM_MODEL_TO_USE=openai-compat
RTVI_VLM_ENDPOINT=http://host.docker.internal:30082/v1
VLM_NAME=<model-or-deployment-id>
OPENAI_API_KEY=<your-api-key>     # client fallback/default credential
RTVI_VLM_API_KEY=<your-api-key>   # mapped to preferred VIA_VLM_API_KEY
# RTVI_VLM_IMAGE_TAG=3.2.1    # pin to a specific image tag; defaults to 3.2.1
```

For an authenticated Method A endpoint, set `OPENAI_API_KEY` and
`RTVI_VLM_API_KEY` to the same endpoint-specific credential. The stock Compose
file maps `RTVI_VLM_API_KEY` to `VIA_VLM_API_KEY`, and the OpenAI-compatible
client prefers `VIA_VLM_API_KEY` over `OPENAI_API_KEY`. Leave both unset only
when the endpoint intentionally accepts unauthenticated requests.

Examples:

```bash
# NIM on the Docker host. Stock Compose maps this name through host-gateway.
RTVI_VLM_ENDPOINT=http://host.docker.internal:30082/v1
VLM_NAME=nvidia/cosmos-reason2-8b

# NVIDIA API catalog (set keys via silent prompt)
RTVI_VLM_ENDPOINT=https://integrate.api.nvidia.com/v1
VLM_NAME=nvidia/cosmos-reason2-8b
OPENAI_API_KEY=<your-nvidia-api-key>
RTVI_VLM_API_KEY=<your-nvidia-api-key>

# OpenAI (set keys via silent prompt)
RTVI_VLM_ENDPOINT=https://api.openai.com/v1
VLM_NAME=gpt-4o
OPENAI_API_KEY=<your-openai-api-key>
RTVI_VLM_API_KEY=<your-openai-api-key>
```

> **Migrating from Method B:** Replace `RTVI_VLM_API_KEY` before recreating
> `rtvi-vlm`; do not leave the previous NGC credential in that variable. If
> `RTVI_VLM_API_KEY` is empty, stock Compose falls back to `NGC_CLI_API_KEY`
> when populating `VIA_VLM_API_KEY`. Because the client prefers
> `VIA_VLM_API_KEY`, a stale NGC key would be sent to the new Method A endpoint
> instead of `OPENAI_API_KEY`. This is both an authentication failure and a
> credential-disclosure risk.

`RTVI_VLM_OPENAI_MODEL_DEPLOYMENT_NAME` is not consumed by the public v3.2.1
blueprint Compose file. Use `VLM_NAME` unless the deployed Compose file has
been intentionally customized.

Verify the configured model identifier from inside the RTVI container. The
conditional `Authorization` header keeps keyless local endpoints working while
authenticating endpoints such as OpenAI and the NVIDIA API Catalog. Endpoints
may advertise multiple models, so confirm the configured ID appears anywhere in
the `/models` response:

```bash
ALERTS_ENV=developer-profiles/dev-profile-alerts/generated.env

# Container-native name for `VLM_NAME`.
EXPECTED_MODEL="$(docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm \
  printenv VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME | tr -d '\r')"
test -n "${EXPECTED_MODEL}"

docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm sh -lc \
  'set --
   if [ -n "${VIA_VLM_API_KEY:-}" ]; then
     set -- -H "Authorization: Bearer ${VIA_VLM_API_KEY}"
   fi
   curl -fsS "$@" "${VIA_VLM_ENDPOINT%/}/models"' \
  | jq -e --arg model "${EXPECTED_MODEL}" 'any(.data[]; .id == $model)' \
  || { echo "Endpoint does not advertise '${EXPECTED_MODEL}'" >&2; exit 1; }
```

`jq -e` exits non-zero when the assertion is `false`, so the check passes only
when the configured model appears somewhere in the list. To see which models
the endpoint offers, drop the `jq` filter and read the raw response.

Treat HTTP 401 from this request as an authentication failure. Only treat a
successful response that fails the `jq` assertion as a model-name mismatch.

### Method B — vLLM inside RTVI container

```bash
RTVI_VLM_MODEL_TO_USE=vllm-compatible
RTVI_VLM_MODEL_PATH=git:https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct

# With HuggingFace token:
# HF_TOKEN=<your-hf-token>

# From NGC:
# RTVI_VLM_MODEL_PATH=ngc:nim/nvidia/cosmos-reason2-8b:hf-1208
```

`MODEL_PATH` prefix conventions (see public
[`services/rtvi/rt-vlm/src/vlm_pipeline/ngc_model_downloader.py`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/v3.2.1/services/rtvi/rt-vlm/src/vlm_pipeline/ngc_model_downloader.py)):
- `ngc:<registry>/<org>/<model>:<version>` — download from NGC
- `git:<url>` — clone from HuggingFace or any git LFS repo
- `/path/to/local/dir` — use pre-downloaded local weights

A bad `MODEL_PATH` aborts startup during download, before the server binds its
port, so the container exits non-zero. v3.2.1 raises a plain `Exception`:
`Failed to download model <name> from <url>` for `git:` paths, and
`Model download failed with status code <code>`, `Could not authenticate with NGC.`,
or `Could not find the model.` for `ngc:` paths. 

---

## Configuring RTVI-VLM Standalone

When running `services/rtvi/rt-vlm/docker/compose.yaml` from the public VSS checkout, use the container-native names:

```bash
# services/rtvi/rt-vlm/docker/.env
BACKEND_PORT=8000

# Method A
VLM_MODEL_TO_USE=openai-compat
VIA_VLM_ENDPOINT=http://host.docker.internal:30082/v1
VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME=nvidia/cosmos-reason2-8b
VIA_VLM_API_KEY=<your-api-key>       # if required; set via silent prompt

# Method B
VLM_MODEL_TO_USE=vllm-compatible
MODEL_PATH=git:https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
```

The standalone Compose file defaults to `nvcr.io/nvidia/vss-core/vss-rt-vlm:3.2.1` and requires `BACKEND_PORT` to be set. All supported `VLM_MODEL_TO_USE` values: `openai-compat`, `vllm-compatible`, `cosmos-reason1`, `cosmos-reason2`, `cosmos-reason3`, `custom`.

Both stock Compose definitions use bridged networking and map
`host.docker.internal` through `host-gateway`. Choose the endpoint host by
where the inference server runs:

- Docker host: `host.docker.internal`
- Same Compose network: the inference server's Compose service name
- Remote machine: a routable hostname or IP address
- `localhost`: only when the RTVI container explicitly uses host networking

For standalone Method A, verify connectivity from inside `rtvi-server`:

```bash
docker compose exec -T rtvi-server sh -lc \
  'set --
   if [ -n "${VIA_VLM_API_KEY:-}" ]; then
     set -- -H "Authorization: Bearer ${VIA_VLM_API_KEY}"
   fi
   curl -fsS "$@" "${VIA_VLM_ENDPOINT%/}/models"'
```

See the environment-variable reference in public
[`services/rtvi/rt-vlm/README.md`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/v3.2.1/services/rtvi/rt-vlm/README.md).

---

## Configuring vlm-as-verifier (`2d_cv` mode only)

`vlm-as-verifier` runs as the `alert-bridge` service and is separate from
RTVI-VLM. Its config is bind-mounted from
`${VSS_PROFILE_DIR}/vlm-as-verifier/configs/config.yml`.

Keep the config parameterized:

```yaml
# ${VSS_PROFILE_DIR}/vlm-as-verifier/configs/config.yml
vlm:
  base_url: ${VLM_BASE_URL}/v1
  model: ${VLM_NAME}
  max_tokens: 4096
```

For the v3.2.1 local-VLM Alerts flow, `dev-profile.sh --vlm-device-id ...`
populates `VLM_BASE_URL` with `http://<host-ip>:8018` and sets `VLM_NAME` in
the generated environment. Remote mode likewise writes the selected endpoint.
Only hardcode these fields when deliberately overriding the profile-generated
values.

The `8018` here is literal in the generator, not derived from
`RTVI_VLM_PORT`. If RTVI-VLM is published on any other port, `alert-bridge`
still dials 8018 until you patch `VLM_BASE_URL` in `generated.env`
([references/port-and-url-wiring.md](references/port-and-url-wiring.md)).

After editing the config or environment, recreate the service:

```bash
cd "${VSS_DEPLOY_DIR}"
docker compose \
  --env-file developer-profiles/dev-profile-alerts/generated.env \
  up -d --force-recreate alert-bridge
```

---

## Configuring vss-agent

`vss-agent` is another separate consumer. The v3.2.1 Alerts profile selects a
block in `${VSS_PROFILE_DIR}/vss-agent/configs/config.yml` using
`VLM_MODEL_TYPE`. The local RTVI-VLM flow uses `rtvi`:

```yaml
rtvi_vlm:  # VLM_MODEL_TYPE=rtvi
  base_url: ${RTVI_VLM_BASE_URL}/v1
nim_vlm:   # VLM_MODEL_TYPE=nim
  base_url: ${VLM_BASE_URL}/v1
openai_vlm: # VLM_MODEL_TYPE=openai
  base_url: ${VLM_BASE_URL}/v1
vllm_vlm:  # VLM_MODEL_TYPE=vllm
  base_url: ${VLM_BASE_URL}/v1
```

For a v3.2.1 local Alerts deployment, keep the generated values, which are
normally equivalent to:

```bash
VLM_MODEL_TYPE=rtvi
VLM_NAME=<model-id>
RTVI_VLM_PORT=8018                                    # from .env
RTVI_VLM_BASE_URL=http://<host-ip>:${RTVI_VLM_PORT}   # from .env
VLM_BASE_URL=http://<host-ip>:8018                    # written by dev-profile.sh
```

`RTVI_VLM_PORT` is required by the `rtvi-vlm` Compose port mapping and has no
Compose default. Define it in the Alerts profile `.env` and regenerate
`generated.env` before using any of these Compose commands. The two URLs have
different owners and agree only at the default port; see
[references/port-and-url-wiring.md](references/port-and-url-wiring.md).

For a remote NIM, OpenAI-compatible endpoint, or external vLLM, select the
matching `nim`, `openai`, or `vllm` profile and set `VLM_BASE_URL` without a
trailing `/v1` because the config appends it. Recreate `vss-agent` after
changing these values.

---

## Health Checks

After changing VLM configuration, recreate the affected service and inspect it
through the same Alerts Compose model:

```bash
cd "${VSS_DEPLOY_DIR}"
ALERTS_ENV=developer-profiles/dev-profile-alerts/generated.env
# Required when an existing `generated.env` does not yet contain the port.
# The persistent fix is to add it to the profile `.env` and regenerate.
export RTVI_VLM_PORT=8018

# Recreate RTVI-VLM and wait up to 10 minutes for its Compose healthcheck.
docker compose --env-file "${ALERTS_ENV}" \
  up -d --force-recreate --wait --wait-timeout 600 rtvi-vlm

# The container must still be running. A failed weights download exits here.
CID="$(docker compose --env-file "${ALERTS_ENV}" ps -aq rtvi-vlm)"
test "$(docker inspect -f '{{.State.Status}}' "${CID}")" = running || {
  docker logs --tail 50 "${CID}" >&2; exit 1; }

# Require the absence of a download or warm-up failure. `Warmup ... done` is
# logged on both paths in v3.2.1, so only the error lines are conclusive.
docker logs "${CID}" 2>&1 | grep -Eq \
  'Failed to download model|Model download failed|Could not find the model|Could not authenticate with NGC|Error during (model )?warmup' \
  && { echo 'rtvi-vlm failed to download weights or warm up' >&2; exit 1; }

# Confirm the readiness endpoint from inside the container.
docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm \
  curl -fsS http://localhost:8000/v1/health/ready

# Capture the implementation and the model ID advertised by the live service.
VLM_METHOD="$(docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm \
  printenv VLM_MODEL_TO_USE | tr -d '\r')"
ACTUAL_MODEL="$(docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm \
  curl -fsS http://localhost:8000/v1/models | jq -er '.data[0].id')"
test -n "${ACTUAL_MODEL}"

# Readiness and /v1/models only report metadata. Require one real inference; a
# text-only chat request needs no media asset and works on both methods.
docker compose --env-file "${ALERTS_ENV}" exec -T rtvi-vlm \
  curl -fsS --max-time 120 -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(jq -nc --arg m "${ACTUAL_MODEL}" '{model: $m, max_tokens: 16, stream: false,
        messages: [{role: "user", content: "Reply with: ok"}]}')" \
  | jq -e '((.choices[0].message.content // "") | length) > 0' > /dev/null \
  || { echo 'inference smoke test returned no content' >&2; exit 1; }
```

Readiness is built from `is_alive()` per process, so it reports `healthy: true`
for a live process whose model never warmed up, and `/v1/chat/completions`
returns HTTP 200 with empty `content` when the pipeline produced no output. The
`jq` assertion on non-empty content is what makes the last check meaningful.

### Model identity differs by method

The two methods advertise their model ID differently, so the follow-up check is
not the same:

- **Method A (`openai-compat`)** — verify that the configured
  `VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME` appears anywhere in the `/v1/models` list using
  the `any(.data[]; .id == $model)` check from the
  [Method A endpoint verification](#method-a--openai-compatible-endpoint) section. Treat
  HTTP 401 as an authentication failure; treat a 200 response where the model is absent
  as a model-name mismatch.
- **Method B (`vllm-compatible`)** — do not compare the advertised ID against
  `VLM_NAME` or `VIA_VLM_OPENAI_MODEL_DEPLOYMENT_NAME`. The resolved model
  directory basename becomes the advertised `/v1/models` ID, so
  `ngc:nim/nvidia/cosmos-reason2-8b:hf-1208` is served as
  `nim_nvidia_cosmos-reason2-8b_hf-1208`. That difference is expected for
  Method B and does not indicate a deployment failure. Point downstream
  consumers (`vlm-as-verifier`, `vss-agent`) at that advertised ID, never at
  the NGC or Hugging Face path.

**Run the per-method verification commands from
[references/health-checks.md](references/health-checks.md)** — covers the
Method A identity check, the Method B inspection, warm-up diagnostics, and
querying a specific external endpoint.
