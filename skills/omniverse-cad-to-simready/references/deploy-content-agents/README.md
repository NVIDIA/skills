# Deploy Content Agents

## When to Use

Use this reference when the user asks to deploy, start, configure, or troubleshoot NVIDIA Omniverse Content Agents services. This repo does not ship a deployment runner or duplicate service runbooks; it selects the target, resolves the upstream checkout, enforces repo-local readiness policy, and hands off to the upstream deployment skills.

This reference is documentation-driven, does not ship `scripts/run.py`, and should not depend on this repository checkout.

## Prerequisites

- A provider API key for the selected Content Agents VLM/LLM backend. Use
  `NVIDIA_API_KEY` from `https://build.nvidia.com` for NVIDIA-hosted defaults;
  other hosted provider keys include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GOOGLE_API_KEY`, and `GEMINI_API_KEY`.
- Docker, Docker Compose v2, NVIDIA Container Toolkit, an NVIDIA driver, and an NVIDIA GPU on the deployment host.
- A normalized upstream checkout of `https://github.com/nvidia-omniverse/content-agents` at the exact ref in `upstream-versions.lock.json` (currently `v0.5.2`).

## Upstream Reference

Use the NVIDIA Omniverse Content Agents deployment skills at the tested manifest pin as the source of truth:

| Target | Upstream skill |
|---|---|
| Material Agent | `https://github.com/nvidia-omniverse/content-agents/blob/v0.5.2/.codex/skills/deploy-material-agent-docker/SKILL.md` |
| Physics Agent | `https://github.com/nvidia-omniverse/content-agents/blob/v0.5.2/.codex/skills/deploy-physics-agent-docker/SKILL.md` |
| Texture Agent | `https://github.com/nvidia-omniverse/content-agents/blob/v0.5.2/.codex/skills/deploy-texture-agent-docker/SKILL.md` |
| OVRTX renderer | `https://github.com/nvidia-omniverse/content-agents/blob/v0.5.2/.codex/skills/deploy-ovrtx-docker/SKILL.md` |

Repository: `https://github.com/nvidia-omniverse/content-agents`, ref `v0.5.2`

If browser or raw-file fetches are blocked, use a local clone checked out to the exact manifest ref and read `.codex/skills/<skill-name>/SKILL.md` from that checkout. Resolve it from `CONTENT_AGENTS_UPSTREAM_ROOT`, then `$OMNIVERSE_CAD_TO_SIMREADY_UPSTREAM_ROOT/content-agents`, then `$HOME/.omniverse-cad-to-simready/upstreams/content-agents`. Do not scan broad developer workspaces.

Do not duplicate full upstream deployment runbooks here.

## Inputs

Collect:

| Input | Requirement |
|---|---|
| `target_service` | `material`, `physics`, `texture`, or `ovrtx`. |
| `content_agents_root` | Optional explicit upstream checkout path. |
| `deployment_mode` | Local container deployment or reuse of healthy existing services. |
| `render_endpoint` | Required for render-dependent agent services. |
| `env_keys` | Provider/API keys required by the upstream skill. Never print or commit secrets. |
| `vlm_backend` / `vlm_model` | Optional deployment-time VLM selection, for example `openai` / `gpt-5.5`. |
| `llm_backend` / `llm_model` | Optional deployment-time LLM selection, for example `openai` / `gpt-5.5`. |

## Instructions

1. Resolve the upstream checkout from the normalized root policy.
2. Read the matching upstream deployment skill before issuing service-specific commands.
3. Confirm required secrets are available from local shell state or a private `.env`; if missing, ask the user and wait.
4. Treat explicit endpoints as user-owned. Probe them and block if they are unhealthy; do not deploy a replacement container for that service.
5. For managed local deployment, deploy missing services in deterministic order: standalone OVRTX when Material or Physics will be deployed, then Material, Physics, and optional Texture one service at a time through the upstream collection deployment helper.
6. Prefer a shared standalone OVRTX renderer plus independently deployed agent services for this workflow. Use bundled agent-specific renderer stacks only when the user asks for isolation or the upstream skill requires it.
7. Preserve upstream secret handling, build steps, image names, health checks, and service-specific environment details.
8. Verify the renderer and selected agent health endpoints before exporting `CONTENT_AGENTS_*_BASE_URL` or `RENDER_ENDPOINT`. For local deployments, `RENDER_ENDPOINT` should point the agent containers at the standalone OVRTX HTTP service; upstream may describe this as the `remote` renderer backend because it is reached over HTTP.
9. Return to the caller only after the requested service is healthy, or report the missing prerequisite as blocked.

## Headless / Nested Host Notes

Use this section only as deployment-readiness guidance for nested or headless
hosts. The upstream `content-agents` skills still own the actual Docker
Compose files, commands, image names, ports inside containers, and
service-specific environment variables.

### Single GPU + Cloud VLM

For local evaluation, prefer one shared standalone OVRTX renderer on the local
GPU plus independent Material, Physics, and Texture Agent service containers
that use a hosted provider key such as `NVIDIA_API_KEY`, `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY` for provider-backed
VLM/LLM calls. This topology does not require a separate local VLM GPU.
Verify the OVRTX host endpoint first, then verify each agent service
reports configured API keys before exporting `CONTENT_AGENTS_*_BASE_URL`.
For managed local deployment, treat OVRTX as part of the Content Agents
deployment contract when Material or Physics will be deployed: start it first,
verify render health, then deploy each missing agent service individually.

For hosted providers, use the upstream collection deployment path with
non-secret model overrides and keep keys in a private `.env`. Supported hosted
backends include `openai`, `anthropic`, and `gemini`. GPT-style OpenAI model
deployments should pin VLM/LLM temperatures to `1.0`, because some GPT models
only accept the default temperature:

```bash
python3 skills/omniverse-cad-to-simready/references/preflight/scripts/preflight.py \
  --content-agents-secret-env-file "$HOME/Codes/.env" \
  --content-agents-vlm-backend openai \
  --content-agents-vlm-model gpt-5.5 \
  --content-agents-llm-backend openai \
  --content-agents-llm-model gpt-5.5
```

For a custom OpenAI-compatible `/v1` VLM endpoint, set
`--content-agents-vlm-backend openai`, then provide
`--content-agents-vlm-endpoint` with the model ID served by that endpoint.
Use `--content-agents-vlm-api-key-env` when that endpoint should read a custom
secret name instead of the default `OPENAI_API_KEY`. Anthropic and Gemini use
their matching upstream key environment names.
Reserve `nim` for actual NVIDIA NIM endpoints.

- Confirm Docker Compose v2 is installed before following the upstream
  deployment skills; the legacy `docker-compose` binary is not sufficient when
  upstream expects `docker compose`.
- In nested Docker environments, overlay storage can fail with `invalid
  argument`. If Docker cannot start containers, check whether the host needs a
  `vfs` storage-driver fallback before retrying the upstream deployment.
- Avoid changing Docker daemon settings in a shared or user-provided SSH
  session until the user approves that risk; prefer a fresh disposable session
  for deployment experiments.
- Treat the Xvfb display number as a configurable deployment input. If OVRTX
  exits during startup and the logs show a display conflict, choose an unused
  display such as `:100` instead of assuming the default display is free.
- Prefer distinct host endpoints for the shared renderer and each independently
  deployed agent service. A proven local layout is OVRTX on `8001`, Material
  Agent on `8100`, Physics Agent on `8200`, and Texture Agent on `8300` when
  texture generation is needed; export the corresponding host URLs only after
  those endpoints respond healthy.
- A container-internal OVRTX healthcheck can be a false negative when the
  externally mapped host `/health` endpoint is healthy. Record both the
  container health result and the host endpoint result before declaring the
  renderer blocked.

## Blocked Deployment Policy

If deployment authentication is missing, ask the user for the matching
provider key and wait; if deployment cannot produce healthy services, report
Content Agents readiness as blocked instead of proceeding. See
`../preflight/README.md` and this reference for provider-key, dotenv, and
custom-endpoint detail.

## Handoff Map

| Target | After deployment |
|---|---|
| Material Agent | Set `CONTENT_AGENTS_MATERIAL_AGENT_BASE_URL`, then use `material-agent-client`. |
| Physics Agent | Set `CONTENT_AGENTS_PHYSICS_AGENT_BASE_URL`, then use `physics-agent-client`. |
| Texture Agent | Set `CONTENT_AGENTS_TEXTURE_AGENT_BASE_URL`, then use `texture-agent-client`. |
| OVRTX | Set `RENDER_ENDPOINT` or `OVRTX_RENDER_ENDPOINT` for render clients. |

## Limitations

- This reference selects targets and readiness gates; upstream deployment skills own commands and service internals.
- It does not call already-running services for asset enrichment.
- It does not publish Docker Compose, `docker run`, port, image-name, or in-container environment recipes. Read those from upstream.

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Upstream skill URL cannot be fetched | Use the local `content-agents` clone checked out to the exact ref in `upstream-versions.lock.json`. |
| Required API key is missing | Ask the user for the provider key matching the selected backend, such as `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY`. Usage tokens for already-running endpoints belong to the client references, not this deployment reference. |
| Service health is not ready | Follow the selected upstream deployment skill's health-check section. |
| Renderer-dependent agent cannot reach OVRTX | Use the upstream renderer and agent deployment skills together; do not patch this repo with local Docker recipes. |
| Nested host cannot start Docker containers | Check Docker Compose v2, GPU visibility, NVIDIA Container Toolkit, and whether the host requires the `vfs` storage-driver fallback. |
| OVRTX exits in a headless session | Check Xvfb logs and retry the upstream deployment with an unused display value instead of assuming `:99` is available. |
| Container health is unhealthy but the mapped host endpoint responds | Treat this as a healthcheck mismatch until the upstream deployment skill confirms the container-internal and host-mapped ports. Use the host endpoint only after `/health` reports renderer readiness. |

## Pass/Fail Policy

Report blocked rather than guessing when:

- the upstream checkout is inaccessible
- Docker/GPU/container prerequisites fail
- required credentials are missing
- the selected upstream deployment skill does not support the requested mode
- health checks do not pass

## Next Steps

After a service is healthy, return to `content-agents` or the selected service wrapper reference for the asset workflow.
