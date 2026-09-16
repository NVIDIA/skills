# CAD to SimReady Preflight

## When to Use

Use this reference before an `omniverse-cad-to-simready` workflow when the host
should have deterministic local dependencies instead of each downstream
reference discovering upstream checkouts independently. It prepares local
upstream checkouts, validates runtime entrypoints, optionally verifies or
deploys Content Agents, and writes a manifest that downstream references can
consume.

This reference is a setup and readiness contract. It is not a monolithic
CAD-to-SimReady workflow runner and it does not run conversion, property
assignment, conformance, validation, rendering, or packaging on an asset.

## Prerequisites

- Python 3.12.
- `uv` when a repository `pyproject.toml` is available and the project Python
  environment should be synchronized.
- `git`, and `git-lfs` when LFS fixtures or source assets must be materialized.
- Network and repository access for the upstream sources listed below.
- Docker, Docker Compose v2, NVIDIA Container Toolkit, an NVIDIA driver, an
  NVIDIA GPU, and at least one configured Content Agents model provider key
  when managed local Content Agents deployment is requested. Supported provider
  keys include `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GOOGLE_API_KEY`, and `GEMINI_API_KEY`.

Windows hosts can use the same Python preflight script and PowerShell wrapper
for checkout and Python-runtime preparation. Managed Content Agents deployment
requires a Linux Docker/GPU host; on Windows use WSL2/Linux Docker or provide
healthy service endpoints.

## Upstream Sources

The preflight installs or verifies local checkouts under
`${OMNIVERSE_CAD_TO_SIMREADY_UPSTREAM_ROOT:-$HOME/.omniverse-cad-to-simready/upstreams}`
unless a per-upstream override is set.

| Area | Upstream | Pinned ref | Default checkout / override |
|---|---|---|---|
| CAD conversion guidance | `https://github.com/NVIDIA-Omniverse/usd-convert-cad` | `v0.2.0` | `usd-convert-cad`, `USD_CONVERT_CAD_ROOT` |
| Gaussian splat conversion | `https://github.com/NVIDIA-Omniverse/usd-convert-gsplat` | `v0.1.15` | `usd-convert-gsplat`, `USD_CONVERT_GSPLAT_ROOT` |
| SimReady validation and FET skills | `https://github.com/NVIDIA/simready-foundation` | `v2026.04.1` | `simready-foundation`, `SIMREADY_FOUNDATION_ROOT` |
| Content Agents services | `https://github.com/nvidia-omniverse/content-agents` | `v0.5.2` | `content-agents`, `CONTENT_AGENTS_UPSTREAM_ROOT` |

The upstream URLs remain documented because they are the source of truth for
external NVIDIA technology. Operationally, downstream references should prefer
the preflight manifest when it is present.

`upstream-versions.lock.json` is the single tested-version manifest. It pins
each Git checkout to both a human-readable tag and an immutable commit, and it
pins the CAD and SimReady Python runtimes to exact direct and transitive package
versions. Preflight checks out managed clones at those commits and never pulls
`main`. Per-upstream overrides are not mutated; they are accepted only when
their current commit matches the manifest. A mismatch blocks with a clear
`requires <ref>, found <commit>` message. `--check-only` and `--no-update` also
verify the pin without changing the checkout.

## CLI Pattern

Linux/macOS:

```bash
.agents/skills/omniverse-cad-to-simready/references/preflight/scripts/preflight.sh \
  --env-file "$HOME/.omniverse-cad-to-simready/state/cad-to-simready-preflight.env" \
  --markdown-report "$HOME/.omniverse-cad-to-simready/state/cad-to-simready-preflight.md"

. "$HOME/.omniverse-cad-to-simready/state/cad-to-simready-preflight.env"
```

Windows PowerShell:

```powershell
.\.agents\skills\omniverse-cad-to-simready\references\preflight\scripts\preflight.ps1 `
  --powershell-env-file "$HOME\.omniverse-cad-to-simready\state\cad-to-simready-preflight.ps1" `
  --markdown-report "$HOME\.omniverse-cad-to-simready\state\cad-to-simready-preflight.md"

. "$HOME\.omniverse-cad-to-simready\state\cad-to-simready-preflight.ps1"
```

Dependency bootstrap without Content Agents service deployment:

```bash
python3 .agents/skills/omniverse-cad-to-simready/references/preflight/scripts/preflight.py \
  --skip-content-agents \
  --env-file "$HOME/.omniverse-cad-to-simready/state/cad-to-simready-preflight.env"
```

Managed Content Agents deployment can read OpenAI, Anthropic, or Google/Gemini
credentials from a private dotenv file without writing secrets to the manifest:

```bash
python3 .agents/skills/omniverse-cad-to-simready/references/preflight/scripts/preflight.py \
  --content-agents-secret-env-file "$HOME/Codes/.env" \
  --content-agents-vlm-backend openai \
  --content-agents-vlm-model gpt-5.5 \
  --content-agents-llm-backend openai \
  --content-agents-llm-model gpt-5.5 \
  --env-file "$HOME/.omniverse-cad-to-simready/state/cad-to-simready-preflight.env"
```

Use `--content-agents-vlm-backend anthropic` with a Claude VLM model, or
`--content-agents-vlm-backend gemini` with a Gemini VLM model, when those keys
are present in the private dotenv. For a custom OpenAI-compatible `/v1` VLM
endpoint, use `--content-agents-vlm-backend openai` and pass
`--content-agents-vlm-endpoint`; reserve `nim` for actual NIM endpoints.

Read-only readiness check:

```bash
python3 .agents/skills/omniverse-cad-to-simready/references/preflight/scripts/preflight.py \
  --check-only \
  --skip-deploy
```

## Manifest Contract

The default manifest path is:

```text
${OMNIVERSE_CAD_TO_SIMREADY_STATE:-$HOME/.omniverse-cad-to-simready/state}/cad-to-simready-preflight.json
```

Set `PHYSICAL_AI_PREFLIGHT_MANIFEST` to point downstream references at a
specific manifest. Set `PHYSICAL_AI_REQUIRE_PREFLIGHT=1` to make downstream
references block instead of falling back to legacy direct discovery when the
manifest is missing or the required component is not ready.

The generated env file exports:

- `PHYSICAL_AI_PREFLIGHT_MANIFEST`
- `PHYSICAL_AI_REQUIRE_PREFLIGHT=1`
- `OMNIVERSE_CAD_TO_SIMREADY_HOME`
- `OMNIVERSE_CAD_TO_SIMREADY_UPSTREAM_ROOT`
- `PATH` with the repository `.venv/bin` prepended when the project virtual
  environment is present, so direct reference scripts can find bundled CLIs
  such as `urdf_usd_converter`
- per-upstream root variables such as `USD_CONVERT_CAD_ROOT`
- prepared runtime variables such as `PHYSICAL_AI_SIMREADY_VALIDATE_VENV`
- ready service endpoints such as `CONTENT_AGENTS_MATERIAL_AGENT_BASE_URL`,
  `CONTENT_AGENTS_PHYSICS_AGENT_BASE_URL`, and `RENDER_ENDPOINT`

Use `--env-file` for POSIX shells and `--powershell-env-file` for PowerShell.

The manifest never writes API keys, bearer tokens, or file-backed secret
contents. Command output is redacted before it is included in the report.
It records `upstream_version_policy: pinned-tested-integration` and the path to
the version manifest so QA evidence identifies the exact dependency set.

## Content Agents Policy

Content Agents readiness is included by default. Preflight treats explicitly
provided `CONTENT_AGENTS_*_BASE_URL` and renderer endpoints as user-owned: it
probes them and blocks when they are unhealthy, but does not replace them with
managed containers. For services without explicit endpoints, preflight deploys
the managed local topology in deterministic order: standalone OVRTX first when
Material or Physics will be deployed, then Material, Physics, and optional
Texture one service at a time through the upstream collection deployment helper.
Material, Physics, and Texture endpoints must also report configured API keys
when their health payload includes `api_keys_configured`; a healthy container
without service credentials is not workflow-ready, except for the provided
already-running endpoints described under the render-key mirror below. Before
deploying, preflight checks Docker/GPU/auth prerequisites with `nvidia-smi`, the
Docker daemon, Docker Compose v2, and a configured provider credential. Use
`--content-agents-secret-env-file` to load credentials such as `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `GEMINI_API_KEY` from a private dotenv
file for deployment only. Use `--content-agents-vlm-*` and
`--content-agents-llm-*` to select hosted (`openai`, `anthropic`, `gemini`),
custom OpenAI-compatible (`openai` plus endpoint), or NIM (`nim` plus endpoint)
VLM/LLM backends and model IDs. Generated collection configs use the selected
backend to choose the default secret environment variable name:
`OPENAI_API_KEY` for `openai`, `ANTHROPIC_API_KEY` for `anthropic`,
and `GOOGLE_API_KEY` or `GEMINI_API_KEY` for `gemini`. Use
`--content-agents-vlm-api-key-env` and `--content-agents-llm-api-key-env` when
the endpoint or provider should read a custom secret name. NIM endpoints remain
NIM-specific and do not use these generic `*_API_KEY_ENV` settings.
For GPT-style OpenAI models, generated deployment config pins
VLM/LLM temperatures to `1.0`, matching models that only accept the default
temperature. The upstream agent collection uses the OVRTX HTTP endpoint through
`RENDER_ENDPOINT`; upstream logs may call this the `remote` renderer because the
renderer is an HTTP service, not because NVCF was selected.
For managed local deployment, known Content Agents credential environment
variables for the selected backend are mirrored into the upstream checkout's
private `.env` file with owner-only permissions so Docker Compose can pass them
to containers. Those values are not written to the preflight manifest or
generated downstream env file. When `NGC_API_KEY` is absent and the render
endpoint is local, the managed local deployment mirrors the first configured
provider credential -- `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`, `GEMINI_API_KEY`, or another known Content Agents provider
name -- into `NGC_API_KEY` inside the upstream `.env`. Upstream Material and
Physics readiness demands a render usage key for any render host it does not
recognize as local, and a host-local renderer is reached from containers through
the `host.docker.internal` Docker host alias, so that mirror keeps a local
renderer from being treated as remote. `localhost`, `*.localhost`, loopback
addresses, unspecified addresses such as `0.0.0.0`, and `host.docker.internal`
all count as local. When the render endpoint is remote, no provider credential
is mirrored; that endpoint needs a real render usage key in `NGC_API_KEY` or
`NVCF_API_KEY`, and preflight names that requirement when an agent reports
`api_keys_configured: false`.

Preflight cannot mirror that credential into an agent it did not deploy. So for
an explicitly provided Material, Physics, or Texture endpoint that is already
running, `api_keys_configured: false` is accepted as ready when the configured
render endpoint is on this machine under a name upstream readiness does not
recognize as local, such as the `host.docker.internal` Docker host alias. The
upstream allowlist gap is enough to explain that combination, and the renderer
still serves over HTTP. The unmodified health payload stays in the report. This
applies only to that gap: a provided endpoint reporting
`api_keys_configured: false` against a renderer upstream already accepts, such
as `localhost` or a loopback address, still blocks, because only a VLM or LLM
credential can explain it there. A remote render endpoint also still blocks and
still needs a real render usage key.

This acceptance is deliberately a false-accept risk, not a proof. Upstream
`has_required_api_keys` returns `vlm_ready and render_ready`, so a failing render
gate masks any VLM or LLM credential gap present at the same time, and `/health`
exposes only the aggregate flag. Preflight cannot tell the two apart, so a
provided agent behind the Docker host alias that is *also* missing a real VLM or
LLM credential now passes this gate and fails later at call time, with the
service error surfaced then. Physics narrows further: it evaluates the render
gate only while `PA_RENDER_BACKEND` is `remote` (its default), so if that is set
to anything else, `api_keys_configured: false` from a Physics agent is always a
credential gap and never the allowlist.

For remote or NVCF-style endpoints that a user explicitly provides, preflight
records the provided endpoint as ready without treating generic unauthenticated
`/health` failures as blockers. Local OVRTX endpoints are probed again by the
selected service wrapper immediately before Material or Physics Agent sessions.

Do not encode service-specific Docker Compose files, image names, ports inside
containers, or deployment runbooks in this repo. If the selected upstream
checkout exposes only documentation-driven deployment skills, preflight reports
Content Agents as blocked and points the user back to the upstream deployment
skills or to provided healthy endpoints.

Use `--skip-content-agents` only when Content Agents are explicitly out of
scope, such as conversion-only, validation-only, or no material/physics
assignment. Use `--skip-deploy` when endpoints should be verified but services
must not be started.

Preflight can reduce dependency checks to the requested workflow target and
source route. Use `--targets conversion`, `--targets validation`, or
`--targets conversion,validation,content-agents` to choose workflow areas. Use
`--source-asset /path/to/input.urdf` or `--source-format urdf` to infer the
conversion route, `--output-root /path/to/output` to verify the output directory
is writable or creatable, or pass `--conversion-tools
repo-python,usd-convert-cad` for an explicit converter set. URDF and MuJoCo/MJCF
routes require only the repo Python conversion tools; CAD and mesh routes
require `usd-convert-cad`; Gaussian splat routes require `usd-convert-gsplat`.
Validation targets also gate OpenUSD Python APIs (`pxr.Usd`, `pxr.UsdGeom`, and
`pxr.UsdPhysics`) and the upstream Asset Validator runtime
(`omni_asset_validate` or `omni.asset_validator`) before SimReady validation.
On every platform, preflight builds the SimReady validation venv from the exact
package set in `upstream-versions.lock.json`, using `usd-exchange==2.3.0` as the
OpenUSD runtime and installing `simready-validate==2026.4.8` without its
unavailable-on-aarch64 `usd-core` dependency. It then rereads all installed
distribution versions and blocks if any direct or transitive pin differs.

If `uv` is missing, the `repo_python` runtime entry includes an install hint:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

## Output Format

The JSON report includes:

- overall `status`: `ready` or `blocked`
- selected `targets`
- selected conversion tools and route-selection reason
- request input readiness for the source asset and output root
- normalized paths for home, state, upstream, venv, project, and output roots
- upstream checkout path, URL, branch, commit, and status
- runtime readiness for repo Python, Git LFS, converters, OpenUSD Python APIs,
  Asset Validator, SimReady validation, and Content Agents
- Content Agents local deployment host diagnostics for `nvidia-smi`, Docker
  daemon access, and Docker Compose v2 when local deployment may be needed
- service readiness for OVRTX, Material, Physics, and optional Texture
- non-secret downstream environment exports
- command steps with redacted output tails
- blocker messages

The Markdown report summarizes the same status for humans.

## Pass/Fail Policy

Return success only when every selected target is ready or explicitly skipped.
Report blocked when a selected runtime, checkout, CLI, service endpoint, or
deployment prerequisite is missing. Do not scan broad developer workspaces or
reuse arbitrary old clones.

## Next Steps

After preflight succeeds, source the generated env file, then run the normal
atomic references in the `omniverse-cad-to-simready` workflow. Downstream
references will consume the manifest and prepared local paths/endpoints before
trying direct legacy discovery.
