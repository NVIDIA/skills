# Environment Overrides

Write the build's complete deployment decisions to
`_builds/<name>/override.env`. The Foundation `.env` and `overrides.env` supply
defaults during resolution; `override.env` is the final, highest-precedence
layer and must be sufficient to explain the selected service graph.

## Always resolve

Include these non-secret deployment values:

| Variable | Value |
|---|---|
| `FOUNDATION` | Exactly one of `base`, `alerts`, `lvs`, or `search` |
| `COMPOSE_PROFILES` | Full effective set of canonical service profile keys |
| `HARDWARE_PROFILE` | Detected or user-selected hardware label |
| `VSS_APPS_DIR` | Absolute path to `deploy/docker` |
| `VSS_DATA_DIR` | Absolute selected data directory |
| `HOST_IP` | Container-reachable host address |

Also include every customized environment value and every value transitively
derived from it. Examples include model slugs derived from model names, public
URLs derived from `EXTERNAL_IP`, and `RTVI_VLM_ENDPOINT` derived from a remote
VLM base URL. Do not copy unchanged Foundation defaults such as stock
host-published ports, model names, or feature toggles into a delta just because
they appear in the Foundation `.env` or `overrides.env`; those layers are read
during resolution. Never rely on shell state to fill a value that belongs to the
build contract.

Credential variables are mode-scoped. Validate them with
[`credentials.md`](credentials.md) before resolution. Do not commit secrets.

## Placement rules

- Treat endpoint variables found in the host environment as candidates, not
  user intent; they may be leftovers from another deployment.
- Default to local or local-shared placement only when the selected model set
  fits the host according to [`sizing.md`](sizing.md).
- Use a remote endpoint only when the user requested it, supplied it, approved
  it after a sizing blocker, or the edge reference requires a standalone local
  service that VSS consumes through a remote-compatible endpoint.
- If a request says only "use build.nvidia.com" or supplies the aggregate
  NVIDIA API endpoint, ask which of LLM, VLM, or both should be remote and
  require the exact model ID.
- Probe every selected remote endpoint with
  `scripts/probe_remote_models.sh` before adding it to `override.env`.
- Record a selected `VSS_CONTAINER_TAG` here, but export it as well: this layer
  is read after `containers.env` derived every `VSS_*_TAG` from it, so the line
  alone moves only `vss-agent-ui` and leaves the rest of the build on
  `develop-latest`. A service-specific pin such as `VSS_RT_CV_TAG` needs no
  export — being read last is exactly what makes it win.

## Common override sets

| User intent | Required values |
|---|---|
| Remote LLM | `LLM_MODE=remote`, `LLM_NAME_SLUG=none`, `LLM_BASE_URL=<host>`, `LLM_NAME=<model>`, and `NVIDIA_API_KEY` when required |
| Remote VLM through RT-VLM | `VLM_MODE=remote`, `VLM_NAME_SLUG=none`, `VLM_MODEL_TYPE=rtvi`, `VLM_BASE_URL=<host>`, `VLM_NAME=<model>`, `RTVI_VLM_ENDPOINT=<host>/v1`, `RTVI_VLM_MODEL_TO_USE=openai-compat`, `RTVI_VLM_MODEL_PATH=none`, and `NVIDIA_API_KEY` when required |
| Dedicated local models | `LLM_MODE=local`, `VLM_MODE=local`, and explicit LLM/RT-VLM device IDs |
| Shared local models | `LLM_MODE=local_shared`, `VLM_MODE=local_shared`, shared device IDs, and explicit utilization limits |
| Different model | Model name, matching slug, model type, serving path, and any derived endpoint or artifact path |

`LLM_BASE_URL` and `VLM_BASE_URL` must not end in `/v1`; the agent appends it.
`RTVI_VLM_ENDPOINT` must end in `/v1` because RT-VLM consumes it verbatim.

## Remote endpoint gate

Before writing a remote model:

1. Obtain the endpoint base URL, exact served model ID, and required key.
2. Strip a trailing `/v1` from the agent-facing base URL.
3. Probe `<base-url>/v1/models` and require the selected model ID to appear.
4. Add the complete row from the table above to `override.env`.
5. Resolve Compose and verify the final URLs, service presence or absence,
   device IDs, and model names in `resolved.yml`.

If remote placement was selected, do not leave a local model profile key in
`COMPOSE_PROFILES` unless the effective architecture deliberately keeps a
local proxy such as `rtvi-vlm`.

## Service-set rule

Unlike the legacy profile helper, this skill writes `COMPOSE_PROFILES`
directly. Start from the Foundation's exact checked-in set, apply only the
requested capability delta, and write the resulting full set to
`override.env`. Never add the build directory name as a profile key.
