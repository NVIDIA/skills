# VLM NIM Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile key |
|---|---|
| Dedicated local VLM | `vlm_local_${VLM_NAME_SLUG}` |
| Shared-GPU local VLM | `vlm_local_shared_${VLM_NAME_SLUG}` |

The Search profile preserves the equivalent dynamic form
`vlm_${VLM_MODE}_${VLM_NAME_SLUG}`.

## Required peers

- Requires the selected model slug to exist under `deploy/docker/services/nim/`.
- Requires the corresponding hardware env file.
- Add this owner only for a standalone local VLM. Integrated RT-VLM is a
  different owner and must not also activate a standalone VLM by default.
- Remote mode activates no VLM NIM profile and requires a reachable endpoint.

## Configuration knobs

| Environment variable | Use |
|---|---|
| `VLM_MODE`, `VLM_NAME`, `VLM_NAME_SLUG` | Select remote, dedicated local, or shared local service. |
| `VLM_DEVICE_ID`, `SHARED_LLM_VLM_DEVICE_ID` | Select dedicated or shared GPU. |
| `VLM_PORT`, `VLM_BASE_URL`, `VLM_MODEL_TYPE` | Configure service access. |
| `HARDWARE_PROFILE`, `VLM_ENV_FILE` | Select checked-in sizing defaults and an optional env override. |
| `VLM_NIM_MODEL_NAME`, `VLM_CUSTOM_WEIGHTS`, `NIM_MODEL_SIZE` | Select a supported model variant or weights. |
| `NGC_CLI_API_KEY` | Pull the image/model for local NIM. |
| `NIM_KVCACHE_PERCENT`, `NIM_MAX_MODEL_LEN`, `NIM_MAX_NUM_SEQS` | Common per-deploy sizing overrides when supported by the selected image. |

## Sources

- `deploy/docker/services/nim/compose.yml`
- `deploy/docker/services/nim/*/compose.yml`
- `deploy/docker/services/nim/*/hw-*.env`
- `deploy/docker/developer-profiles/dev-profile-search/overrides.env`
