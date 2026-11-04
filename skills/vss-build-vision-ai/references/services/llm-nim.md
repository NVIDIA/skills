# LLM NIM Capability Owner

## Capabilities and service keys

| Capability | Canonical service profile key |
|---|---|
| Dedicated local LLM | `llm_local_${LLM_NAME_SLUG}` |
| Shared-GPU local LLM | `llm_local_shared_${LLM_NAME_SLUG}` |

The profile references preserve the equivalent dynamic form
`llm_${LLM_MODE}_${LLM_NAME_SLUG}`.

## Required peers

- Requires the selected model slug to exist under `deploy/docker/services/nim/`.
- Requires the corresponding `hw-${HARDWARE_PROFILE}.env` or
  `hw-${HARDWARE_PROFILE}-shared.env`.
- Shared mode must use a GPU-sharing layout supported by the other selected
  inference owner.
- Remote mode activates no LLM NIM profile and requires a reachable endpoint.

## Configuration knobs

| Environment variable | Use |
|---|---|
| `LLM_MODE`, `LLM_NAME`, `LLM_NAME_SLUG` | Select remote, dedicated local, or shared local service. |
| `LLM_DEVICE_ID`, `SHARED_LLM_VLM_DEVICE_ID` | Select dedicated or shared GPU. |
| `LLM_PORT`, `LLM_BASE_URL`, `LLM_MODEL_TYPE` | Configure service access. |
| `HARDWARE_PROFILE`, `LLM_ENV_FILE` | Select checked-in sizing defaults and an optional env override. |
| `NGC_CLI_API_KEY` | Pull the image/model for local NIM. |
| `NIM_KVCACHE_PERCENT`, `NIM_MAX_MODEL_LEN`, `NIM_MAX_NUM_SEQS` | Common per-deploy sizing overrides when supported by the selected model image. |

## Sources

- `deploy/docker/services/nim/compose.yml`
- `deploy/docker/services/nim/*/compose.yml`
- `deploy/docker/services/nim/*/hw-*.env`
- `deploy/docker/developer-profiles/dev-profile-*/overrides.env`
