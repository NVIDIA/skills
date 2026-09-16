# VSS Source Layout — Blueprint and Standalone RT-VLM

This skill is documentation-only. Blueprint configuration and the
customer-accessible RT-VLM source mirror both live in the public VSS
repository; neither tree exists in the DeepStream monorepo.

```
Repository:   https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization
Release/tag:  v3.2.1
```

Throughout the skill, `${VSS_*}` variables refer to the definitions below.

## Obtaining the checkout

Clone the deployment package described by the
[VSS Quickstart](https://docs.nvidia.com/vss/latest/quickstart.html#download-the-deployment-package)
only if a local checkout is not already present, then run blueprint paths from
that repository root:

```bash
if [ ! -d video-search-and-summarization ]; then
  git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git
fi
cd video-search-and-summarization

# Ensure this is VSS v3.2.1 or a compatible release before continuing.
git lfs install
git lfs pull

VSS_ROOT="$PWD"
VSS_DEPLOY_DIR="${VSS_ROOT}/deploy/docker"
VSS_PROFILE_DIR="${VSS_DEPLOY_DIR}/developer-profiles/dev-profile-alerts"
```

## Official VSS v3.2.1 Alerts profile sources

All links are relative to
`https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization` at tag
`v3.2.1`.

| Path | Contents |
|---|---|
| `deploy/docker/developer-profiles/dev-profile-alerts/.env` | `RTVI_VLM_*`, `VLM_MODEL_TYPE`, `VLM_NAME`, `VLM_BASE_URL` |
| `deploy/docker/developer-profiles/dev-profile-alerts/compose.yml` | service wiring / env mapping |
| `deploy/docker/developer-profiles/dev-profile-alerts/vlm-as-verifier/configs/config.yml` | alert-bridge VLM endpoint |
| `deploy/docker/developer-profiles/dev-profile-alerts/vss-agent/configs/config.yml` | agent UI VLM profiles |
| `deploy/docker/services/rtvi/rtvi-vlm/` | stock `rtvi-vlm` service definition extended by the Alerts profile |
| `services/rtvi/rt-vlm/` | public standalone deployment, documentation, and source mirror |

## Blueprint profile tree

```
${VSS_PROFILE_DIR}/
├── .env                                      # existing; RTVI_VLM_* and vss-agent VLM_* vars
├── compose.yml                               # existing; service wiring / env mapping
├── vlm-as-verifier/configs/config.yml        # existing; alert-bridge VLM endpoint
└── vss-agent/configs/config.yml              # existing; agent UI VLM profiles
```

## Public RT-VLM source and standalone deployment

```
services/rtvi/rt-vlm/
├── README.md                                 # deployment, models, and env reference
├── docker/compose.yaml                       # customer-facing standalone compose
├── docker/Dockerfile                         # optional image for public source edits
├── src/vlm_pipeline/ngc_model_downloader.py  # MODEL_PATH prefix conventions
└── src/models/openai_compat/openai_compat_model.py  # OpenAI-compatible client wiring
```

`services/rtvi/rt-vlm/README.md` is the authoritative environment-variable
reference and carries the supported-model guidance for the bundled vLLM
version.

## Pinned images

`rtvi-vlm` runs `nvcr.io/nvidia/vss-core/vss-rt-vlm:${RTVI_VLM_IMAGE_TAG:-3.2.1}`.
Keep the release tag aligned with the checked-out VSS tag; use the image digest
recorded by your own registry or release process if digest pinning is required.

`alert-bridge` and `vss-agent` are closed-source VSS services, pinned in the
v3.2.1 stack as:

- `nvcr.io/nvidia/vss-core/vss-alert-verification:3.2.0`, from
  `deploy/docker/services/alert/compose.yml`
- `nvcr.io/nvidia/vss-core/vss-agent:3.2.1`, from
  `deploy/docker/services/agent/compose.yml` via `VSS_AGENT_VERSION` in the
  profile `.env`
