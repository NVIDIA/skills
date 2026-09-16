# VSS source location

This skill is documentation-only and does not ship the VSS deployment sources. If there is no local VSS repo, clone the deployment package described by the [VSS Quickstart](https://docs.nvidia.com/vss/latest/quickstart.html#download-the-deployment-package). After cloning or finding the video-search-and-summarization, run all remaining paths and commands from that repository root:

```bash
# Clone only if a local checkout is not already present.
if [ ! -d video-search-and-summarization ]; then
  git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git
  cd video-search-and-summarization
  git checkout v3.2.1
else
  cd video-search-and-summarization
  # Confirm this tree is VSS v3.2.1 or a compatible release before continuing.
fi
git lfs install
git lfs pull

VSS_ROOT="$PWD"
VSS_DEPLOY_DIR="${VSS_ROOT}/deploy/docker"
VSS_APPS_DIR="${VSS_DEPLOY_DIR}"
VSS_PROFILE_DIR="${VSS_DEPLOY_DIR}/developer-profiles/dev-profile-alerts"
```

The official VSS v3.2.1 sources are under [`deploy/docker/developer-profiles/dev-profile-alerts/`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/tree/v3.2.1/deploy/docker/developer-profiles/dev-profile-alerts). The stock release contains:

- [`.env`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/v3.2.1/deploy/docker/developer-profiles/dev-profile-alerts/.env)
- [`compose.yml`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/v3.2.1/deploy/docker/developer-profiles/dev-profile-alerts/compose.yml)
- [`deepstream/configs/`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/tree/v3.2.1/deploy/docker/developer-profiles/dev-profile-alerts/deepstream/configs)
- [`deepstream/init-scripts/ds-start.sh`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/v3.2.1/deploy/docker/developer-profiles/dev-profile-alerts/deepstream/init-scripts/ds-start.sh) — Alerts profile copy; **not** mounted by stock compose
- [`services/rtvi/rtvi-cv/ds-start.sh`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/v3.2.1/deploy/docker/services/rtvi/rtvi-cv/ds-start.sh) — unified entrypoint that stock `perception-alerts` actually runs
- [`services/rtvi/rtvi-cv/compose.yaml`](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/v3.2.1/deploy/docker/services/rtvi/rtvi-cv/compose.yaml), which defines the stock `perception` service extended by the Alerts profile

See [ds-start-entrypoint.md](ds-start-entrypoint.md) before editing any `ds-start.sh`.

The YOLOv11-specific files below are customization targets described by this skill. They are **not present in the stock v3.2.1 VSS checkout**; create them at these locations while modifying the existing files noted above:

```
${VSS_PROFILE_DIR}/
├── Dockerfiles/perception.Dockerfile         # create; extends vss-rt-cv and compiles parser .so
├── deepstream/
│   ├── configs/yolov11.txt                   # create; nvinfer config
│   ├── configs/yolo-coco-labels.txt          # create; 80-class COCO labels
│   ├── custom_parser/nvdsparseyolov11.cpp    # create; custom bbox parser
│   └── init-scripts/ds-start.sh              # existing; edit + remount (stock runs rtvi-cv/ds-start.sh)
├── compose.yml                               # existing; add build, mounts, and environment
└── .env                                      # existing; set MODEL_NAME_2D and NUM_SENSORS
```
