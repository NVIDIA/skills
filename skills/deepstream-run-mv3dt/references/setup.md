# Setup And Readiness

Load this reference when the user asks to install, prepare, or validate MV3DT DeepStream prerequisites, or when another workflow fails due missing setup. Teardown has its own reference in the skill routing table.

## What Setup Covers

The repo script is the source of truth. It checks the OS, NVIDIA driver, Docker GPU runtime, pulls the DeepStream container, extracts `assets/datasets.zip`, downloads models, builds custom parsers, installs or starts Mosquitto and Kafka, creates the Kafka topic `mv3dt`, and creates `mv3dt_venv`.

Default setup is DeepStream Container only. If the user asks for Inference Builder, consult the app checkout’s `docs/step-by-step-inference-builder.md` outside this operational workflow.

Before full setup or a first run, summarize that setup may install packages, pull containers, download models, start Mosquitto/Kafka, and write generated files under the repo and the Kafka base directory (`BASE_DIR`, defaulting to the user's home directory). Ask before using `sudo`, including the `DOCKER_CMD="sudo docker"` fallback.

## Commands

Preflight check before setup or run:

```bash
cd "${REPO_ROOT}"
DOCKER_CMD="${DOCKER_CMD:-docker}"

if [ -f assets/datasets.zip ] && head -c 200 assets/datasets.zip | grep -q 'git-lfs.github.com/spec'; then
  echo "ERROR: assets/datasets.zip is a Git LFS pointer. Install git-lfs, run 'git lfs pull', then retry."
  exit 1
fi

test -w . || { echo "ERROR: repo checkout is not writable: ${REPO_ROOT}"; exit 1; }
mkdir -p experiments/deepstream
test -w experiments/deepstream || { echo "ERROR: experiments/deepstream is not writable"; exit 1; }

if ! $DOCKER_CMD ps >/dev/null 2>&1; then
  echo "ERROR: Docker is not available as '$DOCKER_CMD'. If approved, retry with: export DOCKER_CMD='sudo docker'"
  exit 1
fi
$DOCKER_CMD info --format '{{json .Runtimes}}' | grep -q 'nvidia' \
  || $DOCKER_CMD run --help | grep -q -- '--gpus' \
  || { echo "ERROR: Docker GPU runtime was not detected"; exit 1; }

test -d mv3dt_venv && export PATH="${REPO_ROOT}/mv3dt_venv/bin:${PATH}"
python -c 'import kafka, google.protobuf' 2>/dev/null || echo "WARN: mv3dt_venv is missing or incomplete"

KAFKA_DIR="${BASE_DIR:-$HOME}/kafka_2.13-4.2.0"
if [ -x "${KAFKA_DIR}/bin/kafka-topics.sh" ]; then
  "${KAFKA_DIR}/bin/kafka-topics.sh" --bootstrap-server localhost:9092 --list | grep '^mv3dt$' \
    || echo "WARN: Kafka topic mv3dt is not reachable yet"
fi

if ! find models -name '*.engine' -print -quit 2>/dev/null | grep -q .; then
  echo "INFO: TensorRT engine cache not found; first DeepStream run may spend 10+ minutes building engines."
fi
```

Readiness check only:

```bash
cd "${REPO_ROOT}"
./scripts/setup_prerequisites.sh --check-only
```

Full setup:

```bash
cd "${REPO_ROOT}"
./scripts/setup_prerequisites.sh
```

Before full setup, explain that it may use `sudo`, install packages, download models and containers, and start local services. Fail fast on container-image access before starting a long setup:

```bash
DOCKER_CMD="${DOCKER_CMD:-docker}"
DEEPSTREAM_IMAGE="${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}"
$DOCKER_CMD pull "${DEEPSTREAM_IMAGE}"
```

If the pull returns access denied or authentication errors, stop and report the exact image name and error before running setup.

Optional environment values:

```bash
export BASE_DIR=/path/to/install/kafka
# Default DeepStream image for x86 and Jetson platforms:
export DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:9.1-triton-multiarch
# ARM SBSA override, when needed:
# export DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:9.1-triton-arm-sbsa
```

## Focused Checks

Verify required repo paths are writable before commands that create datasets, models, parsers, or experiment outputs. Ask before any permission fix and keep it scoped to generated directories.

Use these checks when setup fails:

```bash
DOCKER_CMD="${DOCKER_CMD:-docker}"
nvidia-smi
$DOCKER_CMD info | grep -i runtimes
$DOCKER_CMD run --rm --gpus all "${DEEPSTREAM_IMAGE:-nvcr.io/nvidia/deepstream:9.1-triton-multiarch}" nvidia-smi
test -d datasets/mtmc_4cam && test -d datasets/mtmc_12cam
test -d mv3dt_venv
./scripts/mosquitto_test.sh
```

Kafka should serve host BEV consumers at `localhost:9092` with topic `mv3dt`. Container clients additionally require the bridge listeners in [Container Networking](container-network.md); configure and verify Kafka and MQTT before generating run configs. Verify the host endpoint after setup:

```bash
KAFKA_DIR="${BASE_DIR:-$HOME}/kafka_2.13-4.2.0"
"${KAFKA_DIR}/bin/kafka-topics.sh" --bootstrap-server localhost:9092 --list | grep '^mv3dt$'
```

If Kafka was started by `nohup` but is no longer listening in the current environment, start it in a persistent foreground session and leave that session running while DeepStream runs:

```bash
cd "${KAFKA_DIR}"
bin/kafka-server-start.sh config/server.properties
```

Then rerun the topic check above before launching DeepStream.


## Success Criteria

- `./scripts/setup_prerequisites.sh --check-only` passes.
- `datasets/mtmc_4cam` and `datasets/mtmc_12cam` exist.
- `mv3dt_venv` exists and imports Kafka/protobuf dependencies.
- Mosquitto is reachable on port 1883.
- Kafka is reachable on port 9092 with topic `mv3dt`.
- Required model files and custom parser libraries exist under `models/`.

## Troubleshooting

| Symptom | First action |
|---|---|
| Docker GPU test fails | Fix NVIDIA Container Toolkit or Docker runtime before launching samples |
| Dataset directories missing | Confirm `assets/datasets.zip` is materialized, not a Git LFS pointer; install `git-lfs` and run `git lfs pull` if needed, then rerun setup |
| Model download fails | Check network access and retry setup |
| Parser build fails | Capture the Docker build output and confirm the DeepStream image can run |
| Kafka topic missing | Restart setup or create `mv3dt` using the Kafka commands in `docs/manual-setup.md` |
