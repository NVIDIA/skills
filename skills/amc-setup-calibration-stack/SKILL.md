---
name: "amc-setup-calibration-stack"
description: "Launch AutoMagicCalib microservice and web UI from NGC release images via Docker Compose. Use when user says 'deploy auto calibration', 'launch auto calibration', 'launch AMC', 'start MS+UI', or 'set up auto-magic-calib'. Requires NGC API key."
metadata:
  author: "NVIDIA CORPORATION"
  tags: [amc, deepstream, docker, calibration, setup, ngc]
owner: "NVIDIA CORPORATION"
service: "auto-magic-calib"
version: "1.0.0"
reviewed: "2026-04-28"
license: "Apache-2.0"
---

# Skill: Launch AutoMagicCalib Release Containers

Set up the AutoMagicCalib microservice and UI from release containers: resolve an AMC checkout, authenticate to NGC, optionally download VGGT, configure Docker Compose, launch services, and verify readiness.

## Prerequisites
- Docker and Docker Compose installed
- NVIDIA Docker Runtime configured (for GPU support)
- `auto-magic-calib` repo on disk. Step 0b resolves the current repo, DeepStream `tools/auto-magic-calib`, `DEEPSTREAM_REPO_ROOT`, or `~/auto-magic-calib`; otherwise it asks before cloning `https://github.com/NVIDIA-AI-IOT/auto-magic-calib`.
- NGC account with access to NVIDIA container registry
- Docker runnable without `sudo`; verify with `docker ps` before continuing.

## Instructions

### Step 0: Verify Docker Runs Without sudo

```bash
docker ps
```

- If it succeeds → continue.
- If it fails with "permission denied" → the user is not in the `docker` group. Ask the user to run:
  ```bash
  sudo usermod -aG docker $USER && newgrp docker
  ```
  Then ask the user to confirm `docker ps` works before continuing.

> **Agent note**: If `docker ps` cannot be run from within the agent sandbox, ask the user to confirm it works (e.g. "Can you confirm `docker ps` runs without sudo?") before proceeding.

### Step 0b: Resolve Repo Checkout

The skill needs AMC repo assets (`compose/`, sample data, and `models/`). Resolve an existing checkout first; ask before cloning into `~/auto-magic-calib`.

```bash
REPO_URL="https://github.com/NVIDIA-AI-IOT/auto-magic-calib.git"
DEFAULT_CLONE_DIR="$HOME/auto-magic-calib"
CURRENT_GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"

is_amc_checkout() {
  [ -n "$1" ] \
    && [ -f "$1/README.md" ] \
    && grep -q "AutoMagicCalib" "$1/README.md" 2>/dev/null \
    && [ -f "$1/compose/compose.yml" ] \
    && grep -q "auto-magic-calib-ms" "$1/compose/ms/compose.yml" 2>/dev/null \
    && grep -q "auto-magic-calib-ui" "$1/compose/ui/compose.yml" 2>/dev/null
}

REPO_ROOT=""
for candidate in \
  "$CURRENT_GIT_ROOT" \
  "${CURRENT_GIT_ROOT:+$CURRENT_GIT_ROOT/tools/auto-magic-calib}" \
  "${DEEPSTREAM_REPO_ROOT:+$DEEPSTREAM_REPO_ROOT/tools/auto-magic-calib}" \
  "$PWD/tools/auto-magic-calib" \
  "$DEFAULT_CLONE_DIR"; do
  if is_amc_checkout "$candidate"; then
    REPO_ROOT="$candidate"
    echo "✓ Using auto-magic-calib checkout: $REPO_ROOT"
    break
  fi
done

if [ -z "$REPO_ROOT" ]; then
  if [ -n "$CURRENT_GIT_ROOT" ] && [ -d "$CURRENT_GIT_ROOT/tools/auto-magic-calib" ]; then
    echo "Found $CURRENT_GIT_ROOT/tools/auto-magic-calib, but it is not an initialized AMC checkout."
    echo "If running from the DeepStream repository root:"
    echo "  git submodule update --init tools/auto-magic-calib"
  fi

  # Nothing usable on disk — STOP and ask the user for confirmation using the
  # host's question mechanism; if none is available, ask in chat and wait.
  # Do NOT clone silently from this block or clone over a tracked submodule path.
  echo "No usable auto-magic-calib checkout found. Ask the user for confirmation:"
  echo "  Clone $REPO_URL into $DEFAULT_CLONE_DIR? [y/N]"
  echo "On 'y' — run: git clone \"$REPO_URL\" \"$DEFAULT_CLONE_DIR\""
  exit 1
fi

cd "$REPO_ROOT"
export REPO_ROOT
echo "REPO_ROOT=$REPO_ROOT"
```

> **Agent note**: never clone silently. Prefer initialized DeepStream `tools/auto-magic-calib`; do not clone over that submodule path. If it exists but is empty, ask the user to run `git submodule update --init tools/auto-magic-calib`. Honour an alternate AMC path if provided.

### Step 0c: Install Python venv (New Systems Only)

On a fresh system, `pip` and `python3-venv` may not be available. Install them first:

```bash
# Create a venv for HuggingFace CLI (project-local preferred)
REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
HF_VENV="${REPO_DIR}/venv"
python3 -m venv "$HF_VENV" 2>/dev/null || {
  echo "ERROR: python3-venv not available." >&2
  echo "Install it manually: sudo apt install -y python3-venv python3-pip" >&2
  exit 1
}

# Install HuggingFace hub (needed for VGGT download)
"$HF_VENV/bin/pip" install --upgrade pip huggingface_hub
```

> **Note**: Skip this step if a venv with `hf` already exists (check `venv/bin/hf` in the repo root or `~/venv/amc/bin/hf`).

### Step 1: Login to NGC

Ask the user for their NGC API key using the host's question mechanism; if none is available, ask in chat and wait. Then run:

```bash
echo "<NGC_API_KEY>" | docker login nvcr.io --username '$oauthtoken' --password-stdin
echo "✓ NGC authentication complete"
```

### Step 2: Download VGGT Model (If Not Already Present)

```bash
export REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

if [ -f "models/vggt/vggt_1B_commercial.pt" ]; then
  echo "✓ VGGT model already present"
else
  echo "✗ VGGT model not found"
  echo "Options:"
  echo "  1. Continue without VGGT (AMC only - sufficient for most use cases)"
  echo "  2. Download VGGT model (~4.7GB, requires HuggingFace account)"
fi
```

**To download VGGT**: ask the user to accept the license at https://huggingface.co/facebook/VGGT-1B-Commercial and provide a read token from https://huggingface.co/settings/tokens using the host's question mechanism. Pass it through `HF_TOKEN` so it is not exposed in `ps` output:
```bash
REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_DIR"

# Find the HuggingFace CLI binary (named 'hf', not 'huggingface-cli')
HF_BIN="$(find "$REPO_DIR/venv" ~/venv/amc -name hf -type f 2>/dev/null | head -1)"
{ [ -z "$HF_BIN" ] || [ ! -x "$HF_BIN" ]; } && { echo "ERROR: hf binary not found or not executable; install the hf CLI (Step 0c) or set HF_BIN" >&2; exit 1; }

# Do NOT use --token on the command line (leaks via ps/argv). The HF CLI
# reads HF_TOKEN from the environment automatically.
HF_TOKEN="<HF_TOKEN>" "$HF_BIN" download facebook/VGGT-1B-Commercial \
  --local-dir models/vggt/

# Verify
ls -lh models/vggt/vggt_1B_commercial.pt
# Should show ~4.7GB file
```

> **Important**: Download BEFORE setting `chown 1000:1000` on the models directory — the current user needs write access during download. Set permissions in Step 4 after download completes.

### Step 3: Configure Compose Environment Variables

The Compose environment file controls ports and paths. Update it before launching:

```bash
cd $REPO_ROOT/compose

# Find available backend port (8000-8009)
for port in {8000..8009}; do
  if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
    MS_PORT=$port
    echo "Using backend port: $MS_PORT"
    break
  fi
done
[ -z "$MS_PORT" ] && { echo "ERROR: no free backend port in 8000-8009; free one or widen the range." >&2; exit 1; }

# Find available UI port (5000-5009)
for port in {5000..5009}; do
  if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
    UI_PORT=$port
    echo "Using UI port: $UI_PORT"
    break
  fi
done
[ -z "$UI_PORT" ] && { echo "ERROR: no free UI port in 5000-5009; free one or widen the range." >&2; exit 1; }

# Get host IP
HOST_IP=$(hostname -I | awk '{print $1}')
echo "Host IP: $HOST_IP"

# Preserve existing keys and restrict permissions on the Compose environment file.
COMPOSE_ENV_BASENAME="env"
ENV_FILE=".${COMPOSE_ENV_BASENAME}"
if [ -f "$ENV_FILE" ]; then
  BACKUP="${ENV_FILE}.bak.$(date +%s)"
  cp "$ENV_FILE" "$BACKUP"
  chmod 600 "$BACKUP"
fi
touch "$ENV_FILE"
chmod 600 "$ENV_FILE"
set_env_key() {
  local k="$1" v="$2"
  if grep -qE "^${k}=" "$ENV_FILE"; then
    sed -i "s|^${k}=.*|${k}=${v}|" "$ENV_FILE"
  else
    echo "${k}=${v}" >> "$ENV_FILE"
  fi
}
set_env_key AUTO_MAGIC_CALIB_MS_PORT "${MS_PORT}"
set_env_key AUTO_MAGIC_CALIB_UI_PORT "${UI_PORT}"
set_env_key PROJECT_DIR "../../projects"
set_env_key MODEL_DIR "../../models"
set_env_key HOST_IP "${HOST_IP}"

# Keep timestamped Compose environment backups out of git.
GITIGNORE="$REPO_ROOT/.gitignore"
touch "$GITIGNORE"
BACKUP_PATTERN="compose/${ENV_FILE}.bak.*"
grep -qxF "$BACKUP_PATTERN" "$GITIGNORE" || echo "$BACKUP_PATTERN" >> "$GITIGNORE"

echo "✓ Compose environment file updated"
cat "$ENV_FILE"
```

**Important**: `HOST_IP` must be the machine's network IP (not `localhost`) so the UI container can reach the backend from a browser.

Optional: set `VGGT_MODEL_PATH` only if the VGGT model is mounted at a non-default container path; default is `/tmp/vggt_model/vggt_1B_commercial.pt` inside the MS container.

Optional for RTSP calibration: use `skills/amc-run-rtsp-calibration/SKILL.md` after launch. That skill verifies VIOS reachability and, when needed, relaunches the microservice with a temporary compose override that exports `VIOS_BASE_URL` without changing checked-in compose files.

### Step 4: Set Directory Permissions

The containers run as UID/GID 1000. The `projects` and `models` directories must be owned by this UID for containers to read/write properly:

```bash
cd "$REPO_ROOT"

# Create projects directory if it doesn't exist
mkdir -p projects

# Set ownership (required for containers to write calibration outputs).
# Do this AFTER VGGT download is complete (current user needs write access during download).
# Get explicit user confirmation before running sudo chown — it recursively changes
# ownership of $REPO_ROOT/projects and $REPO_ROOT/models to UID/GID 1000.
[ -d projects ] && [ -d models ] || {
  echo "ERROR: expected projects/ and models/ under $REPO_ROOT" >&2; exit 1;
}
echo "About to chown -R 1000:1000 on:"
echo "  $REPO_ROOT/projects"
echo "  $REPO_ROOT/models"
echo "(required because containers run as UID 1000). Confirm before proceeding."
sudo chown 1000:1000 -R projects
sudo chown 1000:1000 -R models

echo "✓ Permissions set"
```

### Step 5: Launch Services

Before pulling, fail fast if the NGC key authenticated in Step 1 but cannot actually access a release image — otherwise `docker compose up` aborts partway with a 401/403 after some work is already done.

```bash
cd $REPO_ROOT/compose

# Fail-fast image-access check: confirm the NGC key can reach every release
# image BEFORE pulling. `docker manifest inspect` checks registry access without
# downloading layers, and the image list is read from the resolved compose so it
# tracks the release tag automatically.
IMAGES=$(docker compose config --images | sort -u)
[ -z "$IMAGES" ] && { echo "ERROR: no images resolved from compose — check the Compose environment settings and chosen profile." >&2; exit 1; }
for img in $IMAGES; do
  echo "Checking access: $img"
  if ! docker manifest inspect "$img" >/dev/null 2>&1; then
    echo "NGC login succeeded, but this key cannot access the required image:" >&2
    echo "  $img" >&2
    echo "Provide an NGC key with access to this image's namespace, then re-run Step 1 (login) and retry." >&2
    exit 1
  fi
done

# Start all services (images pulled automatically on first run)
docker compose up -d

# Check containers are running
docker compose ps
```

The exact image tags change by release; read them from the active compose files instead of hardcoding a version.

### Step 6: Verify Services Are Running

```bash
# Read ports from the Compose environment file.
COMPOSE_ENV_BASENAME="env"
COMPOSE_ENV_FILE="$REPO_ROOT/compose/.${COMPOSE_ENV_BASENAME}"
MS_PORT=$(grep AUTO_MAGIC_CALIB_MS_PORT "$COMPOSE_ENV_FILE" | cut -d= -f2)
UI_PORT=$(grep AUTO_MAGIC_CALIB_UI_PORT "$COMPOSE_ENV_FILE" | cut -d= -f2)
HOST_IP=$(grep HOST_IP "$COMPOSE_ENV_FILE" | cut -d= -f2)

# Wait for microservice readiness. Cold image pulls or first startup can need
# extra time after `docker compose up -d` returns.
READY_URL="http://localhost:${MS_PORT}/v1/ready"
echo "Waiting for microservice readiness at ${READY_URL} ..."
ready_response=""
for attempt in $(seq 1 24); do
  if ready_response=$(curl -fsS --max-time 5 "${READY_URL}" 2>/dev/null) && \
     echo "${ready_response}" | grep -q '"code"[[:space:]]*:[[:space:]]*0'; then
    echo "Microservice ready: ${ready_response}"
    break
  fi
  if [ "${attempt}" -lt 24 ]; then
    printf "  [%02d/24] Microservice not ready yet; retrying in 5s...\n" "${attempt}"
    sleep 5
  fi
done

if ! echo "${ready_response}" | grep -q '"code"[[:space:]]*:[[:space:]]*0'; then
  echo "ERROR: microservice did not report ready within 120 seconds: ${READY_URL}" >&2
  echo "Check status and logs:" >&2
  echo "  cd ${REPO_ROOT}/compose && docker compose ps" >&2
  echo "  cd ${REPO_ROOT}/compose && docker compose logs auto-magic-calib-ms" >&2
  exit 1
fi

# Check UI is serving
UI_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:${UI_PORT}")
if [ "${UI_STATUS}" != "200" ]; then
  echo "ERROR: Web UI returned HTTP ${UI_STATUS}; check docker compose ps and UI logs." >&2
  exit 1
fi
echo "Web UI ready: HTTP ${UI_STATUS}"

echo "Microservice: http://${HOST_IP}:${MS_PORT}"
echo "Web UI:       http://${HOST_IP}:${UI_PORT}"
```

## Success Criteria

- `docker compose ps` shows MS and UI containers `Up`; MS should be healthy.
- `/v1/ready` returns `code:0` and Step 6 prints the microservice and UI URLs.
- Browser access to `http://<HOST_IP>:<AUTO_MAGIC_CALIB_UI_PORT>` works.
- Projects persist under `$REPO_ROOT/projects/`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Docker permission denied | Ask the user to run `sudo usermod -aG docker $USER && newgrp docker`, then retry `docker ps`. |
| `docker login` rejected | Ask for a current NGC key and log in again. |
| Required image inaccessible | The key lacks image namespace access; ask for a key with access, then retry Step 1 and Step 5. |
| `python3 -m venv`, `pip`, or `hf` missing | Install `python3-venv`/`python3-pip`; the HF binary is named `hf`. |
| VGGT permission error | Download VGGT before `chown 1000:1000`; to recover, restore user ownership of `models/` and re-download. |
| Port in use | Pick a free MS port in 8000-8009 and UI port in 5000-5009, then update the Compose environment file. |
| Readiness timeout or exited container | Run `cd $REPO_ROOT/compose && docker compose ps` and inspect `docker compose logs auto-magic-calib-ms`. |
| Project/model permission denied | Re-run Step 4 for `projects/` or `models/` only. |
| UI cannot reach backend | Verify `HOST_IP` in the Compose environment file is the machine network IP, not `localhost`. |
| GPU unavailable | Verify NVIDIA runtime with `docker run --rm --runtime=nvidia --gpus all ubuntu:20.04 nvidia-smi`. |

**Common Fixes**:
```bash
cd $REPO_ROOT/compose

# View logs
docker compose logs -f

# View logs for specific service
docker compose logs -f auto-magic-calib-ms

# Restart all services
docker compose restart

# Stop and remove containers
docker compose down

# Update Compose environment settings and relaunch
docker compose up -d
```

## Stopping the Services

```bash
cd $REPO_ROOT/compose

# Stop all services (containers removed, data persisted)
docker compose down

# Stop and remove volumes
docker compose down -v
```

## Related Skills
- `skills/amc-run-sample-calibration/SKILL.md` - Sanity-check the running stack with the bundled sample dataset
- `skills/amc-run-video-calibration/SKILL.md` - Calibrate from your own pre-recorded MP4s via REST API
- `skills/amc-run-rtsp-calibration/SKILL.md` - Calibrate from live RTSP streams through VIOS capture

<!-- signing marker -->
