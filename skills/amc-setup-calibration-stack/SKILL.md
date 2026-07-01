---
name: "amc-setup-calibration-stack"
description: "Launch AutoMagicCalib microservice and web UI from NGC release images via Docker Compose. Use when user says 'deploy auto calibration', 'launch auto calibration', 'launch AMC', 'start MS+UI', or 'set up auto-magic-calib'. Requires NGC API key."
metadata:
  tags: [amc, deepstream, docker, calibration, setup, ngc]
owner: "NVIDIA CORPORATION"
service: "auto-magic-calib"
version: "1.0.0"
reviewed: "2026-04-28"
license: "Apache-2.0"
data_classification: public
---

# Skill: Launch AutoMagicCalib Release Containers

## When to Use This Skill

Activate this skill when the user wants to bring up the AutoMagicCalib stack. Typical prompts:

- "deploy auto calibration"
- "launch auto calibration" / "launch AMC"
- "start MS+UI" / "bring up the microservice and UI"
- "set up auto-magic-calib"

Prerequisite: NGC API key (the skill prompts for it). For first-time setup on a fresh machine, this also resolves or clones the `auto-magic-calib` repo after explicit user confirmation.

## Overview
Launch the AutoMagicCalib microservice (MS) and UI from pre-built release images via Docker Compose. Use this for production deployments or when you want to run the full stack (backend + web UI) without building images locally.

## Prerequisites
- Docker and Docker Compose installed
- NVIDIA Docker Runtime configured (for GPU support)
- `auto-magic-calib` repo on disk — Step 0b auto-resolves an existing checkout via `git rev-parse --show-toplevel`, or offers to clone `https://github.com/NVIDIA-AI-IOT/auto-magic-calib` into `~/auto-magic-calib` if none is found
- NGC account with access to NVIDIA container registry
- **Docker must be runnable without `sudo`** — verify with `docker ps`. If it fails, ask the user to follow the post-install steps: https://docs.docker.com/engine/install/linux-postinstall/

> **If `docker ps` fails with "permission denied"**, ask the user to run:
> ```bash
> sudo usermod -aG docker $USER
> newgrp docker   # applies group change in current shell without logout
> ```
> Then verify with `docker ps` before continuing.

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

The skill needs `compose/`, `assets/sdg_08_2_sample_data_010926.zip`, and a `models/` mount point — all of which live in the `auto-magic-calib` repo. If you're already inside a checkout, the skill uses it. Otherwise it asks for explicit user confirmation using the host's question mechanism; if none is available, ask in chat and wait before cloning into `~/auto-magic-calib`.

```bash
REPO_URL="https://github.com/NVIDIA-AI-IOT/auto-magic-calib.git"
DEFAULT_CLONE_DIR="$HOME/auto-magic-calib"

# 1. Already inside a usable checkout?
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -n "$REPO_ROOT" ] && [ -f "$REPO_ROOT/compose/compose.yml" ] && [ -d "$REPO_ROOT/assets" ]; then
  echo "✓ Using existing checkout: $REPO_ROOT"

# 2. Default clone dir already populated from a prior run?
elif [ -d "$DEFAULT_CLONE_DIR/.git" ] && [ -f "$DEFAULT_CLONE_DIR/compose/compose.yml" ]; then
  REPO_ROOT="$DEFAULT_CLONE_DIR"
  echo "✓ Found existing clone at $REPO_ROOT"

# 3. Nothing on disk — STOP and ask the user for confirmation using the
#    host's question mechanism; if none is available, ask in chat and wait.
#    Do NOT clone silently from this block.
else
  echo "No auto-magic-calib checkout found. Ask the user for confirmation:"
  echo "  Clone $REPO_URL into $DEFAULT_CLONE_DIR? [y/N]"
  echo "On 'y' — run: git clone \"$REPO_URL\" \"$DEFAULT_CLONE_DIR\""
  exit 1
fi

cd "$REPO_ROOT"
export REPO_ROOT
echo "REPO_ROOT=$REPO_ROOT"
```

> **Agent note**: do **not** clone silently. Ask the user first using the host's question mechanism; if none is available, ask in chat and wait. Example: "auto-magic-calib repo not found. Clone `https://github.com/NVIDIA-AI-IOT/auto-magic-calib` into `~/auto-magic-calib`? (or provide your own path)". Honour an alternate path if the user offers one. The clone is a few hundred MB (compose files + sample dataset + assets).

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

**To download VGGT** (ask the user for a HuggingFace token using the host's question mechanism; if none is available, ask in chat and wait):

**Step 2a: Accept Model License** (required, one-time):
1. Visit: https://huggingface.co/facebook/VGGT-1B-Commercial
2. Log in to your HuggingFace account
3. Click "Agree and access repository" to accept license terms

**Step 2b: Get HuggingFace Token**:
1. Visit: https://huggingface.co/settings/tokens
2. Create new token with "Read" access (starts with `hf_...`)
3. Ask the user for the token using the host's question mechanism; if none is available, ask in chat and wait. Do NOT ask them to run a command

**Step 2c: Download** (pass token via `HF_TOKEN` env var — keeps the secret out of `hf`'s argv / `ps aux`; no interactive login needed):
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

### Step 3: Configure .env Variables

The `.env` file at `compose/.env` controls ports and paths. Update it before launching:

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

# Update .env — preserve any keys the user has already set; back up first.
# .env may contain credentials, so restrict permissions on both the backup and
# the live file (chmod 600). Add `compose/.env.bak.*` to .gitignore.
ENV_FILE=".env"
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

# Keep timestamped .env backups out of git.
GITIGNORE="$REPO_ROOT/.gitignore"
touch "$GITIGNORE"
grep -qxF "compose/.env.bak.*" "$GITIGNORE" || echo "compose/.env.bak.*" >> "$GITIGNORE"

echo "✓ .env updated"
cat .env
```

**Important**: `HOST_IP` must be the machine's network IP (not `localhost`) so the UI container can reach the backend from a browser.

Optional: set `VGGT_MODEL_PATH` only if the VGGT model is mounted at a non-default container path; default is `/tmp/vggt_model/vggt_1B_commercial.pt` inside the MS container.

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
[ -z "$IMAGES" ] && { echo "ERROR: no images resolved from compose — check compose/.env and the chosen profile." >&2; exit 1; }
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

**Expected output**:
```
NAME                    IMAGE                                                              STATUS
auto-magic-calib-ms-1   nvcr.io/nvidia/auto-magic-calib:2.0.0           Up (healthy)
auto-magic-calib-ui-1   nvcr.io/nvidia/auto-magic-calib-ui:2.0.0        Up
```

### Step 6: Verify Services Are Running

```bash
# Read ports from .env
MS_PORT=$(grep AUTO_MAGIC_CALIB_MS_PORT $REPO_ROOT/compose/.env | cut -d= -f2)
UI_PORT=$(grep AUTO_MAGIC_CALIB_UI_PORT $REPO_ROOT/compose/.env | cut -d= -f2)
HOST_IP=$(grep HOST_IP $REPO_ROOT/compose/.env | cut -d= -f2)

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

**Both containers running** — see `docker compose ps` output from Step 5. Both services should show "Up" status; microservice should show "(healthy)" in the STATUS column.

**Microservice healthy** — Step 6 readiness polling returns `code:0` from `/v1/ready` and prints the service URL.

**Web UI accessible**:
- Open browser: `http://<HOST_IP>:<AUTO_MAGIC_CALIB_UI_PORT>`
- Should display the AutoMagicCalib web interface
- Should be able to create projects and run calibration

## Key Output

**Microservice**: `http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>` (default port 8000)
- API docs: `http://<HOST_IP>:<AUTO_MAGIC_CALIB_MS_PORT>/docs`

**Web UI**: `http://<HOST_IP>:<AUTO_MAGIC_CALIB_UI_PORT>` (default port 5000)
- Interactive project management
- File upload interface
- Calibration configuration
- Real-time status monitoring
- Results visualization and download

**Data Persistence**:
- Projects stored: `$REPO_ROOT/projects/`
- State persisted: `$REPO_ROOT/projects/state.json`

## Troubleshooting

| Issue | Symptoms | Solution |
|-------|----------|----------|
| `pip` not found | `pip: command not found` | Run `sudo apt install -y python3.12-venv python3-pip` then create venv (Step 0) |
| `huggingface-cli` not found | `huggingface-cli: command not found` | The binary is named `hf` in the venv. Find it with: `find venv ~/venv/amc -name hf -type f 2>/dev/null \| head -1` |
| `python3 -m venv` fails | "ensurepip not available" | Run `sudo apt install -y python3.12-venv` first |
| Docker permission denied | "permission denied while trying to connect to docker socket" | User not in docker group — ask user to run: `sudo usermod -aG docker $USER && newgrp docker`. See: https://docs.docker.com/engine/install/linux-postinstall/ |
| `docker login` itself rejected | Step 1 login returns an authentication error | The key is invalid or expired. Ask the user for a current NGC key and log in again before continuing. |
| Key logs in but can't access an image | The Step 5 access check stops with "cannot access the required image" / a 401/403 on `docker manifest inspect`, before any pull | The key authenticates but lacks access to that image's namespace. Re-running login with the same key won't help — ask the user for an NGC key with access to the required namespace, re-run Step 1, then retry Step 5. |
| VGGT download permission error | "PermissionError: [Errno 13] Permission denied: 'models/vggt/.cache'" | Download VGGT BEFORE setting `chown 1000:1000` on models. Fix: `sudo chown -R $(id -u):$(id -g) models` then re-download |
| Port already in use | "address already in use" | Find available port in 8000-8009 (MS) or 5000-5009 (UI); update `.env` |
| Microservice readiness timeout | Step 6 reports that `/v1/ready` did not return `code:0` within 120 seconds | Run `cd $REPO_ROOT/compose && docker compose ps`, then inspect logs with `docker compose logs auto-magic-calib-ms` |
| Permission denied (projects) | "Permission denied: 'projects/...'" in MS logs | Run: `sudo chown 1000:1000 -R projects` |
| Permission denied (models) | "Permission denied: 'models/...'" in MS logs | Run: `sudo chown 1000:1000 -R models` |
| UI can't reach backend | Browser shows connection error | Verify `HOST_IP` in `.env` is the machine's network IP, not `localhost` |
| VGGT model not found | Warning in MS logs about missing VGGT | Download model (Step 2) or ignore (AMC works without VGGT) |
| Container exits immediately | Status "Exited" | Check logs: `docker compose logs auto-magic-calib-ms` |
| GPU not available | "CUDA not available" in logs | Check NVIDIA runtime: `docker run --rm --runtime=nvidia --gpus all ubuntu:20.04 nvidia-smi` |
| "No such file or directory" when verifying | After launch, can't find compose directory | Working directory persists after `cd compose`. Use absolute paths or run `cd $REPO_ROOT` first |

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

# Update .env and relaunch
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

<!-- signing marker -->
