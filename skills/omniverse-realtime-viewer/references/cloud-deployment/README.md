# Cloud Deployment

## Triggers

Use this skill for requests mentioning deploy, NVCF self-hosted, OKAS 1, cloud deployment, session APIs, Docker, health checks, Brev, launchables, on-premises Kubernetes, in-place NVCF upgrades, or remote deployment.

## Choose A Deployment Path

- **Brev with standalone ovstream**: a fast demo deployment; use the Brev recipe below. Brev is the hosting provider, not the streaming architecture. For Kubernetes plus NVCF on a Brev machine, select the NVCF path instead.
- **OKAS or another portable session orchestrator**: use the generic session contract below.
- **NVCF self-hosted on AWS, Azure, or on-premises Kubernetes**: read `nvcf-self-hosted.md` first, then follow the current upstream `streaming-self-hosted` orchestration and component skills. Do not duplicate that workflow here.

For NVCF image preparation and viewer-side failures, use
`nvcf-viewer-container-contract.md`, `gpu-container-runtime.md`,
`container-image-build.md`, and `webrtc-network-diagnostics.md` after selecting
the upstream deployment path.

## Brev Launchable Deployment (Recommended for Demo)

Use the permanent `omniverse-realtime-viewer` launchable:
<https://brev.nvidia.com/launchable/deploy?launchableID=env-3EHjQXkUNYv2pOa3idjBeJOauvH>

Instance type: `g5.xlarge` (A10G, 23 GB VRAM).

### Prerequisites

- Brev account with an `omniverse-realtime-viewer-launchable` instance (A10G or better)
- TCP/UDP Port Rules open: `80`, `1024`, `47998`, `49100`
- SSH access configured (uses Brev SSH key, typically `~/.brev/brev.pem`; run `brev refresh` to regenerate)

### Architecture

Two access modes are supported:

#### Option A: HTTPS via Brev domain (port 80 + Caddy WSS proxy)

Brev exposes port 80 as `https://frontend-<id>.brevlab.com` with TLS termination at their edge. Browsers enforce secure WebSocket (`wss://`) — plain `ws://` is blocked as mixed-content. Caddy on port 80 serves both the frontend AND proxies the backend `@nvidia/ov-web-rtc` Direct signaling endpoint used by standalone `ovstream`:

```text
Browser → https://frontend-<id>.brevlab.com (Brev TLS edge)
  └── port 80 → Caddy
        ├── /sign_in* → reverse_proxy localhost:49100 (ovstream WebSocket signaling)
        ├── /*        → file_server (pre-built frontend)
        └── UDP media → <PUBLIC_IP>:47998 (direct, no proxy)
```

The frontend is built with `VITE_SIGNALING_PORT=443` so `@nvidia/ov-web-rtc`
Direct mode connects to
`wss://frontend-<id>.brevlab.com:443/sign_in` (same origin as the page).
Brev routes this → port 80 → Caddy → localhost:49100.

This is the exposed route for standalone `ovstream` Direct signaling. The
deployment layer may provide auth, launch, routing, and lifecycle management,
but the browser WebRTC config still uses `@nvidia/ov-web-rtc` Direct mode with
the exposed signaling endpoint. Do not replace it with a connection profile intended for a different streaming product.

#### Option B: Direct IP access (port 1024 + nginx)

For development or controlled-network testing where TLS isn't needed, access the viewer directly via `http://<PUBLIC_IP>:1024`. nginx on port 1024 proxies both the frontend and signaling:

```text
Browser → http://<PUBLIC_IP>:1024/
  ├── nginx (port 1024) → /            → Vite dev server or static files (port 5173/3000)
  │                     → /sign_in     → ovstream signaling server (port 49100)
  └── UDP media → <PUBLIC_IP>:47998
```

No TLS, no mixed-content issues (both page and WebSocket are plain HTTP).
Frontend uses default `VITE_SIGNALING_PORT=1024` (same port as page).

> **Note:** Do not use Brev's Cloudflare secure link (`https://`) with Option B.
> NVST extracts the client IP from `getpeername()` on the TCP socket — Cloudflare
> in the middle causes NAT hole-punch failure.

### Critical Configuration

#### Option A: Caddyfile (port 80 — frontend + WSS proxy)

```
{
    auto_https off
}

:80 {
    handle /sign_in* {
        reverse_proxy localhost:49100
    }
    handle {
        root * /opt/ov-viewer/clients/webrtc-browser/dist
        file_server
    }
}
```

Install Caddy:
```bash
curl -o /tmp/caddy.tar.gz -sL "https://github.com/caddyserver/caddy/releases/download/v2.8.4/caddy_2.8.4_linux_amd64.tar.gz"
tar -xzf /tmp/caddy.tar.gz -C /tmp caddy
sudo mv /tmp/caddy /usr/local/bin/caddy
```

#### Option B: nginx (port 1024 — direct access)

```nginx
server {
    listen 1024;

    location / {
        proxy_pass http://localhost:3000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
    }

    location /sign_in {
        proxy_pass http://localhost:49100/sign_in;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
    }
}
```

#### Server (`ov_web_viewer_server.py`)

Implement this app entry point using the
[streaming recipe](../streaming-viewer-recipe/README.md) before applying the
launch examples here. Python filenames, container paths, and CLI flags in this
document refer to your generated app; this package does not ship that server.
Keep each launch command aligned with the actual entry point and parser.

The server binds signaling on port 49100 and media on port 47998:

```bash
python3 ov_web_viewer_server.py --port 49100 --public-ip "$PUBLIC_IP" \
    --stage /opt/ov-viewer/samples_data/stage01.usd
```

- `--public-ip` sets the ICE candidate IP in SDP. Required for NAT traversal.
- Media defaults to UDP :47998 — must match Brev port rule.

#### Frontend build (bake signaling port)

The frontend reads `VITE_SIGNALING_PORT` at build time:

```bash
cd clients/webrtc-browser

# Option A (HTTPS via Brev domain):
VITE_SIGNALING_PORT=443 npx vite build

# Option B (direct IP on port 1024):
VITE_SIGNALING_PORT=1024 npx vite build
```

This makes the SDK connect to the matching port for signaling.
With Option A, Caddy proxies the WebSocket transparently.
With Option B, nginx proxies it on the same port.

### Deployment Steps

1. **Build frontend locally:**
   ```bash
   export VIEWER_ROOT=/path/to/generated-viewer
   cd "$VIEWER_ROOT/clients/webrtc-browser"
   npm install --ignore-scripts
   # Choose one:
   VITE_SIGNALING_PORT=443 npx vite build   # Option A (HTTPS)
   VITE_SIGNALING_PORT=1024 npx vite build  # Option B (direct)
   ```

2. **Refresh SSH and rsync payload:**
   ```bash
   brev refresh
   INSTANCE="omniverse-realtime-viewer-launchable-XXXXXX"

   # Wheels (~2.5GB ovrtx + ovstream)
   rsync -az "$VIEWER_ROOT/deps/wheels/" $INSTANCE:/tmp/ov-deploy/

   # Server + samples + frontend
   rsync -az "$VIEWER_ROOT/server" $INSTANCE:/tmp/ov-deploy/
   rsync -az "$VIEWER_ROOT/samples_data" $INSTANCE:/tmp/ov-deploy/
   rsync -az "$VIEWER_ROOT/clients/webrtc-browser/dist" $INSTANCE:/tmp/ov-deploy/frontend-dist
   ```

3. **Remote setup (SSH into instance):**
   ```bash
   # System deps
   sudo apt-get update -qq
   sudo apt-get install -y -qq python3-pip python3-venv python3-dev \
       libgomp1 libatomic1 libgl1 libglx0 libx11-6 libxau6 libxdmcp6 \
       libxcb1 libbsd0 libmd0 libegl1 libglib2.0-0

   # Deploy directory
   sudo mkdir -p /opt/ov-viewer
   sudo chown $(whoami) /opt/ov-viewer
   cp -r /tmp/ov-deploy/server /opt/ov-viewer/
   cp -r /tmp/ov-deploy/samples_data /opt/ov-viewer/
   mkdir -p /opt/ov-viewer/clients/webrtc-browser
   cp -r /tmp/ov-deploy/frontend-dist /opt/ov-viewer/clients/webrtc-browser/dist

   # Install Caddy (Option A only)
   curl -o /tmp/caddy.tar.gz -sL "https://github.com/caddyserver/caddy/releases/download/v2.8.4/caddy_2.8.4_linux_amd64.tar.gz"
   tar -xzf /tmp/caddy.tar.gz -C /tmp caddy
   sudo mv /tmp/caddy /usr/local/bin/caddy
   ```

   Install the generated app's resolved dependencies using
   [the central runtime guidance](../dependencies/nvidia-runtime.md) and its
   manifest/lockfile. The server-start example below assumes its environment is
   at `/opt/ov-viewer/server/.venv`; adjust activation to the app's actual
   environment path. Preserve the selected interpreter and process boundaries:
   USD query workers have their own environment; adding `usd-core` to the
   renderer environment is not a deployment prerequisite. Run the app's
   dependency and first-frame checks on the target GPU before starting clients.

4. **Start server:**
   ```bash
   cd /opt/ov-viewer/server
   source .venv/bin/activate
   export OVRTX_SKIP_USD_CHECK=1
   OVSTREAM_DIR=$(python3 -c "import ovstream, os; print(os.path.dirname(ovstream.__file__))")
   OVRTX_BIN=$(python3 -c "import ovrtx, os; print(os.path.join(os.path.dirname(ovrtx.__file__), 'bin'))")
   export LD_LIBRARY_PATH="${OVSTREAM_DIR}:${OVRTX_BIN}:${LD_LIBRARY_PATH:-}"
   PUBLIC_IP=$(curl -sf ifconfig.me)

   nohup python3 ov_web_viewer_server.py --port 49100 --public-ip "$PUBLIC_IP" \
       --stage /opt/ov-viewer/samples_data/stage01.usd > /tmp/server.log 2>&1 &
   ```

5. **Wait for shader warmup** (~5-10 min cold on A10G):
   ```bash
   # Monitor: port 49100 appears when ready
   watch -n5 'ss -tlnp | grep 49100 && echo READY || echo WAITING'
   ```

6. **Start reverse proxy:**
   ```bash
   # Option A: Caddy on port 80
   sudo caddy run --config /path/to/Caddyfile &

   # Option B: nginx on port 1024 (install + configure per above)
   sudo apt-get install -y nginx
   # Add server block to /etc/nginx/sites-enabled/default, then:
   sudo nginx -s reload
   ```

7. **Access:**
   - Option A: `https://frontend-<id>.brevlab.com/`
   - Option B: `http://<PUBLIC_IP>:1024/`

### Gotchas & Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Mixed Content: `ws://` blocked | Page loaded over HTTPS, SDK uses plain WS | Use Option A (Caddy + `VITE_SIGNALING_PORT=443`) or Option B (direct IP, no TLS) |
| "connection attempts failed, retrying" | Proxy not forwarding the backend Direct signaling endpoint | Verify Caddy/nginx config proxies `/sign_in` to 49100 |
| WebSocket to `wss://` fails on direct IP | SDK tries WSS on HTTPS page | Use Option B with `http://` access (no TLS = no mixed-content) |
| Black screen, input works | NVENC encoder state corruption | Kill and restart server clean |
| `NattHolePunch: Address ... is not valid` | Signaling through Cloudflare hides client IP | Use Option B (direct IP) or Option A (Caddy, doesn't affect UDP) |
| Port 49100 not listening after 10 min | Shader warmup still running | Check `nvidia-smi` — GPU at 0% is normal during compilation |
| `GPU device ID 8759 not white-listed` | A10G not in NVST allowlist | Warning only; NVENC works fine |
| Shader compilation takes 5-10 min | Cold start on A10G (no shader cache) | Wait; GPU util jumps to 50%+ then drops when done |
| Safari shows black video | Missing `autoplay playsinline muted` on `<video>` | Add these attributes to HTML |
| 404 on assets after rebuild | Browser cached old `index.html` with stale hash | Hard refresh (Ctrl+Shift+R) |
| NVST_R_BUSY | Second WebRTC client connected | Only 1 peer at a time; restart server |

### Why Two Options

**Option A (Caddy + HTTPS)** is required for demos and external access. Brev's HTTPS proxy terminates TLS at the edge and forwards to port 80. Browsers enforce `wss://` from HTTPS pages — Caddy solves this by serving both frontend and signaling on the same origin.

**Option B (nginx + direct IP)** is simpler for development/testing. No TLS means no mixed-content issues. Access via `http://<PUBLIC_IP>:1024` bypasses Brev's Cloudflare tunnel entirely. However, NVST cannot determine the client IP through Cloudflare, so never use the Brev `https://` URL with Option B.

Port 1024 and 80 are opened via Brev's "TCP/UDP Port Rules" (actual AWS security group entries), NOT via "Secure Links" (which only proxy TCP through Cloudflare).

---

## Docker Container Deployment

For containerized deployments without a full orchestrator, build a standalone Docker image that bundles the ovrtx server, ovstream, sample data, and a pre-built frontend.

### Base Image And System Dependencies

Select the base image, Python version, native prerequisites, and compatible
package set through [the central runtime guidance](../dependencies/nvidia-runtime.md).
Record the resolved combination in the generated app's manifest and image
recipe. Follow [container image build](container-image-build.md) for build
inputs and [GPU runtime preflight](gpu-container-runtime.md) for the image/host
contract, including graphics, encoder, display, and USD-worker requirements.

### Installing Large Python Wheels Efficiently

If the selected dependency workflow uses large local artifacts, use the build
system's supported temporary mounts so installation archives do not remain in
image layers. Install only the artifacts selected by the app's resolved
manifest; a wildcard wheel directory is not a compatibility check. Keep
acquisition commands and artifact selection in the central dependency guidance.

### .dockerignore

A `.dockerignore` file is critical. Without it, `COPY . /app` sends `node_modules/` to the Docker daemon and can inject platform-incompatible native binaries into the image:

```text
**/node_modules
**/.git
**/__pycache__
*.egg-info
```

### Runtime Requirements

Start the container with GPU access and the display strategy required by the
selected runtime. When that strategy uses a host X11 display, the launch may
include the following socket mount and `DISPLAY`; those settings are not a
universal headless-rendering requirement. Validate the actual image with
[GPU runtime preflight](gpu-container-runtime.md).

```bash
docker run --gpus all \
  -e DISPLAY=:99 \
  -e PUBLIC_IP=<reachable-ip> \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -p 49100:49100 \
  -p 47998:47998/udp \
  -p 8081:8081 \
  ovrtx-viewer:latest
```

| Env Var | Purpose |
|---|---|
| `DISPLAY` | X11 display for GPU rendering (use Xvfb `:99` for headless) |
| `PUBLIC_IP` | WebRTC ICE candidate IP advertised to clients |

| Volume | Purpose |
|---|---|
| `/tmp/.X11-unix` | X11 socket mount (from host Xvfb or display server) |

| Port | Protocol | Purpose |
|---|---|---|
| 49100 | TCP | WebRTC signaling (WebSocket) |
| 47998 | UDP | WebRTC media |
| 8081 | TCP | Health endpoint (`/healthz`) |

### Shader Compilation Cold Start

After a fresh container start, the first scene load triggers GPU shader compilation. Expected times:

| GPU | Approximate shader compilation time |
|---|---|
| L40 / L40S | ~90 seconds |
| A10G | ~240 seconds |
| H100 / A100 (non-graphics) | Not supported for rendering |

Do not connect clients or mark the service as ready until `/healthz` returns `200`. The health endpoint gates on the first successfully rendered and converted frame, which occurs after shader compilation completes; it should not require an attached browser client.

### Entrypoint Pattern

The following wrapper is the X11/Xvfb variant; include its display tools in the
resolved image only when that strategy is selected. For another supported
strategy, omit its display startup block and use the app's matching wrapper.

```dockerfile
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
```

```bash
#!/bin/bash
set -e

# Start Xvfb if no display is available
if ! xdpyinfo -display "${DISPLAY:-:99}" >/dev/null 2>&1; then
  Xvfb ${DISPLAY:-:99} -screen 0 1920x1080x24 &
  sleep 1
fi

export OVRTX_SKIP_USD_CHECK=1
exec python3 /app/server/ov_web_viewer_server.py \
  --port "${PORT:-49100}" \
  --health-port "${HEALTH_PORT:-8081}" \
  --public-ip "${PUBLIC_IP:-$(curl -s ifconfig.me)}" \
  --stage "${STAGE_PATH:-/app/samples_data/stage01.usd}"
```

### Application Image Recipe

Generate the Dockerfile for the selected app and runtime combination using
[container image build](container-image-build.md). Include its server, frontend
output, authorized scene assets, resolved dependency environments, and launch
wrapper. Match paths to the generated app rather than assuming a sample-repo
layout. Preserve `exec`-based server startup so session termination reaches the
viewer, and validate readiness, decoded frames, and shutdown in the final image.

### Build And Run

```bash
# Build (requires BuildKit)
DOCKER_BUILDKIT=1 docker build -t ovrtx-viewer:latest .

# Run
docker run --gpus all \
  -e PUBLIC_IP=$(curl -s ifconfig.me) \
  -p 49100:49100 \
  -p 47998:47998/udp \
  -p 8081:8081 \
  ovrtx-viewer:latest
```

For the X11/Xvfb variant, mount the socket and set `DISPLAY` when using an
existing host display. Otherwise that variant's entrypoint starts its own Xvfb
instance. Other display strategies use their corresponding launch configuration.

---

## OKAS 1 / Generic Session Orchestration

For production-style deployments, keep the Omniverse Realtime Viewer contract portable.
Use OKAS 1 or a generic container/session orchestrator that can start one GPU
container per Omniverse Realtime Viewer session, expose WebRTC signaling and media ports, route the
browser to the frontend, and terminate the container when the session ends.

OKAS is orchestration/session management, not a different WebRTC client profile.
It may allocate GPU resources, start the container, inject environment/config,
publish routes, and manage session lifecycle. After OKAS resolves a session
endpoint, the frontend uses standalone `ovstream` Direct config: `server` and
`signalingPort` point at the exposed signaling endpoint, while media remains
negotiated by WebRTC.

### Registration Contract

Register the Omniverse Realtime Viewer with portable metadata:

```json
{
  "id": "ovrtx-viewer",
  "name": "Omniverse Realtime Viewer",
  "image": "ovrtx-viewer:latest",
  "description": "Omniverse Realtime Viewer using ovrtx rendering with ovstream WebRTC delivery",
  "gpuRequired": true,
  "ports": {
    "signaling": 49100,
    "mediaUdp": 47998,
    "health": 8081
  }
}
```

Keep deployment recipes portable. Do not bind generated apps to app registries,
session-manager paths, sidecars, or caching services unless the selected
deployment target explicitly provides them.

### Launch Contract

A session launcher should run the same server command the Brev path uses:

```bash
export OVRTX_SKIP_USD_CHECK=1
python3 server/ov_web_viewer_server.py \
  --port "${PORT:-49100}" \
  --health-port "${HEALTH_PORT:-8081}" \
  --public-ip "${PUBLIC_IP}" \
  --stage "${STAGE_PATH:-samples/samples_data/stage01.usd}"
```

### Health And Ports

| Port | Protocol | Purpose |
|---|---|---|
| 49100 | WebSocket/TCP | WebRTC signaling |
| 47998 or 47999 | UDP | WebRTC media (deployment-dependent) |
| 8081 | HTTP | health endpoint (`/healthz`) |

### Docker

```bash
cd deploy
docker build -t ovrtx-viewer:latest .
```

`deploy/entrypoint.sh` launches the server with `$PORT` and `$STAGE_PATH`.

| Var | Default | Purpose |
|---|---|---|
| `PORT` | `49100` | signaling port |
| `PUBLIC_IP` | auto-detect via `ifconfig.me` | WebRTC candidate IP |
| `STAGE_PATH` | `samples/samples_data/stage01.usd` | initial stage |
| `HEALTH_PORT` | `8081` | health endpoint |

### Session Lifecycle

```text
POST /sessions {application:"ovrtx-viewer"}
  → spawn one GPU process/container
  → poll GET :8081/healthz
  → mark ready
  → browser connects to frontend and backend Direct signaling
  → WebRTC media flows
  → client requests session shutdown through the orchestrator
  → orchestrator sends SIGTERM to the viewer process
```

## Related

- `streaming-server` for ovstream ServerConfig details and frame handling.
- `streaming-client` for frontend WebRTC SDK usage.
- `streaming-lifecycle` for connection/reconnection behavior.
- `cloud-assets` when deployed sessions load stages from S3/MinIO.
- `nvcf-viewer-container-contract.md` for the application/platform boundary.
- `gpu-container-runtime.md` for GPU image and native-runtime preflight.
- `container-image-build.md` for registry image preparation.
- `webrtc-network-diagnostics.md` for signaling/media failure isolation.
- OKAS 1 or your orchestrator documentation for portal/session APIs.
