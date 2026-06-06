# Tear down an existing VSS deployment

### Step 0 — Tear down any existing deployment

Ask user to confirm to tear down the deployment before you proceed.

Before every deploy, **always** stop any prior VSS stack. This is
mandatory even if you think the host is clean, and especially when
switching profiles (`base` → `search`, `alerts` verification →
`alerts` real-time, etc.). Compose profile flags only *start* the
services listed under the selected profile — they do NOT stop
services from a previously-active profile, so containers from the
prior deploy linger and pass unrelated container-name checks,
contaminate results, and can bind ports the new deploy needs.

```bash
# If a resolved.yml from a prior deploy exists, prefer it — it
# knows about all compose-profile services that were brought up.
if [ -f "$REPO/deploy/docker/resolved.yml" ]; then
  docker compose -f "$REPO/deploy/docker/resolved.yml" down --remove-orphans
fi

# Catch-all: remove every VSS-stack container the dev-profile compose
# files bring up. Without this, leftovers from a prior deploy linger
# (especially the *-smc set, which the alerts compose profile shares
# with the *-dev set on host networking and port 30000) and either:
#   - bind ports the new deploy needs → second sensor-ms fails to bind
#     → /sensor/list returns 502 (issue #151), or
#   - pass the new deploy's container-name health checks while serving
#     stale data from the prior deploy's DB.
# The patterns below cover everything declared under
# deploy/docker/services/ (agent, vios, rtvi, infra, nim, video-summarization, …)
# and deploy/docker/developer-profiles/dev-profile-*/compose files.
docker ps -a --format '{{.Names}}' \
  | grep -E '^(vss-|mdx-|perception-|rtvi-|alert-|nvstreamer-|sensor-ms-|vst-ingress-|vst-mcp-|vst-file-proxy|centralizedb-|storage-ms-|streamprocessing-ms-|sdr-(http|streamprocessing)-|envoy-(http|streamprocessing)-|rtspserver-ms-|recorder-ms-|replaystream-ms-|livestream-ms-|metropolis-vss-ui|phoenix)' \
  | xargs -r docker rm -f
```

# Step 0b - Cleanup previous stale state and local logs, data.

Ask user to confirm to clean up before you proceed.

Use the bundled cleanup helper. It clears every directory whose stale state can poison a fresh deploy: kafka logs, elasticsearch data + logs, redis data + log, behavior-learning data, video-analytics API state, calibration toolkit, VST/nvstreamer recordings, and any blueprint-configurator backup files. The same logic `dev-profile.sh` runs internally between deploys.

```bash
# Step 0 (teardown) runs BEFORE Step 1c initializes generated.env,
# so on a fresh checkout / first deploy generated.env doesn't exist
# yet — fall back to the source .env. Once a prior deploy via this
# skill has run, generated.env carries the actually-deployed paths.
PROFILE_DIR="$REPO/deploy/docker/developer-profiles/dev-profile-<profile>"
ENV_FILE="$PROFILE_DIR/generated.env"
[ -f "$ENV_FILE" ] || ENV_FILE="$PROFILE_DIR/.env"

sudo bash "$REPO/deploy/docker/scripts/cleanup_all_datalog.sh" \
    --env-file "$ENV_FILE"
```
