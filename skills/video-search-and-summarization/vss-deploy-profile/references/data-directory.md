# Deploy — Data directory layout

### Step 1b — Prepare the data directory

**This is the #1 source of silent-deploy bugs. Follow it exactly.**

The stack mounts several subdirs of `$VSS_DATA_DIR` into containers that each
run as a different uid. Docker auto-creates empty bind-mount paths as
`root:root`, which is read-only for the container processes. The reference
`scripts/dev-profile.sh` uses `chmod -R 777` on the relevant subdirs — it
does **not** `chown`.

Run this verbatim before `docker compose up`:

```bash
DATA=$VSS_DATA_DIR      # e.g. <repo>/data
mkdir -p \
  "$DATA/data_log/analytics_cache" \
  "$DATA/data_log/calibration_toolkit" \
  "$DATA/data_log/elastic/data" \
  "$DATA/data_log/elastic/logs" \
  "$DATA/data_log/kafka" \
  "$DATA/data_log/redis/data" \
  "$DATA/data_log/redis/log" \
  "$DATA/agent_eval/dataset" \
  "$DATA/agent_eval/results"
# Profile-specific subdirs:
#   alerts → mkdir -p "$DATA/data_log/vss_video_analytics_api" "$DATA/videos/dev-profile-alerts" "$DATA/models/rtdetr-its" "$DATA/models/gdino"
#   search → mkdir -p "$DATA/models"
chmod -R 777 "$DATA/data_log" "$DATA/agent_eval"
# If you created $DATA/models above, also: chmod -R 777 "$DATA/models"
```

> **FORBIDDEN: `chown -R ubuntu:ubuntu $VSS_DATA_DIR` (or any recursive chown).**
>
> This is "good housekeeping" to a shell-admin instinct but is **the** deploy-
> breaking command in this stack. You will observe a "healthy" deploy
> (containers Up, endpoints 200) while the video pipeline is silently broken.
> Use `chmod -R 777` on the specific subdirs above — nothing else.

**Known per-container uid gotchas** (each uses a bind mount under `$DATA`):

| Container | Image | Runs as | Mount path | Symptom if permissions wrong |
|---|---|---|---|---|
| `vss-vios-postgres` | postgres:17.9-alpine | uid **70** | `$DATA/data_log/vst/postgres/db` | Can't read own PGDATA → VST `sensor_details` query fails → uploaded videos never appear in `/vst/api/v1/sensor/streams` → warehouse E2E check returns empty |
| `redis` | redis:8.2.2-alpine | uid **999** | `$DATA/data_log/redis/log`, `/redis/data` | "Can't open the log file: Permission denied" → redis dies → `sdr-controller` can't reach `WDM_WL_REDIS_SERVER` → SDRC never registers `vss-vios-streamprocessing` on the `:10000` Envoy listener → stream pipeline broken (legacy `envoy-streamprocessing` no longer exists; SDR + Envoy are now consolidated in `sdr-controller` from `services/infra/sdrc/`) |
| `elasticsearch` | elasticsearch | uid **1000** | `$DATA/data_log/elastic/{data,logs}` | "AccessDeniedException" on startup → ES refuses to start |
| `vss-vios-ingress` / `vss-vios-sensor` | vst | uid **1000** | `$DATA/data_log/vst/*` (videos, clips) | 403 on ingest or stream write |

`chmod -R 777 $DATA/data_log` covers all of these. Do NOT chown them to
individual uids — containers that init their own dirs on first start (like
postgres) will then re-chown to their uid and a later chown back to ubuntu
breaks them.

**If postgres is already broken** (common when redeploying without a clean
`data-dir`):

```bash
sudo rm -rf "$DATA/data_log/vst/postgres"  # postgres re-initializes on next start
docker restart centralizedb-dev
```
