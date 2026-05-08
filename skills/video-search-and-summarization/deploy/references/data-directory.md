# Deploy — Data directory layout

### Step 1b — Prepare the data directory

**This is the #1 source of silent-deploy bugs. Follow it exactly.**

The stack mounts several subdirs of `$MDX_DATA_DIR` into containers that each
run as a different uid. Docker auto-creates empty bind-mount paths as
`root:root`, which is read-only for the container processes. The reference
`scripts/dev-profile.sh` uses `chmod -R 777` on the relevant subdirs — it
does **not** `chown`.

Run this verbatim before `docker compose up`:

```bash
DATA=$MDX_DATA_DIR      # e.g. <repo>/data
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
