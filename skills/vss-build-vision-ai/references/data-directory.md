# Data Directory Gate

Run this gate for every stock or delta build once `_builds/<name>/override.env`
exists — a bring-up prerequisite, not a deploy step, so run it whether or not
this run deploys. Any later `docker compose up`, this agent's or hand-run, needs
it: Docker otherwise creates missing bind sources as `root:root`, and stale
dangling symlinks break permission or mount setup. It only prepares the external
`VSS_DATA_DIR`, never the repository tree.

## Check and create

From the repository root:

```bash
REPO="$(git rev-parse --show-toplevel)"
BUILD_DIR="$REPO/_builds/<name>"
ENV_FILE="$BUILD_DIR/override.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing build override: $ENV_FILE" >&2
  exit 1
fi
if [ "$(grep -c '^VSS_DATA_DIR=' "$ENV_FILE")" -ne 1 ]; then
  echo "override.env must contain exactly one VSS_DATA_DIR" >&2
  exit 1
fi
if [ "$(grep -c '^COMPOSE_PROFILES=' "$ENV_FILE")" -ne 1 ]; then
  echo "override.env must contain exactly one COMPOSE_PROFILES" >&2
  exit 1
fi

DATA="$(sed -n 's/^VSS_DATA_DIR=//p' "$ENV_FILE")"
DATA="${DATA#\"}"
DATA="${DATA%\"}"
DATA="${DATA#\'}"
DATA="${DATA%\'}"
COMPOSE_PROFILES="$(sed -n 's/^COMPOSE_PROFILES=//p' "$ENV_FILE")"
COMPOSE_PROFILES="${COMPOSE_PROFILES#\"}"
COMPOSE_PROFILES="${COMPOSE_PROFILES%\"}"
COMPOSE_PROFILES="${COMPOSE_PROFILES#\'}"
COMPOSE_PROFILES="${COMPOSE_PROFILES%\'}"

case "$DATA" in
  /*) ;;
  *) echo "VSS_DATA_DIR must be one absolute path: $DATA" >&2; exit 1 ;;
esac

if [ -L "$DATA" ] && [ ! -e "$DATA" ]; then
  echo "VSS_DATA_DIR is a dangling symlink: $DATA" >&2
  exit 1
fi
mkdir -p "$DATA"

required=(
  data_log/analytics_cache
  data_log/calibration_toolkit
  data_log/elastic/data
  data_log/elastic/logs
  data_log/kafka
  data_log/redis/data
  data_log/redis/log
  data_log/vss_video_analytics_api
  data_log/vst/clip_storage
  data_log/nvstreamer/vst_data
  agent_eval/dataset
  agent_eval/results
  models
)

case ",$COMPOSE_PROFILES," in
  *,nvstreamer-alerts,*)
    required+=(videos/dev-profile-alerts)
    ;;
esac
case ",$COMPOSE_PROFILES," in
  *,perception-alerts,*)
    required+=(
      models/rtdetr-its
      models/gdino
    )
    ;;
esac
case ",$COMPOSE_PROFILES," in
  *,nvstreamer-lvs,*) required+=(videos/dev-profile-lvs) ;;
esac

# Warehouse perception writes its detector ONNX into models/ at ds-start phase 0.
case ",$COMPOSE_PROFILES," in
  *,perception-2d,*|*,perception-3d,*)
    required+=(models)
    ;;
esac

broken_links=()
while IFS= read -r -d '' candidate; do
  [ -e "$candidate" ] || broken_links+=("$candidate")
done < <(find "$DATA" -type l -print0)
if [ "${#broken_links[@]}" -gt 0 ]; then
  printf 'Dangling symlink under VSS_DATA_DIR: %s\n' "${broken_links[@]}" >&2
  echo "Remove or repair each link before deployment." >&2
  exit 1
fi

for relative_path in "${required[@]}"; do
  path="$DATA/$relative_path"
  if [ -e "$path" ] && [ ! -d "$path" ]; then
    echo "Required data path is not a directory: $path" >&2
    exit 1
  fi
  mkdir -p "$path"
done

# Containers use different UIDs. Change only the shared data roots; never
# recursively chown VSS_DATA_DIR to the host user.
chmod -R a+rwx "$DATA/data_log" "$DATA/agent_eval" "$DATA/models"
[ ! -d "$DATA/videos" ] || chmod -R a+rwx "$DATA/videos"

for relative_path in "${required[@]}"; do
  path="$DATA/$relative_path"
  if [ ! -d "$path" ] || [ ! -w "$path" ] || [ ! -x "$path" ]; then
    echo "Required data directory is not writable and traversable: $path" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Warehouse only. This gate CREATES scaffolding, but warehouse also CONSUMES a
# supplied app-data bundle that cannot be created. Deploying without it yields
# 0 streams with every container healthy, so fail here rather than at runtime.
# ---------------------------------------------------------------------------
case ",$COMPOSE_PROFILES," in
  # Every warehouse mode, not just 2d/3d: mv3dt and auto-calibration consume the
  # same bundle. Enumerated rather than globbed -- nvstreamer-alerts/-base/-lvs/
  # -video are developer profiles with a different data layout and no playback/.
  *,nvstreamer-2d,*|*,nvstreamer-3d,*|*,nvstreamer-mv3dt,*|*,nvstreamer-amc,*)
    DATASET="$(sed -n 's/^SAMPLE_VIDEO_DATASET=//p' "$ENV_FILE" | tr -d '"')"
    wh_fail=0
    for d in videos models playback data_log; do
      [ -d "$DATA/$d" ] || { echo "MISSING: \$VSS_DATA_DIR/$d" >&2; wh_fail=1; }
    done
    if [ -z "$(ls -A "$DATA/videos/$DATASET" 2>/dev/null)" ]; then
      echo "MISSING or EMPTY: \$VSS_DATA_DIR/videos/$DATASET" >&2
      wh_fail=1
    fi
    if [ "$wh_fail" -ne 0 ]; then
      echo "STOP — do not deploy. \$VSS_DATA_DIR must point at the extracted bundle" >&2
      echo "(<extract>/vss-warehouse-app-data), not a directory you created." >&2
      echo "See 'Warehouse app data' below to choose a source, then re-run this gate." >&2
      exit 1
    fi
    ;;
esac
```

Do not silently ignore dangling symlinks. A permission walker may skip one in
best-effort mode, but the stale path can still break a later bind mount,
cleanup, or deployment.

## RT-CV model contents

This gate creates the model directories (including `models/rtdetr-its` and
`models/gdino` for `perception-alerts`) and makes them world-writable — the
RT-CV container runs as a non-matching UID and writes generated TensorRT
`.engine` files back into this tree, so the directories must stay `a+rwx`. The
gate does **not** download any model: when the build carries an RT-CV perception
key, the RT-CV container downloads the detector ONNX (and the Search vision
encoder) at first boot (ds-start phase 0) from its mounted `models-download.json`
into this tree and sets their file permissions itself. No host-side staging is
required.

## Existing PostgreSQL failure

If `vss-vios-postgres` already reports corrupted or stale PGDATA, stop the
stack and remove only its resolved Compose volume:

```bash
docker logs vss-vios-postgres
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-vss}"
docker compose -p "${COMPOSE_PROJECT_NAME}" -f "$BUILD_DIR/resolved.yml" down
docker volume ls -q \
  --filter "label=com.docker.compose.project=${COMPOSE_PROJECT_NAME}" \
  | grep 'vios_pg_data$' \
  | xargs -r docker volume rm
```

Do not recursively `chown` the data root and do not delete unrelated Docker
volumes.


## Warehouse app data — check, never create

For `warehouse`, **`${VSS_DATA_DIR}` is not a directory you create — it *is* the
extracted app-data bundle.** Point it at `<extract>/vss-warehouse-app-data`, the
inner directory holding `videos/`, `playback/`, and `data_log/`.

The whole bundle is a **read-only input**: `mkdir` cannot substitute for missing
content, and an empty `videos/` turns a clear "no app data" failure into a
zero-streams symptom. Treat it as a **blocking presence check**, not part of the
`required=()` list above — run it *before* any `docker compose up`.

When `COMPOSE_PROFILES` contains a warehouse nvstreamer key — `nvstreamer-2d`,
`nvstreamer-3d`, `nvstreamer-mv3dt` or `nvstreamer-amc` (auto-calibration) —
verify before deploying:

```bash
DATASET="$(sed -n 's/^SAMPLE_VIDEO_DATASET=//p' "$BUILD_DIR/override.env" | tr -d '"')"

fail=0
# VSS_DATA_DIR must be the extracted bundle, not a directory you just made.
for d in videos models playback data_log; do
  [ -d "$VSS_DATA_DIR/$d" ] || { echo "MISSING: \$VSS_DATA_DIR/$d" >&2; fail=1; }
done
# The selected dataset must actually carry video files.
if [ -z "$(ls -A "$VSS_DATA_DIR/videos/$DATASET" 2>/dev/null)" ]; then
  echo "MISSING or EMPTY: \$VSS_DATA_DIR/videos/$DATASET" >&2
  fail=1
fi
if [ "$fail" -ne 0 ]; then
  echo "Do NOT deploy. \$VSS_DATA_DIR must point at <extract>/vss-warehouse-app-data." >&2
  echo "Docker creates any missing bind source as root:root, which then needs root to" >&2
  echo "repair, and perception reports 0 streams with every container Up." >&2
  echo "Pick an app-data source below, then re-run this gate." >&2
  exit 1
fi
```

Ask the user which app-data source they want, and only run the NGC download when
they explicitly choose it:

| Source | `VSS_DATA_DIR` |
|---|---|
| Repo `data/` | `<repo>/data` |
| Custom local path | user-provided |
| NGC `nvidia/vss-warehouse/vss-warehouse-app-data:3.3.0` | `<extract>/vss-warehouse-app-data` — the **inner** directory, the one holding `videos/`, `playback/`, `models/`, `data_log/` |

Calibration is **not** part of `$VSS_DATA_DIR` and needs no staging here — see
[`profiles/warehouse.md`](profiles/warehouse.md).
