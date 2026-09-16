# Which `ds-start.sh` runs?

VSS ships more than one `ds-start.sh`. Editing the wrong file is a silent failure: the container keeps the stock detector.

## The three paths

| Host path (under `${VSS_DEPLOY_DIR}`) | Role in stock VSS v3.2.1 Alerts |
|---|---|
| `services/rtvi/rtvi-cv/ds-start.sh` | **Actually runs.** `perception-alerts` `extends` the `perception` service, which bind-mounts this file to `/opt/.../metropolis_perception_app/ds-start.sh`. Unified entrypoint: honors `DS_CONFIG_FILE`, copies `mounted-configs/` → `configs/`, dispatches on `DS_MODEL_FAMILY`. |
| `developer-profiles/dev-profile-alerts/deepstream/init-scripts/ds-start.sh` | **Present but unused by stock compose.** Older Alerts-profile script. Does **not** read `DS_CONFIG_FILE` (`CONFIG_FILE=${1:-...}` only). Stock Alerts compose does **not** remount this over the rtvi-cv bind. |
| In-container `/opt/.../metropolis_perception_app/ds-start.sh` | Always the bind-mount target. Whatever compose mounts there is what runs — image contents do not matter if a host bind is present. |

## What this skill does

This skill customizes the **Alerts profile** copy at
`${VSS_PROFILE_DIR}/deepstream/init-scripts/ds-start.sh`, then **overrides** the
stock rtvi-cv bind mount in `${VSS_PROFILE_DIR}/compose.yml` so
`perception-alerts` runs that profile script instead.

That remount is required. Without Step 6's `ds-start.sh` volume override,
edits under `init-scripts/` never execute.

Trade-off: remounting replaces the unified rtvi-cv entrypoint. Keep any
hardware-profile / GDINO / batch-size behavior you still need, or port the YOLO
block into `services/rtvi/rtvi-cv/ds-start.sh` (`start_rtdetr_gdino`) instead and
skip the remount.

## `DS_CONFIG_FILE` pairing

| Entrypoint you run | How config is chosen | YOLO implication |
|---|---|---|
| Stock `services/rtvi/rtvi-cv/ds-start.sh` | Reads `DS_CONFIG_FILE` (Alerts sets `.../configs/run_config-api-rtdetr-protobuf.txt` after copying from `mounted-configs/`). | Add YOLO under `start_rtdetr_gdino`; ensure `yolov11.txt` is copied into `configs/` with the other mounts. |
| Profile `init-scripts/ds-start.sh` (this skill) | Ignores `DS_CONFIG_FILE` unless you apply the Step 5 `CONFIG_FILE=...` fix. Step 6 points `DS_CONFIG_FILE` at `.../mounted-configs/run_config-api-rtdetr-protobuf.txt` so sed patches the bind-mounted tree where `yolov11.txt` lives. | Do not leave stock `DS_CONFIG_FILE` (`.../configs/...`) if YOLO config only exists under `mounted-configs/`. |

## Quick check

After recreate, confirm which script is active:

```bash
docker compose \
  --env-file developer-profiles/dev-profile-alerts/generated.env \
  exec -T perception-alerts \
  sh -lc 'head -n 5 /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/metropolis_perception_app/ds-start.sh'
```

- Mentions unified / `DS_MODEL_FAMILY` → still the rtvi-cv entrypoint (your profile edits are not mounted).
- Mentions Alerts-style `CONFIG_FILE=${1:-...}` / GDINO branches without `DS_MODEL_FAMILY` → profile script is mounted.
