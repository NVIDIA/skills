---
name: "deepstream-run-mv3dt"
description: "Run and operate the DeepStream Multi-View 3D Tracking reference app, also known as MV3DT. Use when the user asks to set up prerequisites, run shipped MV3DT samples, run Multi-View 3D Tracking on custom synchronized MP4 datasets, import camera calibration, delegate missing calibration to AutoMagicCalib, inspect OSD or BEV visualization, consume MV3DT Kafka metadata, or clean up MV3DT run state in the DeepStream MV3DT app directory."
metadata:
  author: "Shubham Agrawal <shuagrawal@nvidia.com>"
  tags: [deepstream, mv3dt, tracking, multi-view, 3d, kafka]
  languages: [bash, python]
  domain: computer-vision
owner: "NVIDIA CORPORATION"
service: "deepstream-tracker-3d-multi-view"
version: "1.0.0"
reviewed: "2026-06-15"
license: CC-BY-4.0 AND Apache-2.0
---

# Skill: Run DeepStream MV3DT

## When to Use This Skill

Activate this skill when the user wants to set up, run, verify, or debug the DeepStream Multi-View 3D Tracking reference app. Typical prompts:

- "set up MV3DT DeepStream"
- "run the 4-camera MV3DT sample"
- "run the 12-camera MV3DT sample"
- "run MV3DT on my synchronized MP4s"
- "I have videos but no calibration; calibrate and run MV3DT"
- "show the BEV visualizer or Kafka metadata"
- "stop MV3DT" / "clean up MV3DT" / "tear down MV3DT"

Do not use this skill for single-view 3D tracking, generic DeepStream app development, or live-stream onboarding unless the user explicitly maps that work to this repo's MV3DT pipeline.

## Examples

- "Deploy the DeepStream MV3DT 4-camera sample and show the OSD and BEV windows."
- "Run the MV3DT 12-camera sample headlessly with RTDETR and save videos."
- "Run MV3DT on synchronized MP4s under /data/mv3dt-demo using PeopleNetTransformer."
- "My custom MV3DT videos do not have calibration; use AutoMagicCalib, then run MV3DT."

## Overview

Operate the Multi-View 3D Tracking reference app in `DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view` using the DeepStream Container path. The skill supports setup, shipped sample runs, custom synchronized MP4 datasets, calibration handoff to AutoMagicCalib, display/headless execution, OSD/BEV outputs, and Kafka metadata inspection.

## Prerequisites

- MV3DT reference app directory on disk under `DeepStream/src/apps/reference_apps/deepstream-tracker-3d-multi-view`; if it is absent, ask before cloning the public DeepStream repo
- Docker with NVIDIA GPU support
- DeepStream container image access
- MV3DT sample datasets, models, custom parser libraries, Kafka, Mosquitto, and `mv3dt_venv` prepared by the repo setup script
- Working X11/VNC display for live OSD/BEV windows, or the saved-output headless path for tiled DeepStream MP4 plus Kafka-derived BEV MP4

## Instructions

### Step 0: Resolve MV3DT App Checkout

The skill can be installed outside the DeepStream repo. Resolve `REPO_ROOT` to the MV3DT app directory, not necessarily the Git top-level directory.

```bash
MV3DT_APP_SUBDIR="src/apps/reference_apps/deepstream-tracker-3d-multi-view"

is_mv3dt_app_dir() {
  test -f "$1/README.md" || return 1
  test -d "$1/config_templates" || return 1
  test -d "$1/scripts" || return 1
  grep -q "Multi-View 3D Tracking" "$1/README.md"
}

GIT_TOP="$(git rev-parse --show-toplevel 2>/dev/null || true)"
CANDIDATES=()
if [ -n "${MV3DT_REPO_ROOT:-}" ]; then CANDIDATES+=("${MV3DT_REPO_ROOT}"); fi
if [ -n "${DEEPSTREAM_REPO_ROOT:-}" ]; then CANDIDATES+=("${DEEPSTREAM_REPO_ROOT}/${MV3DT_APP_SUBDIR}"); fi
CANDIDATES+=("${PWD}")
if [ -n "${GIT_TOP}" ]; then
  CANDIDATES+=("${GIT_TOP}" "${GIT_TOP}/${MV3DT_APP_SUBDIR}")
fi
CANDIDATES+=("${HOME}/DeepStream/${MV3DT_APP_SUBDIR}" "${HOME}/deepstream/${MV3DT_APP_SUBDIR}")

REPO_ROOT=""
for candidate in "${CANDIDATES[@]}"; do
  if [ -n "$candidate" ] && is_mv3dt_app_dir "$candidate"; then
    REPO_ROOT="$(cd "$candidate" && pwd)"
    break
  fi
done

if [ -z "${REPO_ROOT}" ]; then
  cat <<'EOF'
ERROR: MV3DT reference app directory was not found.
Set MV3DT_REPO_ROOT to an existing deepstream-tracker-3d-multi-view app directory, or ask the user to approve cloning the public DeepStream repo and then run:

  DEEPSTREAM_REPO_ROOT="${DEEPSTREAM_REPO_ROOT:-$HOME/DeepStream}"
  git clone https://github.com/NVIDIA/DeepStream.git "$DEEPSTREAM_REPO_ROOT"
  export MV3DT_REPO_ROOT="$DEEPSTREAM_REPO_ROOT/src/apps/reference_apps/deepstream-tracker-3d-multi-view"

Do not clone silently.
EOF
  exit 1
fi

cd "${REPO_ROOT}"
export REPO_ROOT MV3DT_REPO_ROOT="${REPO_ROOT}"
```

If the app directory cannot be resolved, ask the user for an existing checkout path or for approval to clone `https://github.com/NVIDIA/DeepStream`. Do not clone silently.

### Step 1: Select The Primary Workflow

Load exactly one primary reference for the user's current request:

| User intent | Reference |
|---|---|
| Install, prepare, or verify prerequisites | `references/setup.md` |
| Run bundled 4-camera or 12-camera sample | `references/sample-run.md` |
| Run custom synchronized MP4s | `references/custom-dataset.md` |
| Missing custom calibration | `references/amc-calibration-handoff.md`, then return to `references/custom-dataset.md` |
| View OSD, BEV, screenshots, recordings, or Kafka metadata | `references/visualization-metadata.md` |
| Stop a run, clean generated artifacts, or stop prerequisite services | `references/setup.md` |

If setup, datasets, models, Kafka, Mosquitto, Docker GPU runtime, or the Python venv are missing, load `references/setup.md` before continuing to the user's original workflow.

When `sample-run.md` or `custom-dataset.md` needs to regenerate DeepStream configs, load `references/generate-configs.md` as the canonical shared config-generation reference rather than duplicating the shell logic.

### Step 2: Follow The Run Stages

For every run, use this stage order:

| Stage | Action |
|---|---|
| Validate | Check app directory, prerequisites, dataset shape, display/headless mode, Docker GPU support, and output-directory writability. |
| Prepare | State the selected sample/custom dataset, detector, run mode, output directory, expected output surfaces, and the Docker security decision for `--privileged --net=host` before launch. |
| Execute | Generate configs, start offline BEV capture first in headless mode or when saved BEV MP4 is explicitly requested, then run DeepStream or delegate missing calibration to AutoMagicCalib. |
| Verify | Confirm functional readiness with `App run successful`, fresh MP4 artifacts when file output is enabled, Kafka offsets/messages, and BEV message/frame counts when BEV MP4 capture runs. |
| Report | Summarize selected options, generated files, artifact paths, file sizes/counts, and any skipped or failed output surface; after a default 4-camera sample run, mention that the 12-camera sample is also available as a follow-up. |

### Step 3: Apply Defaults Explicitly

- Runtime path: DeepStream Container.
- Display path: if a working X11/VNC display is available, use the repo quick-start path with OSD and BEV windows. The OSD window does not save MP4 by default; when the user explicitly asks to save output in display mode, regenerate configs with both `--enable-osd` and `--enable-file-output` plus `--enable-msg-broker`.
- Headless path: if no working display is available, use saved outputs by default. Generate configs with `--enable-file-output` and `--enable-msg-broker`; start offline BEV capture before DeepStream so the run produces both the tiled DeepStream MP4 and the Kafka-derived BEV MP4.
- Sample dataset: support both shipped 4-camera and 12-camera datasets. If the user does not specify a sample, run the 4-camera sample first, then mention in the final report that the 12-camera sample is also available and can be run next.
- Detector: `PeopleNetTransformer` unless the user asks for `RTDETR` or `PeopleNet2.6.3`; carry the selected `DETECTOR_MODEL` through config generation and AMC `camInfo` modelInfo normalization. For custom calibration handoff, ask the user to choose the AutoMagicCalib detector instead of silently defaulting.
- Custom data source: synchronized MP4 files. Live-stream handling is outside this first-release skill.
- Calibration handoff: use standalone AutoMagicCalib skills instead of duplicating their setup or API workflow.

### Step 4: Preserve Idempotency And User Data

- Readiness checks are safe to rerun.
- Setup may install packages, pull containers, download models, and start services; ask first.
- Config generation and DeepStream runs update generated files under `EXPERIMENT_DIR`; record `RUN_STARTED_AT` and do not report old artifacts as current-run success.
- Normal teardown stops only current run processes and leaves Kafka, Mosquitto, models, datasets, and generated artifacts in place unless the user explicitly asks to stop services or delete files.
- Custom datasets are user data. Copy missing calibration-format variants by default, and ask before renaming, overwriting, clearing, or deleting dataset files.

## Success Criteria

- Prerequisite checks pass or the missing prerequisite is reported with a narrow next step.
- DeepStream run eventually prints `App run successful`.
- Display mode shows the DeepStream OSD grid and live BEV visualizer by default.
- Headless mode produces a fresh `${EXPERIMENT_DIR}/outVideos/tiled_display_raw.mp4` and attempts BEV MP4 capture by default.
- Kafka topic `mv3dt` receives current-run protobuf metadata when message broker output is enabled.
- BEV MP4 is reported as successful only when the separate BEV capture process produced nonzero messages and frames.

## Key Output

- Generated configs: `${EXPERIMENT_DIR}/config_deepstream.txt`, `${EXPERIMENT_DIR}/config_tracker.yml`, `${EXPERIMENT_DIR}/config_msgconv.txt`
- Saved DeepStream tiled MP4 when file output is enabled: `${EXPERIMENT_DIR}/outVideos/tiled_display_raw.mp4`
- Saved BEV MP4 when offline capture is used: `${EXPERIMENT_DIR}/bev_outputs/trajectory_video_<timestamp>.mp4`
- Kafka topic: `mv3dt`
- Sample output roots: `${REPO_ROOT}/experiments/deepstream/4cam` and `${REPO_ROOT}/experiments/deepstream/12cam`

## Troubleshooting

| Issue | First action |
|---|---|
| Setup prerequisites missing | Load `references/setup.md` and run the check-only path before setup. |
| Docker cannot access GPU | Fix NVIDIA Container Toolkit or Docker runtime before launching samples. |
| Display window missing | Check `DISPLAY` and `/tmp/.X11-unix`; use the headless saved-output path when no display is available. |
| DeepStream MP4 missing | Confirm configs were generated with `--enable-file-output` and verify the artifact is newer than `RUN_STARTED_AT`. |
| BEV MP4 missing or zero messages | Start offline BEV capture with `--from-end` before DeepStream, use a long enough `--first-message-timeout`, and verify Kafka offsets move during the run. |
| Kafka client shows no messages | Regenerate configs with `--enable-msg-broker` and verify topic `mv3dt` exists. |
| Custom dataset lacks calibration | Load `references/amc-calibration-handoff.md`; ask detector/settings choices before delegating. |
| Generated files are root-owned | Report the ownership issue and ask before applying a narrow generated-directory permission fix. |

## Safety Notes

- Ask before commands that use `sudo`, install packages, pull containers, download models, start or stop services, change host display access, overwrite dataset files, or clear generated state.
- Treat deleting Kafka, Mosquitto, models, datasets, or experiment outputs as destructive cleanup. Show the exact targets and get explicit confirmation before removing anything.
- Before running `docker run --privileged --net=host`, explicitly state that the container gets broad host, device, network, and mounted-repo access, then get user approval.
- Do not silently clone repositories, change host permissions outside the repo, rename or delete user datasets, or report old artifacts as current-run success.
- Treat custom videos, calibration, saved visualizations, and tracking metadata as potentially sensitive local data. Keep outputs local unless the user explicitly asks to move or share them.
- If permission fixes are needed for generated outputs, propose the narrowest generated-directory-only fix and ask first; never recommend broad world-writable recursive permission changes.

## Related Skills

- `amc-setup-calibration-stack` - Launch the standalone AutoMagicCalib stack when calibration is needed.
- `amc-run-video-calibration` - Generate calibration from synchronized local MP4s before returning to MV3DT.

<!-- signing marker -->
