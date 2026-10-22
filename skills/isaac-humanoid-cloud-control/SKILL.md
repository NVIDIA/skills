---
name: isaac-humanoid-cloud-control
description: Run streamed Unitree G1 cloud-navigation demos in Isaac Sim with guarded gantry release. Use for demo, destination, Mission Control, or cleanup requests.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: "yuzho <yuzho@nvidia.com>"
  tags:
    - isaac-ros
    - isaac-sim
    - humanoid-navigation
    - mission-control
---

# Isaac Humanoid Cloud Control

## Purpose

Run the public NVIDIA Isaac ROS release-4.6 Unitree G1 cloud-control workflow as
an ordered integration. Default to the Isaac Sim full streaming app running
headless, an interactive browser WebRTC view, and a navigation mission. Do not
silently fall back to a local Isaac Sim window. Read
[the release-4.6 runbook](references/release-4.6.md) before executing the
workflow.

## Inputs

Required inputs are the request mode (bring-up, end-to-end demo, destination, or
cleanup) and, before mission submission, an operator-supplied destination and
movement approval. A selected suggestion supplies both for simulation. Remote
missions also require confirmation that the rendered viewport is visible.

Optional or discoverable inputs are robot identity (`g1_0` by default), hosts,
spawn pose, map, and installation paths. Resolve them from active demo state,
explicit operator choices, live observations, discovered installations, then
packaged defaults. Never let a default override an explicit safety boundary.

## Prerequisites

Require an x86_64 NVIDIA GPU with NVENC, Isaac Sim 6.0, the canonical Docker-mode
Isaac ROS workspace, NVIDIA's official web viewer, and the release-4.6 cloud
images. Use [the bootstrap gate](references/bootstrap.md) on a fresh host.
If the CLI or its Docker environment is missing, follow
[CLI installation and Docker setup](references/isaac-ros-setup.md), then rerun
preflight. Missing CLI installation is a recovery step, not a terminal blocker.

## Preserve the target boundary

- Require an NVIDIA GPU with NVENC for the default livestream. Preflight TCP
  49100 for WebRTC signaling, UDP 47998 for media, and TCP 8210 for the browser
  viewer. Keep the endpoints on a private or otherwise trusted network; they
  provide no authentication or encryption by themselves.
- Use the owning cloud skill's broker authentication configuration and supply
  matching credentials to the G1 client. Keep broker ports on the host or a
  trusted private network; authentication alone does not provide TLS.
- Enable Isaac Sim's Python server only for live scene inventory. Keep it on
  loopback at 127.0.0.1:8226; it executes Python and must never be published,
  proxied, or exposed to the browser network.
- Use an x86_64 NVIDIA GPU host and Isaac Sim 6.0 for the sample simulation.
- Support navigation through cloud control in Isaac Sim.
- For Isaac Sim-only navigation, use `enable_camera_localization:=false` and
  `use_static_tf:=true`. Do not launch, configure, validate, or require Visual
  SLAM, visual global localization, RealSense, D435, or camera-localization
  topics; they are outside this simulated-demo path.
- Do not submit humanoid_manipulation objectives to Isaac Sim. GR00T cloud
  manipulation is real-hardware-only in this release.
- Treat real G1 operation as a separate safety-gated path. Require the Unitree
  network setup, D435 localization, CUDA MPS, real-time scheduling, and the
  lower-body-then-upper-body controller enable sequence from the upstream
  GR00T guide. Never infer permission to move real hardware.
- Submit a navigation mission only after the operator has supplied the target
  and explicitly approved the previewed mission payload. Verify occupancy,
  robot-footprint clearance, and Nav2 reachability before submission. A target
  selection alone is insufficient; changed routes require a new approved preview.
- Before invoking [the guarded helper](scripts/guarded_gantry_release.py),
  obtain explicit operator approval for that virtual gantry release attempt.
  A broad demo request is insufficient. State the simulation target and action;
  ask “Release the Isaac Sim virtual gantry now?” unless the operator already
  explicitly approved this specific attempt. Pass the live simulation identity
  interlock and unsupported stability check before offering destinations.
  A retry needs separate explicit approval after reporting the failed attempt
  and confirmed gantry re-engagement. Never apply this automation to hardware.
  Honor an operator's request to use the viewport **G** action instead.
- Keep one robot identity everywhere. Default to g1_0; apply the same value to
  Mission Control, serial_number, MQTT topics, mission payloads, and lifecycle
  verification.
- Use one dedicated, initially empty `ROS_DOMAIN_ID` for Isaac Sim and every
  ROS shell in a demo. Set localhost-only discovery for all demo processes as
  specified in the ordered startup. Confirm that the clock and robot-description
  ROS topics are
  absent on that domain before startup. This prevents transient-local samples
  from a previous crashed run from racing the new controller manager.
- The bundled [occupancy-grid PNG](assets/carter_warehouse_navigation.png)
  supplies obstacle and free-space pixels for the navigation map; its
  [YAML metadata](assets/carter_warehouse_navigation.yaml) defines the map scale
  and origin. The cloud workspace procedure copies this pair into the runtime
  workspace. These are map data, not executable instructions.
- Keep one occupancy map everywhere. The robot launch and Mission Control must
  consume the same PNG and YAML pair.
- Ensure the map YAML and referenced image are readable at the same absolute
  path inside the activated Isaac ROS environment; host-only paths are
  insufficient.
- Create a dedicated runtime `MC_WORKSPACE` using
  [the cloud workspace procedure](references/cloud-workspace.md). Generate map
  and fleet configuration through the owning adapters. Resolve Compose, image
  references, and supporting resources from the pinned upstream skills; never
  modify their checkouts or store generated configuration in this package.

## Resolve upstream scripts

Set `SKILL_DIR` to the absolute directory containing this loaded skill manifest,
using the skill loader's location, and preserve it in every shell invocation.
All bundled paths are relative to it; never derive them from the working directory.
Read [the dependency workflow](references/dependencies.md) before calling another
skill's scripts. It follows the Mission Control showcase composition model:
pinned external checkouts, a persisted manifest, and parent-owned adapters.
Read each pinned upstream instruction manifest from the resolved dependency
record and follow its gates. The exact source paths and commits are disclosed in
[the dependency lock](upstream-versions.lock.json).
Adapter contracts: [cloud](references/bring-up-cloud-stack/README.md),
[map](references/change-map/README.md),
[fleet](references/change-fleet-composition/README.md),
[scene](references/isaac-sim-remote/README.md), and
[installation handoff](references/isaac-sim-installation/README.md), and
[ROS activation](references/isaac-ros-activate/README.md), and
[mission verification](references/submit-and-monitor-mission/README.md).

Resolve dependencies before host preflight; resolution does not launch services:

```bash
python3 "${SKILL_DIR}/scripts/doctor.py" dependencies
```

Normal invocation never downloads missing skills. Explicit dependency preparation
uses `dependencies --prepare` only after authorization as described in the
reference. Use the linked adapters instead of guessed upstream script paths.

## Compose the owning skills

Follow [owning-skill responsibilities](references/dependencies.md#owning-skill-responsibilities)
for each handoff; the upstream skill remains authoritative.

## Available Scripts

| Script | Purpose | Main arguments |
| --- | --- | --- |
| `scripts/doctor.py` | Resolve pinned upstream scripts and write their manifest | `dependencies`; optional `--prepare`, `--check-only`, `--json` |
| [scripts/prepare_cli_repository.py](scripts/prepare_cli_repository.py) | Stage and verify the public APT key and fixed repository source without root | No arguments |
| `scripts/preflight.py` | Discover and validate host, cloud, streaming, and scene-server readiness | `--map`; optional installation paths and `--check-*` gates |
| `scripts/guarded_gantry_release.py` | Set AGILE, verify stability, release the simulated gantry, and verify unsupported balance | `--allow-simulation-gantry-release`; stability limits and durations |
| `scripts/check_runtime.py` | Verify ROS clock, sensors, TF, and Nav2 lifecycle | `--base-frame`, `--timeout` |
| `scripts/scene_inventory.py` | Inventory live-stage objects without modifying USD | injected `output_path`, `max_objects` |
| `scripts/suggest_navigation_missions.py` | Produce occupancy- and footprint-safe object approach poses | `--inventory`, `--map`, optional robot pose |
| `scripts/submit_navigation_mission.py` | Preview or submit a validated Mission Control route | `--host`, `--robot`, waypoints, `--validated-free-space`, `--submit` |
| `scripts/play_viewer.py` | Focus the browser viewer and send Play only as a documented fallback | `--url`, browser and debug options |

Use `run_script()` for these helpers when the runtime exposes it; otherwise run
the bundled helper by its resolved absolute path without reconstructing it.

## Fresh-machine bootstrap gate

For a fresh host, read [the bootstrap gate](references/bootstrap.md), run its
read-only discovery first, and resolve failures in the documented ownership
order. Continue only after preflight has no blocking failures. Report the exact
failed observation rather than claiming a live view is available.

## Instructions

Read and follow [the ordered startup and mission workflow](references/ordered-startup.md)
in full before executing. It contains the exact commands and phase gates:

1. Discover host paths and inspect existing state read-only; resolve upstream
   dependencies before host preflight.
2. Prepare the release package and Deploy extension, and preview and apply requested
   map and fleet changes through the owning adapters.
3. Start cloud services through the cloud adapter, with the runtime workspace
   and no generic robot simulator.
4. Prepare the official browser viewer and publish its reachable URL.
5. Launch rendered headless Isaac Sim paused, stage the activated G1 ROS launch,
   then start the timeline and ROS stack in the required order.
6. Verify AGILE and run the guarded simulation-only gantry release. Require
   unsupported balance before inventorying or offering destinations.
7. Verify runtime and cloud readiness; inventory the live scene through the remote
   adapter and validate navigation suggestions with occupancy and Nav2.
8. After destination approval, validate and submit once, keep the live view
   visible, and apply submit-and-monitor-mission until completion.

The detailed workflow is mandatory; this index does not replace its commands,
startup timing, retry limits, or authorization boundaries.

## Troubleshooting

Apply the [readiness and troubleshooting checklist](references/troubleshooting.md)
before submission and when diagnosing a failed gate.

## Examples

- “Run an end-to-end humanoid cloud-control Isaac Sim demo” starts setup;
  obtain explicit virtual gantry release approval before running the helper.
- “Destination 1: forklift” after live suggestions selects the target.
  Validate the route, preview the mission payload, and obtain submission approval.
- “Bring up the stack for inspection” does not authorize gantry release or a
  mission.
- “Clean up the stack” authorizes teardown of demo-owned resources only.
- “Use GR00T to manipulate a box in Isaac Sim” is outside this release; offer
  supported navigation without inventing manipulation support.

## Output Format

Report these fields in order when applicable:

1. **State** — current phase or first failed gate with measured evidence.
2. **Viewer** — clickable browser URL while live and visibility status.
3. **Mission** — ID, terminal state and timestamp, initial, moving, and final poses,
   Dispatch and Nav2 evidence, and final robot health.
4. **Cleanup** — demo-owned resources removed, reusable resources preserved,
   and confirmation that the previous viewer URL is unavailable.

## Limitations

Support cloud-controlled navigation only in Isaac Sim release 4.6. Do not claim
simulated GR00T manipulation, operate real hardware, expose unauthenticated demo
endpoints publicly, replace missing owning skills, or delete shared data. Treat
camera localization and real-G1 controller sequencing as separate workflows.

## Stop behavior

Keep the completed demo running by default. After a successful mission, leave
the robot launch, headless Isaac Sim, browser viewer, and cloud services alive
so the user can inspect the result or request another mission. Keep publishing
the clickable live-stream URL in the handoff and offer an explicit stack
cleanup. Do not infer cleanup from mission completion, live-view confirmation,
the end of the current turn, or a request that names only one finished mission.

Clean up only on an explicit teardown request. Then read and apply
[the teardown procedure](references/teardown.md). Remove only resources owned by
the demo run, preserve shared data and the reusable Isaac ROS container, verify
the demo endpoints are closed, and invalidate the prior viewer URL.
