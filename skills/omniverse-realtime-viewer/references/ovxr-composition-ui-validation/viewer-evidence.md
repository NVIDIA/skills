<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# XR viewer evidence

Use these gates for the requested headset, simulation and mirror features.
Combine them with [spatial UI validation](README.md) and the generated app's
existing report format. No bundled harness or sample checkout is required.

- XR proof: the OpenXR probe output showing `instance_created` and a runtime
  name, the negotiated system name/view count/swapchain size, a photo or client
  capture of the stereo view on the headset, the steady frame rate at the
  client's negotiated resolution, and one round-trip data-channel message with
  its reply. Note explicitly which runtime `XR_RUNTIME_JSON` selected, and the
  matched ovrtx/ovstage/ovxr build identities and selected OpenXR/CloudXR runtime
  version.
- Tracked-hand physics proof: record one continuous view from open hand through
  approach, contact, close, lift, hold, open/release, gravity-driven fall, floor
  contact, and settle. Keep the hand, target, travel, and floor visible; do not
  cut between grasp and release. Correlate the recording with this causal chain:
  actual native OpenXR joint samples or browser `XRFrame.getJointPose()` samples
  and transmitted byte counts; server
  receipt; coherent kinematic-proxy update; named contact pairs or native
  constraint state; bounded target transform and velocity; physics publication
  ordinal; renderer-consumed ordinal at or after that publication; successful
  submitted frame; and nonblank changing client pixels. Then prove release,
  free-body gravity, floor contact, and settling through newer ordinals and
  pixels. Capture milestone stills and a compact animation for open, first
  contact, grasp confirmed, lifted, held, released, first floor contact, and
  settled. Record lift height, held relative-pose slip, release linear/angular
  velocity, drop height, and settling thresholds. Run repeated trials with
  small approach and timing jitter and report the pass rate. Include negative
  controls: open-hand near pass, close away from the target, insufficient
  contact, explicit release, input loss, stale input, and muted competing tools.
  Record source-frame timestamps and the resulting capture duration. If the
  review artifact is packaged as an animated WebP or GIF, report its playback
  frame rate separately from the measured renderer, physics, and decoded-frame
  cadence. Encoder playback rate is presentation pacing, not performance
  evidence. Preserve the source capture duration by default; if the requested
  playback rate instead shortens time or requires resampling, declare that
  presentation policy and disclose every dropped, duplicated, or synthesized
  source frame.
  Telemetry alone, a synthetic grasp event, a changed USD value, or proxy motion
  without correlated target pixels does not pass.
- XR desktop-mirror proof, when a spectator or mirror window is exposed: the
  capture's width/height and warp state read back from the **runtime**, not
  asserted from the descriptor; verify the selected runtime's effective capture
  extents and warp/unwarp behavior; a frame captured
  from the desktop window in each phase -- no headset, and headset connected --
  showing correct colours in both; evidence that a client disconnect returns the
  window to the interactive product without restarting the process; and a
  measured publish rate alongside the requested cap. Record the process count and
  VRAM to show one renderer served both phases rather than a second `ovrtx`
  session appearing.

Run intrusive proof harnesses on explicit, unused ports and in a dedicated
artifact directory. Snapshot existing listeners and process ownership before
launch, track only the child processes the harness creates, and clean those
children and temporary artifacts in a `finally`/trap path. Treat an existing
renderer, stage, shared-memory service, or XR runtime as external: do not use a
broad process-name kill or restart it to make a proof run reproducible.
