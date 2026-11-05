<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Validation And Review Evidence

Generated apps should produce enough evidence for a reviewer to confirm that the
viewer works without rerunning every step. Store artifacts with the generated
app, test report, or release notes. Do not commit private scene captures,
customer data, local absolute paths, tokens, environment-specific service URLs, or environment
specific logs.

## Routine Evidence Checklist

Capture enough routine evidence to prove the generated viewer works on its
selected delivery path without rerunning every step. Keep it compact:

- startup and dependency command output,
- first nonblank rendered frame,
- camera orbit, pan, zoom, and fit-to-stage,
- object selection with tree/property panel synchronization,
- scene switching with stale selection and hierarchy cleared,
- render setting or AOV changes when requested,
- shutdown, reconnect, cleanup, or sidecar behavior when relevant.

For browser viewers, prefer Playwright screenshots or app-specific end-to-end
tests. For local, Tauri, Electron, C++, and headless viewers, capture frames
from the same `ovrtx` output path used for display or automation.

Validate requested capabilities on the selected delivery path. Skip LAN and
multi-device checks unless that access is requested; localhost browser delivery
does not by itself require LAN deployment.

## Focused Proof Evidence

Use focused proof only for requested capabilities or demo-critical claims. Do
not wire broad artifact generation into routine CI unless explicitly requested.

- Runtime transform proof: one selected prim, before/after transform or pixels,
  the command/result message, and confirmation the renderer stayed live.
- Pick-effect proof: one selected prim, before/after effect state or pixels, and
  the accepted/rejected capability result.
- Physics proof: one focused MP4 or frame sequence plus a concise report showing
  the actual viewer path, selected prim, child/parent OVPhysX process model,
  OVStage write/consume ordinals, `parent_inprocess_attach_ovstage_used` or `bounded_worker_used`,
  nonzero
  displacement/rotation when motion was requested, and server liveness.
- Continuous-simulation proof: the simulation validation ladder result —
  headless behavior, attach and initial publish, first visible transform,
  reset, stage switch with outstanding work, stale-command rejection, and
  delivery-client status confirmation. Record only sanitized artifacts and the
  app declared generation/phase values.
- Asset-fidelity proof: state whether the rendered geometry is the source
  asset's visual mesh or an intentional primitive/substitute fixture. An
  undeclared substitute is a validation failure.
- Deforming-attribute proof when exposed: record the authored prim, attribute,
  render ordinal, expected changed-pixel region, element-count behavior, and
  runtime versions. Treat topology changes as a separate generation/re-anchor
  test.

For requested spectral/LPE features, capture actual output with a fixture that
exercises the selected paths, effective settings, and the declared display/transfer
path. Separate process preparation from live settings and spectral accuracy claims.
For section caps, include a closed solid, a nested hole, and the declared policy
for open/unsupported geometry; show repeated plane changes and cap removal.
For gizmos, prove parameter/transform change, unchanged camera and selection
during the drag, cancellation/stale-intent rejection, and restored input after
release. For runtime export, reopen the saved result and check a nested world
transform. These are focused checks for requested features, not a new mandatory
matrix for every viewer.

## Report Template

Copy this template into the generated app or test output when the app has no
existing validation format.

````markdown
# Viewer Validation Report

## Metadata

| Field | Value |
|---|---|
| App or repo |  |
| Branch or commit |  |
| Date |  |
| Reviewer |  |
| Delivery path | Browser WebRTC / local Python / Tauri / Electron / C++ / headless |
| Runtime environment |  |
| Scene inputs | Sanitized asset names or fixture IDs only |

## Commands Run

| Step | Command | Result | Artifact |
|---|---|---|---|
| Setup |  | Pass / fail / skipped |  |
| Build |  | Pass / fail / skipped |  |
| Runtime launch |  | Pass / fail / skipped |  |
| Validation |  | Pass / fail / skipped |  |

## Evidence Checklist

| Evidence | Status | Artifact | Notes |
|---|---|---|---|
| Startup and dependency output captured | Pass / fail / skipped |  |  |
| First nonblank rendered frame captured | Pass / fail / skipped |  |  |
| Camera orbit, pan, zoom, and fit-to-stage verified | Pass / fail / skipped |  |  |
| Object selection updates viewport, tree, and property panel | Pass / fail / skipped |  |  |
| Scene switch clears stale selection and refreshes hierarchy | Pass / fail / skipped |  |  |
| Render setting or AOV changes verified when requested | Pass / fail / skipped |  |  |
| Focused proof for requested runtime transform, pick effect, or physics capability | Pass / fail / skipped |  |  |
| Continuous-simulation lifecycle and client-status proof when exposed | Pass / fail / skipped |  |  |
| Rendered geometry is the source visual asset or an explicitly declared substitute | Pass / fail / skipped |  |  |
| Deforming or particle attribute publication is pixel-validated when exposed | Pass / fail / skipped |  |  |
| Shutdown, reconnect, or cleanup behavior verified when relevant | Pass / fail / skipped |  |  |

## Issues And Waivers

| ID | Severity | Summary | Owner | Resolution |
|---|---|---|---|---|
|  |  |  |  |  |

## Result

Overall status: Pass / fail / blocked

Reviewer notes:
````

## Done Criteria

The generated app is ready to share only when:

- the renderer path uses `ovrtx`,
- no browser-side USD/3D fallback is present,
- the app can start from documented commands,
- at least one scene produces a nonblank frame,
- requested interactions are demonstrated with artifacts,
- optional runtime transform, pick-effect, and physics claims have focused proof
  when exposed,
- failure cases and runtime requirements are documented,
- validation artifacts are sanitized.

For headset, tracked-hand physics and desktop-mirror evidence, follow
[XR viewer evidence](ovxr-composition-ui-validation/viewer-evidence.md).
