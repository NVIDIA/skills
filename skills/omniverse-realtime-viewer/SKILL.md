---
name: omniverse-realtime-viewer
description: "Use as the top-level router for Omniverse Realtime Viewer USD apps, XR delivery, SPG GPU processing, stylized rendering, and focused viewer reference documents."
version: "0.3.0"
license: Apache-2.0
tools:
  - Read
  - Shell
  - Write
compatibility: >
  Orchestrator skill. Downstream focused references may require NVIDIA GPUs, ovrtx,
  ovstream, ovui, OpenUSD, Python, Node/React, Tauri, Electron, C++, or cloud
  GPU deployment access depending on the selected viewer path.
metadata:
  author: NVIDIA Omniverse
  tags:
    - omniverse
    - usd
    - viewer
    - workflow
  domain: ai-ml
  languages:
    - python
    - typescript
    - cpp
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Omniverse Realtime Viewer

Build responsive USD viewers by composing OVRTX rendering, OVStage scene state,
and the delivery/UI libraries appropriate to the app. Use this package for
viewer architecture and interaction; use each selected SDK's upstream guidance
for installation and exact release-specific APIs.

## Start With The App

1. When delivery is explicit, select its recipe through
   `references/routing.md` and follow its current phase. Otherwise start with
   `references/usd-viewer-app/README.md`. For an existing app, keep its
   architecture and read only the references for the requested change.
   Broad app, conventions and UI catalogs are lookups as needed.
2. Resolve the selected libraries through `references/dependencies/README.md`.
   Preserve the project's manifest unless an upgrade is requested.
3. Build a working frame and camera interaction in the chosen shell, then add
   the requested capabilities. Read `references/conventions.md` for shared input,
   camera, selection, and protocol conventions.
4. Use `references/viewer-ux-workflow/README.md` for broader interface design.
   Loading, selection feedback, errors, and input responsiveness belong in the
   first working app, alongside rendering.
5. Validate the actual delivery path with `references/validation.md`. A browser
   screenshot does not validate a native window or a different transport.

## Architecture

| Owner | Responsibility |
|---|---|
| Viewer runtime | Own the live OVStage, publication clock, OVRTX renderer, scene changes, and render stepping. |
| UI and input adapter | Display frames and copied app state; send user intent to the runtime owner. |
| Storage/query/simulation workers | Perform their assigned work and return results through the selected integration boundary. |
| Delivery adapter | Present locally or stream through ovstream; respect frame ownership and backpressure. |

Use OVRTX for USD/3D rendering. Browser apps display its WebRTC video; local apps
present its output through their native shell. A missing GPU means rendering is
unverified, not a reason to substitute a browser 3D renderer.

Keep viewer cameras, settings, effects, and transient edits in app-owned session
or runtime state so the source USD stays intact. Serialize scene mutation and
render stepping through the runtime owner. Read
`references/ovstage-ovrtx-integration/README.md` for that loop, then the population
or data-plane references when implementing scene loading or attribute writes.

Keep expensive loading and metadata work off the UI and render threads. Keep
transport details behind a delivery adapter so camera, selection, properties,
and loading state can be reused across local and streamed shells.

## Add Capabilities As Needed

Use `references/routing.md` to select camera/picking, scene management, render
settings, cloud assets, simulation, editor tools, deployment, or CAE/CFD
visualization. Read the selected capability's implementation flow and its linked
contract; the full reference catalog is not a prerequisite for a first frame.

For simulation, use the routing reference's Simulation Ecosystem section plus
`references/realtime-sim-integration/README.md`. For UI-driven forces or control,
add `references/interaction-round-trip/README.md`.

For example scenes, use `references/usd-sample-data/README.md`. When a runtime
failure needs diagnosis, use `references/troubleshooting/README.md` and the
selected SDK's matching release guidance before changing the architecture.

## Solid Modeling: Upstream First

For usd-brep, exact BRep, solid modeling, kernel failures, CAD topology, or
BrepArray questions, first read `references/dependencies/README.md` → USD BRep. Follow the
owning repository's current agent guidance, skills, operation docs, and package
API before selecting local integration references. Discover the upstream router
rather than assuming skill names or directory layouts remain fixed. Upstream
owns modeling semantics, schema rules, tuning, and diagnostics; this package
owns OVRTX display, worker handoff, selection mapping, and publication timing.

## GPU Processing And Stylized Viewers

For custom GPU processing, chained outputs and lens sampling, read
`references/spg-processing/README.md`. For toon, ink, technical/X-Ray styles,
geometry strokes or full wireframe, continue to
`references/stylized-visualization/implementation.md`. Its USD/Lua/CUDA starting
node and geometry/depth/composite build sequence work with the selected viewer
recipe; use `references/stylized-visualization/style-recipes.md` for the preset
math and `references/stylized-visualization/README.md` for supporting contracts.
Implement the result inside the existing OVRTX/runtime owner and deliver the
published processed output. These instructions do not require a sample checkout
or a prior stylized-rendering implementation.

For an optional usd-brep geometry source, first read
`references/dependencies/README.md` → USD BRep and the owning repository's
current agent guidance, skills, and operation docs; then read
`references/stylized-visualization/usd-brep-source.md` for the display boundary.

## XR Viewers

For headsets, immersive viewers, or spatial editors, start with
`references/xr-viewer-recipe/README.md`. Resolve SDK setup and contracts through
`references/dependencies/ovxr.md`; the library owns OpenXR lifecycle and frame
submission. For controllers, hands, navigation, or scene tools, add
`references/xr-spatial-input/README.md` and
`references/ovxr-python-interactions/README.md`. For projection-integrated UI,
read `references/ovxr-composition-ui/README.md` and
`references/ovxr-composition-ui-validation/README.md`.
