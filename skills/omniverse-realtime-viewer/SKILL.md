---
name: omniverse-realtime-viewer
description: "Use as the top-level router for Omniverse Realtime Viewer USD app requests and focused viewer reference documents."
version: "0.2.0"
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

This is the top-level entry point for the Omniverse Realtime Viewer skill package.
It is self-contained: all required routing, conventions, and validation
guidance live in the selected references.

Use the focused reference documents as implementation recipes. This file chooses the
right recipes and preserves the architectural rules that must hold across all
generated viewer apps.

## Instructions

Start by classifying the requested viewer, then read only the references needed
for that delivery path and feature set. Implement the render path first, layer
interaction and UI behavior on top of it, and finish by capturing validation
evidence from `references/validation.md`.

## Read Order

1. Read `references/routing.md` to choose the delivery path and focused references.
2. Read `references/conventions.md` before implementing camera, input,
   selection, viewport, streaming protocol, scene loading, or environment
   behavior.
3. For broad viewer requests, read `references/usd-viewer-app/README.md`.
4. If the delivery path is unclear, read `references/streaming-vs-local/README.md`.
5. If the prompt includes layout, panels, controls, inspectors, status, or UX,
   read `references/viewer-ux-workflow/README.md` and then the focused viewer UI references.
   This applies to React/WebRTC, Tauri, Electron, `ovui`, `ovwidgets`, and Dear
   ImGui apps; "frontend" means user-facing UI, not only browser UI.
6. For viewport interaction, read `references/viewer-input-routing/README.md` before
   `references/camera-controls/README.md`, `references/native-picking-selection/README.md`, or `references/object-selection/README.md`.
7. For new viewer apps or ovrtx work, read the `ovstage` references before
   renderer, scene loading, stage management, camera, selection, or transform
   write references.
8. For NVCF self-hosted deployment, read `references/cloud-deployment/nvcf-self-hosted.md` before the selected cloud-deployment and streaming references.
9. Read only the focused capability references needed for the requested app.
10. Use `references/validation.md` to capture review evidence before handoff.

## Non-Negotiables

- Use `ovrtx` for all USD and 3D rendering.
- Browser apps display an `ovstream` WebRTC video stream plus UI. The browser
  does not render USD geometry.
- Do not substitute WebGL, Three.js, Babylon.js, PlayCanvas, A-Frame,
  model-viewer, react-three-fiber, glTF browser viewers, or other client-side
  3D renderers.
- If local validation cannot run because the GPU/runtime environment is absent,
  scaffold the `ovrtx` path and document the runtime requirement. Do not add a
  browser-renderer fallback.
- Keep user USD files unmodified. Viewer cameras, render products, render vars,
  settings, selection metadata, and runtime state belong in session/composite
  layers, OVStage runtime state, or app state.
- Keep one owner for `renderer.step()`, stage mutation, native picking,
  selection writes, OVStage publication, and live attribute writes.
- Use OVStage runtime writes for interchangeable transform animation,
  material/effect attributes, visibility toggles, physics pose samples, and
  reversible viewer-owned state. Keep native OVRTX APIs for rendering, pick
  queues, and selection outline visualization.
- Keep dependency acquisition in `references/dependencies/README.md` and deployment choices in
  `references/cloud-deployment/README.md`; do not duplicate package locations or deployment setup.

## Focused Reference Families

- Entry points and recipes: `references/usd-viewer-app/README.md`, `references/streaming-viewer-recipe/README.md`,
  `references/ovui-local-viewer-recipe/README.md`, `references/streaming-vs-local/README.md`, `references/electron-shm-viewer/README.md`,
  `references/ovwidgets-editor-shell/README.md`.
- Rendering and stage: `references/ovstage-runtime/README.md`, `references/ovstage-population/README.md`,
  `references/ovstage-data-plane/README.md`, `references/ovstage-ovrtx-integration/README.md`,
  `references/ovrtx-rendering/README.md`, `references/stage-loading/README.md`,
  `references/stage-management/README.md`, `references/render-settings/README.md`,
  `references/aov-switching/README.md`, `references/stage-hierarchy/README.md`,
  `references/stage-queries/README.md`, `references/stage-attribute-reads/README.md`,
  `references/camera-auto-select/README.md`, `references/camera-picker/README.md`,
  `references/prim-transform-safety/README.md`, `references/usd-sample-data/README.md`.
- Delivery and runtime: `references/streaming-server/README.md`, `references/streaming-client/README.md`,
  `references/streaming-messages/README.md`, `references/streaming-lifecycle/README.md`, `references/local-viewer/README.md`,
  `references/tauri-local-viewer/README.md`, `references/cpp-native-viewer/README.md`, `references/headless-shm-cli/README.md`,
  `references/viewer-backend-interface/README.md`, `references/webgl-shm-transport/README.md`.
- Viewer UI/UX: `references/viewer-ux-workflow/README.md`, `references/viewer-layout-patterns/README.md`,
  `references/viewer-control-patterns/README.md`, `references/viewer-data-view-patterns/README.md`,
  `references/viewer-feedback-status/README.md`.
- Interaction: `references/viewer-input-routing/README.md`, `references/camera-controls/README.md`,
  `references/object-selection/README.md`, `references/native-picking-selection/README.md`, `references/selection-feedback/README.md`,
  `references/physics-simulation/README.md`,
  `references/selection-animation/README.md`, `references/transform-manipulator/README.md`, `references/gl-viewport-overlay/README.md`,
  `references/ovui-library/README.md`, `references/prim-pick-effects/README.md`, `references/prim-info-display/README.md`,
  `references/viewport-overlays/README.md`, `references/physics-simulation/README.md`.
- Infrastructure: `references/dependencies/README.md`, `references/windows-native-setup/README.md`, `references/cloud-assets/README.md`,
  `references/cloud-deployment/README.md`, `references/cloud-deployment/nvcf-self-hosted.md`,
  `references/cloud-deployment/nvcf-viewer-container-contract.md`,
  `references/cloud-deployment/gpu-container-runtime.md`,
  `references/cloud-deployment/container-image-build.md`,
  `references/cloud-deployment/webrtc-network-diagnostics.md`, `references/troubleshooting/README.md`.
- AI/Agent: `references/a2ui-scene-copilot/README.md`.
- CAE/CFD visualization: `references/cae-cfd-visualization/README.md`,
  `references/cae-cfd-visualization/data-and-operators.md`,
  `references/cae-cfd-visualization/cae-data-ingestion.md`,
  `references/cae-cfd-visualization/temporal-playback.md`,
  `references/cae-cfd-visualization/usd-authoring-and-materials.md`,
  `references/cae-cfd-visualization/ovstage-render-and-camera.md`,
  `references/cae-cfd-visualization/ovui-controls.md`,
  `references/cae-cfd-visualization/glyphs.md`,
  `references/cae-cfd-visualization/volume-rendering.md`,
  `references/cae-cfd-visualization/driving-cae-viz-via-ovstage.md`,
  `references/cae-cfd-visualization/emitter-and-seed-sources.md`,
  `references/cae-cfd-visualization/streaming-cae-viewer.md`,
  `references/cae-cfd-visualization/rtwt-blueprint-recreation.md`,
  `references/cae-cfd-visualization/coordinate-systems-and-up-axis.md`.

## Build Workflow

1. Classify the prompt by delivery path, target user, required capabilities,
   runtime environment, validation needs, and explicit constraints.
2. Select a small reference set. Start with the recipe or routing reference, then add
   focused capabilities such as camera, picking, hierarchy, properties, render
   settings, transform tools, cloud assets, or deployment.
3. Read selected references before writing app code. Follow their build order,
   import order, data-channel contracts, and renderer ownership rules.
4. Implement the OVStage population/data-plane path and core render path first,
   then input routing and camera, then selection and data panels, then
   scene/settings features, then packaging or deployment.
5. Treat the selected references as the behavior contract for API shape,
   compatibility, and generated project structure.
6. Capture validation evidence before calling the viewer ready.

## Examples

- For a browser viewer request, use the streaming recipe references plus camera,
  picking, hierarchy, properties, render settings, and stream-status references.
- For a local workstation viewer request, use the local or native delivery
  references plus renderer setup, stage loading, viewport input, and validation.
- For a "visualize CAE/CFD data in realtime" request, treat CAE/CFD as a layer,
  not a standalone app: first pick the delivery path
  (`references/streaming-vs-local/README.md`), build the base viewer with the
  matching recipe (`references/ovui-local-viewer-recipe/README.md` or
  `references/streaming-viewer-recipe/README.md`), then layer
  `references/cae-cfd-visualization/README.md` (data → operators → USD authoring →
  scene) and wire controls via
  `references/cae-cfd-visualization/driving-cae-viz-via-ovstage.md`.

## Completion Checklist

- Selected references match the user's intent and delivery path.
- No code path uses a browser-side 3D renderer for USD.
- The generated app has one clear owner for render stepping and stage mutation.
- User USD files remain untouched by viewer-owned session data.
- Camera, input, selection, scene loading, and stream behavior follow
  `references/conventions.md`.
- Setup/build/run results and visual interaction evidence are captured with
  `references/validation.md`.

## ovstage References

For all new viewer applications, read these before renderer, scene, interaction, or delivery references:

- `references/ovstage-runtime/README.md` — runtime ownership, ordinals, publication, and recovery.
- `references/ovstage-population/README.md` — composed USD population and scene generations.
- `references/ovstage-data-plane/README.md` — queries, writes, transforms, and runtime DTOs.
- `references/ovstage-ovrtx-integration/README.md` — attached renderer loop and runtime integration contract.
