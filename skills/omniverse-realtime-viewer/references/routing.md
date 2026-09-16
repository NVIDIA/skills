<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Omniverse Realtime Viewer Routing

Use this reference to route plain-language viewer requests into focused references.
This routing reference is self-contained; focused references live in sibling
directories under this `references/` directory.

## Architectural Constraint

All USD and 3D rendering must use `ovrtx`, NVIDIA's RTX renderer.

The pattern is always:

- Server-side: Python or native process owns `ovrtx.Renderer` plus OpenUSD stage
  access, then renders frames on the GPU.
- Browser delivery: `ovstream` WebRTC streams rendered frames to a browser. The
  browser displays video plus UI overlays.
- Desktop delivery: `ovui` native windows, Tauri WebViews, C++ windows, or
  Electron SHM pixel transport display `ovrtx` rendered frames.

If local validation cannot run because the GPU/runtime environment is absent,
scaffold the `ovrtx` code path and document the runtime requirement. Do not
substitute a browser renderer.

## Default UI Preference

If the user requests a viewer or app without naming a UI framework, prefer a
React-based shell. Use WebRTC when browser access, remote GPU execution,
embedding, or web deployment is plausible. Use Tauri when the user clearly
wants a packaged local desktop binary. Preserve the `ovui` path when the user
explicitly requests Python, `omni.ui`, native local UI, or an in-process
viewer.

## How To Use These Skills

When a user describes an Omniverse Realtime Viewer:

1. Route by user intent first.
2. Read `usd-viewer-app/README.md` for broad viewer requests.
3. Add focused references for requested capabilities.
4. Follow each selected reference's implementation notes and gotchas.
5. Capture validation and review evidence before considering the generated app
   ready to share.

## Intent-Based Routing

| User says... | Read these references |
|---|---|
| "I want to visualize USD files" / "build an Omniverse Realtime Viewer" / "3D viewport" | `usd-viewer-app/README.md` first |
| "visualize CAE/CFD data in realtime" / "realtime CAE/CFD viewer" / "visualize my simulation/solver data live" / "show my CFD results interactively" / "visualize CFD/CAE data" / "warp-simdata" / "cae-openusd-plugins" / "iso-surface / slice / streamlines" / "color a mesh by Mach/Pressure/Temperature" / "CGNS/OpenFOAM/VTK/EnSight" | **First** decide the delivery path with `streaming-vs-local/README.md`, **then** build the base viewer with `ovui-local-viewer-recipe/README.md` (local) or `streaming-viewer-recipe/README.md` (browser), **then** layer `cae-cfd-visualization/README.md` and its focused docs on top. CAE/CFD is a layer over a delivery recipe, not a standalone app. |
| "simple interactive viewport" | `streaming-vs-local/README.md` first; for an unspecified framework, prefer `streaming-viewer-recipe/README.md`; use `ovui-local-viewer-recipe/README.md` only when Python/native/in-process local UI is explicit. Then add `ovrtx-rendering/README.md`, `stage-loading/README.md`, `viewer-input-routing/README.md`, and `camera-controls/README.md`. |
| "native desktop app with React UI" / "Tauri viewer" / "Rust OVRTX" | `tauri-local-viewer/README.md`, `ovrtx-rendering/README.md` |
| "C++ viewer" / "native desktop" / "ImGui viewer" / "GLFW viewer" | `cpp-native-viewer/README.md`; add `viewer-control-patterns/README.md` for Dear ImGui controls, toolbars, settings, or dialogs |
| "Electron app" / "Electron viewer" / "SHM viewer" / "shared memory viewer" | `electron-shm-viewer/README.md` |
| "headless automation" / "scripted testing" / "CLI tool" / "SHM automation" | `headless-shm-cli/README.md` |
| "local separate-process viewer" / "process-isolated local viewer" | `electron-shm-viewer/README.md` |
| "reusable UI" / "ViewerBackend" / "shared components" / "cross-transport UI" | `viewer-backend-interface/README.md` |
| "viewer UI" / "frontend UI" / "UX" / "app layout" / "redesign" / "panels" / "toolbar" / "ovui UI" / "ImGui UI" | `viewer-ux-workflow/README.md`, then focused viewer UI references |
| "viewport layout" / "outliner and properties" / "drawer" / "anchored inspector" / "responsive layout" | `viewer-layout-patterns/README.md` |
| "buttons" / "actions" / "forms" / "controls" / "sliders" / "confirmations" | `viewer-control-patterns/README.md` |
| "stage tree UI" / "asset grid" / "property inspector UI" / "JSON tree" | `viewer-data-view-patterns/README.md` |
| "loading state" / "error banner" / "stream health" / "offline" / "lagged" / "status UI" | `viewer-feedback-status/README.md` |
| "stream to a browser" / "browser Omniverse Realtime Viewer" | `streaming-viewer-recipe/README.md`, then `streaming-server/README.md`, `streaming-client/README.md`, `streaming-messages/README.md`, `streaming-lifecycle/README.md`, `viewer-input-routing/README.md` |
| "pick objects" / "click to select" | `viewer-input-routing/README.md`, `native-picking-selection/README.md`, `object-selection/README.md`, `selection-feedback/README.md` |
| "when picked, change a material/visibility/effect attribute" | `prim-pick-effects/README.md`, `object-selection/README.md`, `ovstage-data-plane/README.md` |
| "see info/properties for highlighted objects" | `prim-info-display/README.md`, `stage-attribute-reads/README.md`, `stage-hierarchy/README.md`, `ovstage-data-plane/README.md` |
| "highlight selected objects" | `selection-feedback/README.md`, `native-picking-selection/README.md` |
| "custom segmentation-buffer outline overlay" | `seg-outline-highlight/README.md` |
| "animate selected objects" | `selection-animation/README.md`, `ovstage-data-plane/README.md` |
| "move/rotate/scale selected objects" / "transform gizmo" / "manipulator" | `transform-manipulator/README.md`, `ovstage-data-plane/README.md`; read `prim-transform-safety/README.md` only for legacy direct OVRTX helpers |
| "Tauri SHM transform gizmo" / "client-side gizmo overlay" | `tauri-shm-transform-gizmo/README.md`, plus `tauri-local-viewer/README.md` and `webgl-shm-transport/README.md` |
| "C++ viewport overlay" / "C++ gizmo" / "GL gizmo" | `gl-viewport-overlay/README.md`, `ovui-library/README.md`, plus `cpp-native-viewer/README.md` |
| "switch scenes" / "load different USD files" / "asset browser" | `stage-management/README.md`, `stage-loading/README.md` |
| "rendering settings" / "lighting" / "quality controls" | `render-settings/README.md`, `viewer-control-patterns/README.md` |
| "switch AOVs" / "view normals" / "segmentation render output" | `aov-switching/README.md`, `ovrtx-rendering/README.md`, `streaming-messages/README.md` |
| "settings persist across scenes" | `stage-management/README.md`, `render-settings/README.md`, `viewer-control-patterns/README.md` |
| "scene tree" / "hierarchy" / "variants" | `stage-hierarchy/README.md` |
| "viewport overlays" / "camera gizmo" / "floating panel" | `viewport-overlays/README.md`, plus `camera-controls/README.md` or `prim-info-display/README.md` |
| "switch between cameras" / "camera picker" | `camera-auto-select/README.md`, `camera-picker/README.md` |
| "load from S3/MinIO/cloud assets" | `cloud-assets/README.md` |
| "browse assets with thumbnails" | `cloud-assets/README.md` |
| "deploy with NVCF self-hosted" / "NVCF control plane" / "LLS" / "AWS or Azure NVCF" | `cloud-deployment/nvcf-self-hosted.md`, then upstream `streaming-self-hosted` skills; use the viewer-container and network-diagnostics references for app handoff and validation |
| "deploy with Brev, OKAS, or cloud sessions" | `cloud-deployment/README.md` |
| "physics simulation" / "drop test" / "physics grab" / "pick-driven impulse" | `dependencies/nvidia-runtime.md` → `physics-simulation/README.md` → `object-selection/README.md` / `ovstage-data-plane/README.md`; then inspect the current upstream OVPhysX package/repo `skills/` for exact APIs |
| "import CAD files" / "convert STEP/IGES to USD" | Clone `cad2usd` and check its `skills/` |
| "native Windows setup" | `windows-native-setup/README.md` |
| "full editor with docking/property inspector" | `streaming-vs-local/README.md` first; use `ovwidgets-editor-shell/README.md` for the full editor path, plus `viewer-control-patterns/README.md` and `viewer-data-view-patterns/README.md` for editor controls and panels |
| "scene copilot" / "AI chat for the viewer" / "A2UI agent" / "CopilotKit viewer" / "natural language scene control" | `a2ui-scene-copilot/README.md` |

Target prompt routing:

```text
I want to visualize USD files in a simple interactive viewport, I want to pick
objects and see information about the objects highlighted, and I want to easily
switch between different USD scenes and have some basic rendering and lighting
settings that persist across scenes.
```

Read: `usd-viewer-app/README.md`, `ovui-local-viewer-recipe/README.md`, `local-viewer/README.md`,
`ovrtx-rendering/README.md`, `stage-loading/README.md`, `viewer-input-routing/README.md`,
`camera-controls/README.md`, `native-picking-selection/README.md`, `object-selection/README.md`,
`selection-feedback/README.md`, `prim-info-display/README.md`, `stage-attribute-reads/README.md`,
`stage-management/README.md`, `render-settings/README.md`, `viewer-control-patterns/README.md`, and
`stage-hierarchy/README.md`.

## Capability-Based Routing

| Capability | Skills to read |
|---|---|
| High-level Omniverse Realtime Viewer recipe | `usd-viewer-app/README.md` |
| Core ovrtx renderer construction/step/write APIs | `ovrtx-rendering/README.md` |
| Camera/render product/render var/session stage setup | `stage-loading/README.md` |
| Local desktop end-to-end recipe | `ovui-local-viewer-recipe/README.md`; add `viewer-control-patterns/README.md` for toolbars, forms, render settings, or other user-facing controls |
| Local desktop lightweight ovui shell | `local-viewer/README.md`; add `viewer-control-patterns/README.md` for header, sidebar, toolbar, or inline controls |
| Tauri/Rust native desktop with React WebView | `tauri-local-viewer/README.md` |
| Native C++ OVRTX viewer with ImGui/GLFW | `cpp-native-viewer/README.md`; add `viewer-control-patterns/README.md` for Dear ImGui controls |
| Electron plus SHM local separate-process viewer | `electron-shm-viewer/README.md`, `webgl-shm-transport/README.md` |
| Headless SHM automation and testing | `headless-shm-cli/README.md` |
| ViewerBackend interface and shared React components | `viewer-backend-interface/README.md` |
| SharedArrayBuffer to WebGL pixel transport | `webgl-shm-transport/README.md` |
| Interactive translate/rotate/scale manipulators | `transform-manipulator/README.md`, `ovstage-data-plane/README.md`; add `prim-transform-safety/README.md` only for legacy direct OVRTX helpers |
| Client-rendered transform gizmo for Tauri SHM | `tauri-shm-transform-gizmo/README.md` |
| C++ GL viewport overlays and reusable gizmo math | `gl-viewport-overlay/README.md`, `ovui-library/README.md` |
| Viewer UI intent routing and UX workflow | `viewer-ux-workflow/README.md` |
| Viewport-dominant layout, panels, drawers, responsive shell | `viewer-layout-patterns/README.md` |
| Toolbars, forms, sliders, semantic actions, confirmations | `viewer-control-patterns/README.md` |
| Stage tree, asset browser, property inspector, JSON data views | `viewer-data-view-patterns/README.md` |
| Loading, errors, stream health, lagged/offline status | `viewer-feedback-status/README.md` |
| Full editor shell | `streaming-vs-local/README.md`, `ovwidgets-editor-shell/README.md`, `viewer-control-patterns/README.md`, `viewer-data-view-patterns/README.md` |
| A2UI scene copilot / LLM agent control | `a2ui-scene-copilot/README.md` |
| CAE/CFD simulation visualization (as a layer over a delivery recipe) | `streaming-vs-local/README.md` (delivery), then `ovui-local-viewer-recipe/README.md` or `streaming-viewer-recipe/README.md` (base viewer), then `cae-cfd-visualization/README.md` (CAE layer) |
| Live control → CAE scene write (data-plane vs structural rebuild) | `cae-cfd-visualization/driving-cae-viz-via-ovstage.md` |
| Streaming architecture decision | `streaming-vs-local/README.md` |
| Browser-streamed end-to-end recipe | `streaming-viewer-recipe/README.md` |
| WebRTC/RTSP server and CUDA frame streaming | `streaming-server/README.md` |
| React/AppStreamer browser client for standalone ovstream Direct mode | `streaming-client/README.md` |
| Streaming JSON data-channel protocol | `streaming-messages/README.md` |
| Stream callback/data-channel lifecycle | `streaming-lifecycle/README.md` |
| Viewer input routing / WebRTC input / click-vs-drag / viewport input ownership | `viewer-input-routing/README.md` |
| Orbit/pan/zoom/camera fitting/gizmo | `viewer-input-routing/README.md`, `camera-controls/README.md` |
| Object picking/selection | `viewer-input-routing/README.md`, `native-picking-selection/README.md`, `object-selection/README.md` |
| Selection glow/highlight | `selection-feedback/README.md`, `native-picking-selection/README.md` |
| Custom segmentation-buffer post-process overlays | `seg-outline-highlight/README.md` |
| Transform-safe live prim manipulation | `transform-manipulator/README.md`, `ovstage-data-plane/README.md`, `ovstage-ovrtx-integration/README.md` |
| Selection hover/motion animation | `selection-animation/README.md`, `ovstage-data-plane/README.md` |
| Selected prim info/properties display | `prim-info-display/README.md`, `stage-attribute-reads/README.md`, `ovstage-data-plane/README.md` |
| Scene switching/reload/persistent state | `stage-management/README.md` |
| Render quality/render vars/lighting/settings | `render-settings/README.md`, `viewer-control-patterns/README.md` |
| Browser AOV/render-var switching | `aov-switching/README.md`, `ovrtx-rendering/README.md`, `streaming-messages/README.md` |
| Server-side ovui overlays | `viewport-overlays/README.md` |
| USD hierarchy/properties/variants/bounds | `stage-hierarchy/README.md` |
| OVStage prim discovery/filtering | `stage-queries/README.md`, `ovstage-data-plane/README.md` |
| OVStage scalar/array attribute reads | `stage-attribute-reads/README.md`, `ovstage-data-plane/README.md` |
| Pick-driven runtime/USD attribute effects | `prim-pick-effects/README.md`, `ovstage-data-plane/README.md` |
| S3/MinIO asset loading and browsing | `cloud-assets/README.md` |
| Physics simulation / pick-driven impulse handoff | `dependencies/nvidia-runtime.md` → `physics-simulation/README.md` → `object-selection/README.md` / `ovstage-data-plane/README.md`; use the current OVPhysX package/repo `skills/` for exact APIs |
| CAD-to-USD conversion | Clone `cad2usd`, use its skills |
| Native Windows setup | `windows-native-setup/README.md` |

## Decision Tree

```text
User prompt received
|
+- High-level app request? ("build an Omniverse Realtime Viewer", "visualize USD files")
|  +- READ: usd-viewer-app/README.md
|
+- Visualize CAE/CFD simulation data? ("visualize CAE/CFD data in realtime", "realtime CFD viewer", "visualize my solver data live", CGNS/OpenFOAM/VTK/EnSight, iso-surface/slice/streamlines)
|  +- 1) Delivery path -> READ: streaming-vs-local (browser vs local)
|  +- 2) Base viewer   -> READ: streaming-viewer-recipe (browser) OR ovui-local-viewer-recipe + local-viewer (local)
|  +- 3) CAE layer     -> READ: cae-cfd-visualization/README.md, then its focused docs (data/operators, ingestion, authoring, scene, controls)
|  +- 4) Wire controls -> READ: cae-cfd-visualization/driving-cae-viz-via-ovstage.md (control -> data-plane write vs structural rebuild)
|
+- Delivery method?
|  +- Browser/web -> READ: streaming-viewer-recipe + streaming-server + streaming-client + streaming-messages + streaming-lifecycle
|  +- Desktop/local (React UI, no Python) -> READ: tauri-local-viewer
|  +- Desktop/local (C++, ImGui, no Python/Rust) -> READ: cpp-native-viewer + viewer-control-patterns
|  +- Desktop/local (React UI, Python server, separate process) -> READ: electron-shm-viewer
|  +- Desktop/local (Python, simple) -> READ: ovui-local-viewer-recipe + local-viewer + ovrtx-rendering + stage-loading; add viewer-control-patterns when controls are visible
|  +- Desktop/local (Python, full editor) -> READ: streaming-vs-local + ovwidgets-editor-shell + viewer-control-patterns
|  +- Both/unsure -> READ: streaming-vs-local first
|
+- Viewer/UI work?
|  +- Broad UI/layout prompt -> READ: viewer-ux-workflow
|  +- Panels/drawers/responsive shell -> READ: viewer-layout-patterns
|  +- Toolbars/forms/actions/sliders/confirmations -> READ: viewer-control-patterns
|  +- Trees/asset grids/property inspectors -> READ: viewer-data-view-patterns
|  +- Loading/errors/stream status -> READ: viewer-feedback-status
|
+- Specific feature?
|  +- Object picking -> READ: viewer-input-routing + native-picking-selection + object-selection + selection-feedback
|  +- Pick changes a USD/material attribute -> READ: prim-pick-effects + object-selection + ovstage-data-plane
|  +- Object info panel -> READ: prim-info-display + stage-attribute-reads + stage-hierarchy + ovstage-data-plane
|  +- Camera navigation -> READ: viewer-input-routing + camera-controls
|  +- Transform gizmo/manipulator -> READ: transform-manipulator + ovstage-data-plane
|  +- Tauri SHM client-rendered gizmo -> READ: tauri-shm-transform-gizmo + tauri-local-viewer + webgl-shm-transport
|  +- C++ GL viewport overlay/gizmo -> READ: gl-viewport-overlay + ovui-library + cpp-native-viewer
|  +- Scene switching -> READ: stage-management
|  +- Render quality/lighting -> READ: render-settings
|  +- AOV/render-var switching -> READ: aov-switching + ovrtx-rendering + streaming-messages
|  +- Viewport overlays -> READ: viewport-overlays
|  +- Animation -> READ: selection-animation + ovstage-data-plane
|  +- Custom messages -> READ: streaming-messages
|
+- Infrastructure?
|  +- Cloud assets -> READ: cloud-assets
|  +- Cloud deployment -> READ: cloud-deployment
|  +- Physics simulation -> READ: physics-simulation + object-selection + ovstage-data-plane; clone ovphysx, read its skills/
|  +- CAD file import -> Clone cad2usd, read its skills/
|  +- Windows -> READ: windows-native-setup
|
+- AI/Agent integration?
|  +- Scene copilot / A2UI / LLM agent -> READ: a2ui-scene-copilot
|  +- CopilotKit chat with viewer -> READ: a2ui-scene-copilot + streaming-client
|
+- USD stage work?
   +- Loading scenes -> READ: stage-loading
   +- Hierarchy/queries -> READ: stage-hierarchy
   +- OVStage prim filters -> READ: stage-queries + ovstage-data-plane
   +- OVStage attribute values -> READ: stage-attribute-reads + ovstage-data-plane
```

## Dependencies Model

When a selected reference tells you to install or configure a dependency, read
`dependencies/README.md` first. It is the source of truth for primary NVIDIA
runtime dependencies and their upstream guidance.

| Library | Install method | Notes |
|---|---|---|
| `ovrtx` | See `dependencies/nvidia-runtime.md` | RTX USD renderer; RTX GPU required |
| `ovstream` | See `dependencies/nvidia-runtime.md` | Streaming runtime |
| `ov-web-rtc client` / `@nvidia/ov-web-rtc` | See `dependencies/nvidia-runtime.md` | Browser AppStreamer client for standalone `ovstream` Direct mode; do not use alternate package names, hard-coded client versions, or connection profiles intended for a different streaming product |
| `ovui` | See `dependencies/nvidia-runtime.md` | Native UI toolkit |
| `ovui-data-adapters` | Install from the same `ovui` package set | Local UI adapter contracts |
| Full editor UI package | Install only when current `ovui` dependency guidance explicitly requires it, from the same `ovui` package set | Full editor widgets |
| `ovstorage` | See `dependencies/nvidia-runtime.md` | Cloud asset browsing and cache sync; inspect upstream `AGENTS.md` and `skills/` first |
| `ovphysx` | See `dependencies/nvidia-runtime.md` | Physics simulation; use `physics-simulation/README.md` for viewer integration and OVPhysX skills for tensor APIs |
| `cad2usd` | External checkout | CAD file conversion to USD |
| `pxr` / OpenUSD | `pip install usd-core` | Use version pins from platform skills |
| `numpy` | `pip install numpy` | Array operations |
| `warp` | `pip install warp-lang` | GPU kernels and CUDA buffer utilities |
| `warp-simdata` (CAE GPU operator library) | See `dependencies/README.md` → CAE/CFD Visualization Libraries (`github.com/NVIDIA/warp-simdata`) | CAE/CFD `Dataset` + visualization operators |
| `cae-openusd-plugins` (CAE USD reader/schema library) | See `dependencies/README.md` → CAE/CFD Visualization Libraries (`github.com/NVIDIA-Omniverse/cae-openusd-plugins`) | Optional CAE file readers (CGNS/OpenFOAM/VTK/EnSight/FLASH) |

## Supplemental Guidance

Use the selected references as the implementation contract. When a dependency
reference provides supplemental documentation, use it to clarify API behavior
without changing the selected viewer architecture.

```text
ovui guidance     -> local UI package setup, widgets, overlays, and native UI conventions
ovstream guidance -> streaming runtime setup, SHM behavior, native input, lifecycle, and the ovstream repo's own skills/samples
ovrtx guidance    -> renderer setup, Python/C API behavior, AOVs, picking, and selection
ovstorage guidance -> upstream `AGENTS.md` + `skills/`, then resolver and asset-management patterns
Clone ovphysx    -> check ovphysx/skills/    -> physics simulation, collider cooking, grab, drop
Clone cad2usd    -> check cad2usd/skills/    -> CAD conversion and batch processing
```

## ovstage Runtime Routing

All new viewer implementations begin with `ovstage-runtime`, `ovstage-population`, `ovstage-data-plane`, and `ovstage-ovrtx-integration`. Add them before the focused delivery, camera, selection, hierarchy, and render-setting references. The application owns the runtime scene; ovrtx renders it.
