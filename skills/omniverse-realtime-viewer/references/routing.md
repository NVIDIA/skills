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

## Choose The Delivery And UI

Honor the user's delivery, language and framework requirements; preserve an
existing app's shell. Otherwise choose the simplest suitable framework for the
requested interactions and host/deployment constraints. React and native toolkit
examples illustrate integrations, not preferred frameworks. Use WebRTC for
browser delivery; select a local recipe for a packaged desktop app.

An explicit headset/XR request selects `xr-viewer-recipe/README.md` before
browser/desktop shell selection. A remote GPU or a Quest browser client does not turn the
XR projection path into an ordinary `ovstream` browser viewer. Choose a desktop
companion shell separately, only when requested.

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
| "stream to a browser" / "browser Omniverse Realtime Viewer" | `streaming-viewer-recipe/README.md`; follow its current-work links |
| "pick objects" / "click to select" | `viewer-input-routing/README.md`, `native-picking-selection/README.md`, `object-selection/README.md`, `selection-feedback/README.md` |
| "when picked, change a material/visibility/effect attribute" | `prim-pick-effects/README.md`, `object-selection/README.md`, `ovstage-data-plane/README.md` |
| "see info/properties for highlighted objects" | `prim-info-display/README.md`, `stage-attribute-reads/README.md`, `stage-hierarchy/README.md`, `ovstage-data-plane/README.md` |
| "highlight selected objects" | `selection-feedback/README.md`, `native-picking-selection/README.md` |
| "custom segmentation-buffer outline overlay" | `seg-outline-highlight/README.md` |
| "animate selected objects" | `selection-animation/README.md`, `ovstage-data-plane/README.md` |
| "move/rotate/scale selected objects" / "transform gizmo" / "manipulator" | `transform-manipulator/README.md`, `ovstage-data-plane/README.md`, `prim-transform-safety/README.md` for current local/world, inheritance, and persistence semantics |
| "Tauri SHM transform gizmo" / "client-side gizmo overlay" | `tauri-shm-transform-gizmo/README.md`, plus `tauri-local-viewer/README.md` and `webgl-shm-transport/README.md` |
| "C++ viewport overlay" / "C++ gizmo" / "GL gizmo" | `gl-viewport-overlay/README.md`, `ovui-library/README.md`, plus `cpp-native-viewer/README.md` |
| "Vulkan compositing" / "VulkanSurface overlay" / "raster gizmos over OVRTX" / "composite with the OVRTX Z buffer" | `vulkan-compositing/README.md`, then `vulkan-compositing/interop-and-lifecycle.md`; add `vulkan-compositing/depth-overlays.md` for scene occlusion |
| "switch scenes" / "load different USD files" / "asset browser" | `stage-management/README.md`, `stage-loading/README.md` |
| "rendering settings" / "lighting" / "quality controls" | `render-settings/README.md`, `viewer-control-patterns/README.md` |
| "spectral rendering" / "wavelength bands" / "renderer startup profiles" | `render-settings/README.md`, then the current upstream OVRTX configuration/spectral guidance linked there |
| "lighting contributions" / "light path expressions" / "diffuse/glossy/caustic views" | `aov-switching/README.md`, then the current upstream LPE guidance linked there |
| "section plane" / "cutaway" / "cap cut surfaces" / "slice plane gizmo" | `section-views/README.md`; add `viewer-input-routing/README.md` for interaction |
| "save runtime edits" / "export selected transforms" | `prim-transform-safety/README.md` → Saving runtime edits, then the current upstream OVStage export skill |
| "switch AOVs" / "view normals" / "segmentation render output" | `aov-switching/README.md`, `ovrtx-rendering/README.md`, `streaming-messages/README.md` |
| "SPG" / "GPU post-processing" / "processed render output" / "lens distortion" | `spg-processing/README.md`, then the selected delivery recipe and `aov-switching/README.md` for source/processed output selection |
| "stylized rendering" / "NPR" / "toon shader" / "technical illustration" / "X-ray view" / "feature lines" / "silhouette lines" / "dashed or dotted hidden lines" / "AOV post-processing" | `spg-processing/README.md`, then `stylized-visualization/README.md`; for faithful reproduction follow its four core-contract read order, add the selected delivery recipe, and use `aov-switching/README.md` when the app also exposes raw render-output switching. First inventory the geometry provider for X-Ray: use `stylized-visualization/mesh-xray.md` for ordinary tessellated USD meshes; it is not a BREP requirement. For a new look without a named recipe, also read `stylized-visualization/style-composition-contract.md`. |
| "settings persist across scenes" | `stage-management/README.md`, `render-settings/README.md`, `viewer-control-patterns/README.md` |
| "scene tree" / "hierarchy" / "variants" | `stage-hierarchy/README.md` |
| "viewport overlays" / "camera gizmo" / "floating panel" | `viewport-overlays/README.md`, plus `camera-controls/README.md` or `prim-info-display/README.md` |
| "switch between cameras" / "camera picker" | `camera-auto-select/README.md`, `camera-picker/README.md` |
| "load from S3/MinIO/cloud assets" | `cloud-assets/README.md` |
| "download USD from Hugging Face" / "Hugging Face dataset asset" | `huggingface-usd/README.md`, then `cloud-assets/README.md` if the generated app needs a unified asset browser |
| "dynamic viewport resize" / "responsive stream resolution" / "resizeViewport" | `viewport-resize/README.md`, plus `streaming-server/README.md`, `streaming-client/README.md`, and `streaming-messages/README.md` |
| "browse assets with thumbnails" | `cloud-assets/README.md` |
| "deploy with NVCF self-hosted" / "NVCF control plane" / "LLS" / "AWS or Azure NVCF" / "on-premises NVCF" / "upgrade NVCF streaming stack" | `cloud-deployment/nvcf-self-hosted.md`, then upstream `streaming-self-hosted` skills; use the viewer-container and network-diagnostics references for app handoff and validation |
| "deploy with Brev, OKAS, or cloud sessions" | `cloud-deployment/README.md` |
| "physics simulation" / "drop test" / "physics grab" / "pick-driven impulse" | `dependencies/nvidia-runtime.md` → `physics-simulation/README.md` → `realtime-sim-integration/README.md` → `object-selection/README.md` / `ovstage-data-plane/README.md`; then inspect the current upstream OVPhysX package/repo `skills/` for exact APIs |
| "Newton physics" / "ovnewton" / "cable" / "articulated mechanism" / "robot arm sim" | `newton-physics/README.md`, `realtime-sim-integration/README.md`, `ovstage-data-plane/README.md`; route to <https://github.com/newton-physics/newton> and its `.claude/skills` for Newton itself |
| "click an object to push it" / "flick/throw/knock over a simulated body" / "shake the scene" / "drive a joint from the UI" / "teleop" / "make the sim respond to the user" | `interaction-round-trip/README.md`, then the engine reference (`newton-physics/README.md` or `physics-simulation/README.md`) |
| "I clicked the ball and nothing happened" / "pick returns a mesh but physics wants the body" / "impulse launched the object off screen" / "the physics readout is frozen" | `interaction-round-trip/README.md`, plus `native-picking-selection/README.md` for the pick itself |
| "per-frame controller" / "IK from a hook" / "scripted robot motion" / "control_hook" / "joint targets never change" | `interaction-round-trip/README.md` (perturbation vs command), `newton-physics/README.md`, `realtime-sim-integration/README.md` |
| "the sim runs but nothing moves" / "prims vanished" / "physics is slow" / "solver diverged" / "sim reset does nothing" | `realtime-sim-integration/README.md` first, then the engine reference |
| "step a simulation on the render loop" / "publish poses every frame" / "CUDA graph capture in a viewer" / "catch-up / realtime factor" | `realtime-sim-integration/README.md`, `ovstage-data-plane/README.md` |
| "import CAD files" / "convert STEP/IGES to USD" | Read [NVIDIA-Omniverse/usd-convert-cad — `omniverse-cad-to-usd`](https://github.com/NVIDIA-Omniverse/usd-convert-cad/blob/main/skills/omniverse-cad-to-usd/SKILL.md) |
| "native Windows setup" | `windows-native-setup/README.md` |
| "build XR from libraries" / "managed anchor" / "default anchor" / "user scale" / "live XR settings" | `xr-viewer-recipe/README.md`, `xr-spatial-input/README.md`, `ovxr-python-interactions/README.md` |
| "XR viewer" / "VR" / "AR" / "headset" / "Vision Pro" / "Quest" / "immersive" / "stereo rendering" / "OpenXR" / "CloudXR" | `xr-viewer-recipe/README.md` first; its construction route includes the runtime and OVStage contracts. For an editor use `ovxr-library-runtime/viewer-assembly.md`; add only the requested input, UI, passthrough and mirror features. |
| "projection-integrated spatial UI" / "OpenXR composition UI" / "server-rendered XR panel" / "CloudXR UI" / "Skia Yoga XR UI" / "attach UI to a prim" | `ovxr-composition-ui/README.md`, then `ovxr-composition-ui-validation/README.md` |
| "validate XR spatial UI" / "prove CloudXR UI interaction" / "spatial panel capture" / "UI is head-locked" / "panel is off-frustum" / "scene and UI in one capture" | `ovxr-composition-ui-validation/README.md`; add `ovxr-composition-ui/README.md` for implementation changes |
| "XR controllers" / "PSVR2" / "hand tracking" / "pinch" / "teleport" / "controller ray" / "XR grab" / "tracked prop" / "spatial input" | `xr-spatial-input/README.md`, `ovxr-openxr-runtime/README.md`, `ovstage-data-plane/README.md`, and `ovstage-ovrtx-integration/README.md` |
| "CloudXR server without Kit" / "kitless CloudXR" / "monado-service" / "XR_ERROR_RUNTIME_UNAVAILABLE" / "host the CloudXR runtime" | `cloudxr-opaque-data-channel/README.md` |
| "send messages from a headset" / "XR data channel" / "opaque data channel" / "XR_NV_opaque_data_channel" | `cloudxr-opaque-data-channel/README.md`, then `streaming-messages/README.md` for the dispatch pattern |
| "XR client cannot connect" / "server logs nothing" / "RtspClient connect timed out" / "backend mode" / "nvst-native" / "which port does CloudXR bind" / "CloudXR firewall" / "connect a Vision Pro or Quest client" | `xr-client-connection/README.md` |
| "client message does nothing" / "XR message envelope" / "Type vs type casing" / "12-float matrix" / "scene manifest" / "manipulate a prim from a headset" / "throttle transform messages" | `xr-client-message-contract/README.md`, then `cloudxr-opaque-data-channel/README.md` |
| "foveation" / "warped foveation" / "foveation unwarp" / "inset foveation" / "eye-tracked foveation" / "quad view" / "display mode" / "why is my captured frame distorted" / "render resolution in XR" / "recommended swapchain size" / "runtime PQW" / "foveation_unwarp has no effect" / "the mirror is the wrong size" | `ovxr-foveation-and-display/README.md`, then `ovxr-openxr-runtime/README.md` |
| "headset starts in the wrong place" / "scene is rotated 90 degrees in VR" / "scene is 100x too big in the headset" / "tracking anchor" / "up axis in XR" / "metersPerUnit" / "Z-up stage in VR" / "QR anchor" | `xr-tracking-anchor-and-scale/README.md`, then `ovstage-runtime/README.md` for the ordinal rules on anchor writes |
| "begin_frame" / "bind_frame" / "submit_frame" / "cancel_frame" / "step_result" / "frame readback" / "multi-GPU XR" / "stereo split" / "out of memory in the XR frame loop" | `ovxr-frame-loop-and-readback/README.md`; add `ovxr-foveation-and-display/README.md` when the capture is consumed outside the headset |
| "mirror the XR view" / "spectator view" / "desktop window showing the headset" / "which render product feeds the stream" / "red and blue swapped in the mirror" / "one renderer for a headset and a window" / "the frame rate cap is not the rate I asked for" | `xr-desktop-mirror/README.md`, then `ovxr-frame-loop-and-readback/README.md` and `ovxr-foveation-and-display/README.md`; add `tauri-local-viewer/README.md` or `electron-shm-viewer/README.md` for the consumer shell |
| "AR session" / "passthrough" / "environment blend mode" / "alpha blend" / "black background instead of passthrough" / "VR vs AR in ovxr" / "additive blend" | `xr-session-modes-and-passthrough/README.md` for negotiated blending and `ovxr-library-runtime/ar-compositing.md` for renderer alpha; add `ovxr-library-runtime/live-ar-controls.md` for live switches |
| "live geometry updates" / "streaming parametric geometry" / "write mesh to Fabric" / "real-time mesh swap" / "no stage rebuild" | `ovstage-live-geometry/README.md`, then `ovstage-data-plane/README.md` |
| "library requires different Python version" / "subprocess geometry bridge" / "Python 3.12 subprocess" / "native library version mismatch" | `subprocess-geometry-bridge/README.md` |
| "usd-brep" / "integrate SMLib" / "solidmodeling library" / "omni_solid" / "BRep in ovrtx" / "parametric CAD viewer" | First read `dependencies/README.md` → USD BRep and its current upstream agent guidance, skills, and operation docs; then select the local worker, live geometry, modeling validation, or exact USD integration references. |
| "fillet fails" / "SM_ERR" / "seam edge" / "degenerate solid" / "boolean consumes handle" | First read `dependencies/README.md` → USD BRep and its current upstream agent guidance, skills, and operation docs; then select the local worker, live geometry, modeling validation, or exact USD integration references. |
| "exact BREP" / "BrepArray" / "persist NURBS in USD" / "codeless schema" / "Fabric drops prim" | `exact-brep-in-usd/README.md`, then `ovstage-live-geometry/README.md` |
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
| Native C++ OVRTX viewer (toolkit selected for the app) | `cpp-native-viewer/README.md`; add `viewer-control-patterns/README.md` for Dear ImGui controls |
| Electron plus SHM local separate-process viewer | `electron-shm-viewer/README.md`, `webgl-shm-transport/README.md` |
| Headless SHM automation and testing | `headless-shm-cli/README.md` |
| ViewerBackend interface and shared React components | `viewer-backend-interface/README.md` |
| SharedArrayBuffer to WebGL pixel transport | `webgl-shm-transport/README.md` |
| Interactive translate/rotate/scale manipulators | `transform-manipulator/README.md`, `ovstage-data-plane/README.md`, `prim-transform-safety/README.md` for current local/world, inheritance, and persistence semantics |
| Client-rendered transform gizmo for Tauri SHM | `tauri-shm-transform-gizmo/README.md` |
| C++ GL viewport overlays and reusable gizmo math | `gl-viewport-overlay/README.md`, `ovui-library/README.md` |
| Native Vulkan color/alpha composition, depth-tested raster overlays and CUDA/Vulkan frame ownership | `vulkan-compositing/README.md`; use `vulkan-compositing/depth-overlays.md` for manual metric rejection or D32 seeding |
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
| Switchable navigation / Blender-style controls / Revit-style navigation / grab-the-world / fly speed controls | `viewer-input-routing/README.md`, `camera-controls/README.md`, `navigation-presets/README.md` |
| Object picking/selection | `viewer-input-routing/README.md`, `native-picking-selection/README.md`, `object-selection/README.md` |
| Selection glow/highlight | `selection-feedback/README.md`, `native-picking-selection/README.md` |
| Custom segmentation-buffer post-process overlays | `seg-outline-highlight/README.md` |
| Transform-safe live prim manipulation | `transform-manipulator/README.md`, `ovstage-data-plane/README.md`, `ovstage-ovrtx-integration/README.md` |
| Selection hover/motion animation | `selection-animation/README.md`, `ovstage-data-plane/README.md` |
| Selected prim info/properties display | `prim-info-display/README.md`, `stage-attribute-reads/README.md`, `ovstage-data-plane/README.md` |
| Scene switching/reload/persistent state | `stage-management/README.md` |
| Render quality/render vars/lighting/settings | `render-settings/README.md`, `viewer-control-patterns/README.md` |
| SPG nodes, custom sampling, GPU chaining and processed-output publication | `spg-processing/README.md`, selected delivery recipe |
| SPG stylized image/geometry processing and explicit wireframe | `stylized-visualization/README.md`, `spg-processing/README.md` |
| Browser AOV/render-var switching | `aov-switching/README.md`, `ovrtx-rendering/README.md`, `streaming-messages/README.md` |
| Server-side ovui overlays | `viewport-overlays/README.md` |
| USD hierarchy/properties/variants/bounds | `stage-hierarchy/README.md` |
| OVStage prim discovery/filtering | `stage-queries/README.md`, `ovstage-data-plane/README.md` |
| OVStage scalar/array attribute reads | `stage-attribute-reads/README.md`, `ovstage-data-plane/README.md` |
| Pick-driven runtime/USD attribute effects | `prim-pick-effects/README.md`, `ovstage-data-plane/README.md` |
| S3/MinIO asset loading and browsing | `cloud-assets/README.md` |
| Hugging Face USD dataset download | `huggingface-usd/README.md` |
| Dynamic streamed viewport resolution | `viewport-resize/README.md`, `streaming-server/README.md`, `streaming-client/README.md`, `streaming-messages/README.md` |
| Physics simulation / pick-driven impulse handoff | `dependencies/nvidia-runtime.md` → `physics-simulation/README.md` → `realtime-sim-integration/README.md` → `object-selection/README.md` / `ovstage-data-plane/README.md`; use the current OVPhysX package/repo `skills/` for exact APIs |
| Newton continuous simulation in-process (ovnewton) | `newton-physics/README.md`, `realtime-sim-integration/README.md`, `ovstage-data-plane/README.md`; use the public Newton repo/skills for Newton APIs |
| Engine-agnostic sim-on-the-render-loop patterns: frame contract, divergence guards, catch-up budgets, publish cost, graph-capture lifetime, measurement protocol | `realtime-sim-integration/README.md` |
| Click-to-impulse and other user input INTO a running simulation: hit-to-body resolution, thread-crossing input queues, gravity-relative gesture sizing, per-frame control hooks, honest sim readouts | `interaction-round-trip/README.md`, plus `native-picking-selection/README.md` and `streaming-messages/README.md` |
| CAD-to-USD conversion | Read [NVIDIA-Omniverse/usd-convert-cad — `omniverse-cad-to-usd`](https://github.com/NVIDIA-Omniverse/usd-convert-cad/blob/main/skills/omniverse-cad-to-usd/SKILL.md) |
| Native Windows setup | `windows-native-setup/README.md` |
| Desktop mirror / spectator view of an XR session on one renderer | `xr-desktop-mirror/README.md`, `ovxr-frame-loop-and-readback/README.md`, `dependencies/ovstream.md` |
| XR controllers, hands, tracked props, spatial tools, and lifecycle validation | `xr-spatial-input/README.md`, `ovxr-openxr-runtime/README.md`, `ovstage-data-plane/README.md` |
| Projection-integrated spatial UI with Skia and Yoga | `ovxr-composition-ui/README.md` |
| Emulator, CloudXR, and headset spatial-UI proof | `ovxr-composition-ui-validation/README.md` |
| Live dynamic mesh geometry writes to OVStage Fabric | `ovstage-live-geometry/README.md`, `ovstage-data-plane/README.md` |
| Native library subprocess bridge (Python version mismatch) | `subprocess-geometry-bridge/README.md` |
| SMLib fillet/boolean/tessellation failures and silently wrong results | `smlib-brep-pitfalls/README.md`, `subprocess-geometry-bridge/README.md` |
| Exact BRep persistence alongside a tessellated render mesh | `exact-brep-in-usd/README.md`, `smlib-brep-pitfalls/README.md` |

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
|  +- Browser/web -> READ: streaming-viewer-recipe; follow its current-work links
|  +- Desktop/local (React UI, no Python) -> READ: tauri-local-viewer
|  +- Desktop/local (C++; choose an appropriate native toolkit) -> READ: cpp-native-viewer + viewer-control-patterns
|  +- Desktop/local (React UI, Python server, separate process) -> READ: electron-shm-viewer
|  +- Desktop/local (Python, simple) -> READ: ovui-local-viewer-recipe + local-viewer + ovrtx-rendering + stage-loading; add viewer-control-patterns when controls are visible
|  +- Desktop/local (Python, full editor) -> READ: streaming-vs-local + ovwidgets-editor-shell + viewer-control-patterns
|  +- Headset/XR -> READ: xr-viewer-recipe; use its assembly map for an editor
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
|  +- Projection-integrated XR panel -> READ: ovxr-composition-ui + ovxr-composition-ui-validation
|  +- XR panel off-frustum/head-locked -> READ: ovxr-composition-ui-validation + ovxr-composition-ui/spatial-interaction
|  +- Object picking -> READ: viewer-input-routing + native-picking-selection + object-selection + selection-feedback
|  +- Pick changes a USD/material attribute -> READ: prim-pick-effects + object-selection + ovstage-data-plane
|  +- Object info panel -> READ: prim-info-display + stage-attribute-reads + stage-hierarchy + ovstage-data-plane
|  +- Camera navigation -> READ: viewer-input-routing + camera-controls
|     +- Switchable or tool-specific scheme requested -> also READ: navigation-presets
|  +- Transform gizmo/manipulator -> READ: transform-manipulator + ovstage-data-plane
|  +- Tauri SHM client-rendered gizmo -> READ: tauri-shm-transform-gizmo + tauri-local-viewer + webgl-shm-transport
|  +- C++ GL viewport overlay/gizmo -> READ: gl-viewport-overlay + ovui-library + cpp-native-viewer
|  +- Vulkan raster overlay/compositing/depth -> READ: vulkan-compositing + its selected depth/interop reference
|  +- Scene switching -> READ: stage-management
|  +- Render quality/lighting -> READ: render-settings
|  +- AOV/render-var switching -> READ: aov-switching + ovrtx-rendering + streaming-messages
|  +- SPG/custom GPU processing/lens sampling -> READ: spg-processing; add delivery and aov-switching
|  +- Stylized/NPR/toon/technical/X-ray/feature-line/wireframe visualization -> READ: spg-processing + stylized-visualization; inventory geometry provider, read stylized-visualization/mesh-xray for regular mesh X-Ray, read style-composition-contract for a new unnamed look; add aov-switching when raw render outputs are user-selectable
|  +- Viewport overlays -> READ: viewport-overlays
|  +- Animation -> READ: selection-animation + ovstage-data-plane
|  +- Custom messages -> READ: streaming-messages
|  +- Live mesh geometry swap (no stage rebuild) -> READ: ovstage-live-geometry + ovstage-data-plane
|  +- Native library subprocess (Python version mismatch) -> READ: subprocess-geometry-bridge
|  +- SMLib fillet/boolean failure or silently wrong solid -> READ: smlib-brep-pitfalls
|  +- Exact BRep persisted next to the render mesh -> READ: exact-brep-in-usd
|
+- Infrastructure?
|  +- Cloud assets -> READ: cloud-assets
|  +- Cloud deployment -> READ: cloud-deployment
|  +- Physics simulation (OVPhysX) -> READ: physics-simulation + realtime-sim-integration + object-selection + ovstage-data-plane; clone ovphysx, read its skills/
|  +- Physics simulation (Newton, continuous) -> READ: newton-physics + realtime-sim-integration + ovstage-data-plane; route to the public Newton repo/skills
|  +- User input INTO a running sim (click-to-impulse, shake, teleop, control hook) -> READ: interaction-round-trip + native-picking-selection + streaming-messages
|  +- CAD file import -> READ: https://github.com/NVIDIA-Omniverse/usd-convert-cad/blob/main/skills/omniverse-cad-to-usd/SKILL.md
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
| `ovui-data-adapters-*` | Resolve matching companion distributions through `references/dependencies` | Common interfaces and provider-specific UI adapters |
| Full editor UI package | Install only when current `ovui` dependency guidance explicitly requires it, from the same `ovui` package set | Full editor widgets |
| `ovstorage` | See `dependencies/nvidia-runtime.md` | Cloud asset browsing and cache sync; inspect upstream `AGENTS.md` and `skills/` first |
| `ovphysx` | See `dependencies/nvidia-runtime.md` | Physics simulation; use `physics-simulation/README.md` for viewer integration and OVPhysX skills for tensor APIs |
| `ovnewton` | See `dependencies/nvidia-runtime.md` | Newton physics bridged to OVStage; use `newton-physics/README.md` for viewer integration and the public Newton repo for Newton itself |
| `usd-convert-cad` | Follow [NVIDIA-Omniverse/usd-convert-cad — `omniverse-cad-to-usd`](https://github.com/NVIDIA-Omniverse/usd-convert-cad/blob/main/skills/omniverse-cad-to-usd/SKILL.md) | CAD file conversion to USD; upstream owns installation, versions, platforms, and conversion options |
| `pxr` / OpenUSD | `pip install usd-core` | Use version pins from platform skills |
| `numpy` | `pip install numpy` | Array operations |
| `warp` | `pip install warp-lang` | GPU kernels and CUDA buffer utilities |
| `warp-simdata` (CAE GPU operator library) | See `dependencies/README.md` → CAE/CFD Visualization Libraries (`github.com/NVIDIA/warp-simdata`) | CAE/CFD `Dataset` + visualization operators |
| `cae-openusd-plugins` (CAE USD reader/schema library) | See `dependencies/README.md` → CAE/CFD Visualization Libraries (`github.com/NVIDIA-Omniverse/cae-openusd-plugins`) | Optional CAE file readers (CGNS/OpenFOAM/VTK/EnSight/FLASH) |

## Simulation Ecosystem

These references cover viewer integration, not simulator internals. For Newton, Warp, MuJoCo, USD import, or solver selection, first read the current upstream repository's agent guidance, documentation, package metadata, and examples.

| Need | Read first | Then use here for viewer integration |
|---|---|---|
| Continuous Newton simulation | [Newton repository](https://github.com/newton-physics/newton), its current agent guidance and [docs](https://newton-physics.github.io/newton/stable/) | `newton-physics/README.md`, `realtime-sim-integration/README.md` |
| Input driving a live simulation | The selected engine's current input/control guidance | `interaction-round-trip/README.md` |
| Newton's Warp runtime, graph capture, CUDA timing | [Warp documentation](https://nvidia.github.io/warp/stable/) | `realtime-sim-integration/README.md` |
| Bounded, request-scoped simulation | [OVPhysX guidance](https://github.com/NVIDIA-Omniverse/PhysX/tree/main/ovphysx) | `physics-simulation/README.md` |

For `ovnewton`, inspect the installed wheel's current examples and package metadata before writing an adapter. The application remains the owner of simulation clock, OVStage lifetime, ordinals, render publication, and input routing.

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
CAD conversion -> https://github.com/NVIDIA-Omniverse/usd-convert-cad/blob/main/skills/omniverse-cad-to-usd/SKILL.md
```

## ovstage Runtime Routing

All new viewer implementations begin with `ovstage-runtime`, `ovstage-population`, `ovstage-data-plane`, and `ovstage-ovrtx-integration`. Add them before the focused delivery, camera, selection, hierarchy, and render-setting references. The application owns the runtime scene; ovrtx renders it.

## Solid Modeling: Upstream First

For usd-brep, exact BRep, solid modeling, kernel failures, CAD topology, or
BrepArray questions, first read `dependencies/README.md` → USD BRep. Follow the
owning repository's current agent guidance, skills, operation docs, and package
API before selecting local integration references. Discover the upstream router
rather than assuming skill names or directory layouts remain fixed. Upstream
owns modeling semantics, schema rules, tuning, and diagnostics; this package
owns OVRTX display, worker handoff, selection mapping, and publication timing.

For usd-brep/solid modeling or exact CAD geometry used by stylized viewers,
first read `dependencies/README.md` → USD BRep and the current upstream agent
skills and operation docs, then `stylized-visualization/usd-brep-source.md`.
Discover the owning repository's router rather than hard-coding skill names.
