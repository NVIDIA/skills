# Omniverse Realtime Viewer App

## Triggers

Use for a new USD viewer, interactive viewport, or an app combining scene
browsing, picking, properties, and render controls. For an existing viewer,
apply the relevant capability reference within its established architecture.

## Choose The Delivery Path

| User needs | Start with |
|---|---|
| Browser access or a remote GPU | `streaming-viewer-recipe` |
| Packaged React/Rust desktop app | `tauri-local-viewer` |
| Python desktop or in-process UI | `ovui-local-viewer-recipe` |
| Electron UI with an isolated renderer process | `electron-shm-viewer` |
| C++ native window | `cpp-native-viewer` |
| Headless client or automation | `headless-shm-cli` |
| Headset, immersive viewer, or spatial editor | `xr-viewer-recipe` |

Follow `routing.md` for defaults when the user has not chosen a shell. Resolve
only the dependencies needed for that path through `dependencies`. Deployment
is a separate choice from the viewer's rendering and interaction model.

## XR Recipe

For headsets, immersive viewing or spatial editors, start with
[OVXR library construction](../ovxr-library-runtime/README.md). For the combined
editor feature set use its [assembly map](../ovxr-library-runtime/viewer-assembly.md).
Remote execution and browser-based XR clients still use that XR route, not the
ordinary browser streaming recipe. Add a desktop shell only for a requested
companion/mirror.

Shared scene queries, selection state and UX patterns remain useful, but OVXR
owns eye cameras and OpenXR frames. Use the XR frame/input schedule and settings
lifetimes rather than adding desktop orbit-camera writes, a second render step,
or browser mouse routing to the headset path.

## Build One Working Loop

Start with one scene, one camera, one render product, and a visible frame in the
chosen shell. The viewer runtime owns OVStage population/publication and OVRTX
stepping; the shell displays output and sends commands. Use
`ovstage-ovrtx-integration` for the runtime loop and `stage-loading` for the
viewer-owned camera and render-product setup.

Add orbit, pan, zoom, and fit through `viewer-input-routing` and
`camera-controls`. Route input according to the surface under the pointer:
widgets and gizmos consume their gestures before the camera. Verify a click
selects while a drag navigates before adding more controls.

Keep transport-specific code at the boundary. Reusable panels consume a backend
interface for selection, properties, scene state, and commands; see
`viewer-backend-interface`. The same application state should drive the tree,
viewport feedback, and inspector.

## Add The Requested Features

| Feature | Implementation path |
|---|---|
| Pick and inspect | `native-picking-selection`, `object-selection`, then `prim-info-display` |
| Tree and properties | `stage-hierarchy`, `stage-attribute-reads`; fetch detail on demand |
| Scene switching | `stage-management`; preserve chosen camera/settings policy and clear generation-scoped UI state |
| Quality and AOV controls | `render-settings`, `aov-switching`; expose supported outputs and show pending changes |
| Asset browser | `cloud-assets`; browse metadata first, prepare selected assets in the background |
| Transform tools | `transform-manipulator`, `ovstage-data-plane`; coordinate competing writers |
| Simulation | `realtime-sim-integration` plus the chosen engine reference; use `interaction-round-trip` for user actuation |
| Reversible material feedback | `prim-pick-effects`; preserve authored values |

Use `viewer-ux-workflow` to compose the viewport, toolbar, outliner, and inspector
for the user's task. Keep layout stable while content changes. Show loading,
empty, reconnecting, and failed states explicitly; a stale image should not look
like a live successful render.

## Keep Interaction Responsive

Drain commands at runtime frame boundaries. Coalesce superseded camera/resize
requests while preserving ordered press/release and additive edits. Scene
replacement uses the latest requested scene; cancel or ignore superseded loads. Load assets and query metadata outside the rendering critical section;
only commit prepared work through the runtime owner. For streamed apps, use the
chosen delivery recipe's frame-continuity and buffer-lifetime contract.

## Verify The User Journey

Run the built app in its actual shell: open a scene, navigate, select an object,
inspect it, change a requested setting, and switch scenes. Exercise the delivery
path's resize, disconnect/reconnect, and shutdown behavior where applicable.
Capture working frames and UI states using `validation.md`; report unavailable
runtime coverage separately from successful builds.

## Processed GPU outputs

For custom post-processing, chained resources or lens sampling, read
[SPG processing](../spg-processing/README.md). For stylized image passes,
technical/X-Ray geometry strokes or explicit full wireframe, continue to
[stylized visualization](../stylized-visualization/README.md). Present the actual
published processed output through this viewer's existing owner and transport;
a menu selection or display conversion alone does not implement the effect.
