# Build an XR viewer end to end

Use this route for the combined feature set, not for a minimal headset probe.
The inputs are a matched OVXR/OVRTX/OVStage SDK, the selected OpenXR/CloudXR
runtime, native graphics build tools, and this skill folder. A supplied scene
is content, not an implementation dependency. The application modules below
are yours to create; their names are illustrative, not imports from a sample.

Choose the requested scope before following the table. A minimal viewer needs
bootstrap, a source-safe scene and first stereo pixels. Add input/navigation for
interaction, spatial UI for an editor, and desktop delivery only for a companion.
For user-facing surfaces read [experience quality](../ovxr-composition-ui/experience-quality.md)
alongside the implementation: readable layout, discoverable tools and measured
interaction-time performance are part of the result. Shared UI/scene references
supply semantic patterns; the XR references own eye cameras, frame timing,
physical placement and input capture. Do not graft a desktop render loop onto XR.

| Build increment | Implementation contract | Acceptance before adding the next layer |
| --- | --- | --- |
| Product layout and interaction brief (when UI is requested) | [Experience quality](../ovxr-composition-ui/experience-quality.md), [editor state](../ovxr-composition-ui/editor-document-model.md) | Declared surfaces, anchors, theme, gesture hints and target frame-time budget |
| Package bootstrap, service and one render owner | [Library construction](../ovxr-library-runtime/library-only.md), [dependency owner](../dependencies/nvidia-runtime.md#ovxr) | Native loads, stage population, first stereo pixels and head motion |
| Source-safe scene and navigation | [Session composition](../xr-spatial-input/stage-composition.md), [anchor math](../xr-spatial-input/navigation.md) | Non-origin spawn, Y/Z up, scale and reset without modifying source USD |
| Native controllers and hands | [Input routing](../xr-spatial-input/README.md), [hand validity](../xr-spatial-input/hands.md) | Actual extension/action/pose evidence, one converted snapshot and cancellation on loss |
| Selection and scene tools | [Controllers](../xr-spatial-input/controllers.md), [interaction service](../ovxr-python-interactions/README.md) | Ray hit, one-click selection replacement, no-jump transform, bounded scene-query work |
| Teleport | [Navigation](../xr-spatial-input/navigation.md), [gesture math](../xr-spatial-input/gesture-filtering.md) | Swing steers, relative twist faces, valid release commits, cancel requires both-open/fresh-pinch rearm |
| Retained windows and toolbar | [Composition UI](../ovxr-composition-ui/README.md), [Skia/Yoga](../ovxr-composition-ui/skia-yoga.md), [raster ABI](../ovxr-composition-ui/raster-abi.md) | Crisp text plus matching hit regions; stalled raster work does not block XR |
| SDK frames and an application-owned compositor | [SDK backend](../ovxr-composition-ui/sdk-frame-backend.md), [native compositor construction](../ovxr-composition-ui/library-only-compositor.md) | One begin/bind/submit owner, coordinated stage publication, stereo scene-depth intersections |
| Grab bars, fist movement, layered tools | [Surface interaction](../ovxr-composition-ui/spatial-surfaces-and-interaction.md), [depth tools](../ovxr-composition-ui/depth-tools.md) | Persistent handle selection, scene selection deselects handle, smooth accelerated movement, UI above gizmos |
| Passthrough and supported settings | [Alpha](../ovxr-library-runtime/ar-compositing.md), [AR control policy](../ovxr-library-runtime/live-ar-controls.md) | Correct alpha, material-specific fixtures, live head motion after writes, reconnect persistence |
| Desktop mirror/control shell | [Mirror lifetime](../ovxr-library-runtime/mirror-lifetime.md), [desktop mirror](../xr-desktop-mirror/README.md), selected desktop-shell reference | Same eye source, bounded owned copy, no second XR renderer, truthful stale/unavailable state |
| Performance and closure | [Timings](../ovxr-composition-ui/performance-and-telemetry.md), [spatial validation](../ovxr-composition-ui-validation/README.md) | Matched connected-session capture, counter budgets, loss/reconnect and physical headset checks |

## Select supported capabilities before implementation

This bundle describes application construction; it does not certify every
optional runtime/client feature. Resolve the selected packages and report each
requested capability as supported, conditional, unavailable, or unverified
before promising the integrated result.

| Requested feature | Required implementation boundary |
| --- | --- |
| Depth-aware primary spatial UI | Baseline: Vulkan, nonfoveated projection, required negotiated depth, and application-built OpenXR API layer with attachment-capable swapchains. Follow the compositor construction route. |
| Warped/foveated primary UI | Requires the runtime/vendor's actual projection/warp metadata and mapping. The bundle cannot invent a missing PQW contract; retain the nonfoveated baseline or report the requested mode unavailable. |
| Live AR/material controls | Requires a documented public apply path for the selected renderer/SDK. Otherwise keep policy staged or controls unavailable; arbitrary writes and context recreation do not establish support. |
| Hands, controller profiles and passthrough | Require observed extensions, valid input, client support and target-device evidence. Emulator success is narrower than physical-headset proof. |
| Independent unwarped mirror | Requires a supported negotiated capture mode and measured output; preserve headset rendering and report preview unavailable when capture cannot meet it. |

Native SDK headers/bindings and observed OpenXR create/submission data are the
integration inputs. An internal implementation checkout or an unavailable
source archive is not a prerequisite. A blocked required feature remains
blocked in the handoff; successful startup must not silently reduce its scope.

## Implementation seams

The table above is a build map, not the algorithms. Read these focused contracts
when implementing the corresponding boundary:

| Boundary | Detailed reference |
| --- | --- |
| Predicted input, anchor latch and stage writes | [Frame/input order](../ovxr-library-runtime/frame-input-order.md) |
| Native layer compilation, exports and discovery | [Native build](../ovxr-composition-ui/native-build-and-discovery.md) |
| Clip/FOV/subimage/depth encoding | [Projection and depth](../ovxr-composition-ui/projection-and-depth.md) |
| GPU pass order, barriers and texture retirement | [Compositor passes](../ovxr-composition-ui/compositor-passes-and-sync.md) |
| Stage tree, toolbar and settings state | [Editor model](../ovxr-composition-ui/editor-document-model.md) |
| Multi-window raster/image/hit-map handoff | [Async raster](../ovxr-composition-ui/async-raster-pipeline.md) |
| Pinch/fist features and tool precedence | [Recognizers](../xr-spatial-input/hand-recognizers.md), [arbitration](../xr-spatial-input/gesture-arbitration.md) |
| Walkability, final query and yaw/position commit | [Teleport acceptance](../xr-spatial-input/teleport-acceptance.md) |
| Parented scene manipulation | [Transform safety](../prim-transform-safety/README.md) |
| Supported AR updates versus saved policy | [Available AR apply paths](../ovxr-library-runtime/live-ar-controls.md#select-an-available-apply-path) |

Keep renderer, navigation, selection, input and UI authoritative state in one
application. Transport handlers enqueue intent; they do not mutate SDK objects.
Separate immutable scene extraction, retained raster work, and mirror consumers
from the frame owner. Store stage/context/input generations on async results and
reject stale completions. Pair each raster with its own hit map.

Capabilities must be reported, not assumed: missing native hands, split-frame
exports, compositor build tools, compatible depth attachments or material
settings are explicit blocked features. Additional-layer support alone is not
depth-aware UI. A self-contained recipe cannot supply a missing SDK capability;
do not hide that boundary with private APIs or a second rendering pipeline.

For static review, a developer should be able to name every application module,
its inputs/outputs, and the owning library API without opening the sample repo.
For runtime completion, require the relevant scene/input/UI story on the target
client. Portable mathematical recipes and simulator captures do not prove AVP
or Quest ergonomics, display edges, or controller bindings.

## Generated application handoff

Leave an application-owned build/run entry point, explicit SDK-root configuration,
native layer/shader build, raster-worker lockfile and licensed font/icon assets.
Ship the source and matching ABI declarations for the custom compositor/bridge;
the SDK alone does not contain them. Keep runtime packages external unless their
redistribution is authorized. Record capability failures without installing
private fallback packages or silently removing requested features.

Provide a feature ledger mapping each requested behavior to its module, owning
library boundary, test and current result. Include the three separate functional,
visual and performance results from the experience guide. Run from an unrelated
working directory with explicit configuration to catch sample-relative imports,
machine-specific paths and missing generated shaders/assets. A startup/import
pass is not completion of the native, GPU or physical-client gates.

## Client capability record

Choose transport from the client, not merely the headset name:

| Client path | Service profile | Input/passthrough gate |
| --- | --- | --- |
| AVP native CloudXR client | Matching native profile from the selected CloudXR distribution | Runtime hand extension and controller actions must be observed; alpha must survive client presentation |
| Quest native CloudXR client, if supplied | Compatible native profile from that distribution | Do not infer identical bindings or hand support from AVP |
| Quest controlled browser/WebXR client | Matching WebRTC profile from the selected CloudXR distribution | Native streamed controller input and separately labeled client hand telemetry; request/check browser hand and AR presentation capabilities |

Record client/runtime build, negotiated extensions and blend/depth/foveation,
actual action profile, per-hand validity, and one correlated visual test per
required feature. Unavailable is a supported diagnostic state, not a pass.
Keep device-specific gesture ergonomics configurable; the skill equations do
not override headset system gestures. Reuse the central runtime/dependency
guidance for setup rather than introducing device-specific private SDK paths.
