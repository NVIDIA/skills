# Streaming Omniverse Realtime Viewer Recipe

## Triggers

Use this skill for streaming Omniverse Realtime Viewer, browser-streamed Omniverse Realtime Viewer, complete WebRTC Omniverse Realtime Viewer, build a streaming Omniverse Realtime Viewer, React Omniverse Realtime Viewer over WebRTC, fixed stream resolution, remote GPU browser viewer, or broad WebRTC viewer requests.

Use this as the primary entry point when building a complete Omniverse Realtime Viewer streamed to a browser. Build the server render loop and browser connection first, then add scene interaction features through the data channel.

## Read Order

Load only the reference files needed for the current phase:

| Phase | Read |
|---|---|
| Decide project shape and non-negotiable rules | `project-structure.md` |
| Build Python server, renderer, scene loader, and video stream | `server-runtime.md` |
| Build React client, WebRTC setup, data-channel protocol, UI wiring | `client-protocol.md` |
| Add input routing, camera, picking, selection, scene switching, hierarchy, properties, settings | `interaction-features.md` plus `viewer-input-routing` for transport details |
| Validate behavior and order implementation work | `validation-build-order.md` |

## Critical Rules

- Before writing code, read `dependencies` for exact install commands, package guidance, and supplemental runtime documentation. If a task needs behavior beyond this recipe, keep the implementation within documented APIs and local skill contracts.
- For generated browser viewers, do not report completion after a frontend build alone. Attempt server dependency installation, server startup, and first-frame readiness validation unless the user explicitly opts out or the platform is unsupported. If runtime setup fails, report the exact failing command and the dependency reference to re-check.
- Do not use WebGL, Three.js, Babylon.js, or any browser-side 3D renderer. The browser displays an `ovstream` WebRTC video stream from server-side `ovrtx` rendering.
- Keep the streaming app split into a Python server process and a React browser client.
- Stream rendered pixels through `ovstream`; use JSON data-channel messages only for app state and commands.
- Use NVST native input forwarding for mouse, keyboard, wheel, and touch input. A browser JSON mouse fallback is permitted only after `viewer-input-routing` records a working data channel and a failed native callback test; it replaces, never duplicates, native input.
- Make one render thread the sole owner of `renderer.step()`, stage mutation, native picking, selection outline writes, and live `write_attribute()` calls.
- In OVStage-backed apps, the same runtime owner also owns stage lifetime,
  ordinals, write floors, and renderer publication. Do not pass live OVStage
  handles, mapped buffers, or DLPack objects to the browser, USD workers, or
  physics workers.
- Register ovstream callbacks before starting the server.
- Set `OVRTX_SKIP_USD_CHECK=1` before ovrtx work.
- Keep `pxr`/OpenUSD queries and OVPhysX USD population behind process
  boundaries unless the exact ABI/import path is verified. A failed
  `PhysX.attach_ovstage(stage, read_ordinal=...)` bridge must not fall back to parent-process
  OVPhysX population.
- Keep stream resolution fixed for a session and display video with `object-fit: contain`.
- Treat `/healthz` readiness as server-render proof: return ready after the first valid ovrtx frame has been converted and copied into the app-owned stream buffer, not when a browser connects. Lack of a connected client is a guarded-send condition, not a readiness failure.
- Never modify user USD files when adding viewer camera, render products, render vars, settings, selection metadata, or inline session data.

## Generated App Setup Contract

When generating a browser-streamed viewer, create app-local setup and run wrappers
for the target project. Do not rely on pre-existing applications or
repository-level helper scripts.

The generated setup flow must:

- create a project-local Python virtual environment;
- install `ovrtx`, `ovstage` when the runtime uses OVStage, `ovstream`,
  `warp-lang`, and `numpy` using `references/dependencies`;
- install `ovphysx` only in the selected physics worker or verified
  attached-stage test environment;
- install `usd-core` only when the generated server includes a `pxr` query worker, resolving a release compatible with the selected runtime package set;
- run import and lifecycle checks for `ovrtx`, `ovstream`, `warp`, and the selected `pxr` subprocess path;
- construct an `ovrtx.Renderer` once on the target GPU and attach/use the
  selected OVStage path only after the current dependency guidance has been
  verified;
- record the commands and results in the validation output.

The generated run wrapper must set the required runtime environment, start the
Python server, and expose the server log path. It should derive package paths
from installed Python packages rather than copying paths from local checkouts or
older recipes.

## Build Order

1. Create the project skeleton and establish server/client boundaries.
2. Install dependencies and validate GPU/runtime availability.
3. Build the server runtime shell and renderer construction.
4. Add scene loading and frame streaming.
5. Bring up the React WebRTC client and data-channel router.
6. Add input routing, camera controls, selection, scene switching, hierarchy/properties, and render settings.
7. Wire frontend panels through shared viewer backend concepts.
8. Capture validation and review evidence.

See also: `usd-viewer-app`, `streaming-server`, `streaming-client`, `streaming-messages`, `streaming-lifecycle`, `ovrtx-rendering`, `stage-loading`, `viewer-input-routing`, `camera-controls`, `object-selection`, `selection-feedback`, `stage-hierarchy`, `stage-management`, `render-settings`, and `viewer-ux-workflow`.

For live data/scene editing driven from browser controls — a React control → data-channel message → server op mapping to either a cheap data-plane array/attribute write or a structural rebuild — see `cae-cfd-visualization/driving-cae-viz-via-ovstage.md` (the generalized "control → correct write" pattern) and `cae-cfd-visualization/streaming-cae-viewer.md` (the CAE-specific server runtime, message set, and control → server-op flow that compose over this recipe). To layer CAE/CFD visualization on a browser stream, start at `cae-cfd-visualization/README.md`.
