# Electron + SHM USD Viewer

## Triggers

Use this skill for local separate-process USD viewer, Python ovrtx server, ovstream POSIX shared memory frame transport, Electron main process, N-API SHM client addon, React renderer, WebGL pixel upload/blit, React desktop UI, or separate local process.

Use this when a viewer runs on the GPU workstation, needs Electron/React
desktop UI, and should keep Python/OVRTX isolated from Electron. The Python
server owns OVRTX, OVStage/session state when selected, stage lifetime, camera
state, picking, selection, hierarchy queries, render settings, and any
physics-worker handoff. Electron displays pixels and hosts the UI.

## Read Order

| Need | Read |
|---|---|
| Choose Electron + SHM, understand global rules, architecture, project skeleton | `architecture-project.md` |
| Build Python OVRTX runtime and shared-memory frame server | `python-shm-server.md` |
| Build Electron main process, N-API addon, preload API, React/WebGL blit | `electron-client.md` |
| Wire JSON protocol, input, camera, picking, scene state, lifecycle, dev workflow | `protocol-interaction-lifecycle.md` plus `viewer-input-routing` for gesture semantics |
| Validate behavior and avoid common mistakes | `validation.md` |

## Critical Rules

- Use this path only for local GPU-workstation apps where Python should stay separate from Electron.
- Do not use Electron WebGL for USD rendering; WebGL may only blit already-rendered OVRTX pixels.
- Keep Python/ovrtx as the authoritative owner of USD, picking, camera state, selection, hierarchy, render settings, and renderer mutation.
- Keep OVStage/OVPhysX work behind the Python runtime boundary: the parent
  server owns live OVStage state, and OVPhysX USD population runs in a bounded
  worker unless `PhysX.attach_ovstage(stage, read_ordinal=...)` is verified for the exact installed
  ABI.
- Keep the frontend behind the shared `ViewerBackend` shape where possible so UI can share concepts with streaming and Tauri paths.
- Use native SHM input APIs for local input transport; do not invent JSON mouse input for camera control.
- Read `dependencies` before implementing package setup. For ovrtx renderer
  behavior or ovstream SHM behavior beyond this architecture, use the
  supplemental dependency documentation referenced by `dependencies`.

See also: `webgl-shm-transport`, `viewer-backend-interface`, `headless-shm-cli`, `viewer-input-routing`, `streaming-messages`, `streaming-vs-local`, `ovui-local-viewer-recipe`, `tauri-local-viewer`, and `streaming-viewer-recipe`.
