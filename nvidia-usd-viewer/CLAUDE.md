# USD Viewer — RTX Streaming Tutorial

> Scaffolded from the prompt: **"Build me a USD viewer with RTX rendering, with a browser UI and camera controls."**

## Skill Required

Install the `omniverse-realtime-viewer` skill before working on this codebase:

```bash
npx skills add nvidia/skills --skill omniverse-realtime-viewer --agent claude-code
```

The skill provides the full implementation contract for `ovrtx`, `ovstream`, camera controls,
USD scene loading, and WebRTC browser streaming.

## Architecture

```
Browser (React + @nvidia/ov-web-rtc)
  └─ <video> WebRTC stream ←─ ovstream ←─ ovrtx.Renderer (GPU)
     JSON data channel    ←─→ message_router
     NVST input           ──→ input_router → camera_controller
```

## Runtime Requirements

| Requirement | Version |
|-------------|--------|
| NVIDIA GPU (RTX) | Required |
| Linux x86_64 or Windows x86_64 | — |
| Python | 3.10–3.13 |
| CUDA driver | 12.x |
| Node.js | ≥ 18 |

## Quick Start

```bash
./setup.sh      # Install Python venv + npm deps + validate GPU
./run.sh        # Start streaming server
# Open http://localhost:5173 in a Chromium-based browser
```

## Key Contracts (from omniverse-realtime-viewer skill)

- All rendering is server-side via `ovrtx` — never add Three.js / WebGL / Babylon.js.
- `ovstream` sends BGRA8 frames via WebRTC. Mouse/keyboard input flows via the NVST
  binary input channel (not JSON messages).
- One render thread owns `renderer.step()`, stage mutations, and `write_attribute()`.
- User USD files are **never modified**. Camera and RenderProduct are injected via an
  inline USDA root that sublayers the user file.
- `/healthz` returns `503` until the first valid BGRA frame is buffered; only then `200`.

## File Map

```
server/                      Python streaming server
  ov_web_viewer_server.py    Entry point — startup sequence and render loop
  renderer_runtime.py        ovrtx.Renderer, step(), LdrColor extraction
  scene_loader.py            Builds inline USDA root (sublayers user scene)
  stream_server.py           ovstream WebRTC callbacks and frame submission
  frame_converter.py         RGBA8 → BGRA8 conversion
  camera_controller.py       Orbit / pan / zoom from NVST input events
  message_router.py          JSON data-channel protocol dispatch
  input_router.py            Decodes NVST binary InputEvent structs
  stage_queries.py           USD prim discovery via ovrtx native queries
  settings_store.py          Persistent render settings (JSON)
  config.py                  CLI argument parsing
frontend/                    React + Vite TypeScript
  src/streaming/
    StreamingProvider.tsx    @nvidia/ov-web-rtc AppStreamer lifecycle
    streamingConfig.ts       URL-param based connection config
    messages.ts              Message factory helpers
  src/components/
    Viewport.tsx             <video> element + connection overlay
    Toolbar.tsx              Scene URL input and load button
    StatusBar.tsx            Connection status indicator
  src/types/messages.ts      TypeScript message types
assets/samples/scene.usda   Sample USD scene (cube + lights)
brev.json                    Brev GPU-backed cloud deployment
```

## Deployment (Brev)

See `brev.json` for GPU-backed cloud deployment on AWS g5.xlarge.
Reference: `skills/omniverse-realtime-viewer/references/cloud-deployment/`
