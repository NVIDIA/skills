# USD Viewer with RTX Rendering

A browser-based OpenUSD viewer with RTX-rendered output, real-time WebRTC
streaming, and interactive camera controls — built with NVIDIA Omniverse
libraries and the `omniverse-realtime-viewer` Claude Code skill.

## What this builds

| Component | Technology |
|-----------|------------|
| RTX rendering | `ovrtx` (NVIDIA Omniverse RTX renderer) |
| WebRTC streaming | `ovstream` |
| Browser client | React + `@nvidia/ov-web-rtc` AppStreamer |
| Camera controls | Orbit/pan/zoom via NVST native input |
| USD scene loading | Inline USDA sublayering (non-destructive) |
| GPU deployment | Brev launchable (AWS g5.xlarge) |

## Architecture

```
┌─────────────────────────────────────────┐
│  Browser (React)                        │
│  ┌────────────────────────────────────┐ │
│  │  <video>  WebRTC stream            │ │
│  │  Toolbar  JSON data channel        │ │
│  └─────────────────┬──────────────────┘ │
└────────────────────┼────────────────────┘
                     │ WebRTC / NVST input
┌────────────────────▼────────────────────┐
│  Python Server                          │
│  ovstream  ←←  ovrtx.Renderer (GPU)    │
│  message_router  camera_controller      │
│  input_router    scene_loader           │
└─────────────────────────────────────────┘
```

## Requirements

- **NVIDIA GPU** (RTX series) with CUDA 12.x driver
- Linux x86_64 (or Windows x86_64)
- Python 3.10–3.13
- Node.js ≥ 18
- Chromium-based browser

## Quick start

```bash
# 1. Install everything
./setup.sh

# 2. Start the RTX streaming server
./run.sh

# 3. Open the browser UI
open http://localhost:5173
```

The server streams at 1920×1080, 30 fps by default. The browser displays the
WebRTC video stream and sends JSON commands through the data channel.

## Camera controls (forwarded via NVST input)

| Action | Input |
|--------|-------|
| Orbit  | Left-mouse drag |
| Pan    | Right-mouse drag |
| Zoom   | Scroll wheel |

## Loading a USD scene

Type a path or URL in the toolbar and press **Load** (or Enter):

```
assets/samples/scene.usda
omniverse://my-server/projects/scene.usd
/absolute/path/to/scene.usda
```

Or send a message programmatically:
```javascript
send(JSON.stringify({ event_type: 'openStageRequest', payload: { url: 'path/to/scene.usd' } }))
```

## Server CLI options

```
--width       Stream width  (default 1920)
--height      Stream height (default 1080)
--fps         Target FPS    (default 30)
--signaling-port  WebRTC signaling port (default 49100)
--http-port   Health check port (default 8888)
--public-ip   Public IP for WebRTC ICE (default 127.0.0.1)
--scene       Initial USD scene to load
--asset-root  Root directory for relative scene paths
--gpu         GPU device index (default 0)
```

## Cloud deployment (Brev)

See `brev.json`. One-click deploy on an NVIDIA GPU instance:

```bash
brev open https://github.com/itamariliuk/nvidia-usd-viewer-tutorial
```

## Project structure

```
nvidia-usd-viewer/
├── server/                  Python streaming server
│   ├── ov_web_viewer_server.py  Entry point
│   ├── renderer_runtime.py      ovrtx renderer
│   ├── scene_loader.py          USD loading
│   ├── stream_server.py         ovstream WebRTC
│   ├── frame_converter.py       RGBA→BGRA
│   ├── camera_controller.py     Orbit/pan/zoom
│   ├── message_router.py        JSON protocol
│   ├── input_router.py          NVST input
│   ├── stage_queries.py         USD queries
│   ├── settings_store.py        Persistent settings
│   └── config.py
├── frontend/                React + Vite
│   └── src/
│       ├── streaming/
│       │   ├── StreamingProvider.tsx
│       │   ├── streamingConfig.ts
│       │   └── messages.ts
│       └── components/
│           ├── Viewport.tsx
│           ├── Toolbar.tsx
│           └── StatusBar.tsx
├── assets/samples/scene.usda
├── setup.sh
├── run.sh
└── brev.json
```

## References

- [NVIDIA Omniverse ovrtx](https://github.com/NVIDIA-Omniverse/ovrtx)
- [omniverse-realtime-viewer skill](https://github.com/itamariliuk/skills/tree/main/skills/omniverse-realtime-viewer)
- [Brev deployment](https://brev.dev)
