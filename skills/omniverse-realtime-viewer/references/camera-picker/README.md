# Camera Picker

## Purpose

Stages with multiple authored cameras (e.g., a warehouse with top-down,
perspective, dock-level, and aisle views) should expose those cameras to the
user via a simple dropdown. This skill implements the full round-trip: server
enumerates cameras → frontend renders dropdown → user selects → server switches
active camera → stream updates.

## Triggers

Use when:
- The stage contains two or more `UsdGeom.Camera` prims.
- A user asks for a camera selector, camera picker, view switcher, or viewport
  dropdown.
- Building a viewer that must support multiple viewpoints.
- The `camera-auto-select` skill detected multiple cameras and the app should
  let users explore them.

## Message Protocol

### Server → Client: `camera_list`

Sent once after stage load (or stage switch) alongside or after `push_initial_state`:

```json
{
  "event_type": "camera_list",
  "payload": {
    "cameras": [
      {
        "path": "/World/Cameras/Cam_Persp",
        "name": "Cam_Persp",
        "focalLength": 35.0
      },
      {
        "path": "/World/Cameras/Cam_TopDown",
        "name": "Cam_TopDown",
        "focalLength": 50.0
      },
      {
        "path": "/World/Cameras/Cam_DockLevel",
        "name": "Cam_DockLevel",
        "focalLength": 35.0
      },
      {
        "path": "/World/Cameras/Cam_Aisle",
        "name": "Cam_Aisle",
        "focalLength": 28.0
      }
    ],
    "activeCamera": "/World/Cameras/Cam_Persp"
  }
}
```

### Client → Server: `set_camera`

User selects a different camera:

```json
{
  "event_type": "set_camera",
  "payload": {
    "path": "/World/Cameras/Cam_TopDown"
  }
}
```

### Server → Client: `camera_changed`

Confirms the switch (allows UI to sync if multiple clients are connected):

```json
{
  "event_type": "camera_changed",
  "payload": {
    "activeCamera": "/World/Cameras/Cam_TopDown"
  }
}
```

## Server Implementation

### Attached OVStage Runtime Contract

For an attached OVStage viewer, camera enumeration and selection belong to the
application-owned runtime adapter. It returns copied camera DTOs to the
transport layer and owns all live reads, transforms, lens updates, ordinal
publication, and stage-generation checks. Do not open another `Usd.Stage` or
read/write camera state through standalone renderer APIs in this path.

The `pxr` example below is suitable only for a separate offline/preflight tool.
It must not be called from the live attached viewer.

### Offline / Preflight Enumeration

```python
from pxr import Usd, UsdGeom


def get_camera_list(stage: Usd.Stage) -> list[dict]:
    """Return all authored cameras suitable for the picker."""
    cameras = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            cameras.append({
                "path": str(prim.GetPath()),
                "name": prim.GetName(),
                "focalLength": cam.GetFocalLengthAttr().Get() or 50.0,
            })
    return cameras
```

### Handling `set_camera` in an Attached Viewer

When the server receives `set_camera`:

```python
def handle_set_camera(self, payload: dict) -> None:
    camera_path = payload.get("path", "")
    camera = self.runtime.camera_queries.get_camera(camera_path)
    if camera is None:
        self.send_error(f"Invalid camera path: {camera_path}")
        return

    # The adapter validates the active generation, copies the authored camera
    # into the session/orbit camera, waits, and publishes the resulting ordinal.
    committed = self.runtime.camera_commands.use_authored_camera(camera_path)
    self.active_camera = camera_path
    self.broadcast({
        "event_type": "camera_changed",
        "payload": {
            "activeCamera": camera_path,
            "stageGeneration": committed.stage_generation,
        },
    })
```

`camera_queries` and `camera_commands` are application adapter boundaries, not
literal OVStage API names. Build them from the current OVStage query/data-plane
guidance. The command must publish before the renderer consumes the returned
ordinal.

### Sending Camera List on Stage Load

In the stage load handler, after `push_initial_state`:

```python
def on_stage_loaded(self) -> None:
    # ... existing push_initial_state logic ...

    camera_state = self.runtime.camera_queries.list_authored_cameras()
    cameras = camera_state.cameras
    if cameras:
        active = camera_state.default_camera_path or cameras[0]["path"]
        self.active_camera = active
        self.broadcast({
            "event_type": "camera_list",
            "payload": {
                "cameras": cameras,
                "activeCamera": active,
                "stageGeneration": camera_state.stage_generation,
            },
        })
```

## Frontend Implementation (React)

### CameraPicker Component

```tsx
import React from "react";

interface CameraInfo {
  path: string;
  name: string;
  focalLength: number;
}

interface CameraPickerProps {
  cameras: CameraInfo[];
  activeCamera: string;
  onSelect: (path: string) => void;
}

export function CameraPicker({ cameras, activeCamera, onSelect }: CameraPickerProps) {
  if (cameras.length < 2) return null; // No picker needed for 0-1 cameras

  return (
    <div className="camera-picker">
      <label htmlFor="camera-select">Camera</label>
      <select
        id="camera-select"
        value={activeCamera}
        onChange={(e) => onSelect(e.target.value)}
      >
        {cameras.map((cam) => (
          <option key={cam.path} value={cam.path}>
            {formatCameraName(cam.name)} ({cam.focalLength}mm)
          </option>
        ))}
      </select>
    </div>
  );
}

function formatCameraName(name: string): string {
  // "Cam_TopDown" -> "Top Down", "Cam_Persp" -> "Persp"
  return name
    .replace(/^Cam_?/i, "")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ")
    .trim() || name;
}
```

### Wiring Into the App

```tsx
function ViewerApp() {
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [activeCamera, setActiveCamera] = useState("");

  useEffect(() => {
    // Listen for camera_list from server
    stream.on("camera_list", (payload) => {
      setCameras(payload.cameras);
      setActiveCamera(payload.activeCamera);
    });

    stream.on("camera_changed", (payload) => {
      setActiveCamera(payload.activeCamera);
    });
  }, []);

  const handleCameraSelect = (path: string) => {
    stream.send({ event_type: "set_camera", payload: { path } });
  };

  return (
    <div className="viewer">
      <header className="toolbar">
        <CameraPicker
          cameras={cameras}
          activeCamera={activeCamera}
          onSelect={handleCameraSelect}
        />
      </header>
      <VideoViewport />
    </div>
  );
}
```

### Styling

Place the picker in the toolbar/header bar alongside other controls (render
settings, scene tree toggle, etc.). Keep it compact:

```css
.camera-picker {
  display: flex;
  align-items: center;
  gap: 8px;
}

.camera-picker select {
  padding: 4px 8px;
  border-radius: 4px;
  background: var(--surface-2);
  color: var(--text-primary);
  border: 1px solid var(--border);
  font-size: 13px;
}

.camera-picker label {
  font-size: 12px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
```

## Behavior Rules

1. *Hide the picker when ≤1 camera exists.* If the stage has zero or one
   camera, the dropdown adds no value. The `camera-auto-select` skill handles
   the default; no UI needed.

2. *Show the picker when ≥2 cameras exist.* Even if auto-select picked a good
   default, the user should be able to explore other views.

3. *Include an "Orbit (free)" entry* when the viewer supports free orbit mode.
   Selecting it returns to the user-controlled orbit camera without snapping to
   any authored camera:

   ```tsx
   <option value="__orbit__">Free Orbit</option>
   ```

4. *Preserve orbit state on switch.* When the user selects an authored camera,
   apply its transform to the orbit controller. The user can then orbit from
   that starting point. Switching cameras does not lock the viewport.

5. *Re-emit `camera_list` on stage switch.* If the user loads a different
   stage, the old camera list is stale. Treat it like a fresh load.

6. *SHM/Electron path.* Same message protocol over the SHM JSON channel. The
   Electron renderer handles `camera_list` and `camera_changed` identically to
   WebRTC.

## Keyboard Shortcuts (Optional)

For power users, bind number keys to cameras:

| Key | Action |
|-----|--------|
| `1` – `9` | Switch to camera at that index in the list |
| `0` | Free orbit mode |

Only activate when the viewport has focus (not when typing in a text field).

## Gotchas

- Authored cameras may have different aspect ratios or clipping planes.
  When switching, update the render product resolution or adjust vertical
  aperture to match the stream aspect (see `camera-controls` skill).
- Some stages nest cameras inside referenced assets (props). Filter to
  cameras that are direct children of a `Cameras` Xform or at the scene root
  to avoid showing internal asset cameras.
- The runtime camera adapter must preserve both Y-up and Z-up orientation when
  it converts an authored camera into orbit state.
- Drop delayed selection commands and camera-list messages when their
  `stageGeneration` no longer matches the active runtime.

## See Also

- `camera-auto-select` — picks the initial camera; picker shows the alternatives.
- `camera-controls` — orbit, pan, zoom after a camera is chosen.
- `streaming-messages` — message protocol patterns.
- `stage-loading` — stage open lifecycle.
