# Electron SHM Client

## Electron Main And N-API Addon

Electron main starts the Python sidecar or attaches to an existing SHM session,
waits for `shmReady`, and creates one native client and frame pump. The addon
performs frame waits on a libuv worker; main remains available for window events,
input, and shutdown. Main also forwards app messages through the preload API.
Terminate the sidecar on quit only when the app started it.

Use the environment selected during setup to launch the sidecar; see
`windows-native-setup` for Windows. Package the addon for Electron's runtime ABI.

A compact addon interface is:

```typescript
type NativeShmClient = {
  connect(options: { name: string }): void;
  close(): void;
  waitFrame(): Promise<SharedArrayBuffer>;
  sendMessage(message: string): void;
  onMessage(callback: (message: string) => void): () => void;
};
```

## Unified Native Client And Ownership

Wrap the unified API from the matching ovstream headers and libraries:

```cpp
ovstream_client_config_t config = {};
config.shm.stream_name = OVSTREAM_STRING_LITERAL("viewer-stream");
ovstream_client_t* client = nullptr;
auto result = ovstream_create_client(OVSTREAM_CLIENT_SHM, &config, &client);
// Check result before using client.
```

Use `ovstream_client_wait_frame`, `ovstream_client_send_message`,
`ovstream_client_send_input_event`, and `ovstream_destroy_client` on that handle.
Dependency acquisition and upstream API references live in
[dependencies/ovstream](../dependencies/ovstream.md).

The frame pump validates each frame's source, format, dimensions, and pitch,
then promptly copies rows into an app-owned bounded SharedArrayBuffer ring.
Native `ovstream_frame_t` storage is borrowed and may be recycled independently
of the next wait; retaining a pointer does not pin it for asynchronous JS use.
Pack padded rows before delivery. The 16-byte frontend header used by this
recipe is an app IPC convention, not the native frame struct.

Reuse a ring slot only after its JS consumer finishes reading it. Bound pending
frames and drop stale work when presentation falls behind; a larger queue adds
latency. Keep raw native handles and pointer arithmetic inside the addon.

Use bounded wait timeouts so `close()` can stop and join the pump before
`ovstream_destroy_client`. Destroy outside callbacks, keeping callback objects
and user data alive until destruction completes, including deliveries already
in flight when a callback is replaced.

## Preload API

Expose a narrow `contextBridge` API:

```typescript
export type ShmViewerApi = {
  connect(options?: { name?: string }): Promise<ViewerCapabilities>;
  disconnect(): Promise<void>;
  waitFrame(): Promise<SharedArrayBuffer>;
  sendMessage(message: { event_type: string; payload?: unknown }): Promise<void>;
  onMessage(callback: (message: { event_type: string; payload: unknown }) => void): () => void;
  getStatus(): Promise<BackendStatus>;
};
```

Preload rules:

- `contextIsolation: true`
- renderer `nodeIntegration: false`
- expose functions, not raw `ipcRenderer`
- validate message envelopes before sending to main
- do not expose filesystem, shell, child process, native addon, or environment
  access directly to React
- return a protocol version from `connect()`

## ViewerBackend Interface

`useShmBackend.ts` should implement the shared frontend contract. Keep SHM
specifics inside frame metadata and backend internals.

```typescript
export type FrameData = {
  width: number;
  height: number;
  sequence: number;
  format: 'BGRA8' | 'RGBA8';
  buffer: SharedArrayBuffer;
  pixelsByteOffset: number;
  byteLength: number;
};

export type RenderSettingCapability = {
  key: string;
  label: string;
  control: string;
  applies_at: 'immediate' | 'reload_required' | 'next_scene_load' | 'unsupported';
  apply_path: string;
  validated: boolean;
  validation_evidence: string;
};

export type RenderSettingsState = {
  settings: Record<string, unknown>;
  capabilities: RenderSettingCapability[];
};

export interface ViewerBackend {
  connect(): Promise<void>;
  disconnect(): void;
  loadStage(path: string): Promise<{ path: string; tree?: PrimNode[] } | void>;
  resize(width: number, height: number): Promise<void>;
  setCamera(camera: CameraState): Promise<void>;
  cameraMouseButton(input: PointerInput): Promise<boolean>;
  cameraMouseMove(x: number, y: number): Promise<void>;
  cameraWheel(delta: number): Promise<void>;
  onFrame(callback: (frame: FrameData) => void): () => void;
  onSelectionChanged(callback: (paths: string[]) => void): () => void;
  pick(x: number, y: number): Promise<string | null>;
  getStageTree(rootPath?: string): Promise<PrimNode[]>;
  selectPrims(paths: string[]): Promise<void>;
  getProperties(path: string): Promise<PrimProperty[]>;
  getRenderSettings?(): Promise<RenderSettingsState>;
  setRenderSetting?(key: string, value: unknown): Promise<RenderSettingsState>;
}
```

Behavior:

- `connect()` attaches, subscribes to messages, and starts one frame pump.
- `disconnect()` stops the pump before closing native resources.
- `resize()` updates UI layout unless the server implements RenderProduct
  resolution changes and matching frame-storage updates.
- `onFrame()` fans out frames from the single pump to UI subscribers.
- `pick()` sends a request id and resolves on matching response, with timeout.
- hierarchy, properties, selection, AOVs, and settings use JSON messages.
- Render settings panels must render from `RenderSettingCapability[]`; setting
  changes reject unsupported keys and only report success when active viewer
  state changed or an explicit non-live action was accepted.

## React Renderer And WebGL Blit

The React viewport displays server-rendered pixels:

- maintain canvas size and device pixel ratio
- compute letterboxed content rectangle
- map pointer coordinates into render-product pixels
- upload BGRA/RGBA pixels into a WebGL texture
- draw a full-canvas quad
- render overlays, tree, panels, toolbar, and status as DOM UI
- drop stale frames by sequence number

WebGL setup:

```typescript
const gl = canvas.getContext('webgl', {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
  preserveDrawingBuffer: false,
});
```

RGBA upload fallback:

```typescript
gl.bindTexture(gl.TEXTURE_2D, texture);
gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
gl.texImage2D(
  gl.TEXTURE_2D,
  0,
  gl.RGBA,
  frame.width,
  frame.height,
  0,
  gl.RGBA,
  gl.UNSIGNED_BYTE,
  rgbaPixels,
);
```

Do not add 3D engines, scene graph helpers, material systems, model loaders, or
camera libraries for the viewport. Read `webgl-shm-transport` for extension
checks, shader details, canvas resize behavior, and conversion fallback.
