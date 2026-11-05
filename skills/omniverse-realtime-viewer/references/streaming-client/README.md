# Streaming Client

## Triggers

Use this skill for React streaming client, AppStreamer, DirectConfig,
standalone ovstream Direct WebRTC, browser-streamed Omniverse Realtime Viewer,
remote-video, StreamingContext, VITE_SERVER_HOST, frontend no video, or
object-fit contain.

Use this for the browser side of a WebRTC Omniverse Realtime Viewer backed by
standalone `ovstream`. For standalone ovstream hosted on Brev, OKAS, or another session orchestrator, use the Direct profile when the exposed transport is standalone Direct signaling. For NVCF self-hosted deployment, first read `cloud-deployment/nvcf-self-hosted.md` and follow the upstream workflow for the exposed media and browser connection model; do not guess a client profile.

## NVCF Browser Handoff

For NVCF, read [the deployment handoff](../cloud-deployment/nvcf-self-hosted.md)
before using the Direct example below. Follow upstream deployment skills and
[the browser SDK's NVCF configuration guidance](https://docs.omniverse.nvidia.com/ov-web-sdk/latest/web-streaming-library/overview.html#configure-nvcf),
checking the documentation/types/examples shipped with the selected released
package. Use its supported NVCF profile and the signaling/session routes
provided by that deployment. Record whether session creation is handled by the
application proxy or by the SDK; do not enable a session REST operation without
a backend that implements it. Keep credentials in the deployment's supported
credential boundary rather than embedding service keys in browser assets.

Changing a transport profile requires a fresh SDK stream instance or a fresh
page; verify the effective configuration rather than reusing a cached stream.
Prove decoded frames through NVCF before integrating the full UI. A hosting
provider's HTTPS endpoint alone does not determine the browser profile or prove
UDP media reachability. See also
[network diagnostics](../cloud-deployment/webrtc-network-diagnostics.md).

## AppStreamer Direct API

Install the current released `@nvidia/ov-web-rtc` package as described in
`references/dependencies/nvidia-runtime.md`. For the connection shape, use the
current `ovstream` WebRTC browser client example as the reference:
<https://github.com/NVIDIA-Omniverse/ovstream/tree/main/examples/webrtc_client>

```typescript
import { AppStreamer, StreamType, EventStatus, VideoCodec, type DirectConfig } from '@nvidia/ov-web-rtc';

async function waitForDecodedFrame(video: HTMLVideoElement, timeoutMs = 15_000) {
  const initial = video.getVideoPlaybackQuality?.().totalVideoFrames ?? 0;
  const deadline = performance.now() + timeoutMs;
  while (performance.now() < deadline) {
    const decoded = video.getVideoPlaybackQuality?.().totalVideoFrames ?? 0;
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && (decoded > initial || video.currentTime > 0)) return;
    await new Promise(resolve => window.setTimeout(resolve, 100));
  }
  throw new Error("WebRTC connected but no decoded H264 frame arrived");
}

const video = document.getElementById("remote-video") as HTMLVideoElement;
video.tabIndex = -1;
const focusVideo = () => video.focus();
video.addEventListener("mouseenter", focusVideo);
// Provider teardown removes this listener and terminates its connection.

const streamer = new AppStreamer(); // Retain this instance for the session.
let commandsReady = false;
const config: DirectConfig = {
  videoElementId: 'remote-video',
  audioElementId: 'remote-audio',
  signalingServer: 'localhost',
  signalingPort: 49100,
  width: 1920,
  height: 1080,
  codec: VideoCodec.H264,
  codecList: ['H264'],
  nativeTouchEvents: true,
  fps: 60,
  maxReconnects: 5,
  reconnectDelay: 3000,
  onStart: msg => {
    if (msg.status === EventStatus.ERROR || msg.status === EventStatus.CANCELED) {
      commandsReady = false; // Surface failure and reject pending work.
    }
    if (msg.status === EventStatus.SUCCESS) {
      commandsReady = true; // Query server state here; do not wait for a scene frame.
      video.focus();
    }
  },
  onUpdate: msg => {},
  onCustomEvent: msg => {},
  onStop: msg => { commandsReady = false; },
  onTerminate: msg => { commandsReady = false; },
};

const start = await streamer.connect({ streamSource: StreamType.DIRECT, streamConfig: config });
if (start.status === EventStatus.ERROR || start.status === EventStatus.CANCELED) {
  throw new Error('Stream startup failed');
}
await waitForDecodedFrame(video);
```

`connect()` may resolve with `EventStatus.IN_PROGRESS`; resolution alone is not
command readiness. Follow the selected SDK's confirmed start-success event
(`onStart` with `EventStatus.SUCCESS`) before querying state or sending app
commands. Handle error/canceled events and clear readiness on stop/terminate.
Success is a transport signal; prove decoded frames separately and do not wait
for a scene frame before sending the command that loads that scene. The SDK
forwards application messages; the viewer server must implement their handlers.

Make the video focusable and activate it after stream start and on viewport
entry. This lets the SDK capture native input after page controls release focus.
Keep text fields and other DOM controls outside viewport input ownership; see
[viewer input routing](../viewer-input-routing/README.md).

`onCustomEvent` receives parsed app JSON. Mouse, keyboard, wheel, and touch events are forwarded by the streaming library automatically in native mode. Use the documented `mouseInput` fallback only after the explicit native-callback failure diagnostic.

## React Media Ownership

The browser SDK wraps and reparents the video element to capture input. Give
React a stable, empty media host and create the video/audio children imperatively
before connecting. React controls loading indicators and toolbars as siblings of
that host; the SDK controls everything inside it. A JSX-owned video beside a
conditional overlay can make React target a child the SDK has already moved.

```tsx
import { useCallback } from 'react';

function StreamMedia() {
  const attach = useCallback((host: HTMLDivElement | null) => {
    if (!host || host.querySelector('#remote-video')) return;
    const video = document.createElement('video');
    video.id = 'remote-video';
    video.autoplay = true;
    video.playsInline = true;
    video.muted = true;
    video.tabIndex = -1;
    const audio = document.createElement('audio');
    audio.id = 'remote-audio';
    audio.autoplay = true;
    host.append(video, audio);
  }, []);
  return <div className="stream-media" ref={attach} />;
}

// Keep the host mounted while connecting, disconnected, and reconnecting.
<main className="viewport">
  <StreamMedia />
  {!videoReady && <ConnectionOverlay status={status} />}
</main>
```

The ref is idempotent across development reattachment and leaves SDK wrappers
intact. Keep element IDs unique to the provider's session. Terminate the owned
connection during actual provider teardown; ordinary status changes must not
remove or recreate its media elements. Verify producer-only restart with the
same browser page: the overlay appears, frames resume, and pointer input works.

## Connection Ownership

- Use `streamSource: StreamType.DIRECT` with `signalingServer` and `signalingPort` for
  the standalone `ovstream` server.
- Set an explicit fixed `width`, `height`, `codec: VideoCodec.H264`, and
  `codecList: ['H264']` unless a verified deployment profile selects another
  codec. Do not rely on a default codec value for first-frame validation.
- For standalone `ovstream` Direct deployments, do not use connection profiles or fields intended for a different streaming product. If an orchestrator launches the session, map its exposed endpoint to `signalingServer` and `signalingPort`. For NVCF self-hosted deployment, follow `cloud-deployment/nvcf-self-hosted.md` instead.
- Do not set `mediaServer` or `mediaPort`; SDP discovers the UDP media endpoint. Setting `mediaPort: 49100` sends media to the TCP signaling port and causes connected-with-no-video failures.
- Do not construct a sign-in URL, append a custom signaling path, or add
  auth/session fields to `DirectConfig`. Portal auth and session lifecycle
  belong outside the browser WebRTC client config.
- The `<video id="remote-video">` element must exist in the DOM before `connect()`.
- After confirmed start success, prove video separately: the video element reaches `HAVE_CURRENT_DATA` and decoded-frame count (or `currentTime` where frame counters are unavailable) advances.
- Tune reconnects to avoid log storms: `maxReconnects: 5`, `reconnectDelay: 3000`.
- Keep the React effect that calls `streamer.connect()` stable. It should depend only on immutable connection config, not on stateful message routers that change when app messages arrive.
- Make connection setup and teardown tolerate React development remounts without overlapping sessions.
- `streamer.sendMessage()` returns a Promise. Handle rejection and settle the pending UI action; a transport send is not confirmation that the server applied it.

```tsx
async function sendMessage(message: StreamMessage) {
  if (!connectedRef.current) throw new Error("Viewer disconnected");
  await streamer.sendMessage(message);
}
// The action handler catches errors and clears its pending indicator.
```

## StreamingContext Pattern

Wrap AppStreamer in React context:

```tsx
<StreamingProvider>
  <AppContent />
</StreamingProvider>

const { status, sendMessage, onCustomEvent, errorMessage } = useStreaming();
```

The provider should own one stable `AppStreamer` instance for a given host/port pair. Keep `onCustomEvent` routing behind a stable callback or a ref-backed dispatcher so state updates from server events do not recreate the connection effect and trigger cleanup.

The hook should expose:

| Field | Purpose |
|---|---|
| `status` | `'connecting' | 'connected' | 'failed'`; `connected` means confirmed start success and app commands may be sent. |
| `sendMessage` | send `{event_type, payload}`; reject when unavailable so the UI can recover |
| `onCustomEvent` | subscribe to server messages; returns cleanup |
| `errorMessage` | failure detail |

On connect, request current server state before selecting a scene. Restore the
loaded stage, hierarchy, and AOV state; open a bundled default once only if the
server reports idle. Keep later opens tied to user actions. Follow
[streaming lifecycle](../streaming-lifecycle/README.md) for this handshake and
reconnect behavior.

Clean up handlers:

```tsx
useEffect(() => {
  const unsub = onCustomEvent(event => routeEvent(event));
  return unsub;
}, [onCustomEvent]);
```

Before reconnecting or replacing the session, await `streamer.terminate()` and confirm successful termination. A resolved `IN_PROGRESS` result leaves teardown pending; preserve the owner and follow the selected SDK's lifecycle guidance before starting another session. Do not terminate from ordinary message-handler cleanup. Only terminate when the provider is truly unmounting or the user intentionally disconnects. A common failure is: connect, receive `openStageResult`, rerender, effect cleanup calls terminate, video shows one frame, then goes black.

## Video Layout

The default browser-streamed Omniverse Realtime Viewer uses a fixed server render resolution, typically 1920x1080. Let the page resize the video element with preserve-aspect containment:

```css
#remote-video {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
```

CSS layout changes do not need viewport-size messages. NVST maps native input
through the contained video, including letterboxing. For an explicit change to
rendered resolution, use [viewport resize](../viewport-resize/README.md). Keep
native input as the single gesture path; the diagnostic fallback belongs in
[viewer input routing](../viewer-input-routing/README.md).

For React shells with sidebars, top bars, or inspectors, keep the stream surface
layout-stable while UI state changes:

```css
.viewer-shell {
  height: 100vh;
  overflow: hidden;
}

.viewer-content,
.sidebar,
.viewport {
  min-height: 0;
  overflow: hidden;
}

.viewport {
  position: relative;
}

.stream-media {
  position: absolute;
  inset: 0;
}

#remote-video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
```

Do not let tree expansion, property-panel updates, or selection state change the
viewport container's dimensions. Use flex/grid tracks with constrained overflow
for adjacent panels so the `<video>` element remains pinned while React rerenders.

## Viewport Input Ownership

Browser DOM controls can sit above or beside the streamed video, but native WebRTC
input still belongs to the server. Explicitly arm viewport input only while the
pointer is over the viewport, and disarm it when the pointer enters or presses UI
chrome:

```tsx
function setViewportInputActive(active: boolean) {
  void sendMessage({
    event_type: 'setViewportInputActive',
    payload: { active },
  }).catch(reportStreamError);
}

<aside
  onPointerEnter={() => setViewportInputActive(false)}
  onPointerDown={() => setViewportInputActive(false)}
  onWheel={() => setViewportInputActive(false)}
>
  <StageTree />
</aside>

<main
  className="viewport"
  onPointerEnter={() => setViewportInputActive(true)}
  onPointerDown={() => setViewportInputActive(true)}
  onPointerLeave={() => setViewportInputActive(false)}
>
  <StreamMedia />
</main>
```

The server should implement the matching gate from `viewer-input-routing`, ignore
`on_input` events while this flag is false, and cancel any active camera drag.
This prevents sidebar clicks, tree expansion, inspector scrolling, and top-bar
selection changes from being interpreted as camera orbit, pan, zoom, or pick
gestures.

## Verified JSON Mouse Fallback

Do not add these handlers to a normal browser viewer. Enable them only when the server/app configuration has selected `json-fallback` after the native-input diagnostic. Keep native mode as the default and make mode changes cancel any active gesture on both sides.

Map the CSS pointer position to the visible, letterbox-corrected video rectangle before sending it. Ignore events in black bars. In fallback mode call `stopPropagation()` so AppStreamer/NVST cannot also receive the gesture. Send `mouseInput` with render-product `x` and `y`: use `kind: "mouse_move"` or `"mouse_button"` with DOM `button` and `down`, and `kind: "mouse_wheel"` with `delta_y`. Register move/up/cancel listeners for an active fallback drag and remove them when it ends. Do not send fallback events outside the viewport or while native mode is active.

## Config Resolution

Resolve the signaling host and port independently, before connecting, in this
priority order:

1. URL params: `?server=192.168.1.50&signalingport=49100`
2. Vite build-time env vars: `VITE_SERVER_HOST`, `VITE_SIGNALING_PORT`
3. Runtime [`stream.config.json`](../streaming-viewer-recipe/client-protocol.md#production-streamconfigjson):
   `signalingServer`, `signalingPort`
4. Defaults: `window.location.hostname`, port `49100`

Treat an absent or empty value as unspecified and continue to the next source
for that field. Validate the selected host as a nonempty hostname/IP without a
scheme or path, and convert URL/env port strings to an integer in `1..65535`.
Surface malformed selected values as configuration errors before connecting.
Fetch the runtime JSON before resolution; a missing optional file uses defaults,
while malformed JSON or a failed fetch other than a missing file should surface
a configuration error. Map only the resolved host and port to
`DirectConfig.signalingServer` and `DirectConfig.signalingPort`; the URL parameter
`server` is an app input name, not the deprecated SDK `server` field. Retain the
other settings from the Direct example and omit `mediaServer`/`mediaPort`.

```bash
VITE_SERVER_HOST=192.168.1.100
VITE_SIGNALING_PORT=49100
```

Vite replaces `import.meta.env.VITE_*` at build time, including production builds.
Changing these variables in a container after the static bundle is built has no
effect. Runtime JSON changes require no rebuild and take effect at the next app
startup; ensure the file is fetched fresh and the field is not overridden by a
URL parameter or an embedded build-time value. Leave build-time target values
unset when runtime JSON should control the deployment target.

## Component Layout

```text
App.tsx
  StreamingProvider
    AppContent
      StreamMedia (empty React host; SDK owns media children)
      connection/loading overlay
      stage selector
      StageTree recursive tree
      Inspector selected-prim panel
```

When generating a streaming frontend, create equivalent files such as
`frontend/src/streaming/StreamingContext.tsx`, `frontend/src/App.tsx`,
`frontend/src/components/StageTree.tsx`,
`frontend/src/components/Inspector.tsx`,
`frontend/src/hooks/useWebRTCBackend.ts`, and `frontend/src/types/usd.ts`.

## ViewerBackend Adapter

When using shared UI components, wrap AppStreamer in a `ViewerBackend` adapter instead of letting every component send raw messages. The adapter is promise-based for query responses, observable for selection, and caches tree responses by path.

Required methods:

```typescript
export interface ViewerBackend {
  connect(): Promise<void>;
  disconnect(): void;
  loadStage(url: string): Promise<void>;
  getChildren(path: string): Promise<PrimNode[]>;
  getStageTree(rootPath?: string): Promise<PrimNode[]>;
  getProperties(path: string): Promise<PrimProperty[]>;
  getVariants(path: string): Promise<Record<string, { options: string[]; selection: string }>>;
  setVariant(path: string, variantSet: string, selection: string): Promise<void>;
  selectPrims(paths: string[]): Promise<void>;
  onSelectionChanged(callback: (paths: string[]) => void): () => void;
}
```

Adapter implementation rules:

- Keep resolver buckets keyed by URL or prim path for `openStageResult`, `getChildrenResult`, `getPropertiesResponse`, and `getVariantsResponse`.
- Discover optional feature support from backend capability messages or
  app-specific config before showing controls for runtime transforms, pick
  effects, physics impulses, or other non-baseline commands.
- Add timeouts to promises so dropped data-channel responses do not hang UI components permanently.
- Cache latest children per `prim_path` and return cached rows if a request times out.
- Subscribe to `stageSelectionChanged` and call every selection subscriber with the canonical server path list.
- Treat `getChildrenResult` as a hierarchy read: update the tree/cache and settle its request. Pickability belongs to the runtime scene policy, independently of which branches the client has expanded.
- Accept the current property event name `getPropertiesResponse`; handle `getPropertiesResult` only as a compatibility alias.
- For selected-prim panels, keep the latest selected path in a `useRef` and
  compare `getPropertiesResponse.prim_path` against that ref. Do not compare
  against a `selectedPath` value captured in the React message callback closure;
  fast server responses can otherwise be dropped after a valid
  `stageSelectionChanged` event.
- Keep command readiness tied to confirmed start success and video readiness tied to decoded frames. `EventStatus.ERROR`, `onStop`, and `onTerminate` update session status and reject relevant pending work.
- Keep optional feature commands behind the shared `ViewerBackend` capability
  extension instead of sending library-specific calls from React components.

Resolver pattern:

```typescript
const childResolvers = useRef(new Map<string, Resolver<PrimNode[]>[]>());
const propertyResolvers = useRef(new Map<string, Resolver<PrimProperty[]>[]>());
const variantResolvers = useRef(new Map<string, Resolver<VariantMap>[]>());
const selectionHandlers = useRef(new Set<(paths: string[]) => void>());
const treeCache = useRef(new Map<string, PrimNode[]>());

case 'getChildrenResult': {
  const payload = event.payload as { prim_path?: string; children?: PrimNode[] };
  const key = payload.prim_path || rootPrimPathRef.current;
  const children = normalizeChildren(payload.children || []);
  treeCache.current.set(key, children);
  resolveBucket(childResolvers.current, key, children);
  break;
}
case 'getPropertiesResponse':
case 'getPropertiesResult': {
  const payload = event.payload as { prim_path?: string; properties?: Record<string, unknown> };
  resolveBucket(propertyResolvers.current, payload.prim_path || '', toPrimProperties(payload.properties || {}));
  break;
}
case 'getVariantsResponse': {
  const payload = event.payload as { prim_path?: string; variants?: VariantMap };
  resolveBucket(variantResolvers.current, payload.prim_path || '', payload.variants || {});
  break;
}
```

Track the server-provided root path. `openStageResult` includes `root_prim_path`; use it for the initial hierarchy query and shared backend `getStageTree()` defaults.

```tsx
const rootPrimPathRef = useRef('/World');

case 'openStageResult': {
  const payload = event.payload as { result: string; url: string; root_prim_path?: string };
  if (payload.result === 'success') {
    const root = payload.root_prim_path || '/World';
    rootPrimPathRef.current = root;
    sendMessage({ event_type: 'getChildrenRequest', payload: { prim_path: root } });
    sendMessage({ event_type: 'getPrimCountRequest', payload: {} });
  }
  break;
}
case 'getChildrenResult': {
  const payload = event.payload as { prim_path: string; children: ServerPrim[] };
  const children = (payload.children || []).map(normalizePrim);
  if (payload.prim_path === rootPrimPathRef.current) setPrims(children);
  else setPrims(prev => updatePrimChildren(prev, payload.prim_path, children));
  break;
}
```

If using the shared UI adapter, route property responses by the current event name:

```typescript
case 'getPropertiesResponse': {
  const payload = event.payload as { prim_path?: string; properties?: Record<string, unknown> };
  const properties = Object.entries(payload.properties || {}).map(([name, value]) => ({
    name,
    type: typeof value,
    value: String(value),
  }));
  resolveBucket(propertyResolvers.current, payload.prim_path || '', properties);
  break;
}
```

Do not listen only for the stale `getPropertiesResult` name.

For direct React inspectors without a full `ViewerBackend`, correlate property
responses through a ref-backed selected path:

```tsx
const [selectedPath, setSelectedPath] = useState('');
const selectedPathRef = useRef('');

case 'stageSelectionChanged': {
  const next = payload.prims?.[0] || '';
  selectedPathRef.current = next;
  setSelectedPath(next);
  setProperties(null);
  if (next) {
    sendMessage({ event_type: 'getPropertiesRequest', payload: { prim_path: next } });
  }
  break;
}
case 'getPropertiesResponse': {
  const payload = event.payload as { prim_path?: string; properties?: Record<string, unknown> };
  if (payload.prim_path === selectedPathRef.current) {
    setProperties(payload.properties || {});
  }
  break;
}
```

## Frontend UI Expectations

A full browser client should be a usable Omniverse Realtime Viewer UI, not just
a video tag. Include these pieces when generating a full browser-streamed
Omniverse Realtime Viewer:

- Auto-open the first sample stage once the stream connects when the server has not already pushed a loaded stage.
- Honor server `push_initial_state` on reconnect: do not require a fresh user action to leave loading state.
- Populate the AOV `<select>` from `availableAOVsResult` or `activeAOVState.available`; do not hardcode only `LdrColor`, `NormalSD`, or deprecated `DepthSD`.
- Display prim count from `getPrimCountResult`.
- Display connection status, FPS from video playback quality when available, and latency when WebRTC stats expose it.
- Stage tree supports lazy expansion, search/filter, selected-row state, and prim type icons.
- Use icon sprites or equivalent symbols for `mesh/geom`, `camera`, `light`, `scope`, and `xform`.
- Inspector tracks selected prims from `stageSelectionChanged` and fetches properties with `getPropertiesRequest`.
- Inspector ignores stale property responses by matching `prim_path` to a
  ref-backed current selection, not to closure-captured React state.
- Keep `#remote-video { object-fit: contain; }` inside the SDK-owned media host, and put React controls in sibling overlays with explicit `z-index`.
- DOM controls disable viewport input with `setViewportInputActive {active:false}`;
  the viewport enables it while the pointer is over the stream.

When DOM controls overlay the stream, give them explicit stacking above the `<video>` element:

```css
.toolbar,
.error-banner {
  position: absolute;
  z-index: 3;
}
```

Without explicit stacking, the hardware video layer can visually cover controls once frames arrive, making the app look like it lost the connection UI.

## Recursive Tree Updates

```typescript
function updatePrimChildren(prims: USDPrim[], targetPath: string, children: USDPrim[]): USDPrim[] {
  return prims.map(prim => {
    if (prim.path === targetPath) return { ...prim, children };
    if (prim.children && Array.isArray(prim.children))
      return { ...prim, children: updatePrimChildren(prim.children, targetPath, children) };
    return prim;
  });
}
```

Children semantics: truthy non-array means expandable/not loaded, `null` or absent means leaf, array means loaded children.

The current server worker returns `children: boolean` as expandability metadata, while `StageTree` reads `hasChildren`. Normalize at the boundary:

```typescript
type ServerPrim = Omit<USDPrim, 'children'> & {
  children?: ServerPrim[] | boolean | null;
  has_children?: boolean;
};

function normalizePrim(prim: ServerPrim): USDPrim {
  const childArray = Array.isArray(prim.children) ? prim.children.map(normalizePrim) : undefined;
  const hasChildren = Array.isArray(prim.children)
    ? prim.children.length > 0 || Boolean(prim.hasChildren ?? prim.has_children)
    : Boolean(prim.hasChildren ?? prim.has_children ?? prim.children);
  return { ...prim, hasChildren, children: childArray ?? null };
}
```

See also: `streaming-server`, `streaming-messages`, `streaming-lifecycle`, `stage-management`, `stage-hierarchy`.
