<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Streaming Server Frame Loop And Continuity

## Frame Sources

Raw CUDA buffer:

```python
frame = VideoFrame(buffer=cuda_ptr, width=1920, height=1080, pitch_bytes=1920 * 4)
server.stream_video(frame)
```

From a Warp/CuPy/PyTorch CUDA array with `__cuda_array_interface__`:

```python
frame = VideoFrame.from_cuda_array(cuda_array)  # shape H x W x 4, uint8, BGRA8
```

From a DLPack-producing tensor such as Warp, PyTorch, JAX, or CuPy:

```python
config = ovstream.ServerConfig(width=1920, height=1080, video_input=ovstream.VideoInput.TENSOR)
frame = VideoFrame.from_dlpack(tensor)
```

Pre-encoded bitstream descriptors use `size_bytes` and require a matching
`video_input` such as `H264`, `H265`, or `AV1` at server start:

```python
config = ovstream.ServerConfig(width=1920, height=1080, video_input=ovstream.VideoInput.H264)
frame = VideoFrame(buffer=encoded_host_ptr, width=1920, height=1080, size_bytes=encoded_size)
```

## Submission And Resolution

Use a fixed renderer and stream size for the session, with `webrtc_resizable=False`.
Scale the browser video with `object-fit: contain`. The
[frame ownership contract](../dependencies/ovstream.md#frame-ownership) permits
reusing the app-owned buffer after each submission returns. Synchronize producer
work before submission or pass `sync=ovstream.CudaSync(stream=..., wait_event=...)`
to `VideoFrame.from_cuda_array()` / `VideoFrame.from_dlpack()`.

When live rendered resolution is a product requirement, follow
[viewport resize](../viewport-resize/README.md). It changes both renderer state
and encoder buffer ownership; a CSS layout change alone needs neither.

## Frame Continuity During Idle And Stage Loads

A connected WebRTC session needs continuous valid video frames even when no USD
stage is loaded yet. Allocate a persistent fallback BGRA frame, such as a dark
loading frame, before accepting browser connections. While the server is idle,
loading, or blocked on renderer mutation, submit the fallback or last-good frame
every render tick instead of waiting for the first real stage frame.

Stage loads must not block the message callback thread or stop video output. Asset resolution and shader compilation can outlast the transport liveness window.

Keep one runtime owner for loading and rendering. While a load operation is pending,
poll its progress between stream ticks, or delegate blocking renderer work to the
runtime thread while a separate submission loop repeats an app-owned frame. A
stage lock can serialize those operations; never retain a borrowed renderer
mapping as the loading frame. A lock-based submission loop looks like:

```python
while running:
    if stage_lock.acquire(blocking=False):
        try:
            # Copy/convert into app-owned storage before releasing renderer output.
            frame = render_next_owned_frame() if scene_ready else loading_frame
            last_frame = frame
        finally:
            stage_lock.release()
    elif last_frame is not None:
        frame = last_frame
    else:
        frame = loading_frame

    if server.is_client_connected:
        server.stream_video(frame)
```

This preserves WebRTC heartbeats and avoids stepping ovrtx while `reset_stage()`, `open_usd()`, `open_usd_from_string()`, reference updates, or selection state rebuilds are mutating the renderer.

Log the first valid BGRA8 CUDA mapping or completed required conversion into
the app-owned BGRA stream buffer and set readiness before relying on browser
connection state. This separates renderer/frame-conversion failures from WebRTC
lifecycle failures:

```python
if not logged_first_frame:
    logger.info("First BGRA frame ready: %sx%s", width, height)
    logged_first_frame = True
```
