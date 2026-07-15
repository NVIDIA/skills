# § 8 — Basler Camera Support

> **Pipeline stage**: Stage 1 input — Basler industrial camera as a live video source
> **Critical Rules**: LIVE_REQUIRES_STREAM_TRUE, CAMERA_EMULATION_PYLON_CAMEMU, CLEANUP_ON_DISCONNECT, CAMERA_EMULATION_FRAMES_RGB
> **Related sections**: § 9 (Docker build, Pylon SDK setup, gst-plugin-pylon), § 3 (pipeline camera branch), § 7 (SSE cleanup), § 13 (curl examples)

---

## Docker Build Prerequisites (see § 9)

Pylon SDK binary setup and `gst-plugin-pylon` build conflict resolution are documented
in [`skill_09_docker_build_deploy.md`](skill_09_docker_build_deploy.md) under
**Pre-build: Pylon SDK Binary**. Complete those steps before running `docker build`.

---

## Physical Camera (e.g. serial 40748152)

`config` (PFS file) is **optional** for physical cameras — omit to use camera's current settings.

**IMPORTANT — latency vs. classification tradeoff for live inputs:**
- For **low first-chunk latency** (responsiveness demos, `chunk_e2e` measurement — § 17), use a small
  `max_length_sec` (≤ 5.0, e.g. 2.0). The default 60.0 causes unacceptably long waits before the first SSE chunk.
- For **meaningful SOP classification**, use **action-length chunks** — let DDM segment on boundaries
  (`threshold` ~0.8) with a larger `max_length_sec` (e.g. 15.0). A small `max_length_sec` (2.0) caps every
  chunk at ~2 s, slicing each multi-second action into fragments the VLM cannot recognize → it labels them
  the out-of-scope action even though the pipeline, frames, and DDM are all fine. (This is purely a chunk
  duration effect, NOT a color/camera-path bug.) So do not judge camera classification with a 2 s cap;
  size `max_length_sec` to the expected action length.

```json
{
  "model": "ds_sop_model",
  "messages": [{"role": "user", "content": [{
    "type": "input_camera",
    "input_camera": {
      "camera_id": "40748152",
      "camera_vendor": "Basler",
      "camera_format": "RGB",
      "camera_width": 1280,
      "camera_height": 720,
      "camera_fps_num": 30,
      "camera_fps_den": 1
    }
  }]}],
  "stream": true,
  "chunking_options": {"algorithm": "ddm-net", "threshold": 0.8, "max_length_sec": 2.0}
}
```

---

## Camera Emulation (serial 0815-0000)

Requires `PYLON_CAMEMU=1` in the service environment (CAMERA_EMULATION_PYLON_CAMEMU; the base
compose already defaults it to `1`). Use `Emulation_0815-0000.pfs` (`PixelFormat=RGB8Packed`).
Serial is fixed at `0815-0000`:

```json
{
  "model": "ds_sop_model",
  "messages": [{"role": "user", "content": [{
    "type": "input_camera",
    "input_camera": {
      "camera_id": "0815-0000",
      "camera_vendor": "Basler",
      "config": "/opt/nvidia/nvds_sop/configs/Emulation_0815-0000.pfs",
      "camera_format": "RGB",
      "camera_width": 1280,
      "camera_height": 720,
      "camera_fps_num": 30,
      "camera_fps_den": 1
    }
  }]}],
  "stream": true,
  "chunking_options": {"algorithm": "ddm-net", "threshold": 0.8, "max_length_sec": 2.0}
}
```

**Emulation environment (set in `deploy/compose.yaml` / `.env` — § 9, § 11):**

- `PYLON_CAMEMU=1` — enables the emulated Basler device (default in the base compose).
- `CAMERA_EMULATION_DIR=<host dir>` — bind-mounted to `/opt/nvidia/nvds_sop/streams/simulation`,
  so the emulated source can read frames / `.pfs` assets without rebuilding the image.
- `CAMERA_NUM_BUFFERS=<N>` — makes `pylonsrc` emit EOS after N frames. It is **emulation-only**:
  the pipeline applies it only when `PYLON_CAMEMU=1` (handy for fixed-length emulation runs/tests).

### Generating emulation frames from a video (do NOT ask the user for PNGs)

> **CRITICAL [CAMERA_EMULATION_FRAMES_RGB]**: emit **3-channel RGB** PNGs to match the `.pfs`
> `PixelFormat=RGB8Packed`. The pylon emulation config can expose loaded PNGs as RGB, but
> generate explicit RGB assets anyway so the file set, `.pfs`, and generated instructions stay
> unambiguous. Force RGB:
> `nvvideoconvert ! videoconvert ! "video/x-raw,format=RGB" ! pngenc`.
> (Note: all-out-of-scope VLM output is almost always a chunk-length issue, not a frame-format one — see
> the latency-vs-classification note above; rule out small `max_length_sec` first.)

You do **not** need the user to supply a directory of PNGs. Generate the emulation frames
yourself from any MP4 (e.g. the eval `TEST_VIDEO_PATH` / `Install_6.MP4`) using **GStreamer**,
which ships in the `nvds-sop` image — this mirrors the reference repo's
`scripts/prepare_camera_simulation.sh`. The emulated `pylonsrc` replays the PNGs from
`CAMERA_EMULATION_DIR` in **alphabetical order**, so use a zero-padded sequential name.

**Run the decode INSIDE the already-running service container** (`docker exec`), not a fresh
`docker run`. The image's only H.264 decoder is the GPU `nvv4l2decoder` (NVDEC) — there is **no
`avdec_*` software fallback** — and NVDEC only initializes under the full nvidia runtime +
`ipc:host` + ulimits that compose gives the service. A bare `docker run --gpus all` fails with
`qtdemux ... Internal data stream error (-5)`. The running container already has a working NVDEC,
so generate there and copy the frames out:

```bash
SVC=deploy-nvds-action-sop-1                      # the running service container
VIDEO=/abs/path/to/Install_6.MP4                  # MUST be visible inside the container
SIM_DIR=/abs/host/path/streams/simulation         # becomes CAMERA_EMULATION_DIR (host side)
mkdir -p "$SIM_DIR"

# Decode MP4 -> RGB PNGs inside the service container (working NVDEC).
docker exec "$SVC" bash -c '
  rm -rf /tmp/simgen && mkdir -p /tmp/simgen &&
  gst-launch-1.0 -e filesrc location="'"$VIDEO"'" ! decodebin ! nvvideoconvert ! videoconvert ! \
    "video/x-raw,format=RGB" ! pngenc ! multifilesink sync=false \
    location=/tmp/simgen/sop_sample_frame_%06d.png'
docker cp "$SVC":/tmp/simgen/. "$SIM_DIR/"

N=$(ls -1 "$SIM_DIR"/*.png | wc -l)               # -> set CAMERA_NUM_BUFFERS=$N for one bounded pass
echo "$N frames"
```

Notes:
- **Force `format=RGB` (3-channel).** The pylon emulator config
  (`Emulation_0815-0000.pfs` `PixelFormat=RGB8Packed`) can expose PNGs to the pipeline as RGB, but
  do not rely on implicit conversion in generated assets. The reference emulation frames are
  3-channel RGB; match that and verify VLM output is meaningful. (Do NOT add a
  `width=...,height=...`-only capsfilter — a caps with no `format` breaks
  `nvvideoconvert→pngenc` negotiation with a flow error. To resize, include the format:
  `"video/x-raw,format=RGB,width=1280,height=720"`.)
- `multifilesink location=...%06d.png` zero-pads the index so alphabetical = playback order
  (the reference uses `%04d`, which only sorts correctly up to 9999 frames — prefer `%06d`).
- The video must be readable inside the container — under `MODEL_ROOT_DIR` (already bind-mounted)
  is simplest; otherwise add a `-v` mount when starting the service.
- Then set `CAMERA_EMULATION_DIR=$SIM_DIR` (bind-mounted to
  `/opt/nvidia/nvds_sop/streams/simulation`, which matches `Emulation_0815-0000.pfs`'s
  `ImageFilename ./streams/simulation`) and `CAMERA_NUM_BUFFERS=$N`, then **recreate** the service
  (`up -d`) so the new bind mount attaches. **Never `rm -rf` `$SIM_DIR` while it is bind-mounted**
  into a running container — that detaches the mount (container then sees 0 files); clear it in
  place with `find "$SIM_DIR" -type f -delete` instead.
- After emulation streams, **verify the VLM output is meaningful** (actions classified, not all the
  out-of-scope label) — this is the check that catches the RGB/BGR pitfall above.

---

## Camera Format Support

| Format | CPU Conversion Needed | Pipeline Elements | Notes |
|--------|----------------------|-------------------|-------|
| RGB    | No                   | pylonsrc → caps → nvvideoconvert → NV12 | Use for both physical and emulation cameras |
| NV12   | No                   | pylonsrc → caps → nvvideoconvert → NV12 | Direct NVMM path |
| YUY2   | Yes                  | pylonsrc → caps → **videoconvert** → nvvideoconvert → NV12 | Needs CPU `videoconvert` element (see § 3) |

---

## Live Stream Requirements

- **LIVE_REQUIRES_STREAM_TRUE**: `stream: true` is **required** for live inputs (RTSP/camera). HTTP 400 if `stream: false`.
- **CLEANUP_ON_DISCONNECT**: SSE generator must stop the processor in `try/finally` to release the camera hardware lock.
- **UNIFIED_CLIP_POST_PROCESS**: Clip postprocessor must use incremental mode (`_run_clip_postprocess_live`) for live sources.
- **EVAL_USE_CONFIG_PROMPT**: For camera evaluation and latency, omit request text so the VLM uses `VLM_PROMPT_PATH`.

---

## Troubleshooting: Physical GigE Camera

### Single-connection limit
Basler GigE cameras allow **only one active connection at a time**. If the SOP container
is running with `pylonsrc` open, any second attempt to open the camera will timeout or fail.

**Always stop the SOP container before testing camera reachability:**

```bash
docker compose --env-file deploy/.env -f deploy/compose.yaml stop nvds-action-sop
```

### Verifying camera reachability with pylonsrc
Use `device-serial-number` (not `serial`) as the pylonsrc property. Run inside the container:

```bash
# Basic connectivity check
docker run --rm --network host \
  --entrypoint gst-launch-1.0 nvds-sop:latest \
  pylonsrc device-serial-number=40748152 num-buffers=3 ! fakesink

# Full GPU pipeline check (pylonsrc → nvvideoconvert → nvstreammux)
# Note: nvstreammux requires explicit mux.sink_0 pad — place it first in the pipeline string
docker run --rm --network host --gpus all \
  --entrypoint gst-launch-1.0 nvds-sop:latest \
  nvstreammux name=mux batch-size=1 width=224 height=224 batched-push-timeout=-1 live-source=true ! \
  fakesink sync=false \
  pylonsrc device-serial-number=40748152 num-buffers=30 ! \
  "video/x-raw,format=RGB,width=1280,height=720,framerate=30/1" ! \
  nvvideoconvert nvbuf-memory-type=2 gpu-id=0 compute-hw=1 ! \
  "video/x-raw(memory:NVMM),format=NV12" ! mux.sink_0
```

### Camera evaluation with timeout and hang detection

Send the camera request with a **20 s curl timeout** (`--max-time 20`). Monitor the
**API server log** (not curl output) for hang detection:

- **Restart the container before each attempt** — every failed request leaves the camera occupied in the server; without a restart the next attempt will also fail
- Watch `docker compose logs -f nvds-action-sop` from a second terminal
- If no new log lines appear for **10 s** after the request is sent, the pipeline is hung
- Kill the container (`docker compose stop nvds-action-sop`) to release the camera lock and retry

### pylonsrc hangs at `pipeline.add` — stale GenICam semaphores

**Symptom**: Logs show `pipeline.add pylonsrc serial=<N>` then nothing. `gst-launch pylonsrc`
also hangs via `docker exec` but works in a fresh `docker run --rm`.

**Root cause**: `ipc: host` in `compose.yaml` shares the host's `/dev/shm`. Stale POSIX
semaphores from dead sessions block Pylon's transport layer initialization:
- `sem.*GenICam_XML` — GenICam device XML access locks
- `sem.siso_fglib_mssync_GenTLSession_lock*` — CoaXPress GenTL producer locks

**Fix**: Remove stale semaphores and restart:
```bash
sudo rm -f /dev/shm/sem.*GenICam_XML \
           /dev/shm/siso_fglib_mssync_GenTLSession_shm \
           /dev/shm/sem.siso_fglib_mssync_GenTLSession_lock*
docker compose --env-file deploy/.env -f deploy/compose.yaml restart nvds-action-sop
```
