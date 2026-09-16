# FFmpeg Sidecar

Host-supplied FFmpeg for cosmos-curator **distributable / redistributable**
Docker images. Split, image-annotate, and SAM3 pipelines shell out to
`ffmpeg` and `ffprobe`; the container expects them at `/opt/ffmpeg`.

Full images built from source may still bundle FFmpeg; treat the sidecar as
**mandatory** for registry pulls unless you have verified FFmpeg inside the
image.

---

## Quick install

```bash
make ffmpeg-install                              # default: $HOME/cosmos-curator-ffmpeg
make ffmpeg-install FFMPEG_DIR=/opt/ffmpeg-sidecar
make check-setup                                 # validates sidecar + docker + nvidia
```

Pin: conda-forge `ffmpeg=8.1.1=lgpl*` (NVENC/NVDEC + libopenh264, no libx264).

Verify:

```bash
$FFMPEG_DIR/bin/ffmpeg -version | head -1
$FFMPEG_DIR/bin/ffmpeg -hide_banner -encoders | grep -E 'h264_nvenc|libopenh264'
```

Expected layout:

```text
$FFMPEG_DIR/
├── bin/     ffmpeg, ffprobe
└── lib/     shared objects for LD_LIBRARY_PATH
```

---

## How the mount works

Makefile targets (`run-pipeline`, `run_image_pipeline`, `shell`,
`download-models`) bind-mount the host prefix read-only:

| Host | Container | Purpose |
|------|-----------|---------|
| `$FFMPEG_DIR` | `/opt/ffmpeg` (read-only) | FFmpeg binaries + libs |

Environment inside the container:

- `PATH` prepends `/opt/ffmpeg/bin`
- `LD_LIBRARY_PATH=/opt/ffmpeg/lib`

Override per run:

```bash
make run-pipeline FFMPEG_DIR=/opt/ffmpeg-cuda \
  CONFIG_FILE=cookbook/traffic-video-analytics/split-minimal.yaml
```

---

## Docker direct

Do not hand-roll `docker run`. Makefile targets already mount the sidecar
read-only and set `PATH` / `LD_LIBRARY_PATH`. Use `make run-pipeline` or
`make run_image_pipeline`. There is no untagged image in this workflow:
`make pull` records `IMAGE_NAME=cosmos-curator` and `IMAGE_TAG=2.3.0`.

---

## Acceptance check

After install, confirm the sidecar works **through the container mount**:

1. `make check-setup` — Docker, NVIDIA runtime, and host-side FFmpeg exist.
2. `make check-image` — the pinned `cosmos-curator` tag is present locally.
3. `make shell` then `ffmpeg -hide_banner -encoders` and confirm
   `h264_nvenc` or `libopenh264`.

There is no in-repo E2E or L1 harness. Local proof is Make preflight plus
`uv run pytest tests/unit`. A GPU smoke run uses a reviewed cookbook, for
example `cookbook/traffic-video-analytics/split-minimal.yaml`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ffmpeg: command not found` in container | Sidecar missing or mount omitted; run `make ffmpeg-install` and use Makefile targets |
| `error while loading shared libraries` | Mount `/opt/ffmpeg/lib` and set `LD_LIBRARY_PATH` |
| Transcode very slow | Sidecar lacks NVENC; reinstall LGPL conda-forge build or use CUDA-aware host FFmpeg |
| `libx264` encoder missing | Expected for LGPL build; pipeline defaults to `libopenh264` |
| apt-installed ffmpeg in container | **Do not** `apt install ffmpeg` — shadows the sidecar with CPU-only builds |

---

## Alternatives to conda-forge

- **Existing CUDA FFmpeg at `/usr/local`**: `ln -s /usr/local $HOME/cosmos-curator-ffmpeg`
- **Static CPU-only binaries**: functional but slow; place under `$FFMPEG_DIR/bin/`
- **From source**: only if GPL codecs (e.g. libx264) are explicitly required
