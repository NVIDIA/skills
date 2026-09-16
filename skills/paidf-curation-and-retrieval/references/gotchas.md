# Gotchas

Non-obvious pitfalls when building, configuring, and running cosmos-curator
pipelines from this repo. Read alongside `references/running-pipelines.md`
(invocation/troubleshooting) and `references/sam3-config.md` (SAM3 keys).

## Image and build

- Docker image is `cosmos-curator:<pin>` from `.env.example` (currently
  `2.3.0`) — no separate product engine image.
- Get it via `make pull` from the default NGC registry. Use
  `make clone-curator build` only for custom upstream source builds.
- Docker SHM comes from host RAM. Default is `SHM_SIZE=24gb`; use roughly
  8-16gb for single-GPU runs, 24-64gb for typical multi-GPU runs, and avoid
  exceeding available system memory.
- Distributable images do **not** bundle FFmpeg. Install the host sidecar
  (`make ffmpeg-install`, default `$HOME/cosmos-curator-ffmpeg`) before
  `make run-pipeline`. See `references/ffmpeg-sidecar.md`.
- Do NOT `apt install ffmpeg` inside the container -- it shadows the sidecar
  with a CPU-only build.
- Pre-built distributable images: always `pixi run -e <env> --as-is python ...`.

## Config

- Prefer flat `snake_case` keys. The PAIDF preflight also accepts parameters
  nested under `args` and flattens them before invoking Curator.
- `video_classifier_use_custom_categories: true` is REQUIRED when providing
  custom allow/block lists.
- The `captioning_algorithm` in shard config must match the one used in split.
- Never hardcode machine-specific paths in committed config YAMLs.

## Image pipeline

- Upstream image annotate supports config-file mode. The
  `make run_image_pipeline` target mounts `IMAGE_CONFIG_FILE` and runs
  `python -m cosmos_curator.pipelines.image.run_pipeline`.
- Image pipeline is **resume-aware**: rerunning against an existing
  `output_path` skips already-processed images via `summary.json` / `metas/`.
  Use a fresh `output_path` for clean reruns.
- `image_classifier_use_custom_categories: true` is REQUIRED in image configs
  when supplying a fully custom allow/block taxonomy (analogous to
  `video_classifier_use_custom_categories` for video).

## SAM3 / event captioning canonical keys

- The splitting pipeline's argparse defines `--sam3` and `--event-captioning`
  (dests `sam3` and `event_captioning`). The YAML loader passes keys verbatim
  to `argparse.Namespace`, so writing `enable_sam3: true` or
  `enable_event_captioning: true` puts those names into the Namespace while
  `args.sam3` / `args.event_captioning` are filled with their defaults (False)
  by `fill_default_args`. Result: SAM3 silently never runs and the pipeline
  reports success. Always use `sam3:` and `event_captioning:`. Covered by
  `tests/unit/test_pipeline_config.py`: deprecated `enable_sam3` /
  `enable_event_captioning` keys fail preflight; canonical `sam3:` enables
  the stage.
- Some upstream docs (`cosmos-curator/docs/curator/reference/
  split-pipeline-stages.md`) still reference the old `--enable-sam3` CLI flag.
  That flag does not exist in the parser; the parser only has `--sam3`. The
  mismatch is an upstream doc bug, not our bug.
