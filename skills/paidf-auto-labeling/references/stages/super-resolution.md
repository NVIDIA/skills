# Super Resolution stage

Single-stage reference for the `super_resolution` stage (SeedVR2) that upscales
low-resolution video clips (the original clip) into super-resolved active media
for downstream auto-labeling stages. Video pipelines only.

## When to use / not use

- Use: picking the SeedVR2 resolver/variant, setting the resolution policy, or
  debugging an SR stage that fails, OOMs, or produces empty output.
- Do not use: to run a full pipeline end to end, or to author or restructure a
  whole cookbook.

## Config

Cookbook block `super_resolution:` (essential fields):
- `enabled: true`
- `resolver: seedvr2`
- `variant: seedvr2_3b` (use `seedvr2_7b` only with more GPU memory)

Key `stage_args.super_resolution` flags:
- `--resolution-policy` — pick one: `auto` upscales only low-res clips and passes
  high-res through unchanged; `force` upscales every clip; `off` disables
  upscaling.
- `--min-input-short-side`, `--min-input-long-side` — thresholds that classify a
  clip as low-res under `auto`.
- `--res-h` / `--res-w` — target output resolution.
- `--window-frames`, `--overlap-frames` — temporal window and overlap.
- `--empty-output-policy` — pick one: `fail` stops the run when SR yields no
  output; `warn` logs and continues.

## Instructions

This skill returns configuration or debugging guidance; it does not run the
pipeline. Provide the config immediately - do not gate it behind execution. The
three numbered steps are the only flow nodes; *Config*, *Examples*, *Gotchas*,
and *Guardrails* are reference content and author-time constraints, not flow steps
or loops. The config is returned once.

1. **Classify the task.** Configuring/tuning `super_resolution`, or debugging a
   stage that fails, OOMs, or empties - both in scope. For a mixed request that
   asks for `super_resolution` config AND a full-pipeline run or whole-cookbook
   authoring, answer the `super_resolution` portion here and hand off only the
   run/whole-cookbook portion to the operator or cookbook-authoring skill. STOP
   and hand off entirely only when the request has no `super_resolution`
   config/debug part.
2. **Gather needs.** For a config/tuning request the needed inputs are the
   resolver/variant, resolution policy, target resolution, and window/overlap;
   return the config from these without requiring execution-environment details.
   Ask for runtime details (GPU, model cache, input media) only when they are
   relevant to the requested tuning or to a debug diagnosis (e.g. OOM, empty
   output) - not for a plain config response. If an input that is actually
   required for the request is missing or ambiguous, ask once in a single
   consolidated question; if it stays unresolved after that clarification, state
   the assumption you would need or STOP and report the blocker - do not re-ask
   indefinitely.
3. **Return the config or fix.** Produce the `super_resolution:` block and
   `stage_args` per *Config* (or the debug remedy). Suggest a dry-run
   (`--container-dry-run`) so the caller can verify the generated command before
   running.

**Execution is out of scope.** This skill only produces configuration; it does
not run the stage. Executing (Docker via `workflow-runner`, which requires
explicit user approval) is the operator skill's responsibility.

**When debugging.** On GPU OOM, missing model cache, missing input media, or empty
SR output (with `--empty-output-policy fail`), report the specific cause from the
failing output and the fix, then stop; do not fabricate media or silently retry.

## Examples

Upscale only low-res clips (SeedVR2 3B), pass high-res through unchanged:

```yaml
super_resolution:
  enabled: true
  resolver: seedvr2
  variant: seedvr2_3b
stage_args:
  super_resolution: >-
    --resolution-policy auto
    --min-input-short-side 720 --min-input-long-side 1280
    --res-h 1080 --res-w 1920
    --window-frames 24 --overlap-frames 4
    --empty-output-policy fail
```

Dry-run to inspect the generated Docker command before running:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/video_data_augmentation/configs/pipeline_video.yaml --container-dry-run'
```

## Gotchas

- Heavy GPU stage (SeedVR2) — plan a dedicated GPU; `seedvr2_7b` needs more
  memory than `seedvr2_3b`.
- Docker is required to run this stage.

## Guardrails

Follow [guardrails.md](guardrails.md). SeedVR checkpoints mount via
`<model-cache>`; this stage requires Docker to run.
