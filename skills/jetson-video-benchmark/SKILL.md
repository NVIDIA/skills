---
name: jetson-video-benchmark
license: "Apache-2.0"
description: >-
  Use when measuring Jetson Video Codec SDK or PyNvVideoCodec encode/decode
  throughput, comparing presets or surfaces, testing codec-worker capacity
  with authenticated samples and user media, or producing a clearly labeled
  documentation-derived planning estimate when representative media is absent.
  Also use for a video request limited to PSNR or SSIM, to apply the terminal
  scope response.
metadata:
  author: "Vinit Bansal <vinitkumarb@nvidia.com>"
  tags: [jetson, video-codec-sdk, pynvvideocodec, benchmark, nvenc, nvdec]
  languages: [markdown]
  data-classification: public
---

# Jetson Video Benchmark

## Purpose

Measure codec-stage FPS and megapixels/second for an evidenced target and exact
workload. Supported live routes are encode, decode, P4/P5 comparison, and a
strictly increasing worker-capacity sweep. A separate no-media path calculates
clearly labeled SDK-documentation estimates; it never claims a measurement.

## Terminal gates

1. For a request solely for PSNR or SSIM, state that objective quality is
   outside this skill and needs a separately authorized workflow, then stop.
   Return only that scope response. Do not append an alternative benchmark or
   next step, name another tool, request media, probe, or install anything.
2. Resolve codec direction before probing or calculating. Explicit encode or
   decode wording wins. Otherwise, quality, preset, bitrate, rate control,
   recording, or compressed-output wording implies encode; explicitly
   compressed input, ingest, playback, or IP/RTSP input implies decode. Treat
   generic camera-count wording such as "connect" or "handle", and a codec name
   on a camera or device, as direction-neutral. Set neutral wording aside and
   resolve direction on the remaining cues; a neutral cue never creates a
   conflict. For an encode-led camera estimate, state that NVDEC applies only
   when cameras already emit the named compressed codec, keep encode primary,
   and ask the user to confirm direction after giving the planning scenarios.
   If non-neutral cues conflict or are absent, present both interpretations,
   state that a unique encode estimate also needs an exact preset and measured
   validation needs representative input, then ask which direction applies and
   stop at that gate.
   Transcoding consumes separate decode and encode budgets: measure it as the
   supported decode and encode routes and report each budget separately, never
   as one combined FPS.
3. An explicit live/run/real/actual measurement requires representative input
   for every requested direction: raw frames with format, geometry, and frame
   count for encode; a compressed elementary stream or container for decode.
   Each input must be one exact target-local path or user-supplied HTTP(S) URL.
   If any is absent, return only `input_required` and ask for the missing
   item(s), including a separate exact path or URL for every requested
   direction, before reading workflow references or sibling skills, probing,
   authentication, recipe work, or workspace creation. Never benchmark the
   setup smoke fixture or substitute catalog media.
4. A no-media planning/expected/indicative question follows
   [documented-performance-estimates.md](references/documented-performance-estimates.md).
   It requires the exact documented row and, for a scaled estimate, clock
   provenance; it sets `measurement_performed: false`. Preserve that reference
   byte-for-byte. Omitted preset, per-stream FPS, or stream mix does not block
   planning: enumerate the reference's bounded documented candidates and
   scenarios, disclose every assumption, and ask for the omitted values.

## Live preflight

Read [benchmark-workflow.md](references/benchmark-workflow.md) completely once,
then apply its preflight section. Preserve explicit
`native`, `pynvc`, and `both`; map delegated selection to `auto`. Obtain fresh
read-only readiness from `jetson-video-setup`, passing any exact interpreter
already supplied or established in this conversation. Setup otherwise checks
the conventional profile path. For `auto`, zero eligible surfaces blocks, one
runs, and two returns `selection_required`. Python encode and comparison need a
`full-samples` environment; decode performance may use `pynvc-smoke`.

Encode, compare, and encode-capacity also require a validated schema-2 recipe
from `jetson-video-recipe`. A missing dependency stops that branch with
`dependency_required`; an eligible peer may continue as `partial`.

## Measure

1. Follow that same reference for sample allowlists, builds, direct argument
   lists, frame accounting, option checks, marker grammar, comparisons, and
   worker sweeps.
2. Show the dry-run plan, then use a fresh mode-0700 workspace. Authenticate
   each launcher immediately before use and launch its literal argument list.
3. Run one excluded whole-process warmup and at least three new measured
   processes per variant and surface, each with a finite timeout and without
   `-loop`.
4. Accept the sample-reported FPS only when its positive marker and processed
   frame count match. Apply every identity, workload, marker, and statistics
   check in [benchmark-output-contract.md](references/benchmark-output-contract.md).

## Report

Retain each launch's unedited output and write the compact result described by
[benchmark-output-contract.md](references/benchmark-output-contract.md). Report
every repetition, recomputed mean/minimum/maximum, MP/s only when dimensions
are sample-bound, partial branches, limitations, and retry reason. Label worker
sweeps as codec-stage capacity bounds and preset comparisons as throughput-only.

## Limitations

- Results apply only to the evidenced target, release, clocks, sample, content,
  frame range, and controls; they are not a portable product ceiling.
- Do not interpolate undocumented presets or scale across format, bit depth,
  chroma, codec, rate control, or tuning. Resolution scaling is allowed only as
  the estimate reference's explicitly labeled pixel-area heuristic.
- Keep documented estimates per engine; use an all-engine multiplier only
  under the explicit aggregate-planning rule in the estimate reference.
- The PyNvVideoCodec 2.1 encode-performance helper caps each worker at 1,000
  frames; follow the common-frame rule in the workflow.
- Preserve `input_required`, `selection_required`, `dependency_required`,
  `blocked`, `partial`, and `failed`; do not promote a peer's success.
