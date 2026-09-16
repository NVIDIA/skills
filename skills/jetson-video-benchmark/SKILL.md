---
name: jetson-video-benchmark
license: "Apache-2.0"
description: >-
  Use when measuring Jetson Video Codec SDK or PyNvVideoCodec encode/decode
  throughput, comparing presets or surfaces, testing codec-worker capacity
  with authenticated samples and user media, or producing a documented
  clock-scaled or clock-and-resolution-scaled planning estimate when
  representative content is unavailable.
  Also use for Jetson video requests asking only for PSNR or SSIM results, to
  apply this performance skill's scope-only response.
metadata:
  author: "Vinit Bansal <vinitkumarb@nvidia.com>"
  tags: [jetson, video-codec-sdk, pynvvideocodec, benchmark, nvenc, nvdec]
  languages: [python]
  data-classification: public
---

# Jetson Video Benchmark

## Purpose

Measure codec-stage FPS and megapixels/second on the current Jetson. Use this
skill for encode or decode throughput, P4/P5 comparisons, native-versus-Python
comparisons, and increasing-worker capacity tests. When content is unavailable,
it can instead produce a clearly labeled SDK-documentation estimate for an
exact supported 1080p table row and target maximum video clock. For another
requested resolution, it may additionally apply the bounded pixel-area
heuristic defined in the estimate reference. Never present either estimate as
a target measurement.

## Prerequisites

- For a live measurement:
  - Run on the target Jetson with direct GPU access. A fresh validated
    `nvcodec-environment` identity from `jetson-video-setup` is optional; when
    supplied, it is authoritative and any invalid or stale identity fails
    without local fallback. The agent may obtain that identity from setup's
    public read-only probe; it need not be present in the customer's prompt.
  - Without that identity, native routes inspect only the installed
    `nvidia-video-codec-sdk` APT package and its package-owned official sample
    sources. PyNvVideoCodec routes require an authenticated setup environment
    or the caller's exact absolute `pynvc_interpreter`; never scan for a venv.
    Before asking the customer for that path, invoke setup's public probe when
    that skill is installed and inspect its typed result. These read-only
    checks install, repair, register, and smoke-test nothing. Only when neither
    authority is usable does an explicit `pynvc` or `both` request return
    `input_required`; local `auto` records PyNvVideoCodec as `not_evaluated`
    and may continue an eligible native branch.
  - A PyNvVideoCodec decode-only performance route can use a validated default
    `pynvc-smoke` environment. A Python encode or compare route uses official
    samples that import Torch and therefore requires a separately provisioned
    `full-samples` venv; never upgrade the smoke venv in place.
  - Live encode, compare, and encode-capacity routes require sibling
    `jetson-video-recipe` and an exact portable identity for one of its
    validated schema-2 recipes. If its public validator is absent, preserve
    `dependency_required` and its install-and-retry action. Decode routes do
    not require the recipe skill.
  - When setup is installed, read its shared
    [video content policy](../jetson-video-setup/references/video-content.md)
    before a live measurement. It does not apply to the separate
    documentation-only estimate path. Setup is not required solely for this
    policy: without it, require one exact user-selected path or URL, never
    choose catalog or synthetic media, and preserve source URL, license,
    attribution, path, size, and SHA-256.
  - Apply that policy's input gate before constructing a live benchmark dry
    run. Never choose media for the user or use the setup smoke fixture for
    performance.
- A documentation answer may be produced off-target when the exact platform,
  SDK version, table conditions, and configured maximum video-clock facts are
  supplied with provenance. A clock-scaled estimate also requires a positive
  configured maximum video clock. It requires no media, recipe, sample
  authentication, or codec launch. Without that clock, report only the
  unscaled reference row as `target_clock_unavailable`, not a platform-scaled
  estimate.

## Compose requested sibling stages

Documentation-only estimates and recipe-free live decode do not require a
sibling skill. Live encode, compare, and encode-capacity routes require
`jetson-video-recipe`; SDK installation, repair, a new full-samples Python
environment, or one read-only handoff when registered Python authority is
required belongs to `jetson-video-setup`. Check the agent's installed skill catalog
before either stage. If the sibling is present, read its `SKILL.md` and invoke
its documented public entry point; pass artifacts as data and never import
sibling code. If it is absent, preserve completed input and measurement
evidence and say, using the actual names: `I can run <stage>, but it requires
<skill>, which is not installed. Install <skill> and retry this stage.` Never
require setup when the caller already supplied authenticated setup evidence or
an exact interpreter that passes local authentication, or a recipe for a
decode-only or documentation-only request.

## Instructions

1. Apply the scope boundary first. For a request solely for objective quality
   metrics, including PSNR or SSIM, state only that this performance skill does
   not provide them and that a separately authorized quality workflow is
   required, then stop. Do not name or recommend an external tool, and do not
   offer to configure or run the comparison; do not request media, probe,
   install anything, or launch an operation.
2. Classify an in-scope request as a live `encode`, `decode`, `compare`, or
   `camera_capacity` measurement, or as a `documented_estimate` answer.
   Resolve camera data direction before selecting an encode or decode row.
   Quality, preset, bitrate, rate-control, recording, or requested codec-output
   wording is an encode cue; lead with NVENC and mention decode only as the
   conditional case where cameras already emit the named compressed codec.
   IP/RTSP input, already-encoded input, ingest, playback, or explicit decode
   wording is a decode cue. Explicit transcoding or decode-then-encode uses
   separate NVDEC and NVENC budgets. When no direction cue exists, present the
   encode and decode interpretations conditionally and ask which applies; never
   silently choose one.
3. When media is absent:
   - For an explicit request to run or benchmark, or for actual, real,
     measured, live, or on-this-target FPS, apply the shared policy's
     `input_required` gate and stop before probing, retrieval, authentication,
     workspace creation, dry run, or launch.
     Ask only for the missing media identity at this gate; do not also request
     an interpreter, environment, recipe, or other later-stage field.
   - For a planning, expected, indicative, or achievable-FPS question at any
     positive requested resolution, follow
     [Documented performance estimates](references/documented-performance-estimates.md).
     Use only an exact documented 1080p row and a configured maximum video
     clock as the source basis. For a non-1080p request, apply the reference's
     inverse-pixel-area formula and disclose that it is an additional heuristic
     not stated by the SDK table. If the clock is unavailable, return only the
     unscaled 1080p reference row. Do not invoke the benchmark controller or
     claim a measurement.
4. When local media is supplied, validate and hash that exact selected input.
   For URL input, preserve the exact user-supplied URL and retrieve it only
   after the target, authorization, and runtime-authority gates; then validate
   and hash the retrieved bytes. Preserve source URL (or null for local media),
   license, and attribution. Use the live measurement path, not a documentation
   estimate.
5. Preserve the requested surface for live measurements. Treat “whichever”,
   “best available”,
   “choose for me”, and other unspecified-surface wording as `auto`, never as
   `both`. Reserve `both` for an explicit dual-surface comparison. For `auto`,
   zero eligible surfaces block, one runs, and two return
   `selection_required`; never inspect prior results, invent a preferred SDK,
   or launch a benchmark to make the missing user choice. With two eligible
   surfaces, ask for exactly `native`, `pynvc`, or `both`.
   After the media gate, use valid supplied setup evidence or an exact
   interpreter first. If PyNvVideoCodec may participate and neither is
   supplied, invoke installed `jetson-video-setup` through its public read-only
   `probe_nvcodec.py`: use `--runtime pynvc` for explicit Python or `--runtime
   both` for `both`/`auto`, a fresh `--output`, and never
   `--setup-candidate`. Inspect the fresh artifact; only a live artifact whose
   selected Py surface is installed and whose `pynvc.identity.status` is
   `verified` is usable authority. Snapshot that exact artifact as the
   controller's portable `environment` identity with exactly
   `schema_version`, `kind`, canonical absolute `path`, `size_bytes`, and
   lowercase `sha256`. If setup is absent or reports any not-ready, unreadable,
   stale, binding, or launch failure, request the exact interpreter for
   explicit `pynvc`/`both`; for local `auto`, report Python as `not_evaluated`
   and continue only an eligible native surface. Never pass a blocked probe as
   authority, scan for a venv, or hide an unevaluated peer.
6. Run a `dry_run` first, review every planned argument, then run `execute` in
   a fresh private workspace. Invoke this skill's controller directly:

   ```bash
   python3 -I {baseDir}/scripts/benchmark_controller.py \
     --request request.json --workspace fresh-workspace \
     --output result.json
   ```

7. For native encode, authenticate `AppEncPerf`, inspect both `-h` and `-A`,
   and use only advertised options. Never pass `-loop`. Native decode uses
   authenticated `AppDecPerf`; PyNvVideoCodec uses the wheel-owned performance
   samples under either the setup-evidenced interpreter or the exact absolute
   `pynvc_interpreter` selected by the caller.
8. Retain one whole-process warmup and at least three separate measured
   processes per variant and surface. Every measured command phase is
   `measure`.
9. For a live measurement, list every repetition's FPS and MP/s, then the
   mean, minimum, and maximum of both metrics. PyNvVideoCodec decode omits MP/s
   because its authenticated performance sample does not report dimensions;
   report that reason instead of deriving MP/s from caller metadata. For a
   native/Python comparison, disclose exact projection differences and whether
   they changed the requested intent. For a live preset comparison, report
   only measured throughput differences; do not state or imply a quality,
   compression-efficiency, or storage ordering from preset names or
   throughput. State that quality was not measured when that distinction
   matters. For a documented estimate, report
   the source row, clock provenance, requested resolution, formula,
   assumptions, and every inference label required by the estimate reference,
   plus `measurement_performed: false`; never invent repetitions or measured
   statistics.
10. Label measured concurrency results as codec-stage capacity bounds. For a
    no-media planning question, a resolution-scaled estimate may additionally
    produce a `documented_theoretical_capacity_estimate` when per-stream FPS and
    an explicit or clearly defaulted safety margin are available. Keep encode
    and decode budgets separate; for mixed workloads, sum each stream's
    fractional load against one shared budget instead of granting every
    resolution the complete budget. Call the result a theoretical codec-stream
    bound, never a verified camera count: it excludes capture, ISP, transport,
    AI, display, memory contention, and end-to-end latency. Disclosed 30-fps and
    60-fps scenarios may stand in for an unstated frame rate, but they do not
    replace the question: the final response must still explicitly ask for every
    omitted input it assumed or enumerated, naming the codec direction (encode
    captured frames, decode already-compressed streams, or both), exact preset,
    per-stream FPS, and stream mix or count. End by asking for exact
    representative content and a measured, strictly increasing worker sweep to
    verify capacity. Use a read-only calculator for every documented FPS and
    stream-count calculation, preserve full precision until the final floor or
    display rounding, and apply the reference's two-sided maximum-count check;
    never rely on mental arithmetic for a reported capacity.

## References

- [Benchmark workflow](references/benchmark-workflow.md) defines routes,
  preparation, and execution.
- [Benchmark output contract](references/benchmark-output-contract.md) defines
  result states, repetitions, frame accounting, and statistics.
- [Documented performance estimates](references/documented-performance-estimates.md)
  defines the no-media SDK 13.0 table, clock and pixel-area scaling, theoretical
  stream budgeting, and reporting contract.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/benchmark_controller.py` | Dry-run or execute authenticated encode/decode, comparison, and worker-capacity benchmarks. | `--request`, `--workspace`, and `--output`. |

Inspect the controller's public CLI directly:

```bash
python3 -I {baseDir}/scripts/benchmark_controller.py --help
```

## Limitations

- Results apply only to the evidenced target and workload; they are not a
  portable product ceiling.
- Documentation-derived and resolution-scaled values are indicative per-engine
  planning estimates, not achieved FPS, verified concurrency capacity, or a
  substitute for testing representative content.
- Do not interpolate undocumented presets or scale across formats, bit depths,
  chroma, codecs, rate-control modes, or tuning. Scale resolution only through
  the explicitly labeled pixel-area heuristic. Do not multiply by an engine
  count unless that exact target count is independently authenticated, the user
  explicitly requests multi-session aggregate planning, and the answer remains a
  separately labeled multi-session theoretical bound.
- Objective quality measurement, including PSNR and SSIM, is outside this
  performance skill.
- The released PyNvVideoCodec 2.1 encode performance helper caps each worker
  at 1,000 frames. Follow the output contract rather than silently comparing
  unequal frame counts.
- External codecs and wrapper wall time cannot substitute for official
  sample-reported throughput.

## Troubleshooting

- Preserve `input_required`, `selection_required`, `blocked`, `partial`, and
  `failed` instead of promoting them to completion.
- Reject stale outputs, input or environment identity drift, malformed or
  non-finite metrics, wrong processed-frame counts, unsupported sample
  options, and missing positive markers.
- Retain a successful surface when its peer fails, but never pool or
  substitute their results.
