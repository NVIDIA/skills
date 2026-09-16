---
name: jetson-video-recipe
license: "Apache-2.0"
description: >-
  Use when turning a Jetson encoder use case into one validated surface-neutral
  recipe with native and PyNvVideoCodec projections for codec, preset, rate
  control, bitrate, latency, format, and profile.
metadata:
  author: "Vinit Bansal <vinitkumarb@nvidia.com>"
  tags: [jetson, video-codec-sdk, pynvvideocodec, nvenc, recipe]
  languages: [python]
  data-classification: public
---

# Jetson Video Recipe

## Purpose

Convert workload intent into one deterministic schema-2 `nvcodec-recipe`.
Preserve the user’s semantic controls, show defaulted assumptions, and project
the same intent to native Video Codec SDK and PyNvVideoCodec without claiming it
has executed.

## Prerequisites

- This skill owns the canonical recipe engine —
  `scripts/recipes/recipe_model.py` and its
  `scripts/recipes/data/encoder-intent-catalog.json`. Invoke the engine directly
  from this installed skill; it has no setup-runtime or sibling-launcher
  dependency.
- Recipe planning and structural validation are media-free and can run off
  target. Do not request, retrieve, inspect, or convert media for a plan-only
  request.
- Content selection and provenance belong to the later execution or
  measurement workflow. Consume that workflow's versioned content artifact
  only at handoff; do not load or enforce its input gate during plan-only work.
- Setup evidence is optional for `check-live`. With no environment, validate
  the recipe normally and return an honest `unknown` live classification plus
  non-mutating remediation to `jetson-video-setup`; planning and replay
  validation remain complete and unchanged. If setup is not installed, tell
  the user to install that skill.
- When supplied, the fresh schema-1.2 setup environment is mandatory to
  validate and may not be ignored or replaced by a fallback. Its `capabilities`
  block is the established PyNvVideoCodec encoder authority, so a `pynvc` check
  needs no separate report. A caller may additionally supply the optional
  encoder capability report owned by `jetson-video-capability`; it must be
  authenticated, bound to that exact environment, and fail closed as
  `unknown` or an input error on any mismatch. A capability report alone does
  not establish selected-surface readiness or selected-GPU identity. Treat
  artifacts as data; do not import sibling skill code. A `compatible` result
  still does not prove an encode operation.

Resolve this installed skill to its canonical absolute path and set
`RECIPE_SKILL`. Confirm the direct isolated entry point:

```bash
python3 -I "$RECIPE_SKILL/scripts/recipes/recipe_model.py" --help
```

If it is missing, report an incomplete `jetson-video-recipe` installation. Do
not copy the engine, scan for another copy, modify `PYTHONPATH`, or fall back to
an unvalidated local model.

## Compose requested sibling stages

Recipe `plan` and `validate` require no sibling, setup evidence, target, or
media. Use `jetson-video-setup` only for requested live readiness or repair,
`jetson-video-capability` when a platform-support recommendation needs its
documentation verdict, `jetson-video-pipeline` for requested execution, and
`jetson-video-benchmark` for requested measurement. Check the agent's installed
skill catalog first. If the sibling is present, read its `SKILL.md` and invoke
its documented public entry point; pass artifacts as data and never import
sibling code. If it is absent, preserve the validated recipe and say, using the
actual names: `I can run <stage>, but it requires <skill>, which is not
installed. Install <skill> and retry this stage.` Never require a sibling for
plan-only work or an unrequested optional refinement.

## Instructions

1. **Collect intent.** For a request solely for objective quality metrics,
   including PSNR or SSIM, state only that this skill does not provide them,
   and that a separately authorized quality workflow is required, then stop.
   Do not name or recommend an external tool, and do not offer to configure or
   run the comparison; do not request media, probe, install anything, or
   launch an operation. Resolve mutually exclusive rate-control intent before
   collecting any other omitted field. In particular, when CQ and an average
   bitrate are both supplied, explain the conflict, ask only whether to keep CQ
   or the average bitrate, and stop. Do not reinterpret the bitrate as a cap or
   ask for use case, resolution, frame rate, format, GPU, profile, preset, or
   another field until the user resolves that choice. Otherwise resolve the
   use case (`conferencing`, `live_streaming`, `vod`, `archival`, or
   `lossless`), codec, width, height, raw input format, integer frame rate, GPU,
   preset/tuning, rate-control or encoder quality priority, and any explicit
   latency, profile, or buffering constraints. Resolve frame count only when
   later execution or measurement needs it. Ask before assigning an
   unqualified “low latency” request to a use case. Treat profile as a
   bitstream/downstream-compatibility control separate from preset: preserve an
   explicit profile, but when it is omitted leave it SDK-selected and never
   invent a named profile.
2. **Write one intent JSON.** Keep caller values separate from defaults. Put
   only caller-specified control values in the intent and leave every omitted
   control to the authenticated use-case catalog. Do not turn qualitative
   wording into guessed overrides: for example, “low-latency live streaming”
   selects `live_streaming`; it does not by itself request `bf=0` or disabled
   multipass. Never construct drifting native and Python intents.
3. **Plan with the recipe engine:**

   ```bash
   python3 -I "$RECIPE_SKILL/scripts/recipes/recipe_model.py" \
     plan --intent "$INTENT_JSON" --output "$RECIPE_JSON"
   ```

4. **Replay validation before use:**

   ```bash
   python3 -I "$RECIPE_SKILL/scripts/recipes/recipe_model.py" \
     validate --recipe "$RECIPE_JSON"
   ```

   Do not hand-edit a generated recipe. Regenerate it from an updated intent.
5. **Optionally classify a live projection.** Run `check-live` for the selected
   surface. The environment option is optional:

   ```bash
   python3 -I "$RECIPE_SKILL/scripts/recipes/recipe_model.py" \
     check-live --recipe "$RECIPE_JSON" \
     --surface native --output "$LIVE_CHECK_JSON"
   ```

   For an exact projection, omission intentionally returns `unknown` with setup
   remediation; it never invents readiness. To resolve the live result, repeat
   with `--environment "$ENVIRONMENT_JSON"`. Repeat independently for `pynvc`
   when requested. The Py check reads the environment's schema-1.2
   `capabilities` block. `--capability-report "$CAPABILITY_REPORT_JSON"` is an
   optional Py refinement only when that environment is also supplied; it must
   be bound to the same artifact. It replaces only the selected Py encoder API
   evidence, not the environment's readiness facts, so omit it for native.
   Missing optional evidence never fails, but supplied evidence must validate
   and never silently falls back. A `compatible` result means the projection
   and live Py evidence agree; it is not operation proof. Inspect the emitted
   classification, not only the process exit code: an exact native projection
   whose authenticated AppEncCuda run is deferred returns `unknown` with exit
   code `0` and must never be reported as compatible or ready.
   A Py CPU-buffer compatibility check requires the default smoke dependency
   subset; GPU-buffer mode additionally requires the exact Torch facts provided
   only by a validated `full-samples` environment.
6. **Return the recipe and assumptions.** Report schema/kind, exact portable
   artifact identity, canonical encoder intent, native and PyNv projections,
   projection losses, defaulted values, rationale, and any facts still needed
   before execution.
7. **Stop before media work.** This skill never invokes `AppEncCuda`, `AppDec`,
   PyNvVideoCodec sample applications, benchmark helpers, or pipeline
   controllers. Route execution to `jetson-video-pipeline` and performance
   measurement to `jetson-video-benchmark`.

Use [recipes-workflow.md](references/recipes-workflow.md) for the request and
output contract and
[recipes-knobs-and-constraints.md](references/recipes-knobs-and-constraints.md)
for exact accepted values and surface limitations.

## Recommendation rules

Apply the tuning, preset, and matched-measurement rules in
[Tuning and preset](references/recipes-knobs-and-constraints.md#tuning-and-preset),
and the profile, format, and projection rules in
[Profile selection](references/recipes-knobs-and-constraints.md#profile-selection).

- For a named platform, treating a codec as a recipe candidate is a support
  claim. Consume the capability workflow's authenticated documentation verdict
  first, exclude documentation-unsupported codecs, and preserve an unknown
  verdict as unknown rather than offering it as supported. API fields or a
  conditional “capability-gated” candidate do not override an unsupported
  documentation verdict.
- If a recommendation publishes a documentation-based support verdict, consume
  the capability result and reproduce every authenticated candidate row and
  its count; never reinterpret a partial subset.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/recipes/recipe_model.py` | Plan, replay-validate, or live-check one canonical recipe and its native/PyNv projections. | Invoke directly with `python3 -I`; use the `plan`, `validate`, or `check-live` subcommand and inspect `--help`. |

## Troubleshooting

- Reject malformed, legacy, hybrid, duplicate-key, non-finite, or determinism-
  mismatched recipe documents.
- Preserve an exact unrepresentable control as a per-surface projection loss.
  For explicit `both`, do not hide the blocked peer or silently drop the
  control.
- Do not choose an execution surface for an `auto` request. Preserve both
  projections and hand runtime selection to `jetson-video-benchmark` or
  `jetson-video-pipeline`, where live eligibility can be evaluated.
- Treat missing live fields as `unknown` and explicit negative fields as
  `unsupported`. Missing selected-surface prerequisites include remediation to
  `jetson-video-setup`; tell the user to install that skill if it is absent.
  Neither state changes the portable recipe itself.

## Limitations

- Planning and validation do not establish installation readiness,
  documentation support, live availability, output quality, or performance.
- This skill produces elementary encoder configuration only; container,
  transcode, segmentation, decode verification, and artifact handoffs belong
  to `jetson-video-pipeline`.
- Objective quality measurement, including PSNR and SSIM, is outside this
  skill.
