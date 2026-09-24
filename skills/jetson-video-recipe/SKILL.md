---
name: jetson-video-recipe
license: "Apache-2.0"
description: >-
  Use when turning a Jetson encoder use case into one surface-neutral recipe
  with native Video Codec SDK and PyNvVideoCodec projections.
metadata:
  author: "Vinit Bansal <vinitkumarb@nvidia.com>"
  tags: [jetson, video-codec-sdk, pynvvideocodec, nvenc, recipe]
  languages: [markdown]
  data-classification: public
---

# Jetson Video Recipe

Create one deterministic `nvcodec-recipe` schema 2.0 document. Recipe work is
off-target and media-free: it does not probe, install, encode, decode, or claim
support, quality, or performance.

## Boundaries

- For a request solely about PSNR, SSIM, or another objective quality metric,
  say that a separately authorized quality workflow is required and stop.
- Resolve a supplied CQ plus average or maximum bitrate conflict first. Return
  `input_required`, ask only which one to keep, and stop; do not reinterpret a
  bitrate as a cap or emit a recipe.
- An unqualified “low latency” does not select a use case. Ask whether it means
  conferencing, live streaming, or another contract, return `input_required`,
  and stop without emitting a recipe.
- Preserve every explicit control. If one surface cannot express it, publish a
  projection loss; never silently discard or weaken it.

## Workflow

1. Collect use case, codec, positive integer width/height/fps, input format,
   GPU, and any explicit profile, preset, tuning, rate-control, bitrate, GOP,
   B-frame, lookahead, AQ, multipass, or buffering controls. `frame_count` is
   required before raw encode or measurement, but may remain unknown until an
   authenticated decoder/transcoder reports it for compressed input. If
   `use_case`, `width`, or `height` is missing, return `input_required` naming
   exactly the missing items and stop; apply the documented defaults for every
   other omitted item. Leave omitted profile SDK-selected.
2. Apply the fixed defaults and constraints in
   [recipes-knobs-and-constraints.md](references/recipes-knobs-and-constraints.md),
   then build the schema-2 document exactly as described in
   [recipes-workflow.md](references/recipes-workflow.md). Keep caller values and
   defaults separately attributable.
3. Validate the document structurally: exact schema/kind; finite JSON; positive
   bounded integers; legal enum strings; mutually exclusive rate-control
   fields; format/profile constraints; projections derived from the same
   `encoder_intent`; and every explicit caller control represented in each
   projection or named in that projection's losses. Regenerate rather than
   editing an accepted recipe. If the regenerated document still fails
   structural validation, return `failed` with the exact defect and do not emit
   a recipe.
4. Write canonical, sorted JSON to a fresh path without overwriting anything.
   Record its canonical absolute path, byte count, and SHA-256; every consumer
   rehashes that exact file. If no safe fresh path exists or writing/rehashing
   fails, return `failed` with the exact reason and do not claim a recipe.
5. Return the intent, assumptions/defaults, both projections, all losses, and
   the recipe file's canonical absolute path, byte count, and SHA-256. For
   `both`, retain both outcomes. `auto` means retain both projections without
   selecting either; the pipeline or benchmark selects from fresh live
   eligibility evidence.

For a plan-only recipe, these Markdown rules are the complete authority. Do not
probe the target, inspect installed SDK/sample source, scan the filesystem for
example JSON, or invoke another skill merely to confirm the projection.
Plan-only still requires steps 2-5, including writing and rehashing the fresh
canonical recipe JSON and reporting its absolute path, byte count, and SHA-256;
it forbids target and media operations, not local recipe-artifact creation.

## Live and downstream work

For a requested live classification, obtain a fresh read-only readiness result
from `jetson-video-setup` for the selected product and GPU, plus the applicable
raw/documentation result from `jetson-video-capability`. Missing facts remain
`unknown`; explicit negatives or an unrepresentable projection are
`unsupported`. API-reported capability is not operation proof.

Pass the original recipe identity as data to `jetson-video-pipeline` for
execution or `jetson-video-benchmark` for measurement. Those skills must
rehash it and hold non-compared controls constant. Do not import or recreate a
sibling skill's implementation.

## Required outcomes

- H.264 1920x1080@60, 6 Mbps CBR live streaming defaults to P4,
  `low_latency`, GOP 60, one B-frame, zero lookahead, full-resolution
  multipass, max bitrate 6,000,000, and VBV 3,000,000.
- An explicit H.264 High profile is exact in the native projection and
  `unrepresentable` in the public PyNvVideoCodec 2.1 sample projection.
- A CQ plus average bitrate request produces no recipe until resolved.

## Limitations

This skill produces elementary encoder configuration only. Content selection,
container/transcode work, independent decode, benchmarking, and evidence
capture belong to their owning skills.
