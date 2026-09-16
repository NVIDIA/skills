# Video content policy

This setup-owned reference defines the shared media-selection and evidence
policy for every Jetson video skill. Setup owns the policy and deterministic
smoke fixture; benchmark and pipeline workflows own normal media execution.

## Route matrix

| Workflow | Content rule |
|---|---|
| Setup/readiness verification | A deterministic synthetic raw fixture is allowed; user media is not required. |
| Capability API query | Media-free. A minimal capability-operation smoke test may use setup's deterministic fixture. |
| Recipe planning, validation, or live compatibility check | Media-free. Stop before media work. |
| Documentation-only benchmark estimate | Media-free. Do not launch a controller or claim a measurement. |
| Live benchmark `encode`, `decode`, `compare`, or `camera_capacity` | Require exact user-selected media under the input gate below. |
| Pipeline `encode_decode`, `native_transcode`, `pynvc_segments`, `container_triage`, `av1_verify`, or `acceptance` | Require exact user-selected media under the input gate below. |
| Direct recipe-bound encode/decode execution | Require exact user-selected media, except for the capability-smoke fixture above. |
| Pipeline `content-summary` validation | Consume the already bound external media and evidence; never select or acquire replacement media. |

Never use synthetic, uniform, black, or generated content for recipe
execution, performance/benchmarking, a content-sensitive comparison, or a
pipeline.

## Input gate

Recipe planning and validation, capability queries, and documentation-only
benchmark estimates are media-free. Before any recipe execution, live
benchmark dry run or execution, content-sensitive comparison, or pipeline dry
run or execution, require the user to supply exactly one of:

- an absolute target-local media path; or
- one exact HTTP(S) media URL.

If neither is present, return `input_required` with
`next_action: provide_media_path_or_url`, ask the user for one, and pause before probing,
retrieving, converting, constructing a controller request, or launching an operation. Never browse
for, select, recommend, or retrieve media from a bundled catalog or an alternate source.
The controller's `synthetic_input_allowed: false` field is an agent-routing rule, not a byte-content
classifier: an artifact identity alone does not prove who selected the file or how its frames were
created. Bind request construction to the user's supplied path or URL and never turn the setup
fixture into an execution input.

Treat user selection as operation input, not as proof of copyright or license rights. Preserve
license and attribution exactly when evidenced or supplied. Use the literal string `unknown` when
either is unknown; never infer or invent it.

## Resolve the selected source

- For a local path, require a canonical absolute regular-file path on the target, keep the source
  read-only, and record its current size and SHA-256. Set `source_url` to JSON `null`.
- For a URL, first prove validated target eligibility and an authenticated released route for the
  intended operation. Retrieve only the exact user-supplied HTTP(S) URL into the fresh workspace.
  Record that exact requested URL as `source_url` and preserve the retrieval command evidence.
  Never substitute a mirror, nearby catalog entry, or different agent-selected URL without asking
  the user.
- If the source is missing, unreadable, empty, changes identity, or cannot be retrieved, report the
  evidenced input failure and ask for a replacement. Do not relabel it `unsupported` or
  `operation_failed`.

## Use in workflows

- Classify the selected media as raw, elementary-stream, or container input before choosing a
  released sample. Invoke only the authenticated, release-owned NVIDIA sample selected by the
  workflow for demux, decode, or raw-frame production.
- For an encoder run, create raw frames only through an approved NVIDIA sample that can write the
  required format. Record the source and derived SHA-256, geometry, format, frame rate, frame count,
  and decode evidence. Do not pass a container file to a raw-input runner.
- If the selected NVIDIA sample does not accept the user-selected source, ask for another local
  path or URL, or report the route blocked. Do not choose fallback media or use FFmpeg to replace
  codec work.
- Use the same user-selected content and frame range across comparisons. The controllers validate
  byte identity, frame alignment, metadata, and official-sample evidence; they do not infer scene
  complexity, representativeness, ownership, or nonuniformity. Never promote the setup fixture into
  a performance input. Report results only for the exact user-selected workload and do not
  generalize it into a complete workload suite.

## Evidence

For every selected input, preserve `source_url` (the exact user URL, or JSON
`null` for local media), honest `license` and `attribution` strings (the literal
`unknown` is allowed), canonical path, byte size, observed SHA-256,
codec/container, dimensions, frame rate, pixel format, bit depth, and frame
count. Preserve a published source checksum only when it is actually available.
