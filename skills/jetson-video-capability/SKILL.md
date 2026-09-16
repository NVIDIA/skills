---
name: jetson-video-capability
license: "Apache-2.0"
description: >-
  Use when Jetson codec, profile, chroma, bit-depth, dimension, engine-count,
  or operational support must be reconciled from live APIs, authenticated
  NVIDIA samples, and NVIDIA documentation; also applies the content-DRM scope.
metadata:
  author: "Vinit Bansal <vinitkumarb@nvidia.com>"
  tags: [jetson, video-codec-sdk, pynvvideocodec, nvenc, nvdec, capability]
  languages: [markdown]
  data-classification: public
---

# Jetson Video Capability

Keep three authorities separate:

- API query: raw fields, never operation or product-support proof.
- Authenticated NVIDIA sample: report or exact operation evidence.
- Applicable NVIDIA documentation: product-support verdict.

## Scope gates

- For a request solely for PSNR, SSIM, or other objective quality metrics, say
  this skill does not provide them and a separately authorized quality workflow
  is required; do nothing else.
- For Netflix, Widevine, PlayReady, or clearly content-protected streaming,
  state only that this skill covers hardware encode/decode of user-supplied
  non-DRM bitstreams, not content-DRM playback. Do not claim whether the
  service works, describe Jetson certification, CDM, or secure-playback
  requirements, or recommend a browser, DRM module, workaround, or bypass;
  then stop. An unqualified “DRM” may mean Linux DRM/KMS; ask which meaning if
  context does not resolve it.
- A bare “video SDK” is ambiguous: ask native Video Codec SDK,
  PyNvVideoCodec, or both before probing. An otherwise unqualified capability
  request uses the native-preferred fallback in
  [surface-selection-contract.md](references/surface-selection-contract.md).

## Read-only discovery

Capability work depends on `jetson-video-setup` for a fresh, read-only
installation check. Invoke that skill through public dispatch and consume its
reported exact native package/Samples root or exact PyNvVideoCodec interpreter,
version, and loaded module path. Do not locate or import setup's files. A
missing or mismatched surface is `unknown`/`not_ready`, never codec unsupported;
route repair to setup without mutating anything here.

Engine capability queries belong here, not in setup. For PyNvVideoCodec, use
the exact selected interpreter to call the public `GetEncoderCaps` and
`GetDecoderCaps` APIs as described in
[capability-queries.md](references/capability-queries.md). A broad encoder
catalog covers H.264, HEVC, and AV1; a broad decoder catalog covers all ten
families across four chroma formats and three bit depths (120 exact tuples).
A bounded request queries only named members of the applicable catalog set.
Preserve every scoped record, error, and GPU ordinal. Nonzero-GPU helper results
remain unknown when the public helper selects only GPU 0.

For native reports, reuse authenticated package-owned binaries or build only
the required report target in a fresh user-owned tree, then run
`AppEncCuda -ec` and/or `AppDec -dc` using the build, identity, and grammar
rules in the capability reference. A query-only request does not authorize
package installation or an encode/decode operation. If the package, source,
tool, interpreter, or runtime-library identity cannot be established, report
the result `unknown` and name the missing setup prerequisite.

## Decision flow

1. Classify scope and selected surface before any probe.
2. Query only the selected surface. Preserve raw records, errors, exact GPU,
   release, interpreter/binary identities, argv, and evidence classification.
3. Resolve the most exact live product identity using the device-tree paths and
   NUL handling in
   [capability-queries.md](references/capability-queries.md#documentation-reconciliation),
   then cross-check it against NVIDIA's current
   [Video Encode and Decode Support Matrix](https://developer.nvidia.com/video-encode-decode-support-matrix)
   and the release-matched SDK 13.0
   [NVENC](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html)
   or [NVDEC](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html)
   application note. Never infer SKU from memory, engine count, capability
   fields, or operation behavior. Preserve source URL, retrieval date, table
   title, exact row/column labels, and cell value, using that reference's manual
   capture record.
   Generic family identity stays non-exact; use candidate-row consensus only
   when every authenticated candidate agrees.
4. Report documentation `No` as the final unsupported product verdict even if
   an API or diagnostic operation is positive; this ends the normal
   availability check. Documentation `Yes` establishes documented support;
   live availability additionally needs the matching authenticated operation.
   Missing, unretrievable, or conflicting documentation remains unknown.
   When documentation is unknown, do not present positive capability fields as
   available options: label each affected codec `unknown` beside them and state
   the retrieval failure with the verdict.
5. For a PyNvVideoCodec encode-availability operation, request setup's
   `full-samples` profile and carry its exact interpreter into the recipe and
   pipeline stages. Do not select the smaller decode-performance/smoke profile.
6. Run an exact operation only when the user requests live availability and
   documentation is positive or unknown, or explicitly requests a diagnostic
   despite a negative verdict. A diagnostic under documentation `No` reports
   only `operation_verified` or `operation_failed` for that exact tuple and
   never changes the unsupported product verdict. A tuple is the exact surface,
   GPU, codec, profile, chroma, bit depth, dimensions, input format, and control
   set tested. Resolve one recipe with `jetson-video-recipe`, then use
   `jetson-video-pipeline` for encode followed by independent decode of the
   exact output identity. A failed tuple never generalizes to the product.

## Evidence rules

- `GetEncoderCaps` success is `capability_reported`, `supported=null`,
  `operation_status=not_tested`.
- `GetDecoderCaps` `bIsSupported=1/0` records raw API true/false and whether
  returned limits apply; neither value is the documentation verdict.
- Missing enums, calls, fields, prerequisites, or nonzero-GPU authority are
  unknown, not unsupported.
- Native `-ec/-dc` text is `official_sample_report`; preserve raw values and
  never rewrite them as API or product claims.
- When reporting Main10 support, note that NVENC can convert verified 8-bit
  input internally and that P010 is the exact no-input-bit-depth-conversion
  path; a live claim still requires an authenticated 10-bit operation.
- AV1 output is operation evidence only after the pipeline's package-owned
  decoder consumes the exact artifact and reports the expected frame count.
- For a VP9 encode question, report only that VP9 remains a decoder family and
  no released NVENC route exists. Do not add a generic list of other encoders
  or offer a diagnostic encode operation.
- Presets, tuning, package presence, throughput, and successful concurrent
  streams do not establish codec support or NVENC/NVDEC engine count.
- Codec/API capability, NVENC/NVDEC availability, and successful bounded
  operations do not establish DMA-BUF, NvSciBuf, CUDA-memory sharing, zero
  copy, or cross-stage synchronization compatibility.

For a Python API query, create a short task-local program from this Markdown
and the installed public SDK, run it with the exact selected interpreter, and
keep its raw output with the result. Installation belongs to setup and
operation validation belongs to pipeline. This skill owns engine queries,
classification, documentation reconciliation, and the compact direct report
above.
