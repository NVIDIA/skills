# Encoder recipe contract

## Intent and defaults

Require `use_case`, positive integer `width`, and positive integer `height`.
Accepted use cases are `conferencing`, `live_streaming`, `vod`, `archival`, and
`lossless`. Defaults independent of use case are `codec=h264`, `format=NV12`,
`fps=30`, `gpu=0`, and `output_requirement=elementary_stream`.

Apply this fixed default table only to omitted controls:

| Use case | Preset | Tuning | RC | GOP | BF | Multipass | Lookahead | AQ | Bitrate / max / VBV |
|---|---|---|---|---:|---:|---|---:|---:|---|
| conferencing | p3 | ultra_low_latency | cbr | 30 | 0 | disabled | 0 | — | 3M / 3M / 300k |
| live_streaming | p4 | low_latency | cbr | 60 | 1 | fullres | 0 | — | 6M / 6M / 3M |
| vod | p6 | high_quality | vbr | 120 | 3 | fullres | 20 | 8 | 8M / 12M / 12M |
| archival | p7 | high_quality | vbr | 250 | 3 | fullres | 28 | 8 | 20M / 30M / 30M |
| lossless | p3 | lossless | constqp | 30 | 0 | disabled | 0 | — | `constqp=0,0,0` |

Apply the matching assumptions and retain them verbatim in the recipe's
`assumptions` array. They are preconditions for using a default, not measured
or documented capability claims.

| Use case | Required assumptions |
|---|---|
| conferencing | `latency is prioritized over compression efficiency`; `the application can tolerate strict CBR behavior` |
| live_streaming | `bounded bitrate and moderate latency matter`; `one B frame is acceptable only after latency validation` |
| vod | `offline throughput is secondary to compression efficiency`; `input surfaces remain valid while lookahead consumes them` |
| archival | `quality and size dominate real-time throughput`; `the archive workflow accepts long GOPs` |
| lossless | `the selected codec/profile/surface supports the requested lossless path`; `large output size is acceptable` |

If the caller supplies an average bitrate but omits maximum bitrate, set the
maximum to that bitrate. For conferencing use a 0.1-second VBV; for live
streaming use 0.5 seconds. Record each supplied/defaulted/derived field and its
source. Defaults are starting points, not measured recommendations.

## Schema 2.0

Write one strict JSON object with at least this shape:

```json
{
  "schema_version": "2.0",
  "kind": "nvcodec-recipe",
  "status": "candidate",
  "intent": {"use_case": "live_streaming"},
  "encoder_intent": {
    "codec": "h264", "width": 1920, "height": 1080,
    "format": "NV12", "fps": 60, "gpu": 0,
    "output_requirement": "elementary_stream", "preset": "p4"
  },
  "defaults": {"provided_intent_keys": [], "entries": []},
  "assumptions": [],
  "projection_losses": {"native": [], "pynvc": []},
  "projections": {
    "native": {"status": "exact", "cli_options": []},
    "pynvc": {"status": "exact", "arguments": {}, "config": {}}
  }
}
```

Optional rationale is permitted. All numbers must be finite JSON numbers; all
fields needed by downstream execution remain in `encoder_intent`. Include
`frame_count` once established; never derive it from a media filename.

## Projection rules

Native `cli_options` are an argv array, never a shell string. Begin with:

```text
-s WIDTHxHEIGHT -if FORMAT_LOWER -gpu GPU -codec CODEC
-preset PRESET -tuninginfo TOKEN -rc RC
```

Append present controls using `-profile`, `-fps`, `-gop`, `-bf`, `-multipass`,
`-bitrate`, `-maxbitrate`, `-vbvbufsize`, `-lookahead`, `-aq`, `-cq`, or
`-constqp`; append the operand-free `-temporalaq` only when enabled. Native
tuning tokens are `hq`, `lowlatency`, `ultralowlatency`, `lossless`, and `uhq`.
Native format spellings are `NV12=nv12`, `YUV420=iyuv`, `NV16=nv16`,
`YUV444=yuv444`, `P010=p010`, `P210=p210`,
`YUV444_16BIT=yuv444p16`, `ARGB=bgra`, and `ABGR=abgr`. Never rely on parser
fallback.

PyNv `arguments` carries lowercase `codec`, original `format`, `size`, `gpu`,
and nullable `frame_count`. Its `config` mirrors applicable intent controls,
uses uppercase `P1`–`P7`, `gpu_id`, and typed tuning names. Public 2.1 cannot
express `profile` or intra refresh; record each as a loss and mark the Py
projection `unrepresentable`. Both projections always describe one intent.

## Validation checklist

- `codec`: `h264|hevc|av1`; `preset`: `p1`–`p7`; `rc`:
  `cbr|vbr|constqp`; `multipass`: `disabled|qres|fullres`.
- Width, height, fps, GPU, GOP, BF, lookahead, and frame count are integers;
  `fps/gop/bf <= 2147483647`; bitrate/VBV values `<= 4294967295`.
- CQ is incompatible with average bitrate; const-QP is incompatible with CQ,
  bitrate, maximum bitrate, and VBV. `gop >= bf + 1`; lookahead is
  `0..(31-bf)`; ultra-low-latency requires BF and lookahead zero.
- Apply every format/profile rule in the companion reference. Do not claim a
  live capability or documentation verdict during structural validation.
- `projections.pynvc.config` must exactly equal the JSON passed with the
  official sample's `-json`; benchmark validation depends on this equality.

## File identity and handoff

Create a new file with mode 0600, canonical sorted JSON, and a final newline.
Refuse an existing path or any symlink component. Capture canonical path,
`size_bytes`, and SHA-256 after writing. Pipeline and benchmark consumers must
rehash that same file and reject any difference.

For live compatibility, the fresh setup readiness result must match the recipe
GPU and selected surface. Py capability records remain API facts; native
`AppEncCuda -ec` records remain sample reports. Neither substitutes for the
independent encode/decode proof owned by the pipeline.
