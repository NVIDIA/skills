# NVENC/PyVC knob checklist

## Contents

- [Tuning and preset](#tuning-and-preset)
- [Profile selection](#profile-selection)
- [Rate control and buffering](#rate-control-and-buffering)
- [Latency and quality controls](#latency-and-quality-controls)
- [Surface gates](#surface-gates)
- [Runtime reconfiguration](#runtime-reconfiguration)

## Tuning and preset

The supplied parser defaults `preset` to P4 and, for SDK-10+ presets, defaults an omitted
`tuning_info` to `NV_ENC_TUNING_INFO_HIGH_QUALITY` (`NvEncoderClInterface.cpp:696-724`). It does
not make the workload low latency merely because P1/P2 is selected. Multi-pass encoding is
separately defaulted to disabled unless `multipass` is provided (`:1063-1066`). Use the typed
value `ultra_high_quality` for UHQ. The wrapper preserves that typed value as intent evidence but
translates the executed native CLI and PyNvVideoCodec sample JSON to the parser token `uhq`;
`ultra_high_quality` must never reach the released sample parser unchanged. Public-2.1 Python UHQ
is restricted to HEVC/AV1, Turing-or-newer hardware, and a loaded extension recording linked NVENC
API 12.2 or newer; the `_121` surface does not contain the UHQ token and must remain blocked. Thor
satisfies the generation prerequisite. Other GPU-generation claims remain operation-verified rather
than inferred when authoritative generation evidence is unavailable.

Treat P1–P7 as candidates along a speed/quality trade-off within a
codec/tuning/hardware context. Do not call one faster or higher quality without
matched measurements. For later measurement, let the owning benchmark or
pipeline workflow validate its versioned content-provenance artifact and
measure adjacent presets on the same user-selected content and frame range.

Without matched measurements on identical representative content, do not present qualitative or
numeric cross-codec storage, compression, quality, or output-size rankings as measured or proven.
A documentation/support-based option may be labeled only as a starting candidate for matched
measurement; words such as "better", "best", and "largest" remain unmeasured claims unless the
labeled evidence proves that comparison.
When H.264 and HEVC are both documentation-supported for an 8-bit storage-priority comparison,
HEVC Main is the first matched-measurement candidate and H.264 is the compatibility baseline.
This orders measurement; it does not call either codec a measured storage, compression, quality,
or output-size winner.
For an explicit native HEVC profile, select Main10 when 10-bit output is required and downstream
decoder/container/playback compatibility is confirmed. NVENC can accept authenticated 8-bit 4:2:0
input for Main10 and convert internally; P010 is the exact no-bit-depth-conversion input path. Use
Main for ordinary 8-bit 4:2:0 output or ask for the missing facts. The released PyNvVideoCodec 2.1
sample cannot express `profile`; report that projection limitation instead of claiming that Main or
Main10 was applied.

Use only `p1` through `p7`, exact lowercase tuning values `high_quality`, `low_latency`,
`ultra_low_latency`, `lossless`, or `ultra_high_quality`, and multipass values `disabled`, `qres`,
or `fullres`.
Invalid parser strings can fall back to defaults, so reject them rather than reporting the
requested string as the applied setting.
Schema-1 canonical config and schema-2 `encoder_intent` presets remain lowercase `p1` through
`p7`. The schema-2 Python projection, like public-2.1 executed JSON, uses
uppercase `P1` through `P7`, while native AppEnc CLI argv remains lowercase. Evidence must preserve
both the canonical typed config and the executed config; these are explicit token translations, not
intent changes. Any parser fallback diagnostic is an operation failure.

## Profile selection

Profile is a bitstream-tool and downstream-compatibility choice; it is independent of preset.
Do not call a profile faster, higher quality, or more storage-efficient without matched
measurements. Preserve an explicit caller profile exactly. When the caller and downstream contract
do not require one, omit it and report the result as SDK-selected/default rather than guessing or
claiming a named profile.

| Codec/profile | Selection guidance | Required constraint or gate |
|---|---|---|
| H.264 Baseline | Choose only for a decoder or transport that explicitly requires this restricted legacy profile. | 8-bit 4:2:0; resolved `bf=0`; live profile support and operation verification. |
| H.264 Main | Choose when the downstream contract explicitly requires Main. | 8-bit 4:2:0; live profile support and operation verification. |
| H.264 High | Compatibility-oriented candidate for modern 8-bit 4:2:0 H.264 consumers; this is not a quality claim. | 8-bit 4:2:0; live profile support and operation verification. |
| H.264 High444 | Choose for an 8-bit 4:4:4 bitstream requirement or as one prerequisite of native H.264 lossless. It may also carry 8-bit 4:2:0; the name does not require 4:4:4 input. | Reject 10-bit input in this released Jetson recipe path. When 4:4:4 is requested, require matching input/capability. Lossless also requires QP zero, CONSTQP, transform bypass from the lossless path, and positive lossless capability. |
| HEVC Main | Ordinary 8-bit 4:2:0 output candidate. | Matching codec/profile support and operation verification. |
| HEVC Main10 | Choose when 10-bit 4:2:0 output is required and the downstream consumer supports it. NVENC can convert verified 8-bit input internally, but P010 is the exact no-bit-depth-conversion input path. | Positive 10-bit capability, authenticated input depth, downstream compatibility, and operation verification. |
| HEVC FRExt | Choose for native 4:2:2 or 4:4:4 output, at 8 or 10 bits. | Matching NV16/P210/YUV444/YUV444_16BIT semantics and exact chroma/bit-depth capabilities. |
| AV1 Main | The AppEnc AV1 profile covers 4:2:0 at 8 or 10 bits. | 4:2:0 only; 10-bit needs the exact capability. Release-matched product documentation still governs support. |

The released PyNvVideoCodec 2.1 sample cannot express a named `profile`. Every explicit profile is
therefore a structured Python projection loss, never a silently dropped control. An emitted
bitstream may later reveal the runtime-selected profile, but recipe planning alone does not prove
that profile. Parser acceptance also does not prove a profile/format pair operates; keep structural
validation, live capability, operation, and emitted-bitstream proof separate.

## Rate control and buffering

- `rc=cbr`: bounded-rate starting point for live workloads; set bitrate, max bitrate, and VBV deliberately.
- `rc=vbr`: quality/size starting point when bitrate may vary.
- `rc=constqp`: constant-QP path; the QP value uses the separate `constqp` option. Do not combine with bitrate targets.
- Gate the selected mode against `supported_ratecontrol_modes` from the exact codec query:
  positive mask presence proves const-QP, bit `0x1` proves VBR, and bit `0x2` proves CBR. Missing,
  malformed, or non-positive evidence leaves support unknown.
- RC strings are lowercase-only. An unrecognized value, including `CQP`, falls back to CBR (`NvEncoderClInterface.cpp:912-926`).
- Supplying `bitrate` sets CBR unless an explicit valid `rc` later overrides it (`:1050-1108`).
- Supplying `cq` zeros average and maximum bitrate on the public Python parser; shared recipes
  therefore reject CQ with either bitrate field (`:1096-1103`). CQ is a VBR target-quality value:
  shared recipes use exact integer `0..51` for H.264/HEVC and `0..63` for AV1.
- A direct typed native AppEncPerf/AppEncCuda request may express finite fractional CQ in that
  codec range and may pair it with `maxbitrate` as a cap, but not with average `bitrate`. This
  native-only expressiveness must not be copied into shared public-2.1 Python recipe JSON.
- `constqp` uses one integer or `P,B,I` integers without spaces. Use `0..51` for H.264/HEVC and
  `0..255` for AV1. Do not retain bitrate, maximum-bitrate, VBV, or CQ keys in const-QP mode.
- `qmin`: do not generate in this release because the parser fetches `qmin` and then parses `initqp` at `NvEncoderClInterface.cpp:1119-1122`.

## Latency and quality controls

- `bf`: gate against exact `num_max_bframes`. The resolver enforces `bf=0` for
  `ultra_low_latency`; use `low_latency` when B-frames are required.
- `lookahead`: gate on `support_lookahead` and enforce the API range
  `0..(31 - bf)`. The resolver enforces zero for `ultra_low_latency`.
- `multipass`: `disabled`, quarter-resolution, or full-resolution choices must be throughput-tested. Do not infer support from `support_multiple_ref_frames`; they are different concepts.
- `aq`: spatial AQ strength is 1–15 when enabled. `temporalaq` is a distinct option and is gated on `support_temporal_aq`.
- `temporalaq` is enabled by key presence in the supplied parser. Emit `1`/`true` to enable it;
  omit the key to disable it—do not emit zero or false.
- `gop`: choose from recovery/random-access needs and frame rate; require `gop >= bf + 1`. A longer
  GOP is not automatically better.
- `fps`: the installed public PyNvVideoCodec 2.1 sample/config contract and the native projection
  require an exact positive JSON integer. Reject decimal numbers, decimal strings, and fraction
  strings, and cap native `std::stoi` fields (`fps`, `gop`, `bf`) at `INT_MAX=2147483647`.
  Bitrate/VBV integer fields are capped at `UINT32_MAX=4294967295`. Do not generalize fractional
  handling observed in a newer local/development source tree to the authenticated public 2.1 wheel.
- Intra refresh is unsupported through rel_2_1 PyVC kwargs. Native NVENC
  structs/caps expose it, but `NvEncoderClInterface` only prints the native
  fields and never parses a public kwarg
  (`:1375-1377,1438-1440,1745-1747`). Report it unrepresentable on the
  selected released projection and stop without proposing another interface.

## Surface gates

- `YUV420`, `ARGB`, and `ABGR` are recipe/standalone encoder-input surfaces, not current
  independent-decode pipeline target formats. Native `YUV420` pipeline output would require an
  explicit `AppDec -outplanar` route that is not implemented, public-2.1 PyNvVideoCodec
  `advanced/decode.py` cannot emit planar `YUV420`, and `AppDec` cannot emit matching RGB for
  `ARGB`/`ABGR`. Resolving an encoder recipe for one of these formats does not claim a complete
  content-to-independent-decode pipeline.
- `profile` is native-AppEnc-only because public PyNvVideoCodec 2.1 does not parse it. Native
  profiles are codec-specific: H.264 `baseline|main|high|high444`; HEVC
  `main|main10|frext`; AV1 `main`. H.264 Baseline requires resolved `bf=0`; reject a positive
  caller or catalog/default B-frame value instead of silently rewriting it. Lookahead remains
  independently gated. Use [Profile selection](#profile-selection) for format, depth, lossless,
  omitted-profile, and downstream guidance.
- CPU-buffer sample readiness requires NumPy plus exact `pycuda==2026.1`; GPU-buffer mode also
  requires the validated CUDA-enabled Torch record.

- P010/P016 and other >8-bit surfaces require `support_10bit_encode`.
- AV1 encode is structurally limited to 4:2:0 input. Reject NV16, P210, YUV444, and
  YUV444_16BIT before capability interpretation even when feature fields are positive. P010
  remains eligible when `support_10bit_encode` is positive; H.264/HEVC 4:2:2 and 4:4:4 remain
  live-capability gated.
- H.264/HEVC 4:4:4 surfaces require `support_yuv444_encode`.
- The public 2.1 GPU utility implements H.264/HEVC YUV444 as three full-resolution planes and
  assigns a `width * height * 3` frame size; do not block this path when the live 4:4:4 capability
  is positive.
- H.264/HEVC NV16/P210 4:2:2 encoder surfaces require `support_yuv422_encode`; absence from an
  API-12.1-linked module is unknown, not unsupported. `P216` is decoder-output naming and is not
  accepted as an encode input here.
- 4:2:0 widths and heights must be even.

## Runtime reconfiguration

Do not generalize from the full native API. The supplied Python binding's `structEncodeReconfigureParams` exposes rate-control mode, multipass, average/max bitrate, VBV size/initial delay, frame-rate numerator/denominator, and target quality (`PyNvEncoder.cpp:1172-1198`). Any requested field outside that Python surface requires source/API verification and may require recreating the session.
