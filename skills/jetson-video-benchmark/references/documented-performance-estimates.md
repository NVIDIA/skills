# Documented performance estimates

Use this answer path only when representative content is unavailable and the
user asks for expected, indicative, achievable, planning, specification, or
theoretical FPS or codec-stream capacity. It is not a benchmark execution and
does not satisfy an explicit request to run or measure real content.

## Authoritative SDK 13.0 rows

The source is the NVIDIA Video Codec SDK 13.0
[NVENC application note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html)
and
[NVDEC application note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html).
The values are indicative, per-engine FPS for Jetson Thor at the documented
highest video clock.

Record the source software conditions with the row. The encode note identifies
Video Codec SDK 13.0 and NVIDIA Jetson Linux for Thor without naming an exact
Jetson Linux revision. The decode note identifies Video Codec SDK 13.0 and
Jetson Linux 38.2. If the planned target uses another release, disclose that
software-version difference rather than treating the calculation as a target
measurement.

Match these rows only when the planned platform is Jetson Thor and the codec,
chroma, bit depth, preset, rate control, and tuning are exact. A different
positive resolution may use only the bounded pixel-area heuristic below. For
any other mismatch, return `documentation_estimate_unavailable`; do not
interpolate or substitute.

An omitted field is not a mismatch. Reserve `documentation_estimate_unavailable`
for an explicitly conflicting request, such as an undocumented P2, P4, or P6
preset. When the user omits a field, keep the answer open: map "high quality" to
the exact VBR / high quality rows and enumerate that small documented preset set
instead of silently choosing one, and read "4K" as 3840x2160 only as a disclosed
planning interpretation, noting that DCI 4096x2160 changes the pixel ratio. Show
the resulting scenario math for the enumerated candidates without claiming one
unique answer, then ask for the exact preset, per-stream FPS, stream mix, and
representative content.

A qualitative encode control that is not an exact table value, including
"medium quality", must not be mapped to one preset, rate-control mode, or
tuning. Preserve the encode direction, enumerate a compact set of exact
documented encode candidates without calling any candidate "medium", and ask
for the exact controls. Never switch to a decode row merely because decode has
fewer required controls. Use decode only for already-compressed camera input;
use separate encode and decode budgets when both stages are requested.

Encode rows apply exactly to 1920x1080, YUV 4:2:0, 8-bit input:

| Codec | Preset | Rate control | Tuning | FPS |
|---|---:|---|---|---:|
| H.264 | P1 | CBR | low latency | 724 |
| H.264 | P1 | VBR | high quality | 713 |
| H.264 | P3 | CBR | low latency | 529 |
| H.264 | P3 | VBR | high quality | 527 |
| H.264 | P5 | CBR | low latency | 236 |
| H.264 | P5 | VBR | high quality | 230 |
| H.264 | P7 | CBR | low latency | 219 |
| H.264 | P7 | VBR | high quality | 202 |
| HEVC | P1 | CBR | low latency | 860 |
| HEVC | P1 | VBR | high quality | 850 |
| HEVC | P3 | CBR | low latency | 402 |
| HEVC | P3 | VBR | high quality | 579 |
| HEVC | P5 | CBR | low latency | 279 |
| HEVC | P5 | VBR | high quality | 336 |
| HEVC | P7 | CBR | low latency | 279 |
| HEVC | P7 | VBR | high quality | 148 |

The SDK 13.0 table has no Jetson Thor AV1 encode performance row. It also has
no P2, P4, or P6 row. Never interpolate or substitute a neighboring preset.

Decode rows apply exactly to 1920x1080 YUV 4:2:0:

| Codec/profile | FPS |
|---|---:|
| H.264 | 1434 |
| VP9 8-bit | 1019 |
| VP9 10-bit | 1016 |
| HEVC Main 8-bit | 1293 |
| HEVC Main10 | 1130 |
| AV1 | 794 |

Do not scale a row across chroma, bit depth, codec, preset, rate control, or
tuning. Scale resolution only with the separately labeled pixel-area heuristic
below. Do not multiply by an encoder or decoder engine count by default. The
table does not measure native-versus-PyNvVideoCodec wrapper overhead or prove
simultaneous-stream capacity.

## Select the target clock

Use a positive configured maximum video clock, not an instantaneous idle clock:

1. Prefer a finite `Max Video` value from the target's authenticated
   `nvidia-smi -q -d CLOCK` output. Bind it to the selected GPU identity.
2. If that field is unavailable on Jetson, accept an exact user- or
   CI-supplied configured maximum clock and label it `user_supplied`, or use
   `nvpmodel -q --verbose` only when its current power-mode output explicitly
   labels `PARAM VIDEO`, its `MAX_FREQ` path, and the configured value. Record
   the command, power mode, path, value, and Hz-to-MHz conversion.
3. If no configured maximum is available, report only the unscaled documented
   row and formula with `target_clock_unavailable`. Do not guess.

Keep a current `Video` clock only as a volatile diagnostic. Never use an idle
reading or an instantaneous `NVENC*_FREQ`/`NVDEC*_FREQ` sample as the numerator
for an achievable-FPS estimate.

## Calculate clock scaling

The SDK 13.0 NVDEC note states that the Jetson Thor rows were measured at a
1691 MHz highest video clock and that performance scales almost linearly with
video clock. For an exact row:

```text
clock_ratio = target_configured_max_video_clock_mhz / 1691
scaled_fps = documented_fps * clock_ratio
```

Require a positive target clock. Treat a configured maximum reported within
1 MHz of 1691 MHz as the same reference operating point because integer-MHz
reporting can round it to 1690, 1691, or 1692 MHz. Preserve the raw reported
clock, normalize the calculation to 1691 MHz (`clock_ratio = 1.0`), and label
the normalization. If the target clock is above 1692 MHz, report the reference
row and decline extrapolation beyond the documented point. Preserve full
precision for the calculation and round only the displayed FPS.

The NVENC note instructs clock scaling but omits the numeric Thor reference
clock. Applying the companion NVDEC note's 1691 MHz value to an encode row is a
`cross_note_inference`, not direct NVENC table metadata. The
[Jetson Linux R39.2 power guide](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html)
supports the shared-domain premise by grouping NVDEC, NVENC, OFA, and NVJPG in
one maximum-frequency row and listing 1557 MHz for the T5000 120 W mode. That
guide's 1692 MHz MAXN entry does not replace the performance table's explicit
1691 MHz reference denominator. Disclose the inference. If the user does not
accept it, leave encode FPS unscaled.

## Scale to any requested resolution

When an exact 1080p source row and a valid target clock are available, estimate
another positive width and height with:

```text
pixel_ratio = (1920 * 1080) / (requested_width * requested_height)
estimated_fps = documented_1080p_fps * clock_ratio * pixel_ratio
```

For 3840x2160, `pixel_ratio` is `1/4`. For example, the HEVC P3 VBR
high-quality row at a 1557 MHz configured maximum video clock gives:

```text
579 * (1557 / 1691) * (1920 * 1080) / (3840 * 2160) = 133.28 FPS
```

This inverse-pixel-area relationship is a `derived_pixel_rate_heuristic`; the
SDK 13.0 notes document the 1080p rows and clock scaling, not resolution
scaling. Treat it as a rough, potentially optimistic planning value. It ignores
per-frame fixed cost, memory and copy bandwidth, codec level and reference
buffer constraints, surface overhead, multi-stream scheduling, power and
thermal throttling, and contention from capture, ISP, CUDA, AI, display, or
transport. Preserve full precision until display rounding.

Use `evidence_class: documented_clock_resolution_scaled_estimate`,
`measurement_performed: false`, and `resolution_scaling_documented: false`.
Record the reference and requested dimensions, pixel ratio, clock ratio, full
formula, and both the encode `cross_note_inference` and
`derived_pixel_rate_heuristic` when applicable. Never call the result measured,
verified, guaranteed, or operational capacity.

## Convert an estimate to a theoretical stream budget

Only answer a no-media “how many streams” question as a theoretical codec-stage
planning bound. For one configuration with a known per-stream frame rate:

```text
usable_fps = estimated_fps * safety_margin
theoretical_stream_count = floor(usable_fps / stream_fps)
```

Use a read-only calculator, such as Python, for every documented FPS and stream
count calculation; do not perform the arithmetic mentally. Preserve full
precision through `floor`. For every reported maximum `count`, verify both
bounds before publishing it:

```text
count * stream_fps <= usable_fps
(count + 1) * stream_fps > usable_fps
```

For a resolution-scaled row, also recompute the exact reference-to-requested
pixel ratio used by that row. Reject or correct any table entry that fails one
of these checks.

Require `0 < safety_margin <= 1`. If the user does not provide one, use `0.9`
only as a disclosed planning default. If per-stream FPS is absent, there is no
unique stream count: report the estimated FPS and formula, optionally show
clearly labeled 30-fps and 60-fps scenarios, and ask for the intended FPS.

For mixed resolutions or configurations, never assign the full engine budget
to every group independently. Calculate each group's estimate from its own
exact documented row and use fractional load:

```text
group_load_i = stream_count_i * stream_fps_i / estimated_fps_i
total_encode_load = sum(encode group loads)
total_decode_load = sum(decode group loads)
```

Each resource pool is feasible only when its total load is no greater than the
chosen safety margin. Keep NVENC and NVDEC pools separate. For groups sharing
one row, this is equivalent to a shared 1080p-pixel-rate budget. Report the
remaining headroom and the complete mix equation rather than independently
overcommitting each resolution.

Use `evidence_class: documented_theoretical_capacity_estimate`, retain the
underlying FPS evidence class, and set `measurement_performed: false` and
`capacity_verified: false`. Call it codec-stream capacity, not physical camera
connectivity or end-to-end pipeline capacity. Published rows are per engine.
Multiply by an engine count only when the exact target count is independently
authenticated, the user explicitly requests multi-session aggregate planning,
and the result is labeled `theoretical_multi_session_engine_budget`; a single
session never receives that multiplication.

Every clock-scaled FPS basis must include:

- `evidence_class: documented_clock_scaled_estimate` for an exact 1080p
  result, or `documented_clock_resolution_scaled_estimate` when the pixel-area
  heuristic is applied;
- `measurement_performed: false`;
- source title, SDK version, URL, exact row, and table conditions;
- source and target software versions when known, including any difference;
- documented FPS, 1691 MHz reference clock, target clock and provenance,
  clock ratio, formula, and scaled FPS;
- the encode `cross_note_inference` when applicable;
- no warmup, repetitions, measured minimum/mean/maximum, or achieved-FPS claim;
- a warning that content, controls, power, thermals, software, and concurrency
  can change real throughput; and
- the exact next action: supply representative media and run the authenticated
  benchmark workflow for a measured result.

A non-1080p or theoretical-capacity answer must additionally include every
field and caveat required by the two sections above. Always end by requesting
representative user content for a real benchmark; for capacity, also request
the intended per-stream FPS and stream mix if either was defaulted or omitted.

When the exact row exists but no configured maximum clock is available, return
only `evidence_class: documented_reference_value` with
`measurement_performed: false`, the source row and conditions, and
`target_clock_unavailable`. Do not emit `clock_ratio`, `scaled_fps`, or call it
a platform estimate. When no exact row exists, return
`documentation_estimate_unavailable` instead.
