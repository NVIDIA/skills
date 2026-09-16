# Benchmark workflow

Live measurements invoke authenticated released samples directly. The
documentation-only estimate remains a separate calculation path in
[documented-performance-estimates.md](documented-performance-estimates.md).

## Live preflight

1. Canonicalize the input and preserve URL or `null`, license, attribution,
   bytes, SHA-256, codec, format, width, height, FPS, and frame count. For a URL,
   disable redirects; bound connection time, total time, and accepted bytes
   using an explicit user ceiling or trusted local metadata. Resolve the host
   first and accept only HTTP(S) addresses outside loopback, private,
   link-local, multicast, and reserved ranges. Recheck the connected peer
   address on every connection and preserve hostname-based TLS validation.
2. Carry the surface-selection result from `SKILL.md` and obtain a fresh
   read-only readiness result from setup for every selected branch.
3. Native uses the installed package and allowlisted samples. Python uses the
   exact interpreter, loaded extension, wheel `RECORD`, and wheel-owned sample.
   Decode performance may use `pynvc-smoke`; Python encode or comparison needs
   a separate `full-samples` environment.
4. Encode, compare, and encode-capacity require a validated schema-2 recipe;
   decode routes do not. Hold every non-preset fact equal for P4/P5.

Return `dependency_required` for an ineligible selected branch before a
measurement launch. An independently eligible peer may continue, producing a
`partial` aggregate.

## Exact benchmark allowlist

Native paths are relative to the authenticated SDK build/source roots.

| Sample | Executable | Source | CMake file | Required runtime libraries |
|---|---|---|---|---|
| `AppEncPerf` | `AppEncode/AppEncPerf/AppEncPerf` | `Samples/AppEncode/AppEncPerf/AppEncPerf.cpp` | `Samples/AppEncode/AppEncPerf/CMakeLists.txt` | `libcuda.so.1`, `libnvidia-encode.so.1` |
| `AppDecPerf` | `AppDecode/AppDecPerf/AppDecPerf` | `Samples/AppDecode/AppDecPerf/AppDecPerf.cpp` | `Samples/AppDecode/AppDecPerf/CMakeLists.txt` | `libcuda.so.1`, `libnvcuvid.so.1` |

PyNvVideoCodec wheel members:

| Sample | Additional required `RECORD` members |
|---|---|
| `samples/advanced/encode_perf.py` | `samples/utils/__init__.py`, `samples/utils/Utils.py`, `samples/utils/encode_parser.py`, `samples/utils/frame_utils.py`, `samples/utils/encode_parallel_utils.py` |
| `samples/advanced/decode_perf.py` | `samples/utils/__init__.py`, `samples/utils/decode_parser.py` |

The reference owns this allowlist. The fresh setup readiness result supplies
the package root or exact interpreter and package location. Resolve required
tools directly, and authenticate the selected entries immediately before use.
The allowlist does not prove readiness, support, or operation success.

## Build native benchmark samples

Build the selected target directly. Authenticate one installed public
`nvidia-video-codec-sdk` 13.0.x package and one package-owned, unmodified,
non-symlinked Samples tree with
`/usr/bin/dpkg-query -W`, `/usr/bin/dpkg-query -L`, and silent successful
`/usr/bin/dpkg --verify`. Resolve `cmake`, the C++ compiler, `nvcc`,
`pkg-config`, and the selected generator with standard system commands; record
their canonical paths and versions. Create a fresh mode-0700 `BUILD_ROOT`
outside the SDK. With `TARGET` restricted to `AppEncPerf` or `AppDecPerf`,
invoke:

If the exact package is absent or its 13.0.x ownership resolves to zero or
multiple complete Samples roots, return `dependency_required` and route repair
to setup; never infer another package name or choose one root by path order.
A nonzero configure, build, or native option-validation exit fails that branch:
return `failed` with the exact command and captured output, and never launch or
substitute another binary.

```bash
"$CMAKE" -S "$SDK_ROOT/Samples" -B "$BUILD_ROOT" \
  -G "$GENERATOR_NAME" \
  "-DCMAKE_MAKE_PROGRAM=$GENERATOR" \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_CXX_COMPILER=$CXX" \
  "-DCUDAToolkit_ROOT=$CUDA_ROOT" \
  "-DCUDAToolkit_NVCC_EXECUTABLE=$NVCC" \
  "-DCMAKE_CUDA_COMPILER=$NVCC" \
  "-DPKG_CONFIG_EXECUTABLE=$PKG_CONFIG"
"$CMAKE" --build "$BUILD_ROOT" --target "$TARGET" --parallel 2
```

The only accepted outputs under `BUILD_ROOT` are
`AppEncode/AppEncPerf/AppEncPerf` and `AppDecode/AppDecPerf/AppDecPerf`.
Require confinement to the fresh build root, hash the binary, and prove its
table-row libraries resolve with `ldd` to real non-stub files. Recheck package,
source, tools, binary, and libraries after every warmup/measurement series.

## Direct argument arrays

Build one list of arguments and retain it verbatim with every warmup and
measurement. Never pass `-loop`.

```text
native decode:
  APPDECPERF -i INPUT -gpu GPU -thread WORKERS [-single] [-host]

native encode:
  APPENCPERF -i INPUT -s WIDTHxHEIGHT -if NATIVE_FORMAT -gpu GPU
    -frame FRAMES -thread WORKERS -codec CODEC [RECIPE_CONTROLS...]

Python decode:
  PYTHON -I DECODE_PERF -i INPUT -n WORKERS -m PROCESS_MODEL
    -g GPU -f FRAMES

Python encode:
  PYTHON -I ENCODE_PERF -m PROCESS_MODEL -i INPUT -s WIDTHxHEIGHT
    -if FORMAT -n WORKERS -f FRAMES -g GPU -c CODEC -json CONFIG
```

Native format mapping is exact: `NV12→nv12`, `YUV420→iyuv`, `P010→p010`,
`NV16→nv16`, `P210→p210`, `YUV444→yuv444`,
`YUV444_16BIT→yuv444p16`, `ARGB→bgra`, and `ABGR→abgr`.

Native encode recipe controls are the exact native projection excluding the
base options `-s`, `-if`, `-gpu`, and `-codec`; these are supplied once from
the evidenced workload. Python `CONFIG` is the canonical JSON projection for
that recipe and the sole `-json` operand.

## Native option validation

Run both authenticated `AppEncPerf -h` and `AppEncPerf -A`; both must exit zero.
Collect their complete combined stdout/stderr as the advertised-option set.

Before launching, check the planned argument list against this checklist. Any
failure blocks the launch; report the failing item verbatim.

- The launcher path and every argument are nonempty strings containing no NUL.
- Every emitted option (each argument beginning with `-`, excluding a bare `-`)
  appears in the advertised-option set from the two help texts. Report the full
  sorted list of options that are not advertised.
- `-loop` is never emitted; it is forbidden regardless of advertisement.

Launch exactly the checked list with `shell=False`. Do not reconstruct, edit, or
append to the argument list after checking it -- the checked list and the
launched list must be identical. This is a deterministic procedure the agent
performs, not an automatically enforced runtime hook.

## Frame accounting

The PyNvVideoCodec 2.1 `encode_perf.py` helper caps each worker at 1,000 frames.

- If Python encode participates, use `min(source_frames, 1000)` per worker for
  every compared surface, including native, so the same leading frames run.
- Native-only encode retains all requested frames. Decode is never capped.
- Expected aggregate frames are `effective_frames_per_worker * workers`.
- Preserve the original input identity and source frame count; do not trim or
  rewrite the input or recipe.
- `require_source_frames` is true only when the user explicitly requires exact
  source-frame fidelity; it is false otherwise.
- If `require_source_frames` is true and the cap would apply, stop before any
  launch and require native-only or an input at/below the cap.

For each surface, report source and effective frames per worker, worker count,
expected aggregate frames, whether the cap was applied, and its authenticated
PyNvVideoCodec 2.1 sample authority. A fixed object schema is not required.

## Measured lifecycle

For each variant/surface independently:

1. Authenticate the sample and helper identities and recheck input, recipe, and
   config identities.
2. Run one whole-process warmup. Retain its exact argv and outcome but exclude
   it from statistics.
3. Run at least three new whole processes sequentially. Number repetitions from
   one and label each `measure`.
4. Parse the official terminal markers below, require the expected aggregate
   frame count, and reject non-finite/non-positive or contradictory values.
5. Compute MP/s only with sample-bound dimensions, then apply the concise
   acceptance checklist in the output contract before accepting the result.

Official markers:

- AppEncPerf: `nTotal=N, time=S seconds, FPS=F`; repeated occurrences must
  agree and `N/S` must be consistent with displayed precision.
- AppDecPerf: `Total Frames Decoded=N FPS = F`; also require one matching
  codec, frame-rate, coded-size, display-area, chroma, and bit-depth report per
  worker before deriving dimensions/MP/s.
- Python encode: `^Total frames processed: N$`, `^Duration: S seconds$`, and
  `^Total FPS: F$`, with rate consistency within displayed precision.
- Python decode: `^Total frames decoded: N$`, `^Total FPS: F$`, and
  `^Total wall time: Ss$`. Omit MP/s because this sample does not report
  dimensions.

In every route, reject explicit error/fatal/failed/failure/CUDA/NVENC/FFmpeg
failure lines even when the exit code is zero.

Retry a failed branch once at most, after the input,
sample/package/interpreter, environment, or unavailable resource has
demonstrably changed. Retain both attempts.

## Comparisons and capacity

- P4/P5 comparison accepts exactly two recipes. Reopen and rehash both, compare
  their `encoder_intent` objects after removing only `preset`, and require the
  remaining objects to be identical; the removed values must be exactly `p4`
  and `p5`. Keep surface results separate and disclose projection differences.
  Throughput never proves quality ordering.
- A capacity sweep uses strictly increasing worker counts beginning at one.
  Compute `safety_margin = 1 - margin_fraction`,
  `usable_fps = measured_min_fps * safety_margin`,
  `required_fps = workers * nominal_stream_fps`, and
  `headroom_fps = usable_fps - required_fps`; a 10% margin therefore means
  `safety_margin = 0.9`. Record every point with these values and the maximum
  passing tested count. It is a codec-stage bound, not a camera or end-to-end
  result.
- Explicit `both` runs independent branches and returns `partial` when only one
  passes. `auto` never benchmarks two surfaces to choose between them.
