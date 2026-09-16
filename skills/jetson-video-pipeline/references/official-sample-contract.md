# Official sample and validation contract

The lists below define route selection. Before each operation, bind every
selected native file to the installed `nvidia-video-codec-sdk` package and
every selected Python file to the exact interpreter and PyNvVideoCodec
distribution reported by the fresh setup readiness check. Reject a changed or
unowned file, a stub/missing library, or a different loaded extension.

## Exact pipeline allowlist

Native paths are relative to the authenticated SDK build/source roots.

| Sample | Executable | Source | CMake file | Required runtime libraries |
|---|---|---|---|---|
| `AppEncCuda` | `AppEncode/AppEncCuda/AppEncCuda` | `Samples/AppEncode/AppEncCuda/AppEncCuda.cpp` | `Samples/AppEncode/AppEncCuda/CMakeLists.txt` | `libcuda.so.1`, `libnvidia-encode.so.1` |
| `AppDec` | `AppDecode/AppDec/AppDec` | `Samples/AppDecode/AppDec/AppDec.cpp` | `Samples/AppDecode/AppDec/CMakeLists.txt` | `libcuda.so.1`, `libnvcuvid.so.1` |
| `AppTrans` | `AppTranscode/AppTrans/AppTrans` | `Samples/AppTranscode/AppTrans/AppTrans.cpp` | `Samples/AppTranscode/AppTrans/CMakeLists.txt` | `libcuda.so.1`, `libnvcuvid.so.1`, `libnvidia-encode.so.1` |

PyNvVideoCodec wheel members:

| Sample | Additional required `RECORD` members |
|---|---|
| `samples/basic/encode.py` | `samples/utils/__init__.py`, `samples/utils/Utils.py`, `samples/utils/encode_parser.py`, `samples/utils/frame_utils.py` |
| `samples/advanced/decode.py` | `samples/utils/__init__.py`, `samples/utils/Utils.py`, `samples/utils/decode_parser.py` |
| `samples/basic/create_video_segments.py` | `samples/utils/__init__.py`, `samples/utils/transcode_parser.py`, `samples/basic/segments.txt`, `samples/basic/transcode_config.json` |

The allowlist in this reference owns route selection. The fresh probe supplies
the SDK root, package, tools, interpreter, extension, and dependency evidence;
the agent authenticates the selected source or wheel members immediately
before use. Do not run another sample merely because it exists in a source
tree or wheel.

## Build package-owned native samples

Build the selected target directly. First require `/usr/bin/dpkg-query -W` and
`-L` to identify one installed public `nvidia-video-codec-sdk` 13.0.x package
and its single owned SDK/Samples tree. Require
`/usr/bin/dpkg --verify nvidia-video-codec-sdk` to
exit zero with no output. Reject a symlinked/unowned source, multiple candidate
roots, or any package/source identity change.

The package name and single-root rule are exact. If the package is absent,
renamed, or resolves to zero or multiple complete 13.0.x roots, return
`dependency_required` and route repair to setup; never guess a package or choose
one candidate by path order.

Resolve `CMAKE`, `CXX`, `NVCC`, `PKG_CONFIG`, `GENERATOR`, and
`GENERATOR_NAME` with standard system commands. Take `SDK_ROOT` from the
package-owned Samples path and `CUDA_ROOT` from the resolved `nvcc`; record
canonical paths and versions. Select only `AppEncCuda`, `AppDec`, or `AppTrans`
from the table above as `TARGET`, create a fresh mode-0700 `BUILD_ROOT` outside
`SDK_ROOT`, and invoke these exact argument arrays:

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

The resulting launchers under `BUILD_ROOT` are exactly
`AppEncode/AppEncCuda/AppEncCuda`, `AppDecode/AppDec/AppDec`, and
`AppTranscode/AppTrans/AppTrans`. Require the selected binary to
remain inside `BUILD_ROOT`, hash it, and use `ldd` to prove every library in
its table row resolves to a real non-stub file. Recheck package, sources,
tools, binary, and libraries after execution. A failed target does not erase a
separately successful target; a nonzero configure or build exit fails that
target, returns `failed` with the exact command and captured output, and never
launches or substitutes another binary.

## Direct argument templates

Treat each placeholder as one argument. `NATIVE_RECIPE_ARGS` is the validated
recipe's exact native projection; `CONFIG` is the wheel/config artifact bound
to the same recipe.

```bash
"$APPENC" -i "$RAW" -o "$BITSTREAM" "${NATIVE_RECIPE_ARGS[@]}"
"$APPDEC" -i "$BITSTREAM" -o "$DECODED" -gpu "$GPU"

"$PYTHON" -I "$BASIC_ENCODE" \
  -i "$RAW" -o "$BITSTREAM" -s "${WIDTH}x${HEIGHT}" \
  -m "$BUFFER_MODE" -if "$FORMAT" -f "$FRAMES" -g "$GPU" \
  -c "$CODEC" -json "$CONFIG"
"$PYTHON" -I "$ADVANCED_DECODE" \
  -i "$BITSTREAM" -o "$DECODED" -g "$GPU" -d 0 -f "$FRAMES"

"$APPTRANS" -i "$INPUT" -o "$OUTPUT" -gpu "$GPU" \
  "${TRANSCODE_RECIPE_ARGS[@]}"
"$PYTHON" -I "$CREATE_SEGMENTS" \
  -i "$INPUT" -s "$SEGMENTS" -c "$TRANSCODE_CONFIG" \
  -o "$OUTPUT_TEMPLATE" -g "$GPU"
```

For `AppTrans`, copy the exact native recipe projection except raw-only `-s`,
`-if`, and `-gpu`; add the selected `-gpu` once. Never use string evaluation or
silent option aliases.

## Observable completion

Match anchored lines and require the expected count/path exactly:

- Native encode: `^Total frames encoded: ([0-9]+)$`.
- Native decode count: `^Total frame decoded: ([0-9]+)$`.
- Encode/decode saved output: `^Saved in file (.+) in (NV12|P016|YUV444|YUV444P16|NV16|P216) format$`.
- Transcode saved output: `^Saved in file (.+) of (8|10) bit depth$`.
- Py encode: `^Completed encoding ([0-9]+) frames using (CPU|GPU) buffers$`,
  one `^ENCODING COMPLETE$`, and the requested output path exactly twice as
  `^Output file: (.+)$`.
- Py full decoder: exactly one of
  `^Successfully decoded all ([0-9]+) frames to (.+)$` or
  `^Successfully decoded requested ([0-9]+) frames to (.+)$`.
- Py smoke decoder: exactly one log line ending in
  `Successfully decoded requested ([0-9]+) frames` and exactly one
  `^Total frames decoded: ([0-9]+)$`. Its threaded log may prefix the first
  marker and does not report a decoded output path.
- Native transcode: exactly one total across legacy
  `\(#totFrames=([0-9]+)\)` and current
  `^Total frame transcoded: ([0-9]+)$`; reject duplicates or mixed forms.
- Segments: ordered `^✓ Created: (.+)$` lines exactly matching the validated
  schedule outputs, plus
  `^Summary: ([0-9]+) segments created successfully$`.

In all cases reject an explicit error/fatal/failed/failure/CUDA/NVENC/FFmpeg
failure line, even when the process exits zero. Require fresh nonempty output,
then independent consumption of the same path/size/SHA-256.

## Frame and layout check

Apply this check to the actual raw source and to every decoded output, and
retain the result. It is a mandatory acceptance step, not optional guidance.
Width, height, and frame count must be positive integers. Formats with
subsampling enforce their geometry parity before byte arithmetic.

**Frame-layout rule.**

A raw source file must satisfy the byte-size rule below. A decoded output must
also satisfy the independent decoder checks:

1. `size_bytes == width * height * frames * numerator / denominator` for the
   declared pixel format, using the table below. The product must divide exactly
   -- a non-integral result is a failure, never a rounding case.
2. The package-owned decoder's reported frame count equals `frames`, and its
   reported layout equals the table's decoded layout name.

Fact 1 authenticates raw frame boundaries. For decoded output, both facts are
required; raw encoder input has no decoder marker of its own.

| Pixel format   | numerator/denominator | Width | Height | Decoded layout name |
|----------------|-----------------------|-------|--------|---------------------|
| `NV12`         | 3/2                   | even  | even   | `NV12`              |
| `YUV420`       | 3/2                   | even  | even   | `NV12`              |
| `P010`         | 3/1                   | even  | even   | `P016`              |
| `NV16`         | 2/1                   | even  | any    | `NV16`              |
| `P210`         | 4/1                   | even  | any    | `P216`              |
| `YUV444`       | 3/1                   | any   | any    | `YUV444`            |
| `YUV444_16BIT` | 6/1                   | any   | any    | `YUV444P16`         |
| `ARGB`         | 4/1                   | any   | any    | not a decoded layout|
| `ABGR`         | 4/1                   | any   | any    | not a decoded layout|

Reject any pixel format not in this table rather than guessing a size. Enforce
the width/height parity column; an odd dimension for a subsampled format is a
failure. When reporting a decoded artifact, use the decoded layout name from the
last column -- `ARGB`/`ABGR` are input/packed formats only and must not be
reported as a decoder output layout.

Report the resolved path, pixel format, width, height, frames, expected bytes,
and observed bytes. For a decoded artifact also report the decoder's frame
count and layout. Never infer the frame count from a filename.

Worked example: 302 frames at 1920x1080 `NV12` -> 1920 * 1080 * 302 * 3 / 2 =
939,340,800 bytes, and `AppDec` must report 302 decoded frames.

## AppEncCuda AV1 output

Do not create or run a separate IVF parser. Prove the artifact the same way as
every other output: independently decode the exact produced path with the
package-owned decoder and require a positive frame count equal to the requested
frame count. Report the output path, size, SHA-256, and decoder frame count.

## Evidence classifications

- `api_query_helper`: an API returned fields; never operation proof.
- `official_sample_report`: an authenticated sample's report mode.
- `official_sample_operation`: an authenticated sample was launched and passed
  the observable checks above.
- `documentation_reference`: product-support authority for the exact tuple;
  never live readiness by itself.
- `absent_release_sample`: absent from the authenticated release payload.
- `source_tree_candidate`: present in source but not an authenticated runnable
  route.

Changing an allowlist requires matching direct-command instructions and focused
identity, option, marker, stale-output, and handoff tests.
