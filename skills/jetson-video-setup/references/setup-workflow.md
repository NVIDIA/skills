# Setup workflow

Use these checks directly on the Jetson. Adapt paths to the installed release,
show mutations before running them, and retain the command output needed to
explain the result.

## 1. Confirm the target

Read the target identity before claiming readiness:

```bash
test -r /etc/nv_tegra_release
sed -n '1p' /etc/nv_tegra_release
sed -n '1,12p' /etc/os-release
```

Record the requested GPU ordinal. If these checks are not running on a Jetson,
provide instructions only.

## 2. Inspect the selected product

### Native Video Codec SDK

Use package-manager ownership as the source of truth:

```bash
/usr/bin/dpkg-query -W -f='${Status}\t${Version}\n' nvidia-video-codec-sdk
/usr/bin/dpkg-query -L nvidia-video-codec-sdk
/usr/bin/dpkg --verify nvidia-video-codec-sdk
command -v cmake
command -v g++
command -v pkg-config
command -v ninja
command -v make
command -v nvcc
ls -1 /usr/local/cuda*/bin/nvcc 2>/dev/null
```

Require one installed package, a silent successful `dpkg --verify`, and exactly
one complete package-owned Samples root. The SDK 13 package uses a versioned
root such as `/opt/nvidia/video-codec-sdk/13.0.37/Samples`; derive the actual
path from `dpkg-query -L`. Select the active `nvcc` from `PATH` or a single
versioned `/usr/local/cuda-*/bin/nvcc`, then derive `CUDA_ROOT` from its parent.
Check the AppDec build dependencies directly:

```bash
pkg-config --exists libavcodec libavformat libavutil libswresample
```

Missing tools or dependencies mean `installed`, not `ready`. Do not search for
an unpacked SDK or use an unowned sample tree as a replacement.

### PyNvVideoCodec

Set `PYTHON` to the exact interpreter selected by the precedence in `SKILL.md`.
Run:

```bash
"$PYTHON" -I -c 'import importlib.metadata as m; d=m.distribution("PyNvVideoCodec"); print(d.version); print(d.locate_file(""))'
"$PYTHON" -I -c 'import PyNvVideoCodec as n; print(n.__file__)'
"$PYTHON" -I -m pip check
"$PYTHON" -I -m pip show -f PyNvVideoCodec
```

Require a successful import from that venv, one installed distribution, and a
clean `pip check`. The sample paths used below must appear in the installed
distribution's file list and stay under its package root. Do not set
`PYTHONPATH`, use system-site packages, or switch interpreters after inspection.

For read-only consumer preflight, these direct checks are sufficient. Return
the package/Samples root or exact interpreter, package version, and loaded
module path. The consuming operation provides its own runtime proof.

## 3. Verify the native product

Create a new user-owned build directory outside the package tree. Resolve each
tool with `command -v`; use `Ninja` when available and otherwise use the
matching installed CMake generator. Configure and build only the two official
samples:

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
"$CMAKE" --build "$BUILD_ROOT" --target AppEncCuda --parallel 2
"$CMAKE" --build "$BUILD_ROOT" --target AppDec --parallel 2
```

The expected binaries are:

- `$BUILD_ROOT/AppEncode/AppEncCuda/AppEncCuda`
- `$BUILD_ROOT/AppDecode/AppDec/AppDec`

Use `ldd` to require real `libcuda.so.1` and `libnvidia-encode.so.1` for the
encoder, and real `libcuda.so.1` and `libnvcuvid.so.1` for the decoder. Reject
missing libraries and CUDA stub paths.

Create a fresh 345,600-byte NV12 fixture in a new output directory, then run:

```bash
dd if=/dev/zero of="$RAW" bs=345600 count=1 status=none
"$APPENC" -i "$RAW" -s 640x360 -if nv12 -gpu 0 -codec h264 -o "$BITSTREAM"
"$APPDEC" -i "$BITSTREAM" -o "$DECODED" -gpu 0
```

Accept native readiness only when all of these hold:

- the raw input is exactly 345,600 bytes;
- the encoder exits zero and prints exactly one `Total frames encoded: 1`;
- the bitstream is newly created, regular, and nonempty;
- the decoder consumes that same path, exits zero, and prints exactly one
  `Total frame decoded: 1`;
- the decoded NV12 output is newly created and exactly 345,600 bytes; and
- neither command reports an explicit CUDA, NVENC, NVDEC, fatal, or failure
  message.

## 4. Verify PyNvVideoCodec

Use only wheel-owned files listed by
`"$PYTHON" -I -m pip show -f PyNvVideoCodec`. Locate these members under the
installed distribution root:

- `samples/basic/encode.py`
- `samples/advanced/decode_perf.py`
- `samples/advanced/decode.py` for `full-samples`
- the wheel's `encode_config.json`

Create the same fresh one-frame NV12 input and run the wheel-owned encoder:

```bash
"$PYTHON" -I "$BASIC_ENCODE" \
  -i "$RAW" -o "$BITSTREAM" -s 640x360 \
  -m cpu -if NV12 -f 1 -g 0 -c h264 -json "$CONFIG"
```

For the default smoke profile, independently consume it with:

```bash
"$PYTHON" -I "$DECODE_PERF" \
  -i "$BITSTREAM" -d 1 -f 1 -n 1 -m thread -g 0
```

For a separately provisioned `full-samples` environment, use:

```bash
"$PYTHON" -I "$ADVANCED_DECODE" \
  -i "$BITSTREAM" -o "$DECODED" -d 1 -g 0 -f 1
```

Require one `Completed encoding 1 frames using CPU buffers` marker and a fresh
nonempty bitstream. The smoke decoder must print exactly one
`Successfully decoded requested 1 frames` and one `Total frames decoded: 1` as
literal substring occurrences. The threaded sample prefixes the first marker
with its worker name, so do not require that marker to occupy the whole line.
Require no smoke-decoder worker warning, error, or traceback. The full decoder
must report one requested decoded frame and produce exactly 345,600 bytes. Exit
zero alone is not sufficient because a worker failure may otherwise be hidden.

## 5. Report

Return the concise result defined in
[setup-output-contract.md](setup-output-contract.md). Include actual commands
and logs only to the extent needed to reproduce or diagnose the setup. Create
a checksum package when the user requests one.
