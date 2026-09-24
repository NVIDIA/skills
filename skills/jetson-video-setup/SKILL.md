---
name: jetson-video-setup
license: "Apache-2.0"
description: >-
  Use when installing, repairing, reusing, inspecting, or verifying readiness of
  the native NVIDIA Video Codec SDK or PyNvVideoCodec on Jetson, including the
  one-frame encode/decode smoke test with official samples, and when
  interpreting what those readiness results, including CPU-buffer and
  device-memory sample modes, do and do not establish.
metadata:
  author: "Vinit Bansal <vinitkumarb@nvidia.com>"
  tags: [jetson, video-codec-sdk, pynvvideocodec, setup, nvenc, nvdec]
  languages: [markdown]
  data-classification: public
---

# Jetson Video Setup

## Purpose

Inspect, install, and verify the native NVIDIA Video Codec SDK and
PyNvVideoCodec on a live Jetson. Setup owns product installation and readiness.
Use the sibling video skills for codec support, encoder configuration,
performance measurement, and application pipelines.

Use the standard package manager, Python environment tools, and installed
NVIDIA samples directly, then return a concise readiness result.

## Read before acting

- Read [setup-workflow.md](references/setup-workflow.md) for inspection and
  readiness checks.
- Read [setup-install.md](references/setup-install.md) completely once before
  an APT, venv, or pip change.
- Read [setup-output-contract.md](references/setup-output-contract.md) for the
  concise human-readable result to return.
- Read [video-content.md](references/video-content.md) only when a request also
  supplies or asks to retrieve media.

## Select the product

Resolve the requested product before touching the target:

- “Video Codec SDK”, “VC SDK”, “native SDK”, or
  `nvidia-video-codec-sdk` selects the native product.
- “PyNvVideoCodec”, “PyNv”, “PySDK”, “Python SDK”, or an explicitly Python
  interface selects PyNvVideoCodec.
- Select both only when the user asks for both.

A genuinely bare “video SDK” setup or readiness request is ambiguous. Ask
whether the user wants native Video Codec SDK, PyNvVideoCodec, or both, then
stop. The word “report” does not resolve that ambiguity.

Keep native and Python work independent. A failure on one surface must not
erase a successful result from the other.

## Workflow

1. Confirm commands would run on a Jetson. Inspect `/etc/nv_tegra_release`,
   `/etc/os-release`, and the requested GPU ordinal. On another host, provide
   guidance only and make no readiness claim.
2. Inspect only the selected product with the direct commands in
   [setup-workflow.md](references/setup-workflow.md). For native, identify the
   installed package, package-owned Samples tree, CUDA toolkit, and build
   tools. For Python, use the exact interpreter supplied by the user or
   created during this request and inspect its installed distribution and
   loaded module. Never scan the filesystem for virtual environments.
3. If inspection is all the user requested, report what is installed and stop.
   Package or import presence is `installed`, not `ready`.
4. If the selected product is missing and the user asked to install or repair
   it, follow that install reference. Show the exact
   package or pip commands before mutation. Install and verify system
   prerequisites before creating a final Python environment path. Change only
   the selected product and its missing prerequisites.
5. After installation, repeat the direct inspection. Then run the installed
   release's official one-frame encode followed by independent decode as
   described in [setup-workflow.md](references/setup-workflow.md).
6. Report each selected product separately. Use `ready` only after its official
   encode and independent decode pass all observable checks. Otherwise report
   `installed`, `blocked`, or `failed`, name the failing command or missing
   prerequisite, and give one concrete next action.

When another video skill asks only for readiness, perform steps 1 and 2 and
return the exact package/Samples root or Python interpreter/package path. Do
not run the setup smoke test if that consumer will immediately run its own
authenticated operation.

## PyNvVideoCodec environment selection

Use this interpreter precedence: an explicit user path, an exact path already
established in the current conversation, then the conventional profile path.
The conventional smoke interpreter is `$HOME/.venvs/nvcodec/bin/python`; the
`full-samples` interpreter is `$HOME/.venvs/nvcodec-full/bin/python`. Checking
one of these exact paths is not a filesystem scan. Never select a venv by
directory order or fall back to system Python.

For a new environment, use the applicable conventional path when it is absent,
or an explicit new absolute path in a durable user-owned location. If the
conventional path exists, inspect it first. Reuse it when valid; otherwise
report its exact defect, leave it untouched, and ask for a different new path.
Return the selected interpreter path so downstream skills can use it directly.

Use `pynvc-smoke` for the setup smoke proof, decode-performance work, and a
conventional interface-availability check. That availability check inspects
only the exact smoke path and reports `not_ready` when it is absent or invalid.
Any consumer encode operation, including a capability availability proof, and
work using advanced raw decode, segmentation, or encode-performance samples
selects the separate `full-samples` environment and dependencies described in
the install guide. A media-free capability inventory may use either
conventional profile: inspect smoke first, then `full-samples` when smoke is
absent, and report the exact interpreter that answered.
Apply that either-profile allowance only when the entire request is a
media-free inventory. When a request also seeks an operation or a capability
availability proof, use the profile that request binds; if that profile is not
ready, report `not_ready` and do not query the other conventional profile. The
interface-availability check above remains bound to the smoke path.

## Readiness proof

Both products use one generated 640×360 8-bit NV12 frame (345,600 bytes), H.264
encode, and an independent decode of the fresh bitstream.

| Product | Required evidence |
|---|---|
| Native | Package-owned `AppEncCuda` reports one encoded frame; package-owned `AppDec` consumes that exact nonempty bitstream, reports one decoded frame, and writes a 345,600-byte NV12 output. |
| PyNv `full-samples` | Wheel-owned `basic/encode.py` reports one encoded CPU-buffer frame; wheel-owned `advanced/decode.py` consumes the exact bitstream, reports one frame, and writes a 345,600-byte output. |
| PyNv `pynvc-smoke` | The same encoder proof; wheel-owned `advanced/decode_perf.py` reports one requested decoded frame and a total of one, with no worker error, warning, or traceback. It does not claim a raw decoded file. |

Exit zero or file creation alone is insufficient. Require the expected marker
and count, a newly created nonempty bitstream, and the independent consumer.
Do not require decoded bytes to equal the input because H.264 is lossy.

CPU-buffer and device-memory sample modes establish only the exact readiness
operation. They do not establish external buffer sharing, zero copy, or a
synchronization primitive for another process or pipeline stage. For a request
to confirm an in-process or cross-stage buffer contract, invoke
`jetson-video-pipeline`; require an authenticated operation of the actual
downstream stage that proves the sharing handle, format and layout, ownership
and lifetime, signal and wait behavior, and safe buffer reuse. Do not answer
that request from setup evidence alone.

## Compose requested sibling work

For a request that also asks about support, encoder configuration, throughput,
or an application workflow, invoke only the matching public skill:

- `jetson-video-capability`
- `jetson-video-recipe`
- `jetson-video-benchmark`
- `jetson-video-pipeline`

Pass the selected product and exact local paths as data. Do not read or import
a sibling skill's private files. If a required sibling is unavailable,
preserve completed setup results and name the missing skill.

## Safety

- For a report-only request, perform direct read-only inspection only; never
  build samples, create a workspace or venv, launch a codec operation, or
  mutate packages or an existing Python environment.
- Use only already configured, signature-authenticated package repositories.
  NVIDIA SDK and CUDA packages must come from the public Jetson repository for
  the installed release. Never add or edit a source, key, or trust bypass.
- Before APT installation, inspect the candidate and origin, run the exact
  `apt-get -s` simulation, and reject removals, downgrades, or unexpected
  packages. Apply the reviewed command with noninteractive `sudo -n`; if that
  authorization is unavailable, stop rather than using a password prompt,
  `su`, or another escalation path.
- Run `dpkg --audit` after an APT mutation and stop if it is nonempty or fails.
- Keep native and Python acquisition separate. A Python-only request must not
  install `nvidia-video-codec-sdk`; a native-only request must not create a
  venv.
- Keep credentials out of commands, logs, and reports. Reject symlinked or
  unexpected install targets and use fresh build/output paths.
- Local installation and smoke results do not establish product support or
  that the release is the newest compatible release. Use current official
  NVIDIA documentation for those claims.

For PSNR, SSIM, DRM playback, capture, inference, or display work, state that
setup does not own that workflow and route only an explicitly requested video
codec portion.
