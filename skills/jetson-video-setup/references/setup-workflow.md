# Setup and readiness workflow

## Contents

- [Select scope](#select-scope)
- [Classify intent](#classify-intent)
- [Probe](#1-probe)
- [Plan and validate](#2-plan-and-validate)
- [Execute the reviewed plan](#3-execute-the-reviewed-plan)
- [Verify native](#4-verify-native)
- [Verify PyNvVideoCodec](#5-verify-pynvvideocodec)
- [Report](#6-report)

Use this workflow on the Jetson target. The skill does not open SSH sessions,
copy itself, or treat a GPU-name inference as live evidence.

## Select scope

Before target inspection or the first probe, resolve the requested surface.
Named native and Python products select only their own surface. A bare “video
SDK” setup, install, operation, readiness, or report-only request is ambiguous:
ask whether the user wants native Video Codec SDK, PyNvVideoCodec, or both,
then stop before probing or acting. “Video Codec SDK” is the native product
name, not the bare ambiguous phrase. Report-only intent alone does not select a
surface or authorize broadening to both.

Map the choice to commands as follows:

| User scope | Probe runtime | Plan component |
|---|---|---|
| Native Video Codec SDK | `native` | `native-sdk` |
| PyNvVideoCodec/PySDK/Python | `pynvc` | `pynvc` |
| Explicitly both | `both` | Repeat both component flags |

Use `--runtime both` only when the request explicitly selects both surfaces,
or when a consumer's `auto` gate invokes this read-only probe solely to
authenticate both candidates. That internal candidate probe does not authorize
installing, verifying, executing, or reporting both surfaces.

Evaluate the two surfaces independently. Native Video Codec SDK 13.0.x and
PyNvVideoCodec 2.1.0 are separate products; PyNvVideoCodec does not require the
native developer package.

## Classify intent

- **Report-only/readiness audit:** after the surface is selected, run only its
  read-only probe and reauthentication. Do not build, install, create a venv,
  or launch codec operations. A bare “video SDK” request remains ambiguous:
  ask which surface to inspect and stop before probing.
- **Verify usability:** run the applicable non-installing official-sample
  verifier only when the user authorizes its build/work/output writes.
- **Plan-only:** emit and validate a plan, then stop. The plan marks protected
  commands as requiring confirmation and cannot authorize a transaction.
- **Setup/install:** use `--request-intent setup-install`. The explicit request
  authorizes only the unchanged batches in the reviewed plan.
- **Fresh setup:** additionally use `--fresh-setup`. For Python, require an
  explicit unique, absent `--venv` at an absolute durable path such as
  `/home/ubuntu/.venvs/nvcodec-fresh`, never in the working directory or a
  transient run/evidence tree; the registry outlives that directory. Write the
  `--output` reports somewhere equally durable. Preserve working base packages.

## 1. Probe

Run the selected live probe into a fresh, persistent attempt directory:

```bash
python3 -I scripts/setup/probe_nvcodec.py \
  --runtime native \
  --gpu 0 \
  --output nvcodec-environment-before.json
```

Use `pynvc` or `both` for the corresponding explicitly selected scope. The
probe reads local target, APT, CUDA, Python, package, tool, and import state.
It does not refresh APT, contact a package index, install, build, or launch a
codec. It returns an inventory even when a requested surface is not installed.

Reauthenticate a saved environment with the same public CLI:

```bash
python3 -I scripts/setup/probe_nvcodec.py \
  --reauthenticate nvcodec-environment-before.json
```

This emits `nvcodec-environment-validation`. Do not use a separate validator.
A valid result proves the recorded structure and file identities still match;
it does not prove an encode or decode operation.

### Exact PyNvVideoCodec interpreter

The normal `pynvc` probe resolves the fixed registry at
`$HOME/.local/state/jetson-videosdk/current-pynvc.json` and delegates only to
its authenticated lexical interpreter. It never scans common venv locations
or falls back to system Python.

If the registry is absent or stale and the user says PyNvVideoCodec already
exists, require the exact interpreter or venv path. Probe that lexical
interpreter explicitly as the setup candidate:

```bash
<exact-venv-python> -I scripts/setup/probe_nvcodec.py \
  --runtime pynvc \
  --gpu 0 \
  --setup-candidate \
  --output nvcodec-environment-candidate.json
```

Then run `--reauthenticate` through that same interpreter. Reuse it if valid.
If no path is supplied, ask for one; do not scan, guess, or immediately create
a replacement. Use `--setup-candidate` only for this exact supplied
interpreter or the interpreter created by an authorized setup plan.

## 2. Plan and validate

Generate a plan for only the selected component:

```bash
python3 -I scripts/setup/plan_install.py \
  nvcodec-environment-before.json \
  --component native-sdk \
  --request-intent setup-install \
  --output nvcodec-install-plan.json

python3 -I scripts/setup/plan_install.py \
  validate nvcodec-install-plan.json
```

For PyNvVideoCodec, use `--component pynvc`. For explicit `both`, repeat
`--component` in native-then-Python order. Use `--request-intent plan-only`
when no mutation is authorized.

The `validate` action checks the plan kind/schema, digest, component states,
and authorization fields without mutation. Before an APT transaction,
`plan_install.py` additionally reloads the reviewed sibling
`nvcodec-install-plan.json`, reauthenticates its bound inputs, regenerates the
canonical plan, and requires the same digest. Never execute serialized or
hand-edited caller argv.

If an authorized metadata refresh is required, execute only the plan's emitted
`plan_install.py refresh` command. Discard the old plan, create a fresh probe,
and regenerate it with the exact successful refresh receipt. A refresh can
change all APT candidates, so re-plan every selected package surface.

## 3. Execute the reviewed plan

Read [setup-install.md](setup-install.md), inspect every exact argv, and execute
the plan's batches sequentially. The public owners are:

- `plan_install.py refresh|preview|apply` for APT transactions;
- `lock_pip_reports.py create-venv|materialize-apply` for the isolated Python
  environment and locked pip application;
- `verify_pynvc_sample.py --register-current` for the operation proof and
  registry publication.

Do not reconstruct commands from prose. Preserve the reviewed plan's working
directory and artifact names. Stop on command, candidate, origin, dependency,
hash, setup-mode, or scope drift.

APT packages are eligible only from an already configured, normally
signature-verified source at
`https://repo.download.nvidia.com/jetson/common` or `/som`, exact
`rNN.N/main`, with no trust bypass. Never add, replace, or repair a repository
or signing key.

A blocked component carries no executable command. Continue a separately
actionable selected peer unless a shared APT refresh invalidates both plans.

## 4. Verify native

After native installation or validated reuse, create a fresh native-only
environment and reauthenticate it, then run:

```bash
python3 -I scripts/setup/verify_native.py \
  --environment nvcodec-environment-after-native.json \
  --sdk-root /opt/nvidia/video-codec-sdk \
  --build-dir ./nvcodec-native-build \
  --work-dir ./nvcodec-native-smoke \
  --build --run-encode --run-decode \
  --gpu 0 \
  --output nvcodec-native-verification.json
```

The verifier reauthenticates package ownership, `dpkg --verify`, required
tools, AppDec prerequisites, and real CUDA/NVENC/NVDEC linkage. It builds only
package-owned `AppEncCuda` and `AppDec`. It launches `AppDec` only after the
current `AppEncCuda` run proves exactly one encoded frame, its exact output
marker, and a fresh nonempty H.264 bitstream.

Native readiness requires one decoded frame and an exact 345,600-byte
640×360 NV12 output. H.264 is lossy, so source and decoded pixel hashes need
not match.

## 5. Verify PyNvVideoCodec

For a new environment, execute the plan's exact clean-venv, resolver, and
locked-apply commands. For a valid existing environment, skip all installation
steps and verify only.

Run the final probe and verifier through the exact selected lexical
interpreter:

```bash
<exact-venv-python> -I scripts/setup/probe_nvcodec.py \
  --runtime pynvc --gpu 0 --setup-candidate \
  --output nvcodec-environment-after-python.json

<exact-venv-python> -I scripts/setup/verify_pynvc_sample.py \
  --environment nvcodec-environment-after-python.json \
  --work-dir ./nvcodec-pynvc-smoke \
  --gpu 0 \
  --register-current \
  --output nvcodec-pynvc-verification.json
```

The verifier authenticates the imported module, loaded extension, and every
executed official sample/helper/config file against the installed wheel. It
runs wheel-owned encode before independent decode. Readiness always requires the
exact one-frame encode marker and a fresh nonempty H.264 bitstream. Under
`--profile full-samples` it also requires `advanced/decode.py`'s exact
345,600-byte decoded NV12 output; under the default `pynvc-smoke` it requires
`advanced/decode_perf.py`'s two anchored markers, each exactly once, with no
worker error, warning, or traceback, and produces no decoded output file.

`verify_pynvc_sample.py --register-current` is the only publisher. It first
writes the complete verification result, then publishes the registry only
when that result has `ready=true` and `status=operation_verified`. A failed
proof or publication preserves the previous registry.

## 6. Report

Report:

- target and Jetson Linux release;
- selected surface(s);
- installed/candidate native package and PyNvVideoCodec versions;
- exact Python interpreter for the Python surface;
- independent native and Python inventory and operation verdicts;
- every blocker without suppressing an actionable peer;
- paths, sizes, and SHA-256 values for retained artifacts.

Use `operation_verified` only for a completed official encode→decode proof.
Do not derive codec support from setup inventory. Route capability questions
to `jetson-video-capability`.
