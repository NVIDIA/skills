---
name: jetson-video-setup
license: "Apache-2.0"
description: >-
  Use when installing, repairing, probing, or verifying native NVIDIA Video
  Codec SDK or PyNvVideoCodec on Jetson with official encode-to-decode samples,
  including registered-environment recovery.
metadata:
  author: "Vinit Bansal <vinitkumarb@nvidia.com>"
  tags: [jetson, video-codec-sdk, pynvvideocodec, setup, nvenc, nvdec]
  languages: [python]
  data-classification: public
---

# Jetson Video Setup

## Purpose

Probe, install, and independently verify the native NVIDIA Video Codec SDK and
PyNvVideoCodec surfaces on a live Jetson. Setup owns installation readiness,
not codec-support verdicts, recipes, benchmarks, or application pipelines.

## Read before acting

- Read [setup-workflow.md](references/setup-workflow.md) for surface selection
  and the probe → plan → apply → verify order.
- Read [setup-install.md](references/setup-install.md) before any APT, venv, or
  pip mutation.
- Read [setup-output-contract.md](references/setup-output-contract.md) before
  consuming or reporting an artifact.

## Select the surface

Before step 1 or any probe, resolve the requested surface. "Video Codec SDK",
"VC SDK", "native SDK", or `nvidia-video-codec-sdk` selects native;
"PyNvVideoCodec", "PyNv", "PySDK", or Python selects PyNvVideoCodec. Match a
named product before considering the bare phrase: "Video Codec SDK" is the
native product name even though it contains the words "video SDK". A genuinely
bare "video SDK" setup, install, operation, readiness, or report-only request
is ambiguous: ask only whether the user wants native Video Codec SDK,
PyNvVideoCodec, or both, then stop before probing, acting, or describing future
probes, checks, installation steps, or report contents. Report-only intent
alone does not select a surface or authorize broadening to both.

Select both only when explicitly requested, and reuse the selection for the
rest of the request. One narrow exception applies to a consumer skill's
`auto` selection gate: that consumer may invoke setup's read-only probe with
`--runtime both` solely to evaluate both candidates. This does not select both
for installation, verification, execution, or the final report.

Keep the selected surfaces independent. A native failure must not suppress an
actionable Python surface, and a Python failure must not suppress native.
Report aggregate `both` readiness only after both verification chains pass.

## Compose requested sibling stages

Setup's probe, plan, install, and verification workflow requires no sibling
skill. When a complex request also asks for product capability, recipe,
performance, or pipeline work, add only the corresponding
`jetson-video-capability`, `jetson-video-recipe`, `jetson-video-benchmark`, or
`jetson-video-pipeline` stage. Check the agent's installed skill catalog first.
If the sibling is present, read its `SKILL.md` and invoke its documented public
entry point; pass artifacts as data and never import sibling code. If it is
absent, preserve completed setup results and say, using the actual names: `I
can run <stage>, but it requires <skill>, which is not installed. Install
<skill> and retry this stage.` Never acquire a sibling for an unrequested
stage.

## Workflow

1. Confirm execution is on the Jetson. On a non-Jetson host, produce guidance
   only and make no live readiness claim.
2. Probe the selected surface with `probe_nvcodec.py --runtime
   native|pynvc|both --output ...`. Use `both` only when the request explicitly
   selects both surfaces or for the narrow read-only consumer `auto` candidate
   check above. The probe is read-only. Reauthenticate a saved artifact with
   the same CLI's `--reauthenticate` action.
3. For PyNvVideoCodec, use the fixed validated-venv registry or an exact
   user-supplied interpreter. Never scan for or guess a venv. If the user says
   PyNvVideoCodec is already installed but supplies no exact path and the
   registry is not ready, ask for the path before provisioning anything. A
   missing registered interpreter makes that registry not ready; a registered
   interpreter that cannot be launched blocks the selected Py surface. Never
   scan or fall back to another environment.
4. Generate an install plan with `plan_install.py`, then run
   `plan_install.py validate PLAN`. A report-only request stops after the
   probe; `plan-only` never authorizes mutation. Use `setup-install` intent only
   for an explicit install/setup request; that request authorizes only the
   complete unchanged batches in the reviewed plan.
5. Execute only literal commands from the reviewed `setup-install` plan.
   Invoke every published `argv` verbatim as the current user, including steps marked `privilege: "root"`; never prefix `sudo`, because `plan_install.py` owns the authorized internal `sudo -n` escalation for APT operations.
   `plan_install.py` owns APT refresh, preview, and apply actions;
   `lock_pip_reports.py` owns clean-venv creation and the locked pip apply.
   APT execution regenerates the canonical plan and rechecks live candidate,
   origin, source, and simulation evidence before mutation.
6. Re-probe the completed surface. Run `verify_native.py` for native or
   `verify_pynvc_sample.py` for Python. Each setup proof uses the installed
   release's official samples to encode one 640×360 NV12 frame to H.264, then
   independently decode that fresh bitstream. Native, and Python under
   `--profile full-samples`, decode to exactly 345,600 bytes. The default
   Python profile `pynvc-smoke` decodes one bounded frame with
   `advanced/decode_perf.py`, which writes no raw output, so it proves frame
   production only. A consumer that genuinely needs Torch — Python
   encode-benchmark, pipeline, or the full raw-decode proof — is blocked under
   `pynvc-smoke`; say so and name the remedy: provision a `full-samples` venv
   explicitly with `plan_install.py --profile full-samples`. Exit zero alone is
   never proof: require the profile's exact positive markers and counts. Only a
   passing verifier may promote the selected surface from probe `partial` to a
   final ready verdict.
7. After a ready Python verification, publish the fixed registry only with
   `verify_pynvc_sample.py --register-current --output READY_REPORT`.
8. Report the detected Jetson Linux release, product versions, independent
   surface verdicts, blockers, and artifact identities.

Use `--fresh-setup` only when the user explicitly requests a new setup or
reinstall. It never authorizes removing working base packages. A fresh Python
setup also requires a unique, previously absent `--venv`. That `--venv` must be
an absolute path under a durable location, for example
`/home/ubuntu/.venvs/nvcodec-fresh`; never place it in the current working
directory or any transient run, session, or evidence tree, because the registry
you publish outlives that directory. Relative `--output` names resolve against
the working directory, so write setup reports somewhere equally durable.

## Direct setup scripts

Run every public CLI under `python3 -I` and inspect its `--help` before building
arguments.

| File | Public responsibility |
|---|---|
| `scripts/setup/probe_nvcodec.py` | Emit or reauthenticate the read-only live `nvcodec-environment` schema 1.2 artifact. |
| `scripts/setup/plan_install.py` | Plan and validate selected components; execute only its own reviewed APT refresh/preview/apply actions. |
| `scripts/setup/lock_pip_reports.py` | Create a new venv and materialize/apply the authenticated pip lock. |
| `scripts/setup/verify_native.py` | Build package-owned `AppEncCuda`/`AppDec` and verify the fixed native encode→decode smoke. |
| `scripts/setup/verify_pynvc_sample.py` | Authenticate and run wheel-owned Python encode/decode samples, emit the readiness artifact, and authenticate the validated-venv registry chain. |
| `scripts/setup/setup_contract.py` | Private common mechanics for these setup CLIs: strict JSON, bounded commands, and public-APT binding; never invoke it as a CLI. |

There is no setup dispatcher. Invoke these five public CLIs directly. Setup
must not import Python code from another skill, and another skill must not
import setup's private implementation.

## Readiness and scope

- Inventory or import presence is not operational proof.
- `operation_verified` requires both official operations, their positive
  markers, and a fresh nonempty bitstream. Native and Python `full-samples`
  additionally require the exact decoded frame count and raw-output size;
  Python `pynvc-smoke` instead requires its two exact one-frame production
  markers and claims no raw decoded artifact.
- Exit zero or output-file creation alone is insufficient.
- Setup emits only bounded baseline Py API-query observations and raw native
  sample summaries as supporting readiness evidence. It does not emit the
  complete decoder tuple matrix or a product-support verdict; use
  `jetson-video-capability` for those questions.
- Use `jetson-video-recipe`, `jetson-video-benchmark`, and
  `jetson-video-pipeline` for configuration, measurement, and handoff work.
- A local probe proves only the detected stack and minimum release gate; it
  does not prove release currency or the newest release compatible with this
  target. Call a release `latest` or `newest compatible` only when successfully
  retrieved current official NVIDIA documentation, recorded with URL and
  retrieval date, establishes both release currency and compatibility with the
  authenticated target identity. Otherwise report newest-compatible as
  `unknown` and point to the official compatibility documentation; local APT
  state, a failed source, or either fact alone is insufficient.
- For a quality-only request such as PSNR or SSIM, state that setup does not
  provide it and that a separately authorized quality workflow is required,
  then stop; do not install, invoke, name, recommend, or offer to set
  up an external quality tool.

## Safety

- Accept native SDK/CUDA packages only from the configured,
  signature-authenticated stock public NVIDIA Jetson source
  (`repo.download.nvidia.com/jetson/common` or `/som`, exact `rNN.N/main`).
  Base prerequisites may use another already configured,
  signature-authenticated APT origin. Bind every candidate to its exact source
  record and never add or change a source or key.
- Keep credentials out of argv, logs, artifacts, stdout, and stderr.
- Preserve exact plan, package, interpreter, artifact, and source identities.
- Use fresh output/work/build paths. Never overwrite evidence or reuse it after
  a reflash, driver/package change, or venv replacement.
