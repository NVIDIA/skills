# Public installation workflow

## Contents

- [Product boundaries](#product-boundaries)
- [Authorization and planning](#authorization-and-planning)
- [Stock public APT policy](#stock-public-apt-policy)
- [Native Video Codec SDK](#native-video-codec-sdk)
- [PyNvVideoCodec](#pynvvideocodec)
- [Fresh setup](#fresh-setup)
- [Official operation proof](#official-operation-proof)

## Product boundaries

| Surface | Acquisition | Setup proof |
|---|---|---|
| Native Video Codec SDK | `nvidia-video-codec-sdk` 13.0.x from the configured Jetson APT release | Package-owned `AppEncCuda` → package-owned `AppDec` |
| PyNvVideoCodec | `PyNvVideoCodec==2.1.0` and its dependencies in an isolated venv | Wheel-owned `samples/basic/encode.py` → `samples/advanced/decode_perf.py` by default, or `samples/advanced/decode.py` with `--profile full-samples` |

These are independent products. A PyNvVideoCodec request must not install the
native developer package, and a native request must not create or modify a
Python environment. CUDA, driver, compiler, SDK, and Python package versions
are separate facts; do not require their version strings to match.

Record the detected Jetson Linux release and apply the documented minimum
release gate, but do not impose an exact JetPack/L4T revision. Setup readiness
comes from the applicable installed-artifact operation proof, not an API
capability query.

## Authorization and planning

Always probe before planning. Plan only the selected `native-sdk`, `pynvc`, or
explicitly both components.

- `--request-intent plan-only` creates a review artifact and authorizes no
  protected command.
- `--request-intent setup-install` preauthorizes only unchanged commands in the
  reviewed plan for the selected components.
- `--fresh-setup` requires the complete fresh action for each nonblocked
  selected surface.

The plan groups protected commands into these deterministic batches:

| Batch | Purpose |
|---|---|
| `apt-metadata-refresh` | One authorized metadata refresh, followed by mandatory re-probe/re-plan |
| `native-sdk-install` | Candidate-pinned native package preview and apply |
| `pynvc-bootstrap` | Required candidate-pinned CUDA/Python bootstrap and new-venv creation |
| `pynvc-resolution` | Isolated, non-applying pip resolver reports |
| `pynvc-install` | Authenticated lock materialization and exact locked apply |

Review the plan with `plan_install.py validate`. Execute only the exact
`refresh`, `preview`, or `apply` argv emitted by that plan. Before APT
mutation, `plan_install.py` requires the literal command to occur in the
reviewed sibling `nvcodec-install-plan.json`, reauthenticates the bound inputs,
regenerates the canonical plan, and requires its digest to match. It then
rechecks candidate/source policy and repeats the safe simulation immediately
before apply.

Any change in component, setup mode, command, package, candidate, origin,
source file, dependency set, version, or archive hash invalidates the reviewed
authorization. Re-probe, re-plan, show the difference, and obtain authorization
for the changed scope.

## Stock public APT policy

Accept the native SDK or selected CUDA build candidate only when all of these are
true:

- its configured source is exactly
  `https://repo.download.nvidia.com/jetson/common` or
  `https://repo.download.nvidia.com/jetson/som`;
- its suite/component is exactly `rNN.N/main`;
- APT signature enforcement is active;
- no `trusted`, insecure, weak, downgrade-to-insecure, or unauthenticated
  bypass is enabled;
- the candidate origin is bound to the exact configured-source record and
  source-file SHA-256 captured by the probe;
- the candidate remains unchanged at apply time.

Do not accept an internal mirror, HTTP NVIDIA source, wrong suite/component,
unsigned candidate, trust bypass, or unbound origin. Base Ubuntu prerequisites
such as `pkg-config`, `python3-venv`, and `python3-dev` may use another already
configured APT origin only when its package-signature chain, exact source
record, and source-file SHA-256 are bound by the probe. Never add, replace, or
edit a repository or signing key.

When local metadata has no candidate but configured sources exist, the plan
may emit one `plan_install.py refresh` action. After it succeeds, discard the
old environment and plan, probe again, and build a new plan bound to the
refresh receipt. Do not refresh repeatedly or carry a pre-refresh candidate
forward.

Every APT apply uses an exact `package=version`, `--no-remove`, noninteractive
form after a matching safe simulation. A removal, downgrade, candidate drift,
origin drift, or unsafe simulation blocks the apply.

## Native Video Codec SDK

Use only a stable public 13.0.x
`nvidia-video-codec-sdk=<observed-candidate>`. A normal installed 13.0.x
surface can take the verify-only route. An explicitly fresh native setup emits
the candidate-pinned simulation/apply pair with `--reinstall`; it does not
uninstall the working package.

The final native probe must show:

- the package name, installed status, and version;
- canonical SDK root;
- complete AppDec `pkg-config` prerequisites for `libavcodec`,
  `libavformat`, `libavutil`, and `libswresample`;
- installed CUDA root/version;
- authenticated `cmake`, C++ compiler, `nvcc`, `pkg-config`, and CMake
  generator records.

If a prerequisite is absent, install it only when the selected plan contains
its exact authenticated APT candidate. Never infer missing development
packages from an incomplete or timed-out query.

`verify_native.py` reauthenticates package version/ownership and
`dpkg --verify`, builds only `AppEncCuda` and `AppDec` in a fresh user-owned
directory, and rejects unresolved or stub CUDA/NVENC/NVDEC linkage.

## PyNvVideoCodec

### Reuse

Prefer the exact validated registry environment. A ready
`existing-isolated-environment` route performs no venv, resolver, pip, CUDA,
or native-SDK install; it runs fresh official-sample verification only.

If the registry is absent/stale and the user identifies an existing
installation, require the exact lexical interpreter or venv. Probe and verify
that candidate directly. Never search the filesystem, guess among venvs, or
reinstall a valid candidate.

### New environment

Create only a previously absent target with
`lock_pip_reports.py create-venv`. An existing directory, partial target, or
symlink is blocked and is never deleted, repaired, or reused in place.

The Python route does not depend on `nvidia-video-codec-sdk`. It may install
only its own authenticated prerequisites:

- the version-matched public `cuda-minimal-build-13-N` and
  `libcurand-dev-13-N` packages when the top-level CUDA build probe is absent.
  Derive `13-N` from the authenticated `cuda-toolkit` release candidate and
  authenticate both selected candidates independently. PyCUDA 2026.1 enables
  CURAND in its default source build; do not install the full `cuda-toolkit`
  meta-package merely to compile it;
- Python venv support or development headers when their top-level probe fields
  are missing;
- exact `PyNvVideoCodec==2.1.0`;
- exact `pycuda==2026.1`;
- `numpy>=1.24` and the narrow support set recorded in the plan;
- CUDA-enabled `torch==2.9.1+cu130` under `--profile full-samples` only. The
  default `pynvc-smoke` profile installs no Torch and never reaches the cu130
  extra index, because neither official sample it runs imports Torch.

A minimal CUDA build plus CURAND bootstrap is nonterminal: after applying it, discard the plan,
probe again, and re-plan before creating the venv. Scope `PATH`, `CPATH`, and
`LIBRARY_PATH` to the PyCUDA build only; never expose a stub directory through
runtime loader variables.

Run the plan's isolated pip dry-runs and retain their JSON reports.
`lock_pip_reports.py materialize-apply` then:

1. verifies each required root and merges duplicate records only when package,
   version, filename, type, and SHA-256 agree;
2. accepts only credential-free HTTPS artifacts from its fixed host allowlist;
3. pins each URL and SHA-256, with exact policy for the one permitted PyCUDA
   source archive;
4. writes a fresh lock and manifest;
5. reauthenticates the lock, resolver environment, artifact origins, redirect
   destinations, and source hash before mutation;
6. installs hash-checked wheels before the PyCUDA source stage, with no
   dependency resolution;
7. verifies installed versions and requires `pip check` to pass;
8. emits `pynvc_wheel` evidence for the locked PyNvVideoCodec wheel.

The compact apply contract emits persisted wheel/source stage reports,
installed versions, `pip_check`, and `pynvc_wheel` evidence.

After the final Python probe, run the official operation proof with
`verify_pynvc_sample.py --register-current`. It publishes the registry only
after safely writing a ready verification result. A failed proof or
publication attempt preserves the previous registry.

## Fresh setup

Use fresh mode only when the user explicitly requests another complete setup
attempt.

- Native: reinstall the exact live 13.0.x candidate without removing working
  base packages, then re-probe and verify.
- Python: require an explicit unique absent `--venv` at an absolute durable
  path such as `/home/ubuntu/.venvs/nvcodec-fresh` — never inside the working
  directory or a transient run/evidence tree, since the published registry
  outlives it — perform the complete resolver/locked apply, then re-probe,
  verify, and register. Write the setup reports to an equally durable directory.
- Both: keep each plan and proof independent. Use deterministic native-then-
  Python execution when both are actionable. A blocked peer does not suppress
  the other.
- Shared APT refresh: refresh once, then discard and regenerate all selected
  package plans.

Never reuse prior plans, resolver reports, locks, build directories, work
directories, or operation artifacts for a fresh attempt.

## Official operation proof

Both proofs use one generated 640×360 8-bit NV12 frame (345,600 bytes), encode
H.264 with the installed release's official sample, then independently decode
that fresh bitstream with the corresponding official decoder. The Python decoder
is selected by profile: the default `pynvc-smoke` runs
`samples/advanced/decode_perf.py`, and `full-samples` runs
`samples/advanced/decode.py`.

Require:

- authenticated installed sample ownership;
- zero child exit codes and no recognized failure marker;
- the exact positive marker for one encoded frame before decoder launch;
- a fresh, regular, nonempty bitstream with SHA-256;
- `software_fallback=false`.

Native, and Python under `full-samples`, additionally require the exact positive
marker for one decoded frame and a fresh 345,600-byte decoded NV12 output with
SHA-256. `pynvc-smoke` instead requires both anchored `decode_perf.py` markers,
each exactly once, and rejects any worker error, warning, or traceback; it
writes no raw output, so it claims no decoded artifact and proves frame
production only.

Output-file creation alone is not proof, and for `pynvc-smoke` a zero exit code
alone is not proof either, because the sample swallows worker exceptions. Do not
require the decoded pixels to match the input byte-for-byte because H.264 is
lossy. Setup does not use an external software codec or quality-measurement
tool.

## Primary sources

- https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/read-me/index.html
- https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/index.html
