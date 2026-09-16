# Setup output contracts

## Contents

- [Common rules](#common-rules)
- [`nvcodec-environment` 1.2](#nvcodec-environment-12)
- [Install-plan and APT artifacts](#install-plan-and-apt-artifacts)
- [Python environment and lock artifacts](#python-environment-and-lock-artifacts)
- [Native verification](#native-verification)
- [PyNvVideoCodec verification and registry](#pynvvideocodec-verification-and-registry)

## Common rules

Public setup CLIs emit one strict JSON document to stdout. Persisted outputs are
fresh regular files created without replacing an existing path. Preserve each
artifact's path, size, and SHA-256 at consumer boundaries.

Normalized exits are:

- `0`: the requested positive operation completed; a completed read-only
  inventory also exits 0 even when a surface is absent;
- `2`: a completed blocked, negative, unknown, unsupported, or operation-failed
  result;
- `3`: malformed/unsafe input, a refused write, or an internal contract error.

An `argparse` usage error exits `2` and writes no result artifact. A child
process's original return code remains in its stage evidence.

## `nvcodec-environment` 1.2

`probe_nvcodec.py` owns the only setup environment artifact:

```json
{
  "schema_version": "1.2",
  "kind": "nvcodec-environment",
  "generated_at": "...",
  "mode": "live",
  "requested_runtime": "native",
  "platform": {},
  "selected_gpu": 0,
  "nvidia_smi": {},
  "libraries": {},
  "driver_nvenc_api": {},
  "pynvc": {},
  "installation": {},
  "capabilities": {},
  "warnings": [],
  "readiness": {},
  "command_evidence": []
}
```

These sixteen keys are the closed top-level set; an unknown top-level key is a
validation error. Nine are required: `schema_version`, `kind`, `platform`,
`selected_gpu`, `requested_runtime`, `pynvc`, `installation`, `capabilities`,
and `readiness`. `requested_runtime` is a **string**, exactly `native`,
`pynvc`, or `both` — never a list.

The artifact does not serialize `target`, `requested_surfaces`, `surfaces`,
`cuda`, `python`, or `apt` as top-level keys. `surfaces` may exist only as an
internal normalized representation inside the producer; no consumer reads it
from the artifact. Native surface facts are published under
`installation.native_sdk`, the Python surface under `pynvc`, Python readiness
under `installation.python.packages`, and APT and CUDA observations under
`installation.apt` and `installation.cuda_toolkit`.

`native` and `pynvc` stay independent: either may be absent when not selected,
or present and not installed, without changing its peer.

Consumers validate their required subset and ignore additive optional keys.
They reject an unknown schema major and any non-`live` artifact. A ready Python
probe preserves its baseline API-query fields for downstream compatibility;
those fields are not operation proof or a product-support verdict. Complete
capability matrices and operational classification belong to the capability skill.

### Required top-level identity

| Key | Required content |
|---|---|
| `platform` | `jetson` boolean, `machine` matching `^(aarch64\|arm64)$`, and `jetson_linux.{version, release_line, compatibility.status}` evidence against the minimum release |
| `selected_gpu` | Non-negative integer; GPU identity is the matching `nvidia_smi.gpus[]` record with that `index` |
| `requested_runtime` | Exactly one of the strings `native`, `pynvc`, `both` |

### Required `readiness` block

`readiness.state` is `ready`, `partial`, or `not_ready`. `readiness.layers` carries one layer
per requested surface — `native_sdk` and `pynvc` — each with its own `status`,
`installation`, and `operation` values. Report the layers separately; the probe
proves no codec operation, so a satisfied surface is `partial`, never `ready`,
and there is no aggregate-both readiness claim at this level.

### Required `installation.apt` block

`installation.apt` records local APT observations only:

- `apt_cache`: canonical executable path when available;
- `signature_enforced`: boolean;
- `source_files`: configured source paths;
- `sources`: each source path and SHA-256;
- `candidates`: package-name → candidate record.

A candidate record contains `package`, `query_status`, exact `query_argv`,
`query_exit_code`, `installed`, `candidate`, `candidate_origins`, and
`public_origin`. When native is requested, the
`nvidia-video-codec-sdk` candidate record is required even if no candidate is
visible.

An actionable `public_origin` binds the candidate to:

- `https://repo.download.nvidia.com/jetson/common` or `/som`;
- exact `rNN.N/main`;
- `authentication=apt-signature-chain`;
- exact `configured_source_sha256` and `configured_sources` records;
- no trust bypass.

An absent or untrusted candidate remains inventory and cannot create an APT
apply command.

### Required `installation.cuda_toolkit` block

`installation.cuda_toolkit` is target identity used by Python setup; it is not
borrowed from the native surface.

- Always require `status`, either `installed` or `absent`.
- When installed, require `version`, canonical `root`, and non-empty absolute
  `environment_prefixes.PATH`, `.CPATH`, and `.LIBRARY_PATH` lists.
- When absent, `version` and `root` are null and the prefix lists are empty.

### Required `installation.python` block

`installation.python.executable` is the interpreter that produced the artifact,
after any validated-venv delegation; it must be an absolute, executable file
and it drives `@python-from`. For a new Python environment, also require:

- `system_executable`: the observed canonical system Python when available;
- `venv_module`: `ok` or `missing`;
- `development_headers`: `ok` or `missing`.

These fields select authenticated bootstrap packages at plan time. They do not
authorize scanning for another interpreter.

Python readiness is `installation.python.packages.<name>.{status, version,
requirement_satisfied}`; when Torch is present it additionally carries
`cuda_build`, `cuda_available`, and `sample_readiness`. Torch is required only
by the `full-samples` profile; under the default `pynvc-smoke` profile its
absence is expected and is not a readiness failure.

### Required `installation.native_sdk` block

Always require `installed` as a boolean. When true, require:

- `package.{name,status,version}`, where name is
  `nvidia-video-codec-sdk` and status is `installed`;
- canonical `sdk_root`;
- `build_prerequisites.{status,unresolved_modules}` for the AppDec modules;
- `cuda.{status,version,root}`;
- `tools.cmake`, `tools.cxx`, `tools.nvcc`, `tools.pkg_config`, and
  `tools.generator`, each with `path`, `version`, and `sha256`;
- `tools.generator.name`, either `Ninja` or `Unix Makefiles`.

### Required top-level `pynvc` block

The Python surface is published at top level as `pynvc`. Its established fields
include `imported`, `module`, `module_file`, `module_version`,
`distribution_version`, `linked_nvenc_api`, `identity`, and `errors`.
Consumers validate their required subset and permit additive provenance fields.

Always require `imported` as a boolean. When it is true in a `live` artifact,
require a verified `identity`:

- `identity.status` is `verified`, and otherwise carries only `status` and
  `reason`;
- `identity.interpreter` is the exact lexical interpreter and must equal
  `installation.python.executable`;
- `identity.sys_prefix`, `identity.dist_info_path`;
- `identity.distribution.version` and `identity.module.version`, both exactly
  `2.1.0`;
- `identity.extension` for the loaded extension, whose recorded loaded path
  equals the hashed extension path.

`module_version` must match `distribution_version`. `linked_nvenc_api` records
the linked NVENC API level. The verifier, not the probe, authenticates wheel
`RECORD` ownership of the files it executes.

Python dependency readiness is not part of this block; it is
`installation.python.packages` as described above.

### Reauthentication result

`probe_nvcodec.py --reauthenticate ENVIRONMENT` emits:

```json
{
  "kind": "nvcodec-environment-validation",
  "schema_version": "1.0",
  "valid": true,
  "errors": []
}
```

It validates required structure and rehashes the recorded native tools and
Python interpreter/extension identities. It performs no mutation.

## Install-plan and APT artifacts

`plan_install.py` emits `kind=nvcodec-install-plan`,
`schema_version=1.5`. Its required contract includes:

- `generated_from`: bound environment path, size, SHA-256, kind/mode/schema,
  and canonical JSON digest;
- `plan_inputs`: environment, exact selected components, request intent,
  optional venv, fresh flag, and optional refresh receipt;
- independent `components.native-sdk` and/or `components.pynvc`;
- per-component `plan_status`, selected route, blockers, artifacts, exact
  commands, and final validation commands;
- `preflight_commands`, sequential `execution_batches`, and intent-scoped
  authorization;
- `overall_status`, `mutated=false`, and `plan_digest`.

Component status is `installed`, `ready_to_review`, or `blocked`. A blocked
component has no command. A mixed two-surface plan stays actionable for its
nonblocked peer. `both` readiness is not a plan status.

`plan_install.py validate PLAN` emits
`nvcodec-install-plan-validation` schema 1.0 and checks the plan document,
including its digest and authorization consistency, without mutation.

Transaction actions accept only literal commands present in the reviewed
sibling plan. They regenerate the canonical plan from its bound inputs and
require the same digest before execution. Refresh/environment binding is
performed by the planner through `--apt-refresh-receipt`; the resulting
transaction records `authorizing_plan_digest`. Direct planless transactions
are not supported and carry no separate refresh/environment assertion flags.
APT artifacts use schema 1.0:

- `nvcodec-apt-refresh-receipt`;
- `nvcodec-apt-simulation-receipt`, including live policy evidence,
  transaction summary, and `safe_to_apply`;
- `nvcodec-apt-apply-report`, including the authorizing plan digest, reviewed
  receipt identity, immediate repeated simulation, exact command evidence,
  and final `safe` result.

## Python environment and lock artifacts

`lock_pip_reports.py create-venv` emits
`nvcodec-clean-venv-creation` schema 1.0. `status=created` proves a previously
absent target was reserved and populated without system site packages.
`target_exists` is a complete negative result; the path is not deleted or
modified. A partial failure is preserved and never described as rollback.

`materialize-apply` produces:

- `nvcodec-pip-lock-manifest` schema 1.0 with interpreter/environment
  contract, resolver-report identities, requirements, allowed hosts, exact
  artifact name/version/URL/filename/type/SHA-256 records, lock identity, and
  install policy;
- fresh wheel/source pip stage reports with bounded command evidence;
- `nvcodec-pip-lock-apply` schema 1.0 with source preflight, installed
  versions, `pip_check`, and `pynvc_wheel` evidence for the locked
  PyNvVideoCodec wheel.

Successful apply status is `applied_verified`. A post-mutation failure is
reported as mutation-possible or verification-failed, never clean rollback.

## Native verification

`verify_native.py` emits `kind=nvcodec-native-verification`,
`schema_version=1.5`.

The artifact binds:

- the environment identity and native surface;
- live package version, ownership, and `dpkg --verify` evidence;
- official sample build commands and binary identities;
- real CUDA/NVENC/NVDEC linkage with stub rejection;
- the generated one-frame 640×360 NV12 input;
- exact encode/decode argv, markers, exit codes, log tails, bitstream identity,
  decoded-output identity, and failure reasons.

`ready=true` and `status=operation_verified` require both official operations,
exactly one reported frame at each stage, a fresh nonempty bitstream, and an
exact 345,600-byte decoded NV12 output. A missing prerequisite is `unknown`; a
launched sample failure is `operation_failed`.

## PyNvVideoCodec verification and registry

`verify_pynvc_sample.py` emits
`kind=nvcodec-pynvc-sample-verification`, `schema_version=1.3`.

The artifact binds:

- environment path/hash/canonical digest and selected GPU;
- exact lexical interpreter, prefix, distribution/module version, and loaded
  extension identity;
- wheel `RECORD` ownership for the imported module, loaded extension, official
  encoder/decoder, helpers, and configuration;
- the `verification_profile` and exactly its dependency subset: NumPy and
  PyCUDA readiness always, plus CUDA-enabled Torch readiness under
  `full-samples`;
- the generated one-frame 640×360 NV12 input;
- exact child argv, output logs, positive markers, bitstream identity, decoded
  output identity, and reasons.

Only the known PyNvVideoCodec 2.1.0 stale `RECORD` row for its native extension
may use the narrow recorded mismatch exception; all other executed files must
match their wheel rows.

`ready=true` and `status=operation_verified` require both official operations,
one encoded frame, and a fresh nonempty H.264 bitstream. The decode proof is the
profile's own: `full-samples` runs `advanced/decode.py` and still requires the
exact 345,600-byte decoded NV12 output, while the default `pynvc-smoke` runs
`advanced/decode_perf.py`, which writes no raw output and therefore carries no
decoded-output identity. Its proof is frame production: both anchored markers,
each exactly once, with no worker error, warning, or traceback. A zero exit code
alone is never sufficient, because the sample swallows worker exceptions.

The fixed registry is:

```text
$HOME/.local/state/jetson-videosdk/current-pynvc.json
```

Its schema is `jetson-videosdk/current-pynvc/1`. It records the exact ready
verification-artifact identity and the frozen verified Python surface.
`verify_pynvc_sample.py --register-current` is the only publisher. It writes
the `nvcodec-pynvc-sample-verification` schema 1.3 result before publication,
refuses a non-ready verification, and preserves any prior registry on failure.

A normal PyNv probe reauthenticates the registry, bound verification artifact,
surface, and exact interpreter, then delegates to that lexical interpreter.
It never scans venv directories or falls back to another interpreter. Keep the
registry-bound verification and environment artifacts in persistent storage;
missing or changed evidence makes the registry not ready.
