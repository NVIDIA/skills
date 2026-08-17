# Capability-query rules

## Contents

- [Evidence boundaries](#evidence-boundaries)
- [Optional setup evidence](#optional-setup-evidence)
- [Native official-sample reports](#native-official-sample-reports)
- [Encoder API query](#encoder-api-query)
- [Decoder API query](#decoder-api-query)
- [Setup readiness verification](#setup-readiness-verification)
- [Documentation cross-check](#documentation-cross-check)
- [Exact operations](#exact-operations)
- [Versioned R39.2/SDK 13.0 Thor baseline](#versioned-r392sdk-130-thor-baseline)
- [Not capabilities](#not-capabilities)

## Evidence boundaries

Keep the evidence authorities separate:

| Question | Owner and evidence | Meaning |
|---|---|---|
| What is installed now? | `jetson-video-setup` may publish a fresh `nvcodec-environment` `1.2` artifact. | Optional environment inventory plus bounded baseline Py API-query evidence; no support or operation verdict. |
| What did the native samples summarize? | This skill invokes authenticated `AppEncCuda -ec` and `AppDec -dc` and publishes a native sample capability report. | Raw `official_sample_report` inventory only. |
| What did the Py codec API report? | This skill publishes the encoder report and decoder matrix from `GetEncoderCaps` and `GetDecoderCaps`. | Raw API-query evidence only. |
| Did one exact route run? | Setup proves only its fixed readiness smoke; an exact requested codec operation belongs to `jetson-video-recipe` plus `jetson-video-pipeline`. | Live operation evidence for only the tested route. |
| Does NVIDIA document product support? | The applicable NVIDIA support-matrix row and version-matched SDK application note. | Customer-facing product-support authority. |

An API-positive result is `capability_reported`, not a product-support verdict or operation proof.
A native `yes`, `no`, or numeric `Supported` value remains sample-reported data and is not promoted
to an API, operation, or documentation verdict. A matching official operation may be
`operation_verified` or `operation_failed`, but it does not rewrite either raw report. Publish
product support from the documentation cross-check and live availability only from a successful
exact operation. Preserve disagreements instead of silently reconciling them.

Setup does not publish a capability-skill artifact or a product-support verdict. Its environment
may contain bounded baseline Py API-query fields, and native verification may contain raw
`-ec`/`-dc` aggregate snapshots alongside the fixed H.264 readiness operation. Those supporting
observations are neither the complete decoder tuple matrix nor standalone capability reports.
This skill publishes fresh, independently owned native and Py reports. Setup evidence is optional
for a selected-surface query. “Supplied” includes a fresh artifact obtained by the agent from the
installed setup skill's public read-only probe; the customer does not have to repeat an interpreter
that setup has already authenticated. The producer authenticates supplied evidence and fails closed
on unavailable, stale, malformed, mismatched, or changing evidence; it never discards a supplied
artifact and falls back to local discovery. When setup evidence is absent, the producer authenticates
only its selected local authority. A native-only request requires no Py evidence, a Py-only request
requires no native evidence, and `both` runs the two authorities independently.

## Optional setup evidence

A supplied native input must be setup's exact `kind=nvcodec-native-verification`,
`schema_version=1.5` artifact. The producer requires `ready=true`,
`status=operation_verified`, both fixed H.264 operations verified, verified package ownership,
the bound schema-1.2 environment, unchanged sample identities, and real driver linkage. It
reauthenticates all of those facts before executing a requested report and rehashes the artifacts
and binaries afterward. A valid not-ready or missing verification produces `unknown` and no sample
launch; malformed evidence, a false readiness claim, or identity drift is a contract failure. No
supplied-verification failure authorizes the local build path.

A supplied Py input must be the exact live schema-1.2 `nvcodec-environment` artifact for the
requested GPU. Invoke the query through its declared lexical `pynvc.identity.interpreter`. The
producer validates only its required subset and permits additive optional keys:

- `kind`, `schema_version`, `mode`, and `selected_gpu`;
- `pynvc.imported` and `pynvc.identity.status`;
- `pynvc.identity.interpreter`, `.sys_prefix`, and distribution version;
- optional `pynvc.identity.interpreter_identity.{path,sha256}` when published;
- `pynvc.identity.extension.{path,sha256,loaded_path}`;
- `pynvc.identity.module.{version,path}`.

It compares those declarations with the running interpreter and imported objects, binds the report
to the environment's path, size, raw and canonical SHA-256 values, and authenticates the same
artifact and wheel again after querying. An absent or explicitly unavailable declared Py surface
is `unknown`; malformed, mismatched, or changing supplied evidence fails closed. Neither case may
fall back to another interpreter or omit `--environment` on a retry.

Without a Py environment artifact, invoke the producer under the exact interpreter whose installed
PyNvVideoCodec wheel is being queried and omit `--environment`. Before import, the producer requires
isolated Python and rejects active import-injection variables. Before and after the API calls it
authenticates the resolved invoking interpreter plus exact public PyNvVideoCodec `2.1.0`
distribution/module, the imported initializer, the extension actually loaded, and wheel-`RECORD`
ownership and hashes. The resulting report carries `runtime_binding` instead of `environment`.
The capability producer never reads a setup registry, scans venv directories, selects an
interpreter by name or modification time, searches prior attempts, or falls back to system Python.
The orchestrating agent may instead invoke the installed setup skill's public `probe_nvcodec.py`
without `--setup-candidate`; that probe alone resolves and reauthenticates the fixed registry. Use
`--runtime pynvc` for explicit Py or the unnamed fallback after native is ineligible, and
`--runtime both` for `both` or a genuine `auto` candidate gate. Apply the capability surface
classification first: a bare ambiguous “video SDK” request stops for clarification, while a
support/catalog question that names no SDK surface uses native when eligible and this Py registry
route only when native is ineligible. Serialize a successful choice as explicit `native` or
`pynvc` before applying the four-value routing contract. The probe may return exit zero with a
typed not-ready Py result, so inspect the fresh artifact first and preserve that reason. Require
`mode=live`, the requested GPU,
`pynvc.installed=true`, and `pynvc.identity.status=verified`. If routing selects Py, only then launch
the producer under the lexical `pynvc.identity.interpreter` and pass the raw artifact path through
`--environment`. A blocked artifact is not usable authority.

When the exact selected-surface authority or a build/import prerequisite is unavailable, publish
`unknown`/`not_ready` without mutation and route that surface to `jetson-video-setup`. If that skill
is not installed, tell the user to install it. Do not infer package absence, codec support, or
product support from an import or build-prerequisite failure. For `both`, authenticate and report
the surfaces independently; a peer-only failure must not suppress the other branch.

## Native official-sample reports

Use system Python under isolation; this route does not import PyNvVideoCodec or setup code. When a
fresh setup verification is available, supply it:

```bash
python3 -I "$CAPABILITY_SKILL/scripts/query_native_sample_reports.py" \
  --native-verification "$NATIVE_VERIFICATION_JSON" \
  --gpu 0 --timeout 300 \
  --output "$NATIVE_SAMPLE_REPORT_JSON"
```

When no setup verification is supplied, use a new, previously absent workspace:

```bash
python3 -I "$CAPABILITY_SKILL/scripts/query_native_sample_reports.py" \
  --workspace "$FRESH_NATIVE_REPORT_WORKSPACE" \
  --gpu 0 --timeout 300 \
  --output "$NATIVE_SAMPLE_REPORT_JSON"
```

`--workspace` is mandatory in this mode and is never inferred or reused. The producer
authenticates the fixed installed `nvidia-video-codec-sdk` package through `dpkg`, configures its
package-owned SDK 13.0 sample tree in that workspace, and builds only the targets selected by
`--report`. It performs no package installation, repository or key change, repair, fixture
creation, or media operation; all configure and build products use its fresh workspace.

`--report` is repeatable with closed values `encoder` and `decoder`. Omission selects both in
encoder-then-decoder order; repetition is deduplicated. Each selected command runs at most once:

```text
<authenticated AppEncCuda> -ec
<authenticated AppDec> -dc
```

This route is media-free and query-only, but it opens the native driver. The supplied-verification
path does not build. The no-verification path creates only its explicit fresh build workspace.
When `both` surfaces are requested, run this producer independently from the Py API producers; one
branch's failure must not suppress the other.

### Input authentication

Treat `--native-verification` as a versioned local attestation, not as trusted paths:

1. Strictly read a regular, non-symlink JSON file with no duplicate keys, non-finite numbers,
   unknown schema major, or read-time mutation.
2. Require the frozen schema-`1.5` native verification with `ready=true`,
   `status=operation_verified`, `software_fallback=false`, matching GPU, both fixed H.264
   operations verified, and verified package ownership.
3. Rehash and reauthenticate its exact environment artifact, including the canonical JSON digest.
4. Require installed `nvidia-video-codec-sdk` `13.0.x`; re-run exact `dpkg-query` and silent
   `dpkg --verify`.
5. Rehash each requested absolute package-built sample and require the recorded binary path to
   match its identity.
6. Re-run `ldd` and require real non-stub driver linkage: encoder to `libcuda.so.1` and
   `libnvidia-encode.so.1`, decoder to `libcuda.so.1` and `libnvcuvid.so.1`.
7. Rehash the input artifact, environment, and each executed binary again before publishing.

A valid but not-ready verification yields `unknown`, launches no sample, and exits `2`. Claimed
readiness with malformed structure, identity/package/linkage drift, or tampering launches no sample
and exits `3`. Supplying this option commits the request to that evidence path; the producer never
falls back to a local build if authentication fails.

When `--native-verification` is omitted, authenticate the local native authority instead:

1. Require a fresh explicit `--workspace` whose canonical, non-symlink parent already exists.
2. Resolve exactly `/usr/bin/dpkg-query` and `/usr/bin/dpkg`; require the fixed
   `nvidia-video-codec-sdk` package installed at public SDK `13.0.x`, enumerate its owned files, and
   require silent `dpkg --verify`.
3. Resolve and record the fixed compiler/configuration tools and the first executable `nvcc` in its
   bounded preference order, then require that selected compiler to be CUDA 13.x. Resolve the first
   usable generator in its bounded preference order. Require FFmpeg development modules only when
   decoder reporting is requested.
4. Configure the package-owned `Samples` tree inside the fresh workspace and issue one bounded
   `cmake --build ... --target <target>` for each requested report: `AppEncCuda` for `encoder` and
   `AppDec` for `decoder`. Do not build an unrequested target.
5. Rehash the tools and built binaries, require real non-stub driver linkage, run each requested
   report once, then reauthenticate the fixed package and binary identities before publishing.

An unavailable package, tool, development module, configure/build prerequisite, or requested
authority launches no report and exits `2`; apply the shared missing-prerequisite remediation in
**Optional setup evidence** above. Malformed package ownership, unexpected SDK version, integrity
drift, unsafe paths, or tampering fails closed with exit `3`.

### Artifact and classification

With a supplied verification, the producer publishes this deterministic outer contract:

```json
{
  "kind": "nvcodec-native-sample-capability-report",
  "schema_version": "1.0",
  "status": "complete",
  "authority": "native_official_samples",
  "gpu": 0,
  "requested_reports": ["encoder", "decoder"],
  "native_verification": {
    "initial_identity": {},
    "terminal_identity": {},
    "canonical_sha256": "...",
    "kind": "nvcodec-native-verification",
    "schema_version": "1.5",
    "ready": true,
    "status": "operation_verified",
    "environment_identity": {},
    "package": {},
    "samples": {}
  },
  "aggregate_encoder_capabilities": {},
  "aggregate_decoder_capabilities": {},
  "summary": {"completed": 2, "unknown": 0, "not_requested": 0}
}
```

Without a verification, the same outer report records
`native_verification.status=not_supplied`, `native_verification.ready=false`, and a
`local_authentication` block containing the authenticated package, fresh build, requested targets,
tools, built sample identities, and linkage. That `not_supplied` value is not a readiness failure:
the report's authority comes from the separately recorded local package authentication. It does not
promote the sample output to setup readiness or operation proof.

Top-level `status` is `complete` when all requested reports parse, `partial` when at least one
completes and one is unknown, and `unknown` when none completes. An unrequested family remains
present with `status=not_requested`. A requested family is only `completed` or `unknown`.
`complete` means the official sample's report grammar was captured; it does not mean setup ready,
API-reported support, exact operation success, throughput, or documented product support.

Both aggregate records use:

```json
{
  "evidence_source_type": "official_sample_report",
  "official_sample": true,
  "scope": "sample_reported"
}
```

The encoder classification names `AppEncCuda` and `-ec`; the decoder classification names `AppDec`
and `-dc`. Do not emit `supported=true`, `capability_reported`, `operation_verified`, or a
documentation verdict from either aggregate.

### Accepted report grammars

Every requested report requires exit `0`, no timeout, complete bounded stdout/stderr evidence, no
recognized CUDA/NVENC/NVDEC/error/failure marker, the selected GPU where the format can identify
one, and unchanged binary identity. Preserve the complete streams, sizes, SHA-256 values, display
tails, parsed raw records, and the recognized `format_variant`.

For `AppEncCuda -ec`, accept either:

- the legacy summary grammar: exactly one `Encoder Capability Summary`, one detail hint, unique
  `GPU <ordinal> - <name>` blocks containing the selected GPU, exactly one codec-support section and
  one capability-summary-table section per GPU, and exactly one `H264`, `HEVC`, and `AV1` row; or
- the released R39.2 compact grammar: exactly one `Encoder Capability`, unique
  `GPU <ordinal> - <name>` blocks containing the selected GPU, and exactly one basic `H264`, `HEVC`,
  and `AV1` `yes`/`no` row per GPU. This real format has no synthetic sections or detail hint.

Preserve the basic encoder row values as lowercase `yes` or `no`; they remain
`sample_reported_codec_rows`, never support.

For `AppDec -dc`, accept either:

- the legacy summary grammar: unique `GPU <ordinal> - <name>` blocks containing the selected GPU,
  exactly one `GPU Decoder Capabilities` and `Codec Support Summary` per GPU, and one detail hint;
  or
- the released R39.2 compact grammar: exactly one `Decoder Capability`, exactly one
  `GPU in use: <name>`, and at least one strict
  `Codec ... BitDepth ... ChromaFormat ... Supported ...` row. Because that format carries no GPU
  ordinal, it is eligible only for requested GPU `0`.

Preserve every compact decoder row and its raw numeric `Supported` value. Do not synthesize the
complete `cuvidGetDecoderCaps` tuple matrix or turn any value into product support. The Py decoder
producer remains the distinct API authority for that matrix.

Encoder and decoder reports are independent inventory commands: failure of one does not suppress
the other.

### Exit codes

| Condition | rc |
|---|---:|
| All requested reports completed | `0` |
| Selected native authority or prerequisite unavailable, or any requested report unknown | `2` |
| Malformed arguments/artifact, identity or linkage drift, unsafe output, write failure, or internal contract failure | `3` |

Every handled path prints exactly one strict JSON document. Child exit codes remain inside their
aggregate records and are never returned directly.

## Encoder API query

PyNvVideoCodec exposes `GetEncoderCaps(gpuid, codec)`. The binding selects device 0 internally, so
nonzero-GPU encoder capability remains `unknown`.

Invoke this skill's producer directly under the exact PyNvVideoCodec interpreter to query. Setup
evidence is optional, so the baseline local-authentication command omits `--environment`:

```bash
"$PYNVC_PYTHON" -I \
  "$CAPABILITY_SKILL/scripts/query_encoder_caps.py" \
  --gpu 0 \
  --output "$ENCODER_REPORT_JSON"
```

The producer authenticates that exact lexical and resolved interpreter, `sys.prefix`, public
PyNvVideoCodec `2.1.0` distribution and imported module, initializer, and loaded extension before
and after the query. It requires wheel-`RECORD` ownership and records a `runtime_binding`; it never
scans for a venv or selects another interpreter.

When the caller supplies a setup environment, use its declared lexical interpreter and pass the
artifact explicitly:

```bash
"$VALIDATED_PYNVC_PYTHON" -I \
  "$CAPABILITY_SKILL/scripts/query_encoder_caps.py" \
  --environment "$ENVIRONMENT_JSON" --gpu 0 \
  --output "$ENCODER_REPORT_JSON"
```

The producer must authenticate and bind that exact artifact before and after the query. A supplied
environment is never ignored and an authentication failure never falls back to the local-only
route.

With no repeated `--codec`, the producer queries `h264`, `hevc`, and `av1`. It publishes:

- `kind=nvcodec-encoder-capability-report`;
- `schema_version=1.0`;
- `authority=pynvc`;
- `evidence_classification.encode` with
  `evidence_source_type=api_query_helper`, `official_sample=false`, and
  `api_symbol=PyNvVideoCodec.GetEncoderCaps`;
- the observed linked NVENC API and one `encode.<codec>` record per requested codec.

`GetEncoderCaps` returns capability fields but no explicit boolean codec-support result. A
successful record is therefore `status=capability_reported`, `supported=null`,
`session_status=opened_by_GetEncoderCaps`, and `operation_status=not_tested`; never promote it to
`supported=true`. The loaded extension may expose NVENC API 12.1 or 13.0. Fields absent from a 12.1
header remain unqueryable/`unknown`, not unsupported.

Use the returned minimum and maximum dimensions before planning an exact operation. NV12 dimensions
must be even. Producer exit codes are `0` for complete, `2` for partial/unknown, and `3` for
authentication, malformed-input, or safe-write failure. A declared absent, unimported, or
unverified `pynvc` authority, a local import failure, or unavailable local wheel authority is
`unknown` with exit `2`, not a codec or product-support verdict. Route missing prerequisites to
`jetson-video-setup` without mutation and tell the user to install that skill if it is absent.

## Decoder API query

PyNvVideoCodec exposes `GetDecoderCaps(gpuid, codec, chromaformat, bitdepth)`. It also routes the
query to device 0, so nonzero-GPU results remain `unknown`.

Invoke this skill's producer directly under the exact PyNvVideoCodec interpreter. Omit
`--environment` when no setup artifact is supplied:

```bash
"$PYNVC_PYTHON" -I \
  "$CAPABILITY_SKILL/scripts/query_decoder_caps.py" \
  --gpu 0 \
  --output "$DECODER_MATRIX_JSON"
```

This path performs the same before-and-after local interpreter, wheel-`RECORD`, initializer, and
loaded-extension authentication as the encoder producer and publishes `runtime_binding`. Never
scan for, guess, or silently change the invoking interpreter. If a setup environment is supplied,
invoke its declared lexical interpreter and include `--environment "$ENVIRONMENT_JSON"`; the
producer must validate and bind that artifact and may not fall back if it fails.

The default matrix contains all ten decoder families:

`mpeg1`, `mpeg2`, `mpeg4`, `vc1`, `h264`, `hevc`, `vp8`, `vp9`, `av1`, and `jpeg`.

Each family is queried across `monochrome`, `420`, `422`, and `444` chroma at 8-, 10-, and 12-bit
depth, for 120 tuples. Keep JPEG exactly as the live API reports it; never pre-assume its state.
Repeat `--codec` to request a subset in one fresh artifact:

```bash
"$PYNVC_PYTHON" -I \
  "$CAPABILITY_SKILL/scripts/query_decoder_caps.py" \
  --gpu 0 \
  --codec h264 --codec hevc --codec av1 \
  --output "$DECODER_SUBSET_JSON"
```

Use the complete matrix for an unconstrained explicit Py decoder catalog. Also use it when a broad
unnamed decoder-catalog request reaches Py only because native was ineligible, but summarize that
fallback at family level and do not dump unsolicited tuple claims. For a bounded request such as
H.264, HEVC, and AV1, repeat `--codec` only for those named families; that example produces 36
tuples rather than expanding to all 120.

The producer publishes `kind=nvcodec-decoder-capability-matrix`, its current
`schema_version=1.2`, and the exact classification
`evidence_source_type=api_query_helper`, `official_sample=false`, and
`api_symbol=PyNvVideoCodec.GetDecoderCaps`.

Gate every returned field on `bIsSupported`:

- `bIsSupported=1` becomes raw `status=capability_reported`, `supported=true`, and applicable
  limits. Here `supported=true` records only the API flag; it is not the documented product verdict.
- `bIsSupported=0` becomes raw `status=unsupported`, `supported=false`, with the remaining zeroed
  output fields inapplicable.
- A missing enum, failed call, absent flag, or missing result is `unknown`.

The encoder report and decoder matrix are separate artifacts with separate classifications. Neither
may be substituted by setup inventory or sample output. Missing local import or wheel prerequisites
produce `unknown` plus non-mutating `jetson-video-setup` remediation; if setup is not installed,
tell the user to install it. Malformed or mismatched supplied evidence fails closed and never
authorizes a retry without that evidence.

## Setup readiness verification

Setup verification is an optional fixed installation-readiness proof, not a prerequisite for
capability capture. The PyNv verifier runs the wheel-owned official CPU-buffer encoder and a
profile-selected decoder for exactly one 640x360 NV12 frame of H.264:

```bash
"$VALIDATED_PYNVC_PYTHON" -I \
  "$SETUP_SKILL/scripts/setup/verify_pynvc_sample.py" \
  --environment "$ENVIRONMENT_JSON" --gpu 0 \
  --work-dir "$FRESH_SETUP_WORK_DIR" \
  --output "$FRESH_SETUP_REPORT_JSON"
```

It always requires exactly one encoded-frame marker and a fresh nonempty H.264 bitstream. The
default `pynvc-smoke` profile runs `advanced/decode_perf.py` and requires both anchored one-frame
production markers exactly once, with no worker warning, error, or traceback; that sample writes no
raw output. The explicit `full-samples` profile runs `advanced/decode.py` and additionally requires
one decoded-frame marker plus an exact 345,600-byte decoded NV12 output. The native verifier uses
the same fixed one-frame scope and exact raw-output proof with its package-owned samples. None of
these setup proofs enumerates codecs, emits capability records, proves another tuple, or publishes
product support.

Every verification attempt needs a fresh work directory and output path. A failed attempt consumes
both paths; do not overwrite or reinterpret it as package absence.

## Documentation cross-check

After preserving the raw query, cross-check the target before deciding whether an exact operation
is useful. Use NVIDIA's public
[Video Encode and Decode Support Matrix](https://developer.nvidia.com/video-encode-decode-support-matrix)
and the version-matched Video Codec SDK 13.0
[NVENC](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html)
or
[NVDEC](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html)
application note.

Obtain that live model evidence read-only, never by guessing a SKU. Use a supplied authenticated
`nvcodec-environment` artifact only when it actually carries literal exact model evidence; its
platform record normally reports system, release, machine, Jetson Linux version, and GPU name,
none of which identify an exact board model or SKU. Otherwise read the fixed device-tree identity
sources directly:

```bash
tr -d '\000' </proc/device-tree/model
tr '\000' '\n' </proc/device-tree/compatible
```

Never infer `Jetson T5000` or `Jetson T4000` from engine count, memory size, performance, or any
queried capability field. When these sources yield only a generic family identity, keep the exact
row `unknown` and enumerate the complete authenticated candidate row set as described below.

Record the retrieval date, literal live model evidence, source URL, literal row and field labels,
and either the exact matched row or every authenticated candidate row. Product identity must be
independent of the capability fields and operation result being checked. Do not narrow candidates
using queried dimensions, formats, engine count, operation behavior, performance, or similarity of
specifications.

Treat a documentation row as exact only when immutable live identity and NVIDIA documentation map
the product, model, or SKU one-to-one. A generic family name is not exact. For generic
`NVIDIA Jetson Thor Developer Kit` / `NVIDIA Thor` identity, include every Thor row in the combined
Jetson/IGX table unless an NVIDIA one-to-one product mapping narrows it. When all authenticated
candidates agree for the exact field, publish candidate-row consensus while keeping the exact row
unknown. If they disagree or the field is not documented, the documentation result is `unknown`.

For a dynamic table, capture the literal table title, every header level needed to form the exact
field label, the literal row label, and the associated cell value. Bind by that explicit
row-label → exact column-label → cell-value association, never by a flattened value's ordinal
position. Enumerate all literal current rows matching the authenticated product family; this dated
baseline is not a current-row inventory or exclusion list. If extraction does not preserve the
association, record the current-page field as `unknown` and do not quote a value or claim agreement
or conflict.

If a current official table differs from the dated baseline, preserve both observations and accept
the current value only when the extraction retains that explicit row/column/cell association.
Otherwise record the current-page field as `unknown`. For this release, use only the
version-matched SDK 13.0 application note; do not consult, cite, or mention a note from another SDK
release as corroboration. A field that SDK 13.0 does not establish remains `unknown`.

Use this reconciliation for every tuple:

- Documentation `Yes` establishes documented product support. Report live availability only if the
  matching official operation succeeds.
- An applicable documentation `No`, including unanimous authenticated candidate-row consensus, is
  the final customer-facing `unsupported` verdict. Preserve any positive API or operation evidence
  separately as a discrepancy.
- API negative/unknown plus documentation positive remains documented-supported but unavailable or
  unknown on the tested stack until a matching operation succeeds.
- If applicable documentation is absent, ambiguous, or internally conflicting, keep product
  support `unknown`; do not guess a row.

## Exact operations

Reconcile the raw query with applicable product documentation before launching an operation. A
directly applicable documentation `No` is the final unsupported product verdict and ends the normal
support or availability check without a codec operation. Preserve raw discrepancies, but do not run
an operation merely to challenge a verdict it cannot change. If documentation is supported or
unknown and the user asks whether the exact codec/chroma/bit-depth/format/dimension route is
live-available, run the smallest matching installed NVIDIA sample through the operation owners. A
separate explicit diagnostic experiment may also run despite a documentation-negative verdict, but
its result remains diagnostic and never promotes product support. Resolve a validated recipe with
`jetson-video-recipe`, then invoke the pipeline controller:

```bash
python3 -I "$PIPELINE_SKILL/scripts/encode_controller.py" \
  --request "$OPERATION_REQUEST_JSON" \
  --workspace "$FRESH_OPERATION_WORKSPACE" \
  --output "$FRESH_OPERATION_RESULT_JSON"
```

When setup is installed, apply its shared
[video content policy](../../jetson-video-setup/references/video-content.md). Its absence does not
block capability's local operation path: require an exact user-selected path or URL, never choose
catalog or synthetic media, and preserve provenance and identity. A bounded capability smoke may
use setup's documented deterministic one-frame fixture, but that does not make it representative
content or authorize a performance, quality, or general pipeline claim.

For encode, success requires an independent authenticated decode of the exact encoded path and
SHA-256, with the expected decoded frame count and output size. Exit zero, an encode marker, a
nonempty file, or container structure alone is insufficient. Keep host-output and device-memory
routes distinct. A failed operation describes only the exact tested route and is
`operation_failed`, never a global `unsupported` result. If the recipe or pipeline owner is
unavailable, record `not_tested` and name the missing dependency.

For an AV1 device-memory output, the current capability-owned structure checker may validate the
fresh IVF artifact:

```bash
python3 -I "$CAPABILITY_SKILL/scripts/validate_appenc_av1_ivf.py" \
  --input "$BITSTREAM" --width "$WIDTH" --height "$HEIGHT" \
  --expected-frames "$FRAMES" --output "$IVF_REPORT_JSON"
```

Its `structure_verified` result covers IVF structure only and never establishes
`operation_verified`.

## Versioned R39.2/SDK 13.0 Thor baseline

This compact baseline prevents a dynamic web-table column shift from becoming a product claim. It
is not the complete matrix. It was verified against the official pages on 2026-08-03 and applies to
Jetson Linux R39.2 with Video Codec SDK 13.0. Refresh the applicable official documentation for a
different release, product family, or unlisted field.

The support-matrix encode table uses these named fields for the six generic-Thor candidates:

| Candidate row | H.264 (AVC) YUV 4:2:0 | H.264 (AVC) YUV 4:2:2 | H.264 (AVC) YUV 4:4:4 | H.264 (AVC) Lossless | H.265 (HEVC) 4K YUV 4:2:0 | H.265 (HEVC) YUV 4:2:2 | H.265 (HEVC) 4K YUV 4:4:4 | H.265 (HEVC) 4K Lossless | H.265 (HEVC) 8K | HEVC 10-bit support | HEVC B Frame support | AV1 YUV 4:2:0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jetson T2000 | Yes | No | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | No |
| Jetson T3000 | Yes | No | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | No |
| Jetson T5000 | Yes | No | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | No |
| Jetson T4000 | Yes | No | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | No |
| IGX T7000 | Yes | No | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | No |
| IGX T5000 | Yes | No | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | No |

The SDK 13.0 NVENC application note independently records Jetson Thor H.264
baseline/main/high YUV 4:2:0 as `Y`, HEVC Main YUV 4:2:0 as `Y`, HEVC Main10 as `Y`, HEVC 4:2:2
as `N`, and AV1 Main YUV 4:2:0 as `N`. Keep this observation separate from the support-matrix row.

The support-matrix decoder table lists positive values across its displayed Thor decoder columns,
and the SDK 13.0 NVDEC application note lists supported profiles and features. Neither source
documents every one of the API's 120 codec/chroma/bit-depth tuples. Do not expand a family-level
`Yes` into a tuple-level documentation claim when the exact chroma, bit depth, or profile field is
absent.

## Not capabilities

- Presets and tuning values are configuration choices.
- GPU-family tables are expected-value references, not live gates.
- Runtime library or package presence does not prove a codec tuple or session is available.
- One successful tuple does not generalize to another tuple.
- Concurrent-session policy is not an `NV_ENC_CAPS` field.
- Throughput or successful concurrent streams do not establish NVENC/NVDEC engine count. Keep the
  count `unknown` unless an authenticated API or NVIDIA sample returns that exact field.
