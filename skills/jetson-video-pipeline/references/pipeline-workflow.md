# Pipeline workflow

Use direct, authenticated NVIDIA samples for codec work. The agent owns
planning, execution, artifact handoffs, and the final report.

## Before planning

1. Apply the scope and request-classification gates in `SKILL.md`. Architecture
   questions end with a direct route answer and stated assumptions. Dry runs
   use only user-supplied identity and metadata. The target and media steps
   below apply to execution; recipe-bearing dry runs still use step 4.
2. Preserve an explicit `native`, `pynvc`, or `both`; otherwise use `auto`.
   Obtain fresh direct readiness from setup for the selected local runtime.
   With `auto`, run the only eligible surface, block with zero, or return
   `selection_required` offering `native`, `pynvc`, and `both` when both are
   eligible.
3. Require `full-samples` for Python encode, advanced decode, segmentation, or
   encode-performance. The smaller `pynvc-smoke` profile is sufficient only
   for its documented setup smoke and Python decode-performance route.
4. For encode or transcode dry runs and execution, obtain a validated schema-2
   recipe. Decode-only, segmentation, and container triage do not require one.
5. Canonicalize and hash the exact user media. Preserve URL or `null`, license,
   attribution, path, byte count, SHA-256, and media metadata established by the
   user or a media parser/sample. Never infer metadata from the filename or
   replace the input with setup's synthetic fixture. For a URL, disable
   redirects; bound connection time, total time, and accepted bytes using an
   explicit user ceiling or trusted local metadata. Resolve the host first and
   accept only HTTP(S) addresses outside loopback, private, link-local,
   multicast, and reserved ranges. Recheck the connected peer address on every
   connection and preserve hostname-based TLS validation.

## Plan, then execute

For a dry run, emit a readable `planned` result using the user-supplied media
identity and metadata. Show the intended argument arrays, inputs, outputs,
expected markers, frame counts, producer-to-consumer handoffs, and applicable
`io_contract` from
[buffer-sharing-and-synchronization.md](buffer-sharing-and-synchronization.md).
Do not open the media, inspect the target, authenticate launchers, or claim the
commands were verified.

For execution:

When a native binary still needs compilation, return `dependency_required`
with the exact build command and source/tool identities.

1. Create a new mode-0700 workspace. Require every output path to be absent.
2. Authenticate the selected sample and every required helper as specified in
   [official-sample-contract.md](official-sample-contract.md). Record canonical
   path, size, and SHA-256. Recheck them immediately before launch.
3. Launch the exact argument array directly, without a shell or `eval`, under
   the selected interpreter when Python is used. Retain unedited stdout/stderr
   in separate stage logs and capture exit code, elapsed monotonic time, timeout
   state, and working directory. Use a
   finite per-process timeout: 300 seconds unless the user supplies a different
   finite bound. Record the bound and terminate the launched process group on
   expiry before returning `failed`.
4. Require exit zero, the allowed positive marker set with every expected
   cardinality, count, and path, no explicit failure line, and a fresh nonempty
   output. A timeout, launch error, stale output, contradictory marker, or
   exit-zero unusable file fails that operation.
5. Immediately before a handoff, reopen the original producer output and
   verify its path, size, and SHA-256. The independent consumer must open that
   same path. Rehash it again after the consumer completes. Preserve the
   canonical `io_contract` on successful and failed boundaries and in the
   final result.
6. Apply the reference's byte/layout table to the actual raw source and decoded
   file. Do not infer usability from a marker or byte count alone.
7. Retain a successful branch when another selected surface or segment fails;
   label the aggregate `partial`. Retry once at most, with fresh paths, and only
   after the input, executable/package/interpreter, readiness, or unavailable
   resource has demonstrably changed. Retain both attempts and stop after the
   retry.

## Routes

### Encode and independently decode

- Native: validated recipe → `AppEncCuda` → fresh elementary stream → `AppDec`.
- Python: validated recipe/config → wheel-owned `basic/encode.py` → fresh
  elementary stream → wheel-owned `advanced/decode.py`.
- Require producer and decoder counts to equal the requested frame count.
- Require exact raw-input and decoded-output byte sizes from the format,
  geometry, and frame count. Record the independently decoded layout.
- For `both`, keep each surface's identities, commands, and result separate.

### Native H.264-to-HEVC transcode

Require an H.264 input and an exact native HEVC recipe projection. Do not infer
the input frame count from its filename. Run
`AppTrans` with recipe options except raw-only `-s`, `-if`, and `-gpu`; supply
the selected GPU once. Accept exactly one legacy or current transcode count
marker, never both. Rehash the HEVC output and independently decode it with
`AppDec` for the same positive frame count. If the input frame count was known
before launch, both counts must also equal it; otherwise the two authenticated
sample counts establish the tested frame count.

### PyNvVideoCodec segments

Authenticate the wheel-owned segment sample, `segments.txt`,
`transcode_config.json`, and advanced decoder. Validate every schedule row as
finite `0 <= start < end`. The created-file markers must exactly match the
declared output set and the summary must equal its size. Each segment must be
fresh, nonempty, rehashed, and independently decoded. Report failed segments
without discarding verified peers.

### Container triage

First establish an eligible released Jetson route. Decode the exact local or
retrieved container through `AppDec`'s intrinsic libavformat demux or the
wheel-owned advanced decoder. Record the same content provenance and require a
fresh usable decoded output. This route reports the tested operation only; it
does not generalize hardware support.

### AV1 verification

Require an AV1 recipe with an exact native projection and positive frame count.
Run both requested host-output and video-memory modes independently. For every
fresh IVF output, rehash it and have `AppDec` consume that exact path. Require
the expected positive decoded frame count; do not claim a separate container-
structure verdict from header bytes alone.

## Concise acceptance report

Acceptance is a human-readable reproducibility report. Include only requested
stages, with one row per stage and one of
`complete`, `partial`, `blocked`, `failed`, `not_evaluated`, or `deferred`.
Retain:

- readiness result and selected target/runtime;
- raw API capability facts separately from documentation and operations;
- P4/P5 recipe identities and the exact input identity they share;
- every encode/decode producer/consumer command, marker, frame count, output
  identity, decoded layout, and byte-size validation;
- every benchmark warmup, measured repetition, timing scope, and recomputed
  statistics as required by `jetson-video-benchmark`;
- limitations, failed/retained branches, and exact retry actions.

One report, a command log, and compact JSON evidence are sufficient when they
contain those facts. Write a checksum manifest last if packaging is requested.
Keep media, raw frames, bitstreams, SDK trees, builds, venvs, and caches outside
the small evidence package; bind them by canonical path, bytes, and SHA-256.

## Status meaning

- `planned`: an architecture plan with conceptual steps or a dry run with
  concrete commands was provided; nothing was launched. State unresolved
  assumptions.
- `complete`: every requested branch passed operation and handoff validation.
- `partial`: at least one independent requested branch passed and another did
  not.
- `blocked`: no requested branch could safely run.
- `failed`: launch or observable validation failed.
- `input_required`, `dependency_required`, and `selection_required`: the named
  terminal gate stopped work before launch.
- `operation_verified`: only the exact tested route worked.
- `unsupported`: reserve for directly applicable product documentation; never
  derive it from an API field or failed operation.
