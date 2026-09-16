# Pipeline workflow

## Contents

- [Owner and exact CLI](#owner-and-exact-cli)
- [Pipeline request schema](#pipeline-request-schema)
- [Workflow](#workflow)
- [Acceptance content evidence](#acceptance-content-evidence)

Run compact, authenticated multi-stage Video SDK routes and verify every
producer-to-consumer handoff. This workflow owns encode/decode chains, native
transcode, PyNvVideoCodec segmentation, container decode triage, AV1 verification,
and the customer acceptance package. Each stage uses only released samples; libavformat
embedded in released NVIDIA samples is allowed only for container demux.

## Owner and exact CLI

One controller owns this domain:
**`jetson-video-pipeline/scripts/pipeline_controller.py`**. Launch it directly
in isolated mode:

```text
python3 -I scripts/pipeline_controller.py \
  --request nvcodec-pipeline-request.json \
  --workspace fresh-workspace-dir \
  --output nvcodec-pipeline-result.json
```

All three flags are required (`--help` supported). Exit `0` when `status` is `complete`
or `planned`, `2` on a safe block or stage failure, `3` on malformed input or an
internal error.

## Pipeline request schema

`--request` is a `schema_version: "1.0"`, `kind: "nvcodec-pipeline-request"` object.
`mode` is exactly `dry_run` (plan only) or `execute` (default `execute`). `route` is
exactly one of:

| `route` | Purpose |
|---|---|
| `encode_decode` | Encode then independently decode/verify the same artifact (native, or the same request through native and Python for a dual-surface comparison) |
| `native_transcode` | Native H.264 → HEVC transcode with usable-output proof |
| `pynvc_segments` | Split a clip into independently usable PyNvVideoCodec segments and validate each |
| `container_triage` | Demux a user-supplied local container, or retrieve the exact user-supplied HTTP(S) URL, then decode and report hardware-decode support (`surface` must be `native` or `pynvc`) |
| `av1_verify` | Verify AV1 encode from an AV1 recipe with an exact native projection and positive `frame_count` |
| `acceptance` | Aggregate setup, capabilities, P4/P5 encode/decode, and throughput into one acceptance package |

`container_triage` also requires `target_eligibility` with exactly `eligible`
(boolean) and `reasons` (a list of nonempty strings). Use
`{"eligible": true, "reasons": []}` only after identifying the current target
as an eligible released Jetson route; otherwise set `eligible` false and record
the observed reasons. This gate is request planning context, not an SDK setup
artifact or a codec-support verdict.

For `encode_decode`, the controller consumes one authenticated encode-request artifact
and calls the landed recipe/encode owner. Other routes authenticate their exact sample
set directly. A completed `container_triage` result retains its workspace-relative
decoded-frame evidence and also returns `decode.raw_video` as the exact portable absolute
identity that the encode controller accepts directly. `av1_verify` owns the complete IVF
frame walk and independent AppDec proof;
setup capability queries remain separate context and are never treated as that proof.

`environment` is an optional portable identity for a fresh schema-1.2 setup
artifact. Setup emits raw JSON; the request carries an exact identity with only
`schema_version`, `kind`, canonical absolute `path`, `size_bytes`, and lowercase
`sha256`. When supplied it is authoritative and must validate. When omitted,
the controller authenticates the selected installed surface locally;
PyNvVideoCodec routes additionally require the exact absolute
`pynvc_interpreter`. A controller-produced local binding is private evidence
and is never accepted as a request member.

After the media gate, an agent that needs `pynvc` or `both` and has neither a
supplied environment nor exact interpreter first checks the installed skill
catalog. If `jetson-video-setup` is present, invoke its public
`probe_nvcodec.py` with a fresh output, `--runtime pynvc` for Python-only or
`--runtime both` for `both`/`auto`, and no `--setup-candidate`. Setup alone
resolves and reauthenticates its registry. Inspect the fresh JSON before
constructing the identity: require `mode=live`, the requested GPU,
`pynvc.installed=true`, and `pynvc.identity.status=verified`. Derive the
identity's path, size, and SHA-256 from that exact file and let this controller
authenticate it again. For explicit `pynvc`/`both`, ask for an exact interpreter
only when setup is unavailable or returns any absent, stale, unreadable,
invalid-binding, or launch-failure result. For `auto`, keep Py `not_evaluated`
and continue only an eligible native surface. Never pass a blocked probe as
authority.

## Workflow

1. Apply the media gate before classifying the request into exactly one route
   above. Require its media input to originate from an absolute target-local
   path or one exact user-supplied HTTP(S) URL; never substitute catalog or
   synthetic media, and preserve provenance and identity. When setup is installed, also apply its
   shared [video content policy](../../jetson-video-setup/references/video-content.md).
   A route may consume a prior artifact only when that artifact remains bound
   to the same user-selected source. If no input is supplied, return
   `input_required`, ask for one, and pause before probing, retrieval,
   conversion, authority selection, dry run, or pipeline execution. Ask only
   for media at that gate; never select catalog media or use the synthetic
   setup fixture.
2. Honor an explicit `native`/`pynvc`/`both` surface. Map “whichever”, “best
   available”, “choose for me”, and any other unspecified-surface wording to
   `auto`; never broaden it to `both`. Only an explicit dual-surface request
   selects `both`.
3. After the input gate and surface classification, use a supplied setup
   environment identity, obtain a fresh one through setup's public probe as
   described above, or use local selected-surface authentication. Supplied
   evidence must validate and never falls back. Without it, native derives raw
   dpkg/package-source/toolchain facts; Python derives raw wheel/import facts
   from the exact `pynvc_interpreter`. The consumer stages those facts as
   private evidence, derives its normalized surface view internally, and
   revalidates the binding before launch. It never accepts a serialized local
   binding from the caller and never scans for a Python environment. A missing
   or invalid SDK routes only that surface to `jetson-video-setup`.
4. Apply the `auto` gate using only that authority: zero
   eligible surfaces block, one runs, and two
   return `selection_required` before searching prior results, dry run, or
   codec launch. Eligibility authentication happens first. This gate is
   unconditional when two surfaces are
   eligible: ask for exactly `native`, `pynvc`, or `both`. For `both`, preserve
   every eligible branch and each blocked peer. One surface never authorizes or
   blocks the other. If local `auto` has no exact `pynvc_interpreter`, disclose
   PyNvVideoCodec as `not_evaluated` with the retry action and continue only an
   eligible native branch; explicit `pynvc` and `both` still require setup
   evidence or the exact interpreter. A prompt's claim that a surface is ready
   is never authority. Eligibility is decided at planning time: each surface is
   structurally validated against its own required subset of the schema-1.2
   environment — native `installed`/`package`/`sdk_root`/`cuda`/`tools` versions,
   pynvc `installed`/`version`/`interpreter`/`interpreter_identity`/`sys_prefix`/
   `extension`/`module` — ignoring additive keys. A malformed surface is blocked
   during planning, never deferred to execute-time authentication, so a broken
   surface never consumes a run slot or couples the peer through late failure.
5. Run `dry_run` to review the planned stages and handoffs, then `execute` with a fresh
   output path and bounded per-stage timeouts.
6. Bind each handoff to its original canonical path, size, and SHA-256; reopen and
   rehash the original artifact — never trust copied status prose. A stage is complete
   only when its exact positive marker, count, and fresh nonempty output are proven.

For `acceptance`, the controller writes exactly nine compact pre-seal files and reports
`seal_pending: true`; it intentionally does not create its own checksum manifest or an
inventory. Its `references` object must name `readiness`, `capabilities`, `content`,
`p4_recipe`, `p5_recipe`, `p4_encode_decode`, `p5_encode_decode`, `p4_benchmark`,
`p5_benchmark`, `handoffs`, `commands`, `task_results`,
and `timing`. The task-results JSONL contains exactly one
`{task_id, status: "complete", evidence}` row for each of the ten stage names;
`evidence` is a nonempty list of names from `references`. Recipes must validate as P4/P5,
each encode/decode stage must consume its accepted recipe and the same exact raw input,
and each benchmark variant must retain that exact recipe identity and input identity.
Encode/decode artifacts must be complete, benchmarks must retain a warmup and at least
three measurements; the handoff list must be nonempty and every entry must be verified.
Each accepted benchmark branch retains the successful `warmup` command and
sequential measured repetitions whose command phase is `measure`; every command argv must
equal the authenticated launcher plus `benchmark_arguments`, exit zero, and not time out.
FPS and megapixels/second must be positive finite values, and their repetition count, mean,
minimum, and maximum must agree with the retained measurements.

The acceptance assembler copies the six verified media metadata fields from the `content`
evidence into `evidence/summary.json.representative_content`: exactly
`source_url`, `license`, `attribution`, `path`, `size_bytes`, and `sha256`.
`source_url` is the exact user-supplied HTTP(S) URL for retrieved media and JSON
`null` for target-local media. `license` and `attribution` are nonempty honest
strings; either may be the literal `unknown`. It writes the corresponding exact
direct source record—not a generic reference table—to
`evidence/representative-content.json` with exactly `kind`, those six metadata
fields, `expected_size_bytes`, `expected_sha256`, and `verified`. Require
matching expected size/SHA-256 and `verified: true`. Both files therefore
satisfy the six-field compact metadata contract in
[setup's shared video content policy](../../jetson-video-setup/references/video-content.md)
and can be passed directly to the `content-summary` controller while the
external media remains at its bound canonical path. Retrieval-command or local
user-selection evidence supports acquisition claims separately and is not part
of this six-field validator contract.

### Acceptance content evidence

Keep the source record and fresh validation result in the compact package and
bind both in its manifest. Exclude the media itself from the package, retaining
its canonical path, size, and SHA-256 in the summary. Run
`scripts/validate_representative_content_summary.py` directly before cleaning
up excluded media:

```bash
python3 -I scripts/validate_representative_content_summary.py \
  --summary /absolute/path/to/evidence/summary.json \
  --source-artifact /absolute/path/to/evidence/representative-content.json \
  --output /absolute/path/to/evidence/representative-content-summary-validation.json
```

The validator reads the two JSON files directly, requires strict shapes and
values, and freshly rehashes the current content file without modifying it. It
proves current byte identity and exact cross-record metadata equality only. It
does not prove URL origin, publisher-supplied expectations, license,
attribution, representativeness, or nonuniformity. Preserve a retrieval command
record for URL media or user-selection evidence for local media when making an
acquisition claim; these are separate from the six-field metadata consistency
check and need not be copied into the compact package. Claim content
characteristics only when separate observed evidence supports them.

Exit `0` means this limited consistency check is verified, exit `2` means
invalid or unauthenticated input, and exit `3` means validator or fresh-output
failure. The compact package cannot rerun the check after excluded media is
removed unless that external media is restored at its bound path. Do not pipe a
log or stdout record into this check, and do not use command substitution in
place of either file.

Set `validation_result_filename` to one fresh plain JSON filename and pass the matching
workspace-relative path to the CLI, for example
`--output evidence/customer-validation.json`. The controller writes the other eight files;
the mandatory `--output` is the ninth validation-result file. It does not contain its own
artifact identity. After the result is complete and every JSON/JSONL file parses, seal the
final small package from its root:

```bash
(
  cd "$PACKAGE_ROOT"
  find . -type f ! -path './evidence/manifest-sha256.txt' -print0 |
    LC_ALL=C sort -z | xargs -0 sha256sum
) > "$PACKAGE_ROOT/evidence/manifest-sha256.txt"
```

Do not place media, raw frames, streams, SDK trees, build directories, virtual
environments, or caches in the package; record their external path, size, and SHA-256.

The allowed operation routes are defined by
[official-sample-contract.md](official-sample-contract.md).
