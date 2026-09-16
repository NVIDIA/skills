# Platform Execution Contract

Use this reference for every TAO-backed PAS stage (`pool_embed`, `evaluate`,
`target_embed`, `knn`, `train`, and the three visualization embeddings). Host
stages remain local and use `run_pas_stage.py`.

## One producer, five consumers

`run_deft_action.py prepare` is the only producer. It validates the requested
image, argv, stage scope, token forwarding, output set, and immutable run
hashes; deletes the exact stale outputs; then writes:

```text
<stage>/<name>.action.json
```

The request contains a `tao-artifacts` schema-valid `spec_bundle`, the selected
platform, logical mounts, non-secret environment settings, approved credential
variable names, expected fresh outputs, the exact job-record state directory,
and an attempt number. Never edit it.
The selected platform skill is the consumer:

| state platform | required platform skill | native lifecycle |
|---|---|---|
| `docker` | `tao-run-on-docker` | `docker` |
| `slurm` | `tao-run-on-slurm` | `ssh` + `sbatch/squeue/sacct/scancel` |
| `kubernetes` | `tao-run-on-kubernetes` | `kubectl` Job |
| `brev` | `tao-run-on-brev` | `brev exec` + remote Docker |
| `virtualenv` | `tao-run-on-virtualenv` | vendored `virtualenv_runner.py` |

Read that platform's complete `SKILL.md`, run its preflight, and use its exact
`submit`/`status`/`logs`/`cancel` contract. Do not translate one platform into
another or fall back to Docker.

## Prepare

Each stage reference supplies the exact arguments:

```bash
ACTION_JSON=$(
  "$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
    "$SKILL_ROOT/scripts/run_deft_action.py" prepare \
      --results-dir "$RESULTS_DIR" --image <pyt|ds> \
      --stage-dir "$STAGE_DIR" --name <stable-name> \
      "${HF_ARGS[@]}" \
      --fresh-output "$EXPECTED_OUTPUT" \
      -- <exact TAO argv>
)
ACTION_REQUEST=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["request"])' \
  <<<"$ACTION_JSON")
```

Repeat `--fresh-output` in the documented order when an action has several.
Preparation is part of the approved launch and is a mutation: do not call it
before the launch review.

`prepare` is crash-safe. Repeating it before finalization returns the existing
immutable request and does not delete outputs or mint a new attempt. Inspect
the embedded `reconciliation` result, or run the equivalent explicit check:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" reconcile \
    --request "$ACTION_REQUEST"
```

`NO_JOB_RECORD` means it is safe to open the record. `JOB_OPENED_UNBOUND`
means bind that exact record before submitting. `BOUND` means a native handle
is already recorded: never submit a second native job, and poll/finalize the
bound job instead. `BOUND_BACKEND_RECONCILIATION_REQUIRED` is the crash window
after the immutable pre-submit binding but before a native handle became
durable. Query the selected backend for the deterministic `JOB_ID`/object name
and repair the existing record only when that native object is proven to
exist. If absence cannot be proven, stop for operator reconciliation; never
blindly submit a replacement. Stop on any malformed, duplicate, or mismatched
record rather than guessing.

## Stage inputs and preserve logical paths

Read `mounts` from the request. The same source may have two targets by design:

- workflow results at `/results` and at their approved absolute path;
- the dataset parent at `/data` and at its approved absolute path;
- immutable config at `/specs`;
- compatibility code at `/patches`;
- persistent model cache at `/cache`.

The producer emits only inputs used by that action. Text-only `pool_embed`,
`target_embed`, and `knn` requests intentionally omit the dataset parent;
evaluation, training, and image-embedding requests include both dataset
aliases because they dereference image paths. Consumers must stage every
declared input and must not add undeclared workflow-wide inputs.

Embedded parquet rows and generated TAO specs can contain the approved absolute
dataset path, while TAO commands use `/results`, `/data`, and `/specs`. Every
container platform must therefore render **all** request mounts, including the
absolute-path aliases. Do not remove an alias because it looks host-specific.

Local Docker and virtualenv are tier A and use local paths. Docker through an
approved remote `DOCKER_HOST`, SLURM, Kubernetes, and Brev use the request's
remote staging contract: use `tao-data-io` to stage every declared input, then
render the remote source at each request target. The workflow results tree is
mutable shared application state, so synchronize it to the compute side before
each action and synchronize it back after terminal success.

Freshness on a remote platform is an absence contract, not a clock comparison:
mirror the prepared local results tree with delete semantics so every path in
`staging_absent_paths` (all `fresh_outputs` plus the native action log) is
absent on the compute side before submit. Never use an incremental copy that
can retain stale evidence. The finalizer then accepts the fetched artifact
without comparing clocks, because remote and launcher clocks need not agree.
Local Docker and virtualenv instead require local mtime after preparation.
Remote Docker never uses launcher-local mtimes. The producer also gives every
bundle action a digest suffix bound to the run, stage, attempt, and preparation
time; do not shorten or rederive it when opening the job-record.

After the platform-native absence check succeeds and **before** opening the
job-record, persist that result. Repeat every request output in order and use a
non-secret, canonical durable results scope. It must be the exact absolute
compute-side action-stage path (for example `/lustre/.../embeddings/source`) or
a credential-free persistence URI (for example
`s3://bucket/runs/<run>/embeddings/source`); relative labels, credentials,
queries, fragments, and `..` traversal are rejected:

```bash
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" attest-staged \
    --request "$ACTION_REQUEST" \
    --backend-scope "$BACKEND_SCOPE" \
    --absent-path "$EXPECTED_OUTPUT" \
    --absent-path "$ACTION_LOG"
```

The finalizer rejects a remote action without this digest-bound staging receipt
or when the job-record was opened before the receipt. This prevents a retained
remote artifact from being accepted merely because clocks differ.

## Submit and job record

Validate the bundle before submit. Open the job-record before native launch:

```bash
JOB_ID=$("$BANK/scripts/tao_job_record.py" open \
  --platform "$PLATFORM" \
  --image "$RECORD_IMAGE" \
  --network-arch "$NETWORK_ARCH" \
  --action "$ACTION" \
  --storage-tier "$STORAGE_TIER" \
  --results-dir "$JOB_RESULTS_SCOPE" \
  --upload-exclude .tao-runtime/ \
  --upload-exclude '*.launch.lock')
```

Pass every `spec_bundle.upload_excludes` item, in order, as one
`--upload-exclude`; do not copy a static list from this example. For local
Docker and virtualenv, `JOB_RESULTS_SCOPE` is the exact local
`request.stage_dir`. For remote Docker, SLURM, Kubernetes, and Brev it is the
exact `backend_scope` already recorded by `attest-staged`. The producer rejects
a different path or exclusion list even if the rest of the record matches.

Take `RECORD_IMAGE`, `NETWORK_ARCH`, and `ACTION` from `record_image` and
`spec_bundle`; never rederive them. Immediately bind the still-PENDING record,
before any native submit:

```bash
RECONCILIATION=$("$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" reconcile \
    --request "$ACTION_REQUEST")
JOB_RECORD=$(python3 -c \
  'import json,sys; print(json.load(sys.stdin)["job_record"])' \
  <<<"$RECONCILIATION")
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" bind-job \
    --request "$ACTION_REQUEST" --job-record "$JOB_RECORD"
```

Binding is serialized by a request-scoped lock. It validates the request-owned
record path, platform, image, action, upload exclusions, submission time, exact
results scope, and (for remote platforms) the staging receipt, then persists
an immutable identity digest. It fails once a backend handle or any post-open
transition exists; a distinct concurrent record cannot replace the first
binding. Only after it succeeds may the platform skill submit with `JOB_ID`,
every rendered mount, the bundle command (`command` followed by `args`),
`compute_shape.gpus`, the non-secret `environment` values, and only the
variable names in `forward_env`. Mark `RUNNING` with the native handle.

For Docker, SLURM, Kubernetes, and Brev the executable argv is the bundle argv
inside the pinned image. Virtualenv mirrors the two image families with two
immutable profiles. `image_kind=pyt` selects the approved `pyt` environment
for `clip`; `image_kind=ds` selects the approved `ds` environment for
`embedding` or `tmm`. A single environment is compatible only when it passes
both contracts. Config preparation checks Python 3.12/platform ABI, pinned
distribution versions, exact console-script metadata ownership/target,
profile imports, PyTorch CUDA build, and `pip check`. Action preparation
rechecks the selected action entrypoint and runs the real per-GPU CUDA tensor
probe so a stale or replaced environment cannot reach submit.

Use `request.virtualenv` (not a shell default or the control `.venv`) as the
runner `--venv`, use the attempt-specific `request.platform_runtime_dir` as
the runner `--job-dir`, and submit the packaged shim:

```text
script = request.virtualenv_shim
args   = --request <action.json> -- <bundle command> <bundle args...>
```

Pass each token as a separate runner `--arg` (use `--arg=TOKEN` for tokens
beginning with `-`). The shim verifies the request digest, full immutable
workflow state and paths, exact argv, exact equality between `VIRTUAL_ENV` and
the action's selected profile, and the prepared console-script content digest.
It opens the entrypoint without symlink traversal, verifies its executable file
type and profile-bound shebang, and executes the pinned file descriptor so
replacement races cannot redirect execution. It translates compute-frame paths
and YAML values to local paths, sets local cache locations, constructs a
minimal child environment (approved runtime tuning, validated certificate
paths, and explicitly forwarded `HF_TOKEN` only), and executes only the
allowlisted TAO CLI at the exact `<selected-profile>/bin/<command>` path.

## Status, logs, synchronization, and finalize

Poll the backend, not the record, and map native state to the fixed vocabulary.
On terminal state, mark the job-record. Preserve the ordered
`PENDING -> RUNNING -> terminal` transition lineage; finalization rejects an
earlier terminal state, a missing transition, or timestamps that move
backwards. Capture the selected platform's complete action log at the exact
request `log_path`; a missing, empty, symlinked, or credential-bearing log is
failure. Redact native logs before finalization when the repository credential
linter reports literal secret material.

On resume, a successful request-owned `<name>.status.json` means finalization
already completed. Treat it as durable terminal stage evidence and proceed to
commit/audit; never inspect a launcher process handle and never prepare or
submit that action again. While no finalized action status exists, reconcile
the request and poll only the bound platform backend. A shell PID or tool-call
handle is neither a job identity nor completion evidence.

For a remote `COMPLETE`, synchronize the workflow results tree back before
finalization and verify all declared outputs arrived. Do not tear down tier-C
storage or an ephemeral backend object before output synchronization and log
capture.

Then bind native and artifact evidence:

```bash
JOB_RECORD="${TAO_STATE_DIR:-$HOME/.tao}/jobs/$JOB_ID.json"
"$SKILL_ROOT/scripts/deft_python.sh" --workspace "$WORKSPACE" \
  "$SKILL_ROOT/scripts/run_deft_action.py" finalize \
    --request "$ACTION_REQUEST" \
    --job-record "$JOB_RECORD" \
    --native-exit-code "$NATIVE_EXIT_CODE"
```

Finalization verifies the pre-submit job binding, exact job-record path and
results scope, platform/image/action identity, backend handle, terminal state,
native exit code, immutable request digest, log, and all fresh outputs. It
writes `<stage>/<name>.status.json`; only that status may be passed to
`commit_stage.py`. Commit and audit additionally require the evidence platform
to equal the platform initialized in workflow state; legacy schema-v1
container evidence is accepted only for Docker runs.

On `ERROR` or `CANCELED`, still capture logs and call `finalize` (omit a native
exit code only when the backend cannot report one). It records terminal error
evidence and returns nonzero. A retry is permitted only from that complete,
schema-valid status plus its immutable request, binding, terminal job-record,
backend handle, and log. The producer archives attempt-1 status as
`<name>.attempt-1.status.json`, retains its request/binding/log, and creates
attempt-specific request, receipt, binding, log, and platform-runtime paths for
attempt 2. Repeating preparation returns the same attempt-2 request. The second
attempt receives a new job-record and can never re-submit under the old job id;
a third attempt is rejected.
