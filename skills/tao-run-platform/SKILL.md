---
name: tao-run-platform
description: TAO Execution SDK for submitting and monitoring GPU training jobs on supported platforms (Brev, SLURM,
  local Docker, Kubernetes). Use when the user wants to run TAO jobs through the SDK, get job tracking, S3 I/O wrapping,
  multi-node distributed training, or platform-specific features that docker-run can't provide. Trigger phrases include
  "use the TAO SDK", "call tao_sdk", "AutoMLRunner", "ActionWorkflow", "Job handles", "S3 I/O wrapping", "TAO platform run".
license: Apache-2.0
compatibility: Requires Python 3.10+ and the nvidia-tao-sdk package (pip install nvidia-tao-sdk[all]).
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- platform
- tao
- sdk
---

# TAO Execution SDK

The SDK is the **optional** Python layer for users who need job handles, S3 I/O wrapping, or platform-specific features (SLURM/Lustre queues, Kubernetes Jobs, local Docker debugging, Brev instance reuse). Most TAO skills run with just `docker run` and don't need it. Reach for the SDK when:

- You want a `Job` handle to poll status and stream logs over time.
- You need S3-aware input download / output upload baked into the entrypoint.
- You're chaining multiple jobs and want persisted state.

## Preflight

Install `nvidia-tao-sdk[all]` before using this platform — the `[all]` extra pulls in every platform-specific dependency (Brev, S3 utilities, etc.). If it is missing, install it by default in the active Python environment and rerun the import check:

```bash
python -c "import tao_sdk" 2>/dev/null || {
  echo "Installing missing Python requirement: nvidia-tao-sdk[all]"
  python -m pip install "nvidia-tao-sdk[all]"
}
python -c "import tao_sdk"
```

The package index is environment-specific — the runner/container is expected to have a working `pip` configuration (e.g. `~/.pip/pip.conf`, `PIP_INDEX_URL`, `PIP_EXTRA_INDEX_URL`, or proxy). If the install fails for index/network reasons, that's a runner setup issue; this skill stays agnostic to the registry.

Missing pip requirements are installed automatically by default and reported in the run log. Non-pip/system prerequisites still require a normal preflight failure and user-visible remediation.

## Setup

Credentials come from **environment variables** — read from the session environment (export them in your shell before launching).

```python
from tao_sdk.platforms.brev   import BrevSDK     # Brev GPU instances

sdk = BrevSDK()      # reads BREV_API_TOKEN (optional — falls back to brev login)
```

The SDK validates credentials lazily on first use and raises `CredentialError` with a clear message if a required env var is missing. Required env vars:

| Platform | Required | Optional |
|---|---|---|
| Brev | — (manual `brev login` works) | `BREV_API_TOKEN` |
| S3 I/O (any platform) | `S3_BUCKET_NAME`, `ACCESS_KEY`, `SECRET_KEY` | `S3_ENDPOINT_URL`, `CLOUD_REGION` |
| Container env | `NGC_KEY` | `HF_TOKEN` |

The agent never reads credential values — it only checks presence with `[ -n "$VAR_NAME" ]`.

## Workflow Launch Intake

For any TAO workflow or action launch, first confirm the user goal. Then ask
for platform and monitoring preferences before credentials or launch details.
Generate the supported platform choices from the packaged helper, not by
scanning platform docs or folders:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --format text
```

Ask:

1. Which supported platform should run this workflow?
2. Should long-running monitoring stay enabled? Default: enabled. This means
   the agent remains attached and posts status until terminal state, including
   long `PENDING` queue waits.
3. How many minutes between status updates? Default: 5 minutes.

After the model/action are known, resolve the default container image from the
packaged metadata and ask the user to confirm it or provide `image=<override>`
before creating runner files:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/resolve_tao_image.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --model <network_arch> --action <action> --format text
```

For train-capable model workflows, inspect model-level AutoML metadata before
creating a plain training job:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_models.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --scope automl --format json
```

If the selected model has `automl_enabled: true` and a valid train schema,
route training through `skills/applications/tao-run-automl` by default with
`automl_policy: on`. A workflow should only bypass AutoML when its run settings
include `automl_policy: off`, the user explicitly asks for a plain run, or the
model metadata says AutoML is enabled but the train schema is not packaged yet.

After the platform is selected, get the credential filter:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --platform <platform> --format text
```

Ask only for credentials returned for the selected platform. For example, SLURM
needs `SLURM_USER` and `SLURM_HOSTNAME`; it does not need Brev credentials.
Kubernetes and local Docker do not need Brev or SLURM credentials. Ask storage
credentials such as S3 keys only when the selected platform and the data/result
URIs require them.

## Core API

All platform SDKs implement the same core shape:

```python
sdk.create_job(image, command, gpu_count=1, env_vars=None, inputs=None, outputs=None, **kwargs) -> Job
sdk.get_job_status(job_id) -> JobStatus
sdk.get_job_logs(job_id, tail=None) -> str
sdk.cancel_job(job_id) -> bool
sdk.get_failure_analysis(job_id) -> dict | None
sdk.get_job_results_dir(job_id) -> str
sdk.check_path(remote_path) -> bool
sdk.list_path(remote_path) -> list[str]
```

Brev-only:
- `sdk.delete_instance(instance_id)` — clean up an ephemeral instance.
- `sdk.list_instances()` — list active instances.

## Submitting a Job

The agent always **constructs the container command via `build_entrypoint`** before calling `create_job`. The agent reads the action's schema from `skill_info.yaml` (`command`, `mode`, `config_format`, `inputs`, `outputs`, `upload_excludes`) and passes those fields as kwargs. `build_entrypoint` then bakes:

1. The in-container `script_runner` runtime (inlined as a base64 heredoc — no need for `tao_sdk` to be installed in the container).
2. The CLI invocation that, at runtime in the container, will: download declared inputs (S3 / HF-Hub / NGC), write the spec file at `{config_path}` with remote URIs rewritten to local paths, run the user command, and upload outputs.

Output destinations are resolved at runtime from env vars the SDK injects (see "Where outputs go" below). The platform SDK's `create_job` runs the resulting command **as-is** — no inputs/outputs kwargs, no implicit wrapping. The data flow is visible in the agent's code.

### Where outputs go (resolved at runtime — agents don't manage it)

The SDK injects `TAO_JOB_ID` (matches `Job.id`) and, when a persistent mount is attached, `TAO_RESULTS_ROOT` into the container env. Inside the container, `script_runner` resolves output destinations:

| Container env | Result |
|---|---|
| `TAO_RESULTS_ROOT` set (Lustre / PVC / bind / NFS) | Outputs at `{TAO_RESULTS_ROOT}/<job_id>/<key>/`; no upload |
| `S3_BUCKET_NAME` set (cloud, no mount) | Outputs at `s3://{bucket}/results/<job_id>/<key>/`; uploaded at end of run |
| Neither | Outputs at `/results/<job_id>/<key>/` (container-ephemeral) with a loud end-of-run warning |

Per-platform policy:

| SDK | What gets injected |
|---|---|
| `SlurmSDK` | `TAO_RESULTS_ROOT={SLURM_BASE_RESULTS_DIR}/results` (always — Lustre, never S3, avoids GPU-idle scheduler kill) |
| `KubernetesSDK` / `DockerSDK` / `BrevSDK` | `TAO_RESULTS_ROOT=/results` if a mount targets `/results`; otherwise S3 fallback |

Agents who want a custom destination can put an `s3://...` URI or absolute path directly at the output spec key — explicit values override the auto-fill. Otherwise, model-natural defaults like cosmos-rl's `output_dir: "output"` or DINO's empty `results_dir` are auto-rewritten by `script_runner`.

### The spec is nested dicts, NOT flat dotted keys

This is the most common mistake when constructing a spec. The dotted notation that appears in `skill_info.yaml`'s `inputs:` / `outputs:` blocks (e.g. `section.subsection.key`) is a **path into** a nested spec — `script_runner` looks values up at that path. It's not the spec's own shape. The spec mirrors whatever shape the model's container expects (typically a nested TOML/YAML).

```python
# ✓ CORRECT — nested dicts
specs = {
    "section": {
        "subsection": {"key": "value"},
    },
}

# ✗ WRONG — flat top-level key with dots. TOML/YAML emits this as a
# quoted bare-string key, the model sees an empty `section` table, and
# any input declared at "section.subsection.key" silently fails to
# download because _get_nested(specs, "section.subsection.key") → None.
specs = {
    "section.subsection.key": "value",
}
```

The two shapes look superficially similar but mean different things. When in doubt, open the model's `references/` directory (e.g. a default-spec TOML or YAML) — that's the literal nested structure the spec dict needs to mirror. The `inputs:` / `outputs:` declarations in `skill_info.yaml` are *paths into* the nested spec, not key names.

### Constructing the spec / args

The skill's action declares its config mechanism in `skill_info.yaml`'s `actions.<action>.mode` field. Treat missing `mode` as invalid metadata and fix the skill instead of inferring a default. Read `actions.<action>.mode` first, then pass the matching argument shape to `build_entrypoint`:

| Declared mode | What the agent passes |
|---|---|
| `config` | `specs=...` with spec-keyed `inputs` / `outputs`; the helper writes the spec file, rewrites URIs, and runs the command |
| `args` | `args=...` with optional spec-keyed `inputs` / `outputs`; the helper substitutes CLI args into the command template |
| `passthrough` | path-keyed `inputs=...` and/or `outputs=...`; the helper downloads to listed paths, runs the command, and uploads listed outputs |

Do not infer mode from missing metadata. Missing `mode` means the skill contract is stale.

See [`references/spec-construction.md`](references/spec-construction.md) for the per-mode construction strategy, the recommended decision order, and worked `build_entrypoint` examples for spec-driven jobs (config file) and path-keyed jobs (no config file).

## Resolving container images

Skills declare images either by key (`tao_toolkit.pyt`) or as an absolute URI (`nvcr.io/...`). Use `resolve_container_image()` to handle both:

```python
from tao_sdk.versions import resolve_container_image
image = resolve_container_image(skill_info["container_image"])
```

Behind the scenes it walks `versions.yaml` for keys; absolute URIs are returned as-is.

## Monitoring

```python
status = sdk.get_job_status(job.id)
print(status.status)   # Pending, Running, Complete, Error, Canceled
print(status.message)  # platform-specific detail

logs = sdk.get_job_logs(job.id, tail=200)
print(logs)
```

On failure, `get_failure_analysis()` classifies the root cause:

```python
analysis = sdk.get_failure_analysis(job.id)
if analysis:
    print(analysis["err_class"])   # ERR_PROGRAM, ERR_INFRA, etc.
    print(analysis["suggestion"])  # human-readable fix
    for event in analysis.get("job_failure_by_node_event", []):
        print(event["node_event_name"], event["message"])  # OOM, GPU error, etc.
```

## Polling pattern

For interactive runs where the user wants to watch:

```python
import time
status_interval_minutes = status_interval_minutes or 5
while True:
    status = sdk.get_job_status(job.id)
    if status.status in ("Complete", "Error", "Canceled"):
        break
    print(f"  {status.status}")
    time.sleep(status_interval_minutes * 60)

if status.status == "Error":
    print(sdk.get_job_logs(job.id, tail=100))
    print(sdk.get_failure_analysis(job.id))
```

With long-running monitoring enabled, do not stop after 30 minutes or after a
few unchanged polls. Keep emitting updates every `status_interval_minutes`
until the job finishes, fails, is canceled, or the user asks to detach/stop.
If the chat/runtime cannot remain open that long, say so explicitly and provide
the durable workflow/log path for manual status refresh.

Do not use a final response for non-terminal monitored jobs. Finalizing the
turn detaches the chat watcher. Keep non-terminal status messages in progress
updates and continue polling; only finalize at terminal state, explicit user
detach/stop, or a real runtime limit that prevents further polling.

For background runs, persist `job.id` and the `state_file` path, then re-attach later by constructing the same SDK and calling `get_job_status(job_id)` — job state is read from the on-disk store.

## Orchestration patterns

Multi-step workflows, parallel sweeps, and run-folder durability via
`ActionWorkflow` live in
[`references/orchestration-patterns.md`](references/orchestration-patterns.md).
Read it before chaining `create_job` calls, sweeping a parameter, or
persisting run state across context breaks.

## Dataset utilities

When the skill's documented filenames don't match the user's layout, list the dataset to confirm:

```python
assert sdk.check_path("s3://my-bucket/coco/")
files = sdk.list_path("s3://my-bucket/coco/train/")
# Use the actual paths to set spec fields.
```

For S3 paths, strip trailing slashes when concatenating to avoid `//`:

```python
base = dataset_uri.rstrip("/")
specs["dataset"]["train_csv"] = f"{base}/train.csv"   # nested — see "spec is nested dicts"
```

## Platform-specific notes

See [`references/platform-notes.md`](references/platform-notes.md) for per-platform behavior, kwargs, and credential scoping: Brev (`instance_id`/`gpu_type`/`cloud_cred_id`/`workspace_group_id`, ready-wait timeouts), SLURM (sbatch over SSH, Lustre paths, queue defaults), Kubernetes (kubeconfig, GPU Operator), and local Docker (single-host, multi-GPU).

## Error patterns

SDK error → root cause → fix mappings are in
[`references/error-patterns.md`](references/error-patterns.md). Read when
you hit a `CredentialError`, image-pull failure, stuck-Pending job, or
similar — the entries map exception text to the underlying cause.

## What the SDK does NOT do

The SDK does not read/interpret skills, run AutoML on its own, decide spec contents, select platforms, or orchestrate multi-step workflows — those stay the agent's responsibility. See [`references/scope.md`](references/scope.md) for the full scope guardrails, including the model-level AutoML policy (`automl_enabled: true` → `skills/applications/tao-run-automl` unless `automl_policy: off` or the user asks for a plain single run).
