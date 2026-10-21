---
name: cdf-video-data-augmentation
description: Run the CDF Video Data Augmentation Airflow workflow on Kubernetes or OSMO to create Cosmos video-to-video variants and auto-label both source and generated videos. Use for VDA payload authoring, preflight, execution, monitoring, output verification, or result retrieval. OSMO requires external endpoints. Standalone augmentation or auto-labeling configuration belongs to the respective component skill. Not for retrieval-driven CDF pipelines or image-to-video event generation.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  owner: NVIDIA
  service: cosmos-data-factory
  version: "1.0.0"
---

# Run CDF Video Data Augmentation

Run the existing Video Data Augmentation workflow end to end. Read the
[getting-started guide](../../docs/video-data-augmentation/getting-started.md) and
[payload guide](../../docs/video-data-augmentation/payload-guide.md) before authoring a payload.
Treat the checked-in DAG, payload model, and platform manifest as the source of truth if the docs
and implementation differ.

Use the bundled references progressively:

- Read [setup-and-preflight.md](references/setup-and-preflight.md) when preparing input or checking
  a controller/compute environment.
- Read [payload-contract.md](references/payload-contract.md) before rendering or changing a
  payload.
- Read [airflow-direct-api.md](references/airflow-direct-api.md) before triggering or monitoring a
  run through the API.
- Read [outputs.md](references/outputs.md) before downloading or interpreting artifacts.
- Read [troubleshooting.md](references/troubleshooting.md) when a check or run fails, or when the
  user asks to stop a run.

The workflow prepares source videos, runs the full auto-labeling and training-export chain on the
source branch, generates Cosmos Transfer variants, runs the same chain on passing generated
videos, validates storage outputs, and shuts down run-owned services. It does not produce one
combined final dataset manifest.

## Component details

Distinguish the configuration layers before authoring a payload:

| Requested work | Route to |
|---|---|
| Augmentation container input/config YAML, model parameters, distributions, or evaluator settings | `cdf-augmentation` |
| Auto-labeling container input/config, stage options, prompts, or question banks | `cdf-auto-labeling` |
| Existing VDA DagRun `conf.payload`, mapping supported component options into it, triggering, monitoring, or results | This skill and its workflow references |
| Kubernetes/Airflow controller setup, environment audit, connections, or deployment readiness | `cdf-orchestration-setup` |
| A requested new Kubernetes DAG with added, removed, or reordered stages that no existing DAG covers; or DAG parity review against a specification | `cdf-orchestration-write-dag` |

Consult the component skill before authoring or changing its container configuration, even when
that configuration is used inside a full VDA run. Component YAML and the Airflow trigger payload
are different contracts; do not put an entire container config directly into `conf.payload`.
Changing supported payload values alone does not require a new DAG. Route structural changes to
the DAG-authoring skill only when requested; setup/deployment remains separately authorized.

When the request needs component configuration or diagnosis, consult that component's
documentation, and do not invent keys or values. The links below are pinned to the latest `main`
commit of each component repository as of 2026-10-20; a component skill installed separately may
describe a different version.

| Component | Covers | Documentation at that commit |
|---|---|---|
| `cdf-augmentation` | Cosmos configuration, distributions, captioning, evaluator setup, and component errors | [`cdf-augmentation` skill][augmentation-skill] |
| `cdf-auto-labeling` | Stage configuration, prompts/question banks, stage outputs, and component errors | [`cdf-auto-labeling` skill][auto-labeling-skill] |

[augmentation-skill]: https://github.com/NVIDIA/cdf-augmentation/tree/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation
[auto-labeling-skill]: https://github.com/NVIDIA/cdf-auto-labeling/tree/6c75c76c52a6c76c3866a38d30885e11123b0981/skills/cdf-auto-labeling

Use only documentation at these commits; do not substitute a moving branch or another version.
If the documentation is inaccessible, report the exact reference and continue only work supported
by the local workflow guidance; do not invent component settings.

The orchestration routes are sibling skills in this repository:
[`cdf-orchestration-setup`](../cdf-orchestration-setup/SKILL.md) for environment preparation and
Airflow controller setup, and [`cdf-orchestration-write-dag`](../cdf-orchestration-write-dag/SKILL.md)
for custom DAG composition and specification parity review.

The pinned component skills own container schemas, model options, prompt authoring, and evaluator
internals; use their referenced documentation rather than maintaining local copies. This skill
owns the VDA trigger contract and the constraints imposed by its checked-in DAG. Preserve those overrides in
[payload-contract.md](references/payload-contract.md); component capabilities do not automatically
become supported DAG options. Keep execution through Airflow for a workflow request.
If a pinned component contract differs from the deployed image or DAG integration, report the
version mismatch before proposing settings; do not silently update an image or bypass an override.
Routine payload rendering, status checks, and GPU sizing do not require loading both components.

## Select the execution target

| Target | DAG ID | Service modes |
|---|---|---|
| Kubernetes | `video_data_augmentation_dag_k8s` | Internal or external inference services |
| OSMO | `video_data_augmentation_dag_osmo` | External inference and evaluator endpoints only |

List the DAGs Airflow actually loaded and check import errors before selecting a DAG. A platform
DAG is registered only when its manifest exists. Never invent a DAG ID or reinstall Airflow merely
because the requested DAG is absent; report that the existing installation does not contain it.

Use Kubernetes unless the user explicitly requests OSMO or supplies an OSMO execution context.
For OSMO, require `external_services: true`, keep every `service_lifecycle.*.enabled` value false,
and supply all VLM, LLM, Cosmos Transfer, and required Cosmos Evaluator endpoints. The OSMO
manifest cannot deploy any of these services.

## Minimum compute

For Kubernetes, let `C` be the number of enabled managed evaluator checkers, from zero to three.
Each managed checker—hallucination, attribute verification, and physical plausibility—requires one
GPU. Use `C=0` when the evaluator is external or all checks are disabled.

| Mode | Absolute GPU floor | Practical default-graph minimum |
|---|---:|---:|
| Kubernetes, external inference services | `C + 1` | `C + 2` |
| Kubernetes, internal inference services | `C + 4` | `C + 5` |
| OSMO, all endpoints external | One 1-GPU job slot | Two concurrent 1-GPU job slots |

The absolute floor reserves one GPU for a runnable batch task; the practical minimum reserves two
because source auto-labeling and Cosmos/output processing can overlap. Internal Kubernetes adds one
GPU for the shared language endpoint and two for Cosmos3-Nano Transfer. Thus the default internal
payload with all three bundled checks enabled needs **7 GPUs to make forward progress and 8 GPUs
for the default parallel graph**. With physical plausibility disabled but managed hallucination
and attribute verification enabled, the corresponding floors are **6 and 7 GPUs**. OSMO assigns
each active task 1 GPU, 11 CPUs, 100 GiB RAM, and 100 GiB storage. Read
[setup-and-preflight.md](references/setup-and-preflight.md#minimum-compute) for assumptions,
scaling, and the external-service examples.

## Manual payload entry in the Airflow UI

If the user explicitly wants to enter their own payload in the Airflow UI, confirm controller
readiness, make the UI reachable, and report the URL. Do not render, confirm, or submit a second
payload on their behalf. Once they say they triggered a run, locate it through the API and resume
monitoring from the recorded DagRun configuration.

## Collect the run contract

Before building a payload, confirm:

- `input_path`: an accessible single video or directory containing `.mp4`, `.avi`, `.mov`,
  `.mkv`, or `.webm` files.
- `output_directory`: a writable, run-appropriate storage prefix.
- Target platform and `external_services` mode.
- `max_videos`: use a positive integer for a bounded run. Use `null`, zero, or a negative integer
  for all videos; validation normalizes every unlimited form to `null` before input discovery.
- `cosmos.num_augmentation` and a non-empty, video-appropriate
  `cosmos.variable_distribution`.
- Enabled evaluator checks and whether the evaluator is managed or external.
- For external mode, the real service URLs used by `cosmos`, `captioning`, `visual_qa`, and
  `reasoning`. `cosmos.cosmos_transfer_service_url` is operationally required even though the
  current schema does not reject its absence.

Never invent a path or endpoint — use only values the user supplied, including ones they
supplied earlier for a previous run.
Do not ask the user to paste credentials; inherit them from the
configured Airflow connections, Kubernetes secrets, OSMO configuration, or shell environment.

For a local file or flat video directory, validate before upload:

```bash
python3 scripts/upload_videos.py --path /path/to/videos --validate-only
python3 scripts/upload_videos.py \
  --path /path/to/videos \
  --destination-path video-data-augmentation/my-run
```

The upload command requires a configured `UPLOAD_DESTINATION` and the standard AWS credential
chain.

## Bundled tools

- `scripts/upload_videos.py`: validate and optionally upload one local video or a flat directory.
- `scripts/payload.py`: render or validate standalone Kubernetes and OSMO payloads.
- `scripts/summarize_results.py`: summarize a downloaded run's candidate, evaluator, labeling,
  export, and report files without loading media.
- `assets/variable-distribution.json`: editable weather/time-of-day distribution for the bundled
  VDA prompt.

Run commands from this skill directory. The scripts never need credentials in command arguments.

## Author and validate the payload

Start from the smallest payload that expresses the requested run, then validate it with
`scripts/payload.py`. When the repository is available, also validate with
`VideoDataAugmentationDagConfig` from
`airflow/dags/workflows/video_data_augmentation_dag/models/payload.py`; it is authoritative for
the deployed code.

Preserve these invariants:

- Top-level `external_services` controls the VLM, LLM, and Cosmos Transfer lifecycle. Nested
  service modes and output directories must agree with the top level.
- Internal Kubernetes mode enables the VLM, LLM, and Cosmos Transfer lifecycle entries. The
  checked-in Kubernetes manifest uses one shared language-model endpoint and a two-GPU
  Cosmos3-Nano Transfer endpoint.
- External mode supplies every endpoint used by each enabled stage. If caption summaries are
  disabled, the captioning LLM endpoint is not required.
- Whenever a `cosmos` object is present, include an explicit non-empty
  `variable_distribution`; otherwise the shared model can fall back to an unrelated clothing
  distribution.
- The bundled transfer config fixes the generation model to `nvidia/Cosmos3-Nano`;
  `cosmos.cosmos_transfer_model` does not override it.
- The DAG replaces `visual_qa.question_bank_file` with its bundled question bank. Do not claim
  that a payload value changes VDA questions.
- Unknown fields may be ignored by the schema. Review field names rather than treating successful
  parsing as proof that every requested option took effect.
- On Kubernetes, enabled evaluator checks may use either an explicit
  `cosmos.cosmos_evaluator_service_url` or managed evaluator components. On OSMO, every enabled
  check requires an external evaluator URL reachable from OSMO tasks; alternatively disable all
  checks explicitly.

Show the resolved, validated payload to the user without secrets and obtain confirmation before
submitting a run. Submission authorization covers one run only.

Typical internal Kubernetes rendering:

```bash
python3 scripts/payload.py render \
  --platform k8s --service-mode internal \
  --input-path s3://bucket/vda/input/ \
  --output-directory s3://bucket/vda/output \
  --max-videos 1 --num-augmentation 1 \
  --variable-distribution assets/variable-distribution.json \
  --output /tmp/vda-payload.json
```

## Preflight the existing installation

Do not deploy or reinstall unless the user explicitly asks. Before triggering:

1. Inspect the active repository branch/worktree and the live controller state. Confirm controller
   pods are healthy, the target DAG is loaded and unpaused, and no relevant import error exists.
2. Verify the required Airflow pools and open slots. Kubernetes uses `k8s_rtx6000_1`,
   `default_pool` (internal-mode augmentation also runs here), and
   `cosmos_transfer_3_service_pool` for external mode; OSMO uses `osmo_rtx6000_1`,
   `default_pool`, and `cosmos_transfer_3_service_pool`.
3. Verify the selected task platform, object-storage access, endpoint reachability, image/model
   compatibility, cache/storage headroom, and required Airflow connections or secrets without
   printing credential values.
4. Check actual free GPUs as well as advertised capacity and running consumers. Kubernetes batch
   stages use GPU task profiles in both service modes. Internal mode additionally deploys the
   shared language endpoint and two-GPU Cosmos Transfer endpoint. Each enabled managed checker
   consumes one GPU. Enforce the minimum-compute floor above, then size for the live execution
   graph, mapped inputs, replicas, and pool concurrency.
5. Report stale failed pods or prior run-owned resources. Clean up only resources belonging to a
   run the user owns and only with authorization.
6. From the repository root, confirm the deployment maps every bucket the payload reads or
   writes:

   ```bash
   python3 skills/cdf-orchestration-preflight/scripts/storage_check.py --payload <payload.json>
   ```

   It must print `Storage mapping: ready`. An `UNMAPPED` line means the run would fail at
   runtime; report the unmapped field and the configured prefixes to the user. Using that
   bucket requires changing the controller's storage configuration and redeploying it; with the
   user's authorization, route that change to
   [orchestration setup](../cdf-orchestration-setup/SKILL.md), then rerun the check.

If the existing installation is not ready, report the failed prerequisite. Deployment, controller
reinstallation, pool resizing, and destructive cleanup require separate explicit authorization.
Use [airflow-direct-api.md](references/airflow-direct-api.md) for the concrete checks; do not infer
readiness only from a healthy Airflow UI.

## Submit and monitor

Submit exactly one run with the confirmed payload under `conf.payload`, an explicit unique run ID
when the interface supports it, and the required logical date for the Airflow REST API. Record the
DAG ID, DagRun ID, platform, service mode, input, output prefix, requested input limit,
augmentation count, and enabled evaluator checks. If the submission response is uncertain, look
up the same run ID before retrying; do not create a duplicate run.

Monitor until the DagRun reaches `success` or `failed`. Inspect task instances and logs when state
stalls or fails, especially:

- `auto_labeling_input` for the source-video branch.
- `cosmos_augmentation` and evaluator tasks for generated candidates.
- `auto_labeling_output` for passing generated videos.
- `validated_output` for final artifact checks.
- Service shutdown tasks for cleanup, without confusing successful cleanup with workflow success.

For a requested stop, follow
[Stopping a run](references/troubleshooting.md#stopping-a-run). Use the rest of that reference for
evidence-driven failure routing.

## Verify and report the result

For `<output_directory>/<dag_run_id>/`, verify:

- `cosmos/<video_key>/<augmentation_index>/` contains `output.mp4`, `caption.txt`, and
  `metadata.json` for each successful candidate. Account separately for evaluator-rejected or
  incomplete candidates.
- `auto_labeling_input/` and `training_export/auto_labeling_input/` contain source-branch output.
- `auto_labeling_output/` and `training_export/auto_labeling_output/` contain generated-branch
  output for passing candidates.
- Generated media fully decodes and the observed video/candidate counts agree with the confirmed
  payload and evaluator outcomes.
- Run-owned managed services were removed while external services remained untouched.

Return the terminal state, failing task and concise root cause when applicable, candidate and
evaluator counts, artifact prefixes, service-cleanup state, and any validation gap. State clearly
that the two training-export branches are separate. Download or analyze artifacts only when the
user requests it. When they do, read [outputs.md](references/outputs.md), download only the named
run prefix, and use `scripts/summarize_results.py` before deeper media inspection.

## Boundaries

- Route standalone generation to `cdf-augmentation` and standalone labeling to
  `cdf-auto-labeling` using the pinned references above. The existing VDA DAG has no
  generation-only switch; do not trigger the full graph for a subset request without acceptance.
- Use the `autonomous_vehicle_dataset_generation_dag_k8s` workflow for retrieval-driven CDF-AV processing and
  saved dataset or retrieval manifests.
- Use the event-video-generation workflow for image-to-video anomaly generation from seed images.
- Do not silently turn a run request into a deployment, DAG edit, custom workflow, endpoint
  replacement, or quality-policy change.
