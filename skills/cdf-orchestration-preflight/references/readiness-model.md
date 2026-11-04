# Readiness model

## Status semantics

Keep installation and run readiness separate:

- `READY`: all required evidence for this gate or mode was observed.
- `PARTIAL`: at least one packaged DAG is ready and at least one is blocked or unverified.
- `BLOCKED`: a known missing or incompatible prerequisite prevents the operation.
- `UNVERIFIED`: the prerequisite may exist, but the audit could not prove it safely.
- `NOT_PACKAGED`: the code may support a DAG variant, but its required manifest is absent.

For a DAG with multiple modes, the DAG is runnable when at least one mode is `READY`. Still report
every unavailable internal, external, Kubernetes, or OSMO mode. Overall run readiness is `READY`
only when every packaged DAG has a runnable mode.

## Expected DAG catalog

The machine-readable source is `config/readiness-profiles.json`. It currently covers:

| Platform | Packaged DAGs |
| --- | --- |
| Kubernetes | `image_attribute_augmentation_dag_k8s`, `event_video_generation_dag_k8s`, `video_data_augmentation_dag_k8s`, `defect_image_generation_smt_dag_k8s`, `autonomous_vehicle_dataset_generation_dag_k8s` |
| OSMO | `event_video_generation_dag_osmo`, `video_data_augmentation_dag_osmo` |
| Operational | `k8s_resource_cleanup_dag` |

`image_attribute_augmentation_dag_osmo` has a code path but no checked-in manifest, so it is not a
packaged DAG. If Airflow exposes a DAG not present in the catalog, report it as
`UNPROFILED/UNVERIFIED` and inspect its manifest or DAG before adding a profile.

## Evidence by layer

### Installation

- Required repository files and supported local tool versions.
- A single user-supplied kubeconfig with a non-loopback API endpoint that is reachable from an
  existing ready NVIDIA GPU Operator toolkit pod, or the first explicit `--probe-pod` container
  using Python 3 sockets. Explicit containers must be ready and non-hostNetwork. Probes do not
  create cluster resources; missing or inconclusive probe capability remains `UNVERIFIED`.
- A supported Kubernetes distribution; K3s is currently unsupported by this repository.
- Ready nodes, required authorization, and an NVIDIA device resource for GPU workflows.
- `libnvcuvid.so.1` on every GPU node, verified through existing GPU Operator toolkit pods without
  creating diagnostic workloads. Explicit `--probe-pod` GPU containers instead test library loading
  on their nodes; every GPU node needs coverage. Missing inspection coverage remains `UNVERIFIED` for workflow
  modes whose readiness profiles require GPU video decode.
- StorageClasses referenced by PostgreSQL, Airflow logs, and model cache.
- Required secret names populated without exposing their values.
- Either a configured Airflow image repository/tag that the user can push for a local build, or an
  explicit prebuilt controller image whose manifest is accessible with current Docker
  authentication. The prebuilt path does not require Docker daemon or registry push access.

Build image settings come from `AIRFLOW_IMAGE_REPOSITORY`/`AIRFLOW_IMAGE_TAG` in `make/common.mk`;
exported variables of the same names take precedence. The checked-in repository value is a
placeholder and is always `BLOCKED`. Push access becomes `READY` only with an exact matching `--confirmed-push-repository`;
the report labels this as user confirmation, not a performed push. Never infer permission from
login or manifest reads. S3 access and model-license acceptance still need separate evidence.

### Live controller

When `AIRFLOW_URL` and `AIRFLOW_TOKEN` are available, verify the DAG list, paused state, import
errors, required pools, and positive pool capacity. Also inspect Helm state, controller pods, and
PVC binding through Kubernetes. A healthy UI alone does not prove workflow readiness.

### Workflow modes

Use the profile catalog's absolute GPU floor to decide whether work can make progress and its
practical GPU target to warn about likely serialization or startup timeouts. The AV
profiles distinguish retrieval (one GPU) from generation (four GPUs, including
physical-plausibility evaluation); both use workflow-managed services. Calculate free GPU
capacity from allocatable GPUs minus active pod requests; total allocatable capacity is not free
capacity.

A mode's `required_pvcs` lists operator-provisioned claims in `sdg-workflow` that its tasks mount
but the Helm chart does not create. Each entry has a `name` and a `remediation` that the report
appends to the blocker. A claim that is absent or not `Bound` blocks the mode, and uninspected PVCs
leave it `UNVERIFIED`. The AV modes require `shared-ffmpeg`, the FFmpeg sidecar that
`run_curator` and the media probe pods mount read-only at `/opt/ffmpeg`. Its remediation points to
the requirements in
[the AV prerequisites](../../../docs/autonomous-vehicle-dataset-generation/prerequisites.md#shared-ffmpeg-installation). The audit
checks binding only; the workflow's startup check verifies the claim holds `ffmpeg`, `ffprobe` and
an H.264 decoder.

External modes require real endpoints reachable from the task network. OSMO modes additionally
require the OSMO token, URL and compute pool settings, matching Airflow pool, and external
inference/evaluator endpoints. The
generic audit cannot prove payload-specific endpoints, so those modes remain `UNVERIFIED` until an
endpoint-aware check supplies that evidence.

The cleanup DAG requires a healthy controller, `default_pool`, and Kubernetes cleanup RBAC but no
GPU. It must be included in the report even though it is operational rather than data-generating.
