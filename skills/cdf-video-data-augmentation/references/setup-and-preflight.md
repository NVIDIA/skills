# Setup and preflight

## Required run inputs

Collect these values before building a payload:

| Value | Requirement |
|---|---|
| Input | Accessible `s3://`, HTTP, or HTTPS path to one video or a directory of videos |
| Output | Writable `s3://` prefix |
| Platform | Kubernetes or OSMO |
| Service mode | Kubernetes: internal or external; OSMO: external only |
| Input limit | Positive `max_videos` for a bound; `null`, zero, or negative for all videos |
| Variant count | `cosmos.num_augmentation`, at least one |
| Distribution | Explicit video-appropriate `cosmos.variable_distribution` |
| Evaluator policy | Enabled checks and managed/external placement |
| External endpoints | VLM, LLM, Cosmos Transfer, and evaluator when required |

Never invent an endpoint URL or bucket path; use only the ones the user supplied. They are
deployment-specific and may be stale even when they pass schema validation.

## Local input preparation

VDA accepts `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm`. A local directory must be flat: every
supported file directly below the directory is an independent input video.

```bash
python3 scripts/upload_videos.py --path /path/to/videos --validate-only

export UPLOAD_DESTINATION="s3://bucket/prefix"
python3 scripts/upload_videos.py \
  --path /path/to/videos \
  --destination-path video-data-augmentation/my-run
```

The uploader uses the standard AWS credential chain. Never paste credentials into prompts or
store them in the skill.

## Cluster and controller checks

Use the cluster context explicitly configured for the installation. Confirm:

- Controller pods with release label `sdg-workflow-controller` are Running.
- The target DAG is loaded, unpaused, and has no relevant import error.
- Required PVCs are Bound.
- `kubernetes_remote` is present in the scheduler environment for Kubernetes runs.
- `default_pool` and the selected platform pool have open slots: `k8s_rtx6000_1` for
  Kubernetes or `osmo_rtx6000_1` for OSMO. External inference also uses
  `cosmos_transfer_3_service_pool`; OSMO does not require the Kubernetes GPU pool.
- For OSMO, the OSMO credentials are ready.
- Input listing and output write access work with the controller's storage configuration.
- External endpoints are reachable from the task network, not only from the operator's laptop.

Read [airflow-direct-api.md](airflow-direct-api.md) for concrete API checks.

## Minimum compute

The Kubernetes manifest puts VDA batch components on `k8s_gpu_task`; GPU use is not zero in
external-service mode. The source and generated auto-labeling branches include super resolution,
detection/tracking, captioning, visual QA, reasoning, and training export, while augmentation also
uses a GPU task profile.

Internal mode additionally deploys:

- One shared Qwen language endpoint on one GPU per `vlm_service` replica. The payload retains an
  `llm_service` lifecycle entry, but the checked-in Kubernetes manifest aliases the VLM endpoint
  for the LLM role rather than deploying a separate LLM component.
- Cosmos3-Nano Transfer on two GPUs per replica.
- One GPU for each enabled managed checker: hallucination, attribute verification, and physical
  plausibility. The evaluator database and arbitrator are CPU-only.

Let `C` be the number of enabled managed checkers, from zero to three. Use `C=0` when an external
evaluator is supplied or all checks are disabled.

| Platform and mode | Resident endpoint GPUs | Runnable batch GPUs | Absolute floor | Practical minimum |
|---|---:|---:|---:|---:|
| Kubernetes, external inference | `C` | 1 | `C + 1` | `C + 2` |
| Kubernetes, internal inference | `3 + C` | 1 | `C + 4` | `C + 5` |
| OSMO, external endpoints | 0 in OSMO | One 1-GPU slot | One 1-GPU slot | Two 1-GPU slots |

The **absolute floor** is the minimum capacity that leaves one GPU for a batch task after resident
endpoints start. The **practical minimum** leaves two batch GPUs because the source-labeling branch
and Cosmos/output branch can overlap. With only the absolute floor, work may serialize and parallel
pods can remain pending long enough to hit startup or DAG timeouts.

Concrete Kubernetes examples:

| Configuration | Absolute floor | Practical minimum |
|---|---:|---:|
| External inference with an external evaluator or all checks disabled | 1 GPU | 2 GPUs |
| External inference with one, two, or three managed checkers | 2, 3, or 4 GPUs | 3, 4, or 5 GPUs |
| Internal inference with an external evaluator or all checks disabled | 4 GPUs | 5 GPUs |
| Internal inference with one, two, or three managed checkers | 5, 6, or 7 GPUs | 6, 7, or 8 GPUs |

These numbers assume one shared language-model replica and one Cosmos Transfer replica. Each
additional language replica adds one resident GPU; each additional Cosmos Transfer replica adds
two. Each enabled managed checker adds one resident GPU. Higher mapped-task concurrency requires
one GPU per additional simultaneously running batch pod. The Kubernetes manifest targets RTX PRO
6000 nodes but does not declare a numeric CPU or RAM minimum, so do not invent one; also verify
model memory, PVC/cache headroom, and pod placement on the live cluster.

Each OSMO task requests **1 GPU, 11 CPUs, 100 GiB RAM, and 100 GiB storage**. One slot can serialize
the graph; two slots cover the workflow's normal two-branch overlap. All inference and evaluator
endpoints remain external to OSMO and need their own separately provisioned capacity.

Check actual free GPUs, running consumers, node placement, and Airflow pool concurrency rather than
relying only on allocatable totals.

## Safe local checks

```bash
python3 scripts/upload_videos.py --path /path/to/videos --validate-only
python3 scripts/payload.py validate --platform k8s --payload /tmp/vda-payload.json
```

These commands do not trigger a workflow. Deployment, reinstall, pool resizing, and cleanup of
existing resources require explicit user authorization.

## OSMO capacity

Use the selected manifest's `deployment.profiles` to distinguish the Airflow concurrency pool
(`osmo_rtx6000_1` currently) from the OSMO compute pool, which the `osmo_pool` Airflow variable
sets from `OSMO_POOL`. With the configured OSMO CLI, check `osmo pool list` and
`osmo resource list --pool <osmo-pool> --mode free`. Require available quota and an
eligible node with the task's requested GPU, CPU, memory and storage. Airflow open slots alone
do not establish OSMO compute availability.
