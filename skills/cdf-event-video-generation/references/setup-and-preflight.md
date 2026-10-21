# Setup and preflight

## Collect from the user before doing anything else

Ask the user for every value in this table before building a payload or triggering a run. Endpoint
URLs and S3 bucket paths are deployment-specific: use only the ones the user supplied, and
**never** invent one.

| # | Value | Notes |
|---|---|---|
| 1 | **Input path** | `s3://` (or HTTP/HTTPS) URL to a single seed image or a directory of seed images |
| 2 | **Output directory** | Writable `s3://` URL where the DAG will write results |
| 3 | **Service mode** | `external` or `internal` — see SKILL.md for GPU implications |
| 4 | **VLM endpoint URL** | External mode only; HTTPS URL for the VLM inference service |
| 5 | **LLM endpoint URL** | External mode only; HTTPS URL for the LLM inference service |
| 6 | **Image2video endpoint URL** | External mode only; HTTPS URL for the Cosmos3 image-to-video inference service |
| 7 | **`max_images`** | Optional; positive = limit images taken from a sorted directory listing, 0/negative = process all |
| 8 | **`num_augmentation`** | Optional; generated videos per image, default 1 |

If a required value is missing (1–3 always; 4–6 in external mode), ask for it. Do not substitute
a default or guess. Values the user explicitly supplied earlier, including for a previous run,
may be reused when clearly applicable to the current target and request. Ask for clarification
when the target or context is ambiguous.

## Repository layout

This skill lives at `skills/cdf-event-video-generation/` and is surfaced to agents
through the `.agents/skills`, `.claude/skills`, and `.cursor/skills` symlinks at the repository
root. It is self-contained: it does not require the repository `airflow/` directories at runtime.

Export credentials in the parent shell before launching the agent. Never paste secret values into
the agent prompt or store them in the skill.

## Required environment

The controller deploys Airflow. `make port-forward` exposes Airflow at `http://localhost:8080`
(bound to `0.0.0.0`). Trigger and monitor runs through the Airflow API — see
[airflow-direct-api.md](airflow-direct-api.md).

- `HF_TOKEN`: HuggingFace token with read access to the deployed model repos. Required only for
  internal service mode (in-cluster VLM/LLM/image2video). It is pre-configured during environment
  setup (`make setup`) and is not read or loaded by anything in this skill.
- `UPLOAD_DESTINATION`: writable `s3://bucket/prefix/`, required only for local uploads.
- Standard AWS credential chain: required only for local uploads and result downloads.
- `AWS_REGION` or `AWS_DEFAULT_REGION`: optional; when unset the uploader falls back to the
  standard AWS configuration chain to resolve the region.

The upload command imports `boto3`. Other bundled scripts use only the Python standard library.

## Cluster access

The cluster is reached only through a credential file the user supplies. It carries a cluster
address and admin credentials, so it is never part of the repository, never logged, and never read
from a default location without the user naming it. The agent must:

1. Check whether the cluster credential path is already set in the environment.
2. If not, ask the user for the absolute path before running any cluster command.
3. Export it as the active cluster context for all subsequent `kubectl` calls.

Never guess a path, assume a repository-relative location, or silently fall back to any on-disk
default. All `kubectl` and readiness commands use the path the user supplied unless another is
explicitly configured.

## Dataset preflight

There is no person-ID subdirectory convention here — this workflow's input is an image, not a
video, and every accepted file is treated as one independent input:

```text
dataset-root/
├── warehouse-entrance.jpg
├── loading-dock.png
└── aisle-3-camera.jpeg
```

Accepted extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tiff`, `.webp`. `input_path` may
instead name one image file directly, in which case `max_images` has no effect. When it names a
directory, the DAG lists matching files, sorts them, and takes the first `max_images` (default 10;
0 or negative processes all). The bundled validator rejects nested directories, symlinks, empty
files, unsupported extensions, and extension/signature mismatches for JPEG and PNG. Hidden entries
are ignored.

## GPU requirements

Derive these from the manifest (resolve each component's `deployment_profile` against
`deployment.profiles`; a profile with `configuration.gpu: true` claims `gpu_count` GPUs, default
1). The figures below are what the currently checked-in K8s manifest resolves to — re-derive them
whenever the manifest or the pool names change, since both are deployment-specific.

- There is **one** deployed inference VLM, not a separate VLM and LLM: `service_lifecycle` has only
  `vlm_service` and `image2video_service`, and in internal mode the payload model points
  `llm_service_url` at `vlm_service_url`. `vlm_service` claims **one** GPU per replica from the
  manifest's GPU pool (`k8s_rtx6000_1` today).
- Image2video claims **two** GPUs per replica (`gpu_count: 2`, `host_ipc: true` in
  `event_video_generation_k8s_manifest.yaml`'s `k8s_gpu_endpoint_2` profile) — it deploys
  `nvidia/Cosmos3-Super-Image2Video` with `--cfg-parallel-size 2 --use-hsdp --hsdp-shard-size 2`.
- Internal mode needs **three** GPUs for the inference endpoints (1 `vlm_service` + 2
  `image2video_service`). Raising any `replicas` value increases this proportionally.
- In **both** modes, the default Cosmos Evaluator checks add **two** in-cluster GPUs
  (`cosmos_evaluator_attribute_verification` and `cosmos_evaluator_physical_plausibility`;
  postgres/database/arbitrator are CPU-only) unless the payload supplies
  `cosmos.cosmos_evaluator_service_url` or disables the checks.
- Batch task pods need **one** more GPU in either mode: `detection_and_tracking`, `captioning` and
  `visual_qa` run on `k8s_gpu_task` and **`augmentation` runs on the `augmentation_task` profile,
  which is also `gpu: true`**, but the stages run one at a time. Only
  `event_and_person_attribute_search` is CPU-only (`k8s_cpu_task`). Note this differs from IAA,
  where the augmentation task *is* CPU-profiled.
- Floors on the default payload: **six** GPUs internal and **three** external, or four and one
  without managed checkers.

## Agent permissions

The agent needs scoped outbound access to the Airflow API host and, for local uploads or result
downloads, the relevant S3 hosts. Approve only the concrete API and storage operations required for
the requested run. Do not work around a sandbox denial by revealing credentials or copying data to
another endpoint.

## Safe preflight

```bash
python scripts/upload_images.py --path /path/to/seed-images --validate-only
python scripts/payload.py validate --payload /tmp/evg-payload.json
```

For Airflow-level preflight (DAG loaded, pools have slots, pods healthy), follow
[airflow-direct-api.md#preflight-direct-path](airflow-direct-api.md#preflight-direct-path).

## OSMO capacity

Use the selected manifest's `deployment.profiles` to distinguish the Airflow concurrency pool
(`osmo_rtx6000_1` currently) from the OSMO compute pool, which the `osmo_pool` Airflow variable
sets from `OSMO_POOL`. With the configured OSMO CLI, check `osmo pool list` and
`osmo resource list --pool <osmo-pool> --mode free`. Require available quota and an
eligible node with the task's requested GPU, CPU, memory and storage. Airflow open slots alone
do not establish OSMO compute availability.
