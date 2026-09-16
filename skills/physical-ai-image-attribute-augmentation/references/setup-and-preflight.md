# Setup and preflight

## Collect from the user before doing anything else

Ask the user for every value in this table before building a payload or triggering a run. The
repository's checked-in dev and CI payloads contain deployment-specific endpoint URLs and S3 bucket
paths that must **never** be used for user runs.

| # | Value | Notes |
|---|---|---|
| 1 | **Input path** | `s3://` (or HTTP/HTTPS) URL whose immediate subdirectories are person-ID folders |
| 2 | **Output directory** | Writable `s3://` URL where the DAG will write results |
| 3 | **Service mode** | `external` or `internal` — see SKILL.md for GPU implications |
| 4 | **VLM endpoint URL** | External mode only; HTTPS URL for the VLM inference service |
| 5 | **LLM endpoint URL** | External mode only; HTTPS URL for the LLM inference service |
| 6 | **Image-edit endpoint URL** | External mode only; HTTPS URL for the image-edit inference service |
| 7 | **`max_imgs`** | Optional; positive = limit person-ID count, 0/negative = process all |
| 8 | **`num_augmentation`** | Optional; clothing variants per person, default 1 |

If any required value (1–6) is not provided, ask for it. Do not substitute a default, guess, or
reuse a value from a previous run without the user's explicit confirmation.

## Repository layout

This skill lives at `skills/physical-ai-image-attribute-augmentation/` and is surfaced to agents
through the `.agents/skills`, `.claude/skills`, and `.cursor/skills` symlinks at the repository
root. It is self-contained: it does not require the repository `airflow/` directories at runtime.

Export credentials in the parent shell before launching the agent. Never paste secret values into
the agent prompt or store them in the skill.

## Required environment

The controller deploys Airflow. `make port-forward` exposes Airflow at `http://localhost:8080`
(bound to `0.0.0.0`). Trigger and monitor runs through the Airflow API — see
[airflow-direct-api.md](airflow-direct-api.md).

- `HF_TOKEN`: HuggingFace token with read access to the deployed model repos. Required only for
  internal service mode (in-cluster VLM/LLM/image-edit). It is pre-configured during environment
  setup (`make setup`) and is not read or loaded by anything in this skill.
- `UPLOAD_DESTINATION`: writable `s3://bucket/prefix/`, required only for local uploads.
- Standard AWS credential chain: required only for local uploads and result downloads.
- `AWS_REGION` or `AWS_DEFAULT_REGION`: optional; when unset the uploader falls back to the
  standard AWS configuration chain to resolve the region.
- `WEBSERVER_ENDPOINT` and `NGC_API_KEY`: required only by `scripts/workflow.py`, which drives the
  webserver API instead of Airflow. Not needed for the standard Airflow-direct workflow.

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

Require this exact directory depth:

```text
dataset-root/
├── person-001/
│   ├── front.jpg
│   └── side.png
└── person-002/
    └── view-01.jpeg
```

Each immediate subdirectory is one person ID. Image filenames are not constrained — the DAG combines
every image in a person directory into a single horizontal strip. The bundled validator rejects
root-level files, deeper directories, symlinks, empty images, unsupported extensions, and
extension/signature mismatches. Hidden entries are ignored. Upload preserves each
`person_id/<image>` relative path.

## Agent permissions

The agent needs scoped outbound access to the Airflow API host and, for local uploads or result
downloads, the relevant S3 hosts. Approve only the concrete API and storage operations required for
the requested run. Do not work around a sandbox denial by revealing credentials or copying data to
another endpoint.

## Safe preflight

```bash
python scripts/upload_images.py --path /path/to/crops --validate-only
python scripts/payload.py validate --payload /tmp/iaa-payload.json
```

For Airflow-level preflight (DAG loaded, pools have slots, pods healthy), follow
[airflow-direct-api.md#preflight-direct-path](airflow-direct-api.md#preflight-direct-path).
