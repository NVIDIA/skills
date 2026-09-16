---
name: tao-data-io
description: The data-mover for TAO jobs — decides the storage tier (A pre-positioned mount with zero fetch / B volume-from-S3 / C ephemeral in-compute fetch), stages inputs (bulk + annotation-selective + archive extract + HF/NGC PTM), maps credentials to env, routes outputs 3-way with upload-excludes, and runs the compute-frame verify gate. A support skill other platform skills (docker, kubernetes, slurm, brev, virtualenv) call to get data to and from the compute container without the TAO SDK. Trigger phrases include "stage inputs", "mount the dataset", "upload TAO results", "download only referenced files", "resolve results_dir", "verify the container can read the data".
license: Apache-2.0
compatibility: Requires aws CLI or s5cmd on the staging host, plus Python 3.10+ with boto3 and pandas/pyarrow for annotation-selective download. No nvidia-tao-sdk, no fsspec/s3fs. Credentials are read from the process environment, whether exported in the user's shell or sourced from a user-approved env file.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- platform
- storage
---

# tao-data-io

Get data to and from the compute container. Decide the storage tier first —
under **strategy A (pre-positioned mount) no bytes move at all** — and when a
fetch is needed, move it **host-side** with `aws`/`s5cmd`/`boto3`/`huggingface-cli`/`ngc`
directly — no `nvidia-tao-sdk`, no in-container runtime. Other platform skills
call this skill to stage inputs before launch and sync outputs after. It never
launches a container itself. The chosen tier is stamped into the job-record at
submit.

**When NOT to invoke this skill:** if the inputs are already readable from the
compute frame (a local path on the execution host, an existing Lustre/PVC/bind
mount), that IS tier A — record it and skip this skill entirely; there is
nothing to move. **Air-gapped hosts:** tier A is the *only* tier — never
attempt an S3/HF/NGC fetch; anything missing (datasets, checkpoints, and the
container images themselves) must be pre-positioned by the operator, and the
preflight's readability check is the only data step that runs.

## Credentials (env vars; values never written to disk by this skill)

S3 credentials use the **officially documented AWS env vars**, read from the
session environment: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and (for
S3-compatible stores) `AWS_ENDPOINT_URL`, `AWS_DEFAULT_REGION`. The `aws`
CLI and `boto3` pick the variables up natively — **never** run `aws configure`
and **never** write `~/.aws/credentials`:

```bash
set -a; source /path/to/.env; set +a   # omit if already exported
aws s3 ls "s3://$S3_BUCKET_NAME/..."   # reads AWS_* from the environment
```

If a session provides only the legacy TAO names (`ACCESS_KEY`, `SECRET_KEY`,
`S3_ENDPOINT_URL`, `CLOUD_REGION`), map them once, scoped to the command:
`AWS_ACCESS_KEY_ID="$ACCESS_KEY" AWS_SECRET_ACCESS_KEY="$SECRET_KEY" aws s3 ...`

`HF_TOKEN` / `NGC_KEY` pass through unchanged for PTM pulls. Never pass a
credential as a CLI argument (`-p`, `--token`, `-e KEY=value`); use
`--password-stdin` or `-e VAR` (no value).

## Storage strategy (agent decides; the gate verifies)

Pick per backend from what the cluster/daemon actually offers:

- **A — pre-positioned mount** (PVC / NFS / Lustre / bind mount already holding the data): mount it, author the *mount paths* into the spec. **No S3 fetch.** Also the air-gap answer.
- **B — volume populated from S3**: an initContainer / stage step fills a durable volume; compute mounts it; a final step drains it back to S3 if needed.
- **C — ephemeral + in-compute fetch**: no persistent mount — fetch into the compute container and upload out at the end (today's K8s default).

## Verify-before-launch gate (the one invariant)

> The path the spec references is readable **from the compute frame** (not the
> launcher's), and the output destination **persists after the container exits**.

Probe in the compute's frame of reference (in-container `aws s3 cp`/`touch` on
the resolved `results_dir`, or a `kubectl run`/`srun` probe) — a green `aws s3
ls` on the launcher is **not** proof the pod can read the data (managed
backends inject different creds into the compute container).

## Input staging

- **Bulk folder:** `s5cmd cp 's3://.../*' <stage>` or `aws s3 sync`.
- **Single file:** `aws s3 cp`.
- **Annotation-selective** (download only files referenced by an annotation): use `references/selective_download.py` (below).
- **Archive:** `tar -xzf X -C <dir> --strip-components=1` guarded by a `.extracted` marker (idempotent).
- **PTM (`ngc://` / `hf://`):** `huggingface-cli download` / `ngc registry model download-version`, then author the local path.

After staging, author the spec with **local paths** and run the verify gate.

## Output routing (3-way) + upload

- `TAO_RESULTS_ROOT` set → write to that mount, **no upload**.
- else `S3_BUCKET_NAME` set → upload to `s3://$S3_BUCKET_NAME/results/$TAO_JOB_ID/`.
- else → **loud ephemeral warning**.

SLURM: never set `S3_BUCKET_NAME` (Lustre-only); run any upload on the login
node, not inside the GPU allocation. Upload with excludes:
`aws s3 sync <local>/ s3://... --exclude '.tao/*' <upload_excludes...>`.

## Quick Start — annotation-selective staging

Download only the files an annotation references (e.g. the `video` column),
preserving relative paths, into a local staging dir:

```bash
set -a; source /path/to/.env; set +a   # omit if already exported
python references/selective_download.py \
  --annotation /path/to/annotation.parquet \
  --key video \
  --bucket "$S3_BUCKET_NAME" --src-prefix datasets/clips \
  --dest /data/stage/clips
```

`--key` is repeatable or comma-separated; `--format` overrides extension
inference (`parquet`/`jsonl`/`json`/`csv`).

## Helpers

- `references/selective_download.py` — annotation-driven selective download (boto3 + pandas). Unit tests live in `references/tests/`; run with `python -m pytest`.
