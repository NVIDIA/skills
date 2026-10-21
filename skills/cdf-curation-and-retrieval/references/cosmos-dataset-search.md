# Cosmos Dataset Search

Task-oriented reference for public GA CDS bring-up and CE1 ingest/search.
Make is the local operator façade; agents and orchestrators should invoke the equivalent
installed `cdf_curation_and_retrieval` CLI.

In Curation and Retrieval 1.2, the adapter, CLI, Make targets, and handoff
docs provide invocation coverage for the GA CDS core. This is not a
substitute for live CDS deployment acceptance.

## Hardware constraint

CDS does not run on Blackwell (B-series) GPUs. The published Cosmos-Embed1
container has no kernels for compute capability 10.0 (`sm_100`: B100, B200,
GB200) or 12.0 (`sm_120`: RTX PRO 6000 Blackwell); the CDS API does not use
the GPU. Select a Hopper, Ada, or
Ampere class GPU with 16GB+ VRAM for Path D. Curator and TAO are
unaffected.

## Select the image

`CDS_PROFILE=public` uses the public GA pin
`nvcr.io/nvidia/blueprint/cosmos-dataset-search:1.2.0`.

Make loads the committed example environment file first, then the
gitignored operator overlay (the overlay wins). The default
`CDS_PROFILE=public` then assigns the GA pins, so image pins in those files
apply only when one of them sets another `CDS_PROFILE`. Shell prefixes such as
`CDS_PROFILE=public make ...` do **not** override a profile set there. Pass
Make arguments: `make pull-dataset-search CDS_PROFILE=public
DATASET_SEARCH_TAG=1.2.0 DATASET_SEARCH_REGISTRY=nvcr.io/nvidia/blueprint/cosmos-dataset-search`.

Operator Path D (compose, wait for health, collection **id**, lab ingest):
`docs/user-guide/operations-cds.md` and repository `README.md` Path D.
CDS deployed elsewhere (cluster, shared lab host): this repository does not
deploy CDS outside compose. Clone CDS from
`https://github.com/NVIDIA-Omniverse-blueprints/cosmos-dataset-search.git`
and deploy it by whichever route that repository documents for the target. CDS documents AWS EKS; it
claims generic-Kubernetes charts but ships no guide for them, so do not
promise a generic-cluster recipe. Then treat CDS as a prerequisite endpoint:
`export CDS_URL=<that API>` and run `make ds-health` before any other Path D
target. On Kubernetes that URL ends in `/api`, because CDS sets
`FASTAPI_ROOT_PATH=/api` and its ingress routes `/api/` without rewriting;
compose has no prefix. Never offer a Helm install from this repo.
Public GA uses `CDS_PROFILE=public` and image `1.2.0`. Do not
skip `make up-dataset-search` on a laptop. Do not call `ds-health` until
`curl -sf http://localhost:8888/health` succeeds.

```bash
# Public GA — Make arguments beat the gitignored operator overlay
make pull-dataset-search \
  CDS_PROFILE=public \
  DATASET_SEARCH_REGISTRY=nvcr.io/nvidia/blueprint/cosmos-dataset-search \
  DATASET_SEARCH_TAG=1.2.0
make up-dataset-search \
  CDS_PROFILE=public \
  DATASET_SEARCH_REGISTRY=nvcr.io/nvidia/blueprint/cosmos-dataset-search \
  DATASET_SEARCH_TAG=1.2.0

```

Search needs a Cosmos-Embed1 service on host port 9000 to embed the text
query. `CDS_PROFILE=public` starts it: `up-dataset-search` adds the embed
overlay and the API reaches it via `COSMOS_EMBED_URI`.
The image cannot download its own weights: set `COSMOS_EMBED_MODELS_DIR`
to a snapshot of `nvidia/Cosmos-Embed1-224p` at revision
`787e0b996f5260a71ad474a283c90539a2e12986`. Any other revision fails the
container's checksum check.

## Verify core CDS operations

Public GA pipeline id is `cosmos_video_search_milvus`. Confirm
`"enabled": true` with `curl -sS http://localhost:8888/v1/pipelines`
(`make ds-pipelines` lists names only).
Create-collection returns an **id**; ingest and search must use that id,
not `NAME=traffic`.

```bash
make ds-health
make ds-pipelines
make ds-collections
make ds-create-collection \
  NAME=traffic \
  PIPELINE=cosmos_video_search_milvus
make ds-get-collection COLLECTION=<id-from-create>
make search COLLECTION=<id-from-create> QUERY='pedestrian crossing the road'
```

## CE1 artifact policy

CDS-bound Curator ingest accepts CE1 only:

- Use `EMBEDDING_BACKEND=ce1`, or `auto` only when CDS-bound ingest selects CE1.
- Declare `CDS_EMBEDDING_FAMILY=ce1` for bulk insert.
- Curator IV2 remains valid for TMM, not CDS ingest.
- Vectors do not identify their producer; the family declaration is mandatory.

## Public GA bulk-ingest workflow

Follow this order:

1. Confirm CDS is healthy.
2. Convert and validate a Curator CE1 export.
3. Stage the validated parquet in CDS-readable object storage.
4. Create the collection and keep the returned **id**.
5. Submit one non-retried `/insert-data` request.
6. Poll the returned job to a successful terminal state.
7. Search only after successful completion.

Use pipeline `cosmos_video_search_milvus` on public 1.2.0.

```bash
make ds-health

make ingest-curator \
  CURATOR_DIR=/path/to/curator/ce1-output \
  OUT=$PWD/data/cds-ce1.parquet \
  EMBEDDING_BACKEND=ce1

make stage-cds-artifact \
  STAGE_SOURCE=$PWD/data/cds-ce1.parquet \
  STAGE_DESTINATION=s3://bucket/cds-ce1.parquet \
  OBJECT_STORE_ENDPOINT=https://objects.example

make ds-create-collection \
  NAME=traffic \
  PIPELINE=cosmos_video_search_milvus

make ds-bulk-insert \
  COLLECTION=<id-from-create> \
  PARQUET_URI=s3://bucket/cds-ce1.parquet \
  CDS_EMBEDDING_FAMILY=ce1

make ds-job-status JOB_ID=<job-id> WAIT=1
make search COLLECTION=<id-from-create> QUERY='pedestrian crossing the road'
```

`ingest-curator` in convert-only mode validates the export while creating the
CDS-shaped parquet. Capture `job_id` from the bulk-insert JSON result. A waited
job that times out or reaches failed/cancelled/error exits nonzero.

### Fail closed on `/insert-data` 404

HTTP 404 from `POST /insert-data` fails closed by default. On local compose
the body is often `Collection not found` (name used as id) or `File not
found` (host path). That is not a missing route.

Lab document ingest (`ALLOW_LAB_DOCUMENT_FALLBACK=1`) with the collection
**id** is the Path D smoke path in `docs/user-guide/operations-cds.md`.
A fallback result is never evidence of GA bulk-ingest success. Agents
must not present lab ingest as GA `/insert-data` qualification.

## Secure object-store staging

- Use an absolute HTTPS endpoint by default.
- Permit HTTP only in an isolated lab with
  `ALLOW_LAB_HTTP_ENDPOINT=1`.
- Pass credential environment-variable **names** through
  `OBJECT_STORE_ACCESS_KEY_ENV` and `OBJECT_STORE_SECRET_KEY_ENV`.
- Never put credential values in command arguments, docs, or logs.
- Both referenced variables must be set together. Staging failures redact
  worker output.

Example using operator-owned environment names:

```bash
make stage-cds-artifact \
  STAGE_SOURCE=$PWD/data/cds-ce1.parquet \
  STAGE_DESTINATION=s3://bucket/cds-ce1.parquet \
  OBJECT_STORE_ENDPOINT=https://objects.example \
  OBJECT_STORE_ACCESS_KEY_ENV=CDF_S3_ACCESS_KEY \
  OBJECT_STORE_SECRET_KEY_ENV=CDF_S3_SECRET_KEY
```

## Machine-readable automation

Prefer the installed `cdf_curation_and_retrieval` CLI for automation; Make
may add operator-oriented output.

- CDS and integration success payloads are JSON.
- `ingest-curator --ingest` emits one JSON object per completed phase.
- Waited job failures emit JSON before exiting nonzero.
- Click usage/parsing and direct `ClickException` paths may emit
  human-readable stderr.

Treat every nonzero exit as failure. Do not infer success from partial output,
and never retry POST submissions automatically.
