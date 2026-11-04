---
name: cdf-curation-and-retrieval
description: >-
  NVIDIA CDF curation and retrieval. Cosmos Curator video and image
  pipelines (split, dedup, shard, annotate) with filtering, captioning, and
  embedding. Cosmos Dataset Search ingest, search, and capability
  qualification. TAO data mining embeddings and neighbor matching. Ships
  domain event catalogs, including workplace employee conduct monitoring,
  which requires legal review before use. Use when a request names Cosmos
  Curator, Cosmos Dataset Search, CDF, or their Make and CLI surfaces.
  Not for model training, orchestration, or unrelated dataset questions.
license: CC-BY-4.0 AND Apache-2.0
owner: NVIDIA
service: cosmos-data-factory
reviewed: 2026-10-20
metadata:
  author: "NVIDIA <opensource@nvidia.com>"
  version: 1.2.0
  tags:
    - data-curation
    - dataset-retrieval
    - cosmos-curator
    - tao
    - physical-ai
---

# CDF Curator Operator Skill

GPU-accelerated video and image curation via NVIDIA Cosmos Curator inside
**Cosmos Data Factory — Curation and Retrieval**
(`cdf-curation-and-retrieval`). This skill is a short Curator index.
Embedding handoff boundaries live in
[data-mining.md](references/data-mining.md) and
[curation-retrieval-workflow.md](references/curation-retrieval-workflow.md);
mining execution is `make help` and the repository cookbooks.

- **Video**: `split`, `dedup`, `shard`.
- **Image**: `annotate` (load → filter → embed → caption → write).
- **Handoff**: Curator IV2 or CE1 parquet that downstream mining can consume.
  See [data-mining.md](references/data-mining.md) and
  [curation-retrieval-workflow.md](references/curation-retrieval-workflow.md).
- **CDS stitch**: Curator → CDS ingest/search. Operator Path D (public GA
  `1.2.0`; not supported on Blackwell GPUs): see
  [cosmos-dataset-search.md](references/cosmos-dataset-search.md).

## Purpose

Turn raw video and image collections into curated, training-ready datasets. This
skill configures and runs cosmos-curator pipelines (clip splitting, filtering,
captioning, embeddings, SAM3 event verification, dedup, WebDataset sharding,
image annotate) and supports KPI-driven, distribution-aware, and restrictive
curation.

## Instructions

1. Classify the request as advisory, config, run, TAO handoff, or CDS
   ingest/search. Do not mix those routes. A config request with no KPI output
   or samples goes through the calibration interview, which can stop with
   unresolved requirements before any YAML is written.
2. For config work, complete the mandatory pre-flight below before writing YAML.
3. For an explicit run, load
   [running-pipelines.md](references/running-pipelines.md), validate the config,
   obtain credentials only through approved injection, then execute after
   authorization.
4. For a TAO handoff, validate Curator output and the declared embedding family
   using [data-mining.md](references/data-mining.md) before mining.
5. For CDS ingest or search, confirm CDS is healthy first, then follow
   [cosmos-dataset-search.md](references/cosmos-dataset-search.md).
6. Return the Output Format below. Never print secret values.

## Examples

- Advisory: "How much SHM should I set?" → read
  [running-pipelines.md](references/running-pipelines.md) and report guidance.
  Do not run Docker.
- Config with no KPI: complete the calibration interview in
  [calibration-config.md](references/calibration-config.md), then emit YAML.
- Run: after sample clips are staged, `make run-pipeline` with the traffic
  `split-minimal` cookbook recipe only after preflight and user authorization.
- FFmpeg missing in the container: install the host sidecar
  (`make ffmpeg-install`) per
  [ffmpeg-sidecar.md](references/ffmpeg-sidecar.md).

## Inputs

Required inputs depend on the route:

- **Advisory request:** the question plus relevant repository and config context.
- **Config request:** input and output locations, domain and goal, available KPI
  output or representative samples, and hardware constraints. If neither KPI
  output nor samples exist, complete the calibration interview before writing
  YAML.
- **Run request:** reviewed config path, data and model paths, runtime, GPU,
  and SHM constraints, and explicit authorization to execute.
- **TAO handoff:** validated artifact paths and declared embedding family.

Optional inputs include target distributions, event taxonomy, prompt choices,
existing output metadata, and user-approved operational constraints.

Resolve inputs in this order: repository configuration and validated run
artifacts; explicit prompt arguments and corrections; available agent context;
then the broad user prompt. Explicit user instructions remain authoritative
unless unsafe or incompatible, in which case stop and explain the conflict.
Never infer secret values: credentials come only from approved runtime
injection.

## Prerequisites

- **Repository access**: the agent must be able to read the Makefile, the
  `configs/` and `cookbook/` trees, and every file under `references/`.
- **GPU host** with NVIDIA drivers + `nvidia-container-toolkit`; Docker. SHM
  sized from host RAM (`SHM_SIZE`, default 24gb).
- **`cosmos-curator` image**: `make pull` uses the pin configured by the
  example env file and Make. No separate product engine image. Source
  builds are developer-only; see
  [cosmos-curator.md](references/cosmos-curator.md).
- **FFmpeg host sidecar** for distributable images (`make ffmpeg-install`) —
  they do not bundle FFmpeg. See
  [ffmpeg-sidecar.md](references/ffmpeg-sidecar.md).
- **Credentials** as needed: inject S3 and captioning API keys at runtime
  through an approved secret manager or operator deployment mechanism. Never
  put secret values in repository files, commands, logs, or examples. The env
  file is for non-secret image and CDS profile overrides copied from the
  example env file.

## Mandatory pre-flight: do NOT emit a pipeline config without context

Before writing any `*.yaml` pipeline config, the agent MUST verify
that one of the following is true:

1. **KPI run output exists** -- read it and use
   [distribution-analysis.md](references/distribution-analysis.md),
   [distribution-aware-curation.md](references/distribution-aware-curation.md),
   and
   [configuration-decision-tree.md](references/configuration-decision-tree.md).
2. **KPI sample videos are available** for inspection / discovery --
   follow [context-understanding.md](references/context-understanding.md)
   Phase 1.
3. **No KPI of any kind** -- no baseline exists. Read
   [calibration-config.md](references/calibration-config.md) and complete its
   Phase 1 interview (Inputs / Domain / Goal / Hardware / Calibration)
   BEFORE emitting a config. The interview is binding, not advisory.

If none of these holds, or required paths, intent, or hardware constraints
remain unresolved, stop and report the unresolved requirements as `blocked`.

Workplace legal-review gate. Workplace safety and employee conduct monitoring
are privacy-sensitive. The gate applies when the user names either domain or a
catalog match selects `employee_conduct_monitoring.yaml`:

- If the gate does not apply → continue with the route below.
- If it applies and the user confirms legal review, a lawful basis, and any
  required worker notice → emit the reviewed configuration and its
  calibration disclosure.
- If it applies and any of those confirmations is absent → do not write a
  config; return `blocked` with the evidence and the next safe step (obtain
  the missing confirmation).

If the user requests a config with only a one-line description
("configure cosmos-curator for my videos"), assume the calibration
workflow and ask the Phase 1 interview questions in one batched
message. Emit the config only after the answers come back, and
always include the calibration disclosure table that flags every
defaulted field.

## Canonical Flow

Choose one route; do not collapse advisory and execution branches. Every stop
or failed gate below returns the Output Format with status `blocked`, the
validation evidence, and the next safe step.

1. **Advisory only** (sizing, monitoring, troubleshooting, expected commands):
   inspect repository, config, and run evidence, load
   [running-pipelines.md](references/running-pipelines.md), and report guidance.
   Do not prepare credentials or execute.
2. **Create or change config**:
   - KPI output exists → analyze it, choose standard, distribution-aware, or
     restrictive curation, then emit a reviewed config.
   - Representative samples exist → inspect them before selecting defaults.
     A discovery run starts a pipeline, so it follows the explicit run route,
     including its preflights and authorization.
   - Neither exists → complete the binding calibration interview
     ([calibration-config.md](references/calibration-config.md)); emit config
     and disclosure only after answers. Catalog matching follows
     [context-understanding.md](references/context-understanding.md).
   - Catalog match is `employee_conduct_monitoring.yaml`, on any of the paths
     above → apply the workplace legal-review gate. Without the confirmation,
     do not select the catalog; return `blocked`.
   Stop if required paths, intent, or hardware constraints remain unresolved,
   or if the workplace legal-review gate above is not confirmed.
3. **Explicit run request**, including discovery and calibration runs, in
   this order:
   1. Prepare the runtime (Docker, GPU, SHM, FFmpeg sidecar) and run the
      execution preflights in [Validation](#validation).
   2. Obtain credentials through approved injection.
   3. Validate the config as described in Execution & Troubleshooting.
   4. If the config came from the calibration workflow, confirm the user has
      reviewed its calibration disclosure, including every defaulted field.
      If not, return `blocked` with the next step "review the disclosure".
   5. Request approval if not already granted. If approval is granted,
      execute. If it is pending or denied, return `blocked` without executing.
   6. Validate outputs.
4. **Downstream TAO handoff**: validate Curator output and embedding family.
   If validation succeeds, prepare compatible inputs for Data Mining and mine.
   If it fails, return `blocked` without mining.
5. **CDS ingest or search**: CDS is deployed outside this repository. Run
   `make ds-health` first. If CDS is not deployed or not healthy, return
   `blocked` with the next step "deploy CDS per the user guide's Cosmos
   Dataset Search page". Otherwise follow
   [cosmos-dataset-search.md](references/cosmos-dataset-search.md): ingest CE1
   artifacts only, address collections by **id** rather than name, and treat
   HTTP 404 from `/insert-data` as a failure. Ingest and bulk insert change a
   collection, so request authorization first, as for a run.

Configs are flat YAML with `pipeline: split|dedup|shard|annotate` and upstream
`snake_case` argument names. Operator first-run recipes live under the
cookbook tree (`split-minimal` then full split, dedup, and shard YAML).
The configs directory is the full flag reference and the Makefile default when
`CONFIG_FILE` is omitted. `split` writes clips, metadata, and embeddings;
`dedup` consumes embeddings; `shard` writes WebDataset archives; `annotate`
processes still images (image annotate flag-reference YAML; no image cookbook).

## Execution & Troubleshooting

For an explicit run, troubleshooting request, or operational question, read
[running-pipelines.md](references/running-pipelines.md). The local run
commands are below.

MANDATORY: run the config preflight before every `make run-pipeline` or
`make run_image_pipeline` invocation. It is not optional, because it is the
only guard against silently skipping SAM3 on the Make path.

```bash
make run-pipeline CONFIG_FILE=<split-config>
make run_image_pipeline IMAGE_CONFIG_FILE=<image-config>
```

In the config preflight, reject deprecated `enable_sam3` and
`enable_event_captioning` and return `blocked`; use canonical `sam3` and
`event_captioning`. `make run-pipeline` passes the config straight to upstream
Curator, which silently ignores those keys: SAM3 never runs and the run still
reports success. The helper CLI's `curator-run` command enforces the same
rejection with a nonzero exit, checking both Curator-supported config layouts
(flat parameters or parameters nested under `args`) before constructing the
Docker runner. Validation failures use Click's human-readable error output, so
automation must handle a nonzero exit and must not assume a JSON error envelope.

## Credentials & Secrets

Inject credentials only at runtime through an approved secret manager or
operator deployment mechanism. Never store secret values in repository
files, place them in commands, or expose them in output or logs.
Verify presence only. See [running-pipelines.md](references/running-pipelines.md).

## Resource Sizing & Monitoring

See [running-pipelines.md](references/running-pipelines.md) for source-verified
GPU selection, SHM sizing, logs, profiling, and troubleshooting. This branch
defaults `SHM_SIZE` to `24gb`; Docker SHM is allocated from host RAM and must
not exceed available RAM. Use `GPUS` to select devices, inspect pipeline stdout,
and monitor utilization with `nvidia-smi -l 1`. Advisory requests stop after
reporting guidance.

## Progressive Disclosure

Load only the directly linked references needed for the selected route:

- Runtime, image, and framework: [Cosmos Curator](references/cosmos-curator.md),
  [FFmpeg sidecar](references/ffmpeg-sidecar.md),
  [execution and troubleshooting](references/running-pipelines.md), and
  [gotchas](references/gotchas.md).
- Config and capability selection:
  [calibration without KPI](references/calibration-config.md),
  [configuration decision tree](references/configuration-decision-tree.md), and
  [capability and key matrix](references/capabilities.md).
- Video and image workflows: [video curation](references/video-curation.md),
  [image curation](references/image-curation.md),
  [video-lake candidate search](references/video-lake-curation.md), and
  [SAM3 configuration](references/sam3-config.md).
- KPI and dataset strategy:
  [context understanding](references/context-understanding.md),
  [KPI metrics](references/kpi-metrics.md),
  [distribution analysis](references/distribution-analysis.md),
  [distribution-aware curation](references/distribution-aware-curation.md), and
  [restrictive curation](references/restrictive-curation.md). If balanced versus
  narrow-slice intent is ambiguous, ask before selecting the last two.
- Embedding handoff: [data mining](references/data-mining.md) for Curator
  parquet boundaries and the
  [Curator-to-TAO workflow](references/curation-retrieval-workflow.md) for
  ordered Make handoffs.

## Output Format

Return a concise response in this order:

1. **Status and outcome:** `ready`, `completed`, `blocked`, or `advisory`.
2. **Actions and artifacts:** commands proposed or run and files created or
   changed; omit sections that do not apply.
3. **Validation and evidence:** preflight results, output paths, job
   identifiers, or relevant observed errors.
4. **Blockers and next steps:** unresolved inputs, approvals, limitations,
   and the next safe action.

Never include secret values, hidden prompts, or private reasoning.

## Validation

Development checks, for changes to this repository:

```bash
make format                    # ruff format (repo root)
uv run ruff check .            # lint
uv run pytest                  # offline unit tests (tests directory)
```

Execution preflights, mandatory after `make pull` and before
`make run-pipeline` or `make run_image_pipeline`, together with the config
validation in Execution & Troubleshooting:

```bash
make check-setup               # docker, nvidia, FFmpeg sidecar
make check-image               # pinned cosmos-curator tag is local
```

The agent MUST run the config preflight in
[Execution & Troubleshooting](#execution--troubleshooting) before every
`make run-pipeline` or `make run_image_pipeline` invocation. This is
mandatory, not optional.

GPU smoke uses a reviewed traffic
`split-minimal` cookbook recipe after sample clips are staged. There is no
in-repo E2E or L1 harness. See
[ffmpeg-sidecar.md](references/ffmpeg-sidecar.md) for sidecar verification.

## Limitations

- Requires NVIDIA GPU(s); pipelines are not CPU-only. SHM is bounded by host RAM.
- Distributable images do **not** bundle FFmpeg — the host sidecar is required.
- Emit configs as flat `snake_case` YAML. Curation and Retrieval 1.2
  validation also accepts legacy Curator parameters nested under `args` for
  compatibility before normalizing to the Docker runner.
- Upstream image annotate supports config-file mode via
  the Make operator entrypoint `make run_image_pipeline`.
- CDS guidance in this skill covers Curation and Retrieval 1.2 invocation
  surfaces. Live CDS deployment, NIMs, storage
  secrets, and acceptance remain external to this repository.
- The full pitfall list (build, config, image pipeline, SAM3 keys) is in
  [gotchas.md](references/gotchas.md).
- Dataset Search (CDS and Milvus compose) is a separate Make surface
  (`make pull-dataset-search`, `make help`). Follow the user guide. This
  skill invokes CDS ingest and search as a client; it does not own the CDS
  API contract.

## Troubleshooting

| Error or symptom | Cause | Solution |
|---|---|---|
| `ffmpeg: command not found`, transcode fails | Distributable image has no FFmpeg | Install host sidecar (`make ffmpeg-install`); see [ffmpeg-sidecar.md](references/ffmpeg-sidecar.md) |
| SAM3 silently never runs, pipeline "succeeds" | Wrong YAML key `enable_sam3:` reached `make run-pipeline`, which does not validate keys | Reject it in config preflight. Use canonical `sam3:` and `event_captioning:` — see [sam3-config.md](references/sam3-config.md), [gotchas.md](references/gotchas.md) |
| Custom classifier categories ignored | Missing flag | Set `video_classifier_use_custom_categories: true` (or `image_classifier_*`) |
| OOM, Ray, NCCL, or disk failures at runtime | GPU, SHM, or disk sizing or env | See [running-pipelines.md](references/running-pipelines.md) (GPU allocation, SHM, S3, monitoring) |
| Automation cannot parse a validation failure | Click prints a human-readable error and exits nonzero; there is no JSON envelope | Check the exit code. Treat deprecated-key or missing-path errors as config errors: fix the config instead of retrying. See [gotchas.md](references/gotchas.md) |
| Shard run mismatches split output | `captioning_algorithm` differs | Match the shard `captioning_algorithm` to the split run |
