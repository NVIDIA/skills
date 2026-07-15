---
name: tao-mine-aoi-images
description: Runs the DEFT embed-then-mine workflow for VCN AOI iterations — embeds the gap-analysis target parquet, embeds a source pool, and mines nearest-neighbour source images for downstream augmentation. Use as the immediate next step after `tao-route-visual-changenet-samples` when expanding a real-image augmentation queue from the mining subset.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit and a CUDA GPU. Pulls the `tao_toolkit.data_services` image declared in `versions.yaml` at the skill bank root.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- data
- mining
- embedding
- vcn
- aoi
- sda
---

# DEFT Mining and Embedding Skill

You are the operator of the DEFT embed-then-mine workflow for VCN AOI. Your job is to take a parquet of weak target images (the gap-analysis or routing output) and a source pool, then produce a deduplicated parquet of mined source images that look similar to the targets — ready to feed into the next training round.

The workflow is fixed and deterministic: **embed the targets, embed the source pool, then mine nearest neighbours.** Each step's output parquet is the next step's input. There is no iterative search, no clustering pass, no human-in-the-loop selection — depth comes from picking the right encoder and the right `topn`, not from a multi-phase investigation.

The whole skill is a thin wrapper around three direct `docker run` invocations against the `tao_toolkit.data_services` image declared in `versions.yaml` (resolved at runtime — see Setup). The container's entrypoint takes `<category> <action> -e <spec.yaml> [hydra overrides...]` — pass `embedding image_embeddings -e <embedding_spec.yaml> …` for embedding and `tmm nearest_neighbors -e <mining_spec.yaml> …` for mining. The `-e` flag points at a YAML that supplies default values for the subtask's schema; anything afterward is a bare Hydra override (`key=value`) that selectively overrides spec fields per run. (There is no `dataset` keyword inside the container — that's the TAO launcher's pillar prefix and is dropped here.) Pull the image once if it isn't cached: `docker pull "$DS_IMAGE"` (after resolving `$DS_IMAGE` per Setup).

Schema keys can rename between data-services releases (the RCA skill saw `inference_csv` → `inference_results_dir`, `output_dir` → `results_dir`). When in doubt, introspect the actual schema once per image: `docker run --rm "$DS_IMAGE" embedding image_embeddings --cfg=job` and `... tmm nearest_neighbors --cfg=job`.

---

## Inputs

1. **Target parquet** — the gap-analysis output, typically `mining_gaps.parquet` from `tao-route-visual-changenet-samples` (or `gaps.parquet` from `tao-analyze-gaps-visual-changenet` if routing was skipped). Required column: `filepath`. If `label` is also present, label-aware filtering during mining is available; otherwise the mining task silently no-ops the filter.
2. **Source pool** — a parquet of candidate images to mine against, with a `filepath` column. If the user only has a CSV, convert it to a parquet **with the same columns** before Step 2. For label-aware filtering, the pool must also carry a `label` column.
3. **Embedding spec file** — a YAML containing `model`, `model_path`, `batch_size`, and (only when `model_path` is a TAO `.pth`/`.ckpt`) `model_config_path`. Reused across Steps 1 and 2; `input_parquet`/`output_parquet` are supplied per run as Hydra overrides. The **same** spec MUST drive both embedding steps — embeddings from different encoders are not comparable, and mismatched encoders are the most common cause of "the mined images look unrelated" reports.
4. **Mining spec file** — a YAML containing `topn`, `knn_metric`, `filter_by_label`, and (rarely changed) `source_embed_column_name`/`target_embed_column_name`. `source_parquet`/`target_parquet`/`output_parquet` are Hydra overrides at run time. SigLIP and CLIP embeddings should use `knn_metric: cosine`. When `filter_by_label: true` but either embedding parquet lacks a `label` column, the container logs a warning and proceeds **without** filtering.

---

## Setup

Resolve the concrete `tao_toolkit.data_services` URI from `versions.yaml` once at the top of the run, then confirm Docker, the NVIDIA container toolkit, and a GPU are present before doing anything else. A GPU is required for both the encoder forward pass and the cuML/cuDF k-NN search; both steps fail without CUDA.

```bash
# Resolve tao_toolkit.data_services → concrete nvcr.io/... URI from versions.yaml
DS_IMAGE=$(python3 -c "import yaml,os; print(yaml.safe_load(open(os.environ['TAO_SKILL_BANK_PATH']+'/versions.yaml'))['images']['tao_toolkit']['data_services'])")
echo "DS_IMAGE=$DS_IMAGE"

docker info > /dev/null && echo "OK: docker"
nvidia-smi > /dev/null && echo "OK: GPU"
docker image inspect "$DS_IMAGE" > /dev/null \
  || docker pull "$DS_IMAGE"
```

Every host path the container reads or writes must be bind-mounted. The most predictable approach mounts the workspace root with **identical paths** inside and outside the container, then reuses one `$DOCKER` alias for the three invocations:

```bash
WORKSPACE=<absolute path that contains all parquets, outputs, and the source-pool images>
DOCKER="docker run --gpus all --rm --ipc=host -v $WORKSPACE:$WORKSPACE -w $WORKSPACE $DS_IMAGE"
```

Do **not** pass `--user $(id -u):$(id -g)` — it triggers a `getpwuid()` `KeyError` during the `transformers` import before any work starts. The container runs as root; chown outputs back to the host UID afterward.

Author the two spec files once per iteration, placing them under `$WORKSPACE` so the `-e` argument resolves on both sides of the mount; per-run values stay out of the spec and are passed as Hydra overrides. If the source pool is a CSV, convert it to parquet up front (preserving `filepath`, and `label` if present). The default `embedding_spec.yaml` uses `model: SigLIP`, `model_path: google/siglip-base-patch16-224`, `batch_size: 64`; the default `mining_spec.yaml` uses `topn: 5`, `knn_metric: cosine`, `filter_by_label: "false"` (quoted — the schema reads it as a string).

See `references/setup.md` for the full environment notes, `TAO_SKILL_BANK_PATH` handling, the path-mounting rationale, the `getpwuid` chown workaround, the CSV-to-parquet snippet, and the verbatim spec-file authoring blocks.

---

## Method

Three commands, in order. Each command's output parquet is the next command's input. Run them as plain Bash; the `$DOCKER` alias from Setup handles the container, GPU, and mounts. Every invocation follows the same shape: `-e <spec>` for the baked-in defaults, then a handful of Hydra overrides for the run-specific paths.

### Step 1 — Embed the target images

```bash
$DOCKER embedding image_embeddings \
    -e <embedding_spec.yaml> \
    input_parquet=<target_parquet> \
    output_parquet=<target_embeddings_parquet>
```

Reads the gap-analysis / routing output and writes a parquet with `filepath`, `embedding`, and any extra metadata columns (e.g. `label`, `siamese_score`, `weakness`) carried forward verbatim from the input. Print the output schema (`pd.read_parquet(...).columns`) to stdout so the script-check hook can confirm the embedding column exists.

If you need to override `model` / `model_path` / `batch_size` for one run without editing the spec, append them as Hydra overrides (e.g. `model_path=...`).

### Step 2 — Embed the source pool

```bash
$DOCKER embedding image_embeddings \
    -e <embedding_spec.yaml> \
    input_parquet=<source_pool_parquet> \
    output_parquet=<source_embeddings_parquet>
```

Same command shape as Step 1, applied to the source pool. Use the **identical** `embedding_spec.yaml` as Step 1, and do not override `model` / `model_path` / `batch_size` differently here — mismatched encoder configs across the two steps produce non-comparable embeddings.

### Step 3 — Mine nearest neighbours

```bash
$DOCKER tmm nearest_neighbors \
    -e <mining_spec.yaml> \
    source_parquet=<source_embeddings_parquet> \
    target_parquet=<target_embeddings_parquet> \
    output_parquet=<mined_parquet>
```

For each target embedding, finds the `topn` closest source embeddings under the chosen metric, deduplicates across targets, and writes a single-column (`filepath`) parquet of unique mined source paths. The container also drops a `mining_summary.txt` next to the output parquet with: query count, neighbour count, duplicates removed, and (when label filtering is on) kept-vs-dropped pair counts. Tweak `topn`, `knn_metric`, or `filter_by_label` via inline Hydra override when sweeping (e.g. `topn=10`) — no need to rewrite the spec.

When `filter_by_label=true` but one of the embedding parquets is missing the `label` column, the container logs a warning and proceeds without filtering. If the mined output looks larger than expected or contains cross-label pairs, scan the docker log for that warning before assuming the task did the right thing.

See `references/reference-invocation.md` for the minimal paste-and-edit end-to-end recipe (resolves `$DS_IMAGE`, writes both specs, runs all three steps, chowns outputs, and prints row counts) to run as a single streamed Bash block.

---

## Outputs and report

Write everything into a timestamped folder under the experiment / iteration directory. Get the real timestamp by running `date +%Y-%m-%d_%H%M%S` in Bash — do NOT hardcode or guess. If the user specifies a custom output path, use it directly but maintain the same internal layout. The packaging hook adds `mining_config/` and `claude_session.jsonl` automatically when `Mining_Report.md` is written.

The mined parquet is the artifact downstream training consumes. The two embedding parquets are intermediate but worth retaining — reusable across multiple mining runs against the same source pool, and the only place to look when a "looks unrelated" report needs encoder-level debugging.

See `references/outputs-and-reporting.md` for the full output-directory layout and the verbatim `Mining_Report.md` template (Verdict, Inputs, Encoder Consistency, Mining Run, Per-Label Breakdown, Output Sanity, Recommended Actions; keep it 600–1200 words).

---

## Common pitfalls

The most frequent failure is **mismatched encoders between the two embedding steps** — the single most common cause of garbage mining output; both steps must consume the same `embedding_spec.yaml`. Other recurring traps: passing `--user` (the `getpwuid` `KeyError`), skipping an embedding step, a missing `label` column silently no-oping `filter_by_label=true`, spec files outside `$WORKSPACE`, unresolved `???` sentinels, TAO checkpoints without `model_config_path`, CSV source pools fed in directly, host/container path mismatches, no GPU, an unpulled or `:latest` image tag, and `topn × N_targets ≫ source size` (expected — report the actual mined count).

See `references/troubleshooting.md` for the full pitfall list with the exact errors, causes, and fixes.

---

## Execution Order

1. Resolve `DS_IMAGE` from `versions.yaml` (`images.tao_toolkit.data_services`), then run `docker info`, `nvidia-smi`, and `docker image inspect "$DS_IMAGE"` (pulling if missing) once to confirm the environment. Abort with a clear message if any fail.
2. Run `date +%Y-%m-%d_%H%M%S` to get the timestamp; create `<output_dir>/mining_results/<timestamp>/`.
3. Write `embedding_spec.yaml` and `mining_spec.yaml` into the timestamped dir, filling in the encoder choice and mining knobs. Keep these under `$WORKSPACE` so the `-e` path resolves inside the container.
4. If the source pool is a CSV, convert to parquet first (preserve `filepath` and `label`).
5. Run Step 1 (embed targets) via `docker run … embedding image_embeddings -e embedding_spec.yaml input_parquet=… output_parquet=…`. Print the output parquet's row count and columns to stdout.
6. Run Step 2 (embed source pool) with the **identical** `embedding_spec.yaml` as Step 1. Print output row count and columns.
7. Run Step 3 (mine nearest neighbours) via `docker run … tmm nearest_neighbors -e mining_spec.yaml source_parquet=… target_parquet=… output_parquet=…`. Confirm `mining_summary.txt` was written next to `mined.parquet`.
8. Compute the per-label breakdown (Section 5) by joining the target embeddings parquet with the mined output on filepath, if both carry `label`.
9. Write `Mining_Report.md` last — writing it triggers the packaging hook, which copies session logs and skill config alongside.
