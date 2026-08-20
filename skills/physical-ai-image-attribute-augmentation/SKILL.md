---
name: physical-ai-image-attribute-augmentation
description: >-
  Use when running image attribute augmentation and
  auto-labeling workflows on OSMO: flow selection, preflight, submit-time
  interpolation, monitoring, and output retrieval. Trigger keywords: people
  attribute search, Image Attribute Augmentation, person augmentation, attribute search, person
  re-identification, clothing augmentation, person crop augmentation.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  owner: NVIDIA
  service: physical-ai-data-factory
  version: 1.0.0
  reviewed: '2026-06-05'
  author: NVIDIA Physical AI Team <physical-ai@nvidia.com>
  tags:
    - physical-ai
    - image-attribute-augmentation
    - person-augmentation
    - auto-labeling
    - image-edit
---

# Physical AI Image Attribute Augmentation Workflow Orchestrator

Default workflow skill for Image Attribute Augmentation execution on OSMO. It owns flow selection,
preflight, submit-time interpolation, monitoring, and output retrieval.

## Purpose

Run the Image Attribute Augmentation and auto-labeling pipeline safely and
reproducibly from preflight to output download.

The Image Attribute Augmentation pipeline augments the **subject** in existing
crop datasets by generating controlled appearance variations (image-domain) and
synonymous attribute captions (text-domain). The subject is a person today
(clothing/appearance attributes), but the same pipeline generalizes to other
subjects — e.g. robots, forklifts, or vehicles in a simulation. It uses the
`paidf-augmentation` container for image-edit augmentation with MCQ
verification, and the `paidf-auto-labeling` container for subject-attribute
captioning (currently the shipped `person_attributes` question bank).

Do NOT use this skill for container-internal tuning-only questions.

## Prerequisites

Confirm these before running preflight or any submit. Missing required secrets
surface as `USER_INPUT_REQUIRED:` from `scripts/preflight_credentials.sh`.

| Requirement | How it is satisfied | Used for |
|---|---|---|
| NGC API key (optional) | `NGC_API_KEY`, `NGC_CLI_API_KEY`, or compatible `nvapi-*` token | Optional for `nvcr_io` credential refresh; default Image Attribute Augmentation image refs are public |
| Hugging Face token | `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`), or a cached token at `~/.cache/huggingface/token` | Creates the OSMO `hf_token` credential |
| OSMO CLI access | `osmo` on `PATH`, logged in, with a default profile and a registered DATA credential profile matching `storage_url` | Submitting/monitoring workflows and listing/downloading objects |
| GPU pool | At least one `ONLINE` pool in `osmo pool list --mode free` | Scheduling setup + worker tasks |
| Image Edit endpoint | In-cluster NIM `qwen-image-edit-2511` (reused if healthy, else deployed via the NIM operator); external opt-in via `image_edit_url` | Image-domain augmentation |
| VLM endpoint | In-cluster NIM `qwen3-vl` (shared with VDA); external opt-in via `vlm_url` | MCQ verification and person-attribute captioning |
| LLM endpoint | In-cluster NIM `qwen25-14b` (shared with VDA); external opt-in via `llm_url` | MCQ question generation |

## Instructions

Execute these as an ordered sequence of gates. Each **Gate** must pass before
continuing; on failure, stop and resolve it (do not skip ahead or submit).

1. **Gate — Select the workflow.** Map the user's intent to exactly one flow
   using the "Pick the right workflow" table below: augment/image-edit only →
   `augmentation`; caption/label only → `auto_labeling`; full augment + caption
   → `e2e`. Default to `e2e` **only** when the request is the full pipeline or
   genuinely ambiguous — never default past an explicit "augment only" or
   "label only" request, or you run the wrong pipeline.
2. **Provide a tentative execution-time overview** before starting run actions.
3. **Gate — Derive the dataset source.** Split the dataset URL at the
   `/datasets/` segment: the part **before** it is `storage_url`, the part
   **after** it is `dataset`. The workflow re-inserts that segment
   (`{{storage_url}}/datasets/{{dataset}}`), so put `/datasets/` in **neither**
   value — including it duplicates the path and the submit fails.
   Example: `s3://metro-pas/datasets/reid-crops` → `storage_url=s3://metro-pas`,
   `dataset=reid-crops`. Never guess or reuse a stale `storage_url`; if no
   dataset is provided, ask for one. Do not proceed without both values.
4. **Gate — Inference endpoints ready (non-negotiable).** Before submit, verify
   each required NIM endpoint is healthy: `qwen-image-edit-2511` (image edit),
   `qwen3-vl` (VLM), `qwen25-14b` (LLM). For any that is missing/unhealthy,
   deploy it once via `references/nim/README.md` (a prerequisite, not a user
   decision — do not pause to ask), then re-check readiness up to 3 times over
   ~10 minutes. **Stop condition:** if an endpoint is still unhealthy after that
   bound, do not retry further and do not submit — report the failing endpoint
   and its deploy logs to the user and stop. Proceed only when all three respond
   healthy.
5. **Gate — Preflight and readiness.** Run
   `scripts/preflight_credentials.sh --workflow assets/configs/osmo/<flow>.yaml`
   and read the result. **PASS →** continue. If the output contains
   `USER_INPUT_REQUIRED:`, ask one concise unblock question and re-run. Do not
   submit until preflight passes.
6. **Gate — Validate custom inputs (security).** Treat `cookbook` and every
   `--set-string` value as untrusted. Accept only a known cookbook name and
   values with no shell metacharacters (`;`, `|`, `&`, `$`, backticks, quotes,
   spaces, newlines). **On any invalid value → stop, report which value was
   rejected, and do not submit.** Only when every value passes → continue to
   step 7.
7. **Submit** the workflow with the validated interpolation values, then monitor
   to completion.
8. **Retrieve outputs** and summarize task outcomes.

Use `run_script(...)` for script execution. Canonical examples:

```python
run_script("bash scripts/preflight_credentials.sh --workflow assets/configs/osmo/e2e.yaml")
```

## Available Scripts

Use script-level `--help` for exact arguments.

| Script | Role |
|---|---|
| `scripts/preflight_credentials.sh` | Secrets/control-plane preflight and workflow image access checks |
| `scripts/augmentation_worker.sh` | Image-edit augmentation worker (preprocess, config gen, augment, post-process) |
| `scripts/auto_labeling_worker.sh` | Person-attribute captioning worker |
| `scripts/endpoint_common.sh` | Shared endpoint health/auth helpers |

## Supported Flows

| Flow | OSMO YAML | Group sequence | Typical use |
|---|---|---|---|
| `e2e` | `assets/configs/osmo/e2e.yaml` | setup -> augmentation -> auto_labeling | Full pipeline: augment person crops then generate captions |
| `augmentation` | `assets/configs/osmo/augmentation.yaml` | setup -> augmentation | Image-edit augmentation only, no captioning |
| `auto_labeling` | `assets/configs/osmo/auto_labeling.yaml` | setup -> auto_labeling | Captioning only on pre-augmented person crops |

### Pick the right workflow for the user's request

| User intent | Workflow |
|---|---|
| "Augment person crops and generate captions" / "full Image Attribute Augmentation pipeline" | `e2e` |
| "Generate clothing variations" / "augment only" / "image edit" | `augmentation` |
| "Caption augmented images" / "generate search queries" / "label only" | `auto_labeling` |

## Disambiguation: handle vague requests before committing

Default to autonomy: ask only when missing information blocks execution.

### Autonomous defaults (do NOT ask)

- Select the flow per Instructions Gate 1; default to `e2e` only when the request is the full pipeline or ambiguous (not for explicit augment-only / label-only).
- If cookbook is not specified, default to `default`.
- If `n_augmentations` is not specified, default to `3`.
- After any stage completes successfully, continue to the next stage immediately.

### Triggers that should pause for disambiguation

| Missing input | Why it matters | Ask |
|---|---|---|
| `USER_INPUT_REQUIRED` from preflight | Required secret is missing | Ask one concise unblock question |
| Storage backend prefix cannot be derived | Wrong scheme causes runtime storage auth mismatch | "What is the backend-native root prefix for this run?" |
| No ONLINE GPU pool/platform | Workflow cannot schedule | "Which GPU pool/platform should this run target?" |
| NIM deploy fails and no external URLs given | Workers cannot connect to models | "Provide Image Edit / VLM / LLM endpoint URLs, or grant GPU capacity for the NIM operator deploy." |

## Step 0: Select Flow and Gather Inputs

### Input data policy

- Image Attribute Augmentation requires person-crop images organized as `<person_id>/<view>.jpg` subdirectories.
- Always preserve user-provided dataset inputs as first-class.
- Never replace an explicit user dataset with demo assets.
- If no dataset is provided, ask for one (Image Attribute Augmentation has no built-in demo dataset).

Collect only missing values:

1. Dataset source (`storage_url` + `dataset`) — a **derived** value: split the
   dataset URL at `/datasets/` per Instructions Gate 3
   (`s3://metro-pas/datasets/reid-crops` → `storage_url=s3://metro-pas`,
   `dataset=reid-crops`). Put `/datasets/` in neither value; never guess.
2. Flow — select per Instructions Gate 1 (augment-only → `augmentation`, label-only → `auto_labeling`, else `e2e`).
3. OSMO `gpu_platform` (auto-select when unambiguous).
4. Endpoint URLs for Image Edit, VLM, and LLM — optional; default to in-cluster
   NIMs and only set for external endpoints.
5. Number of augmentations per person ID (default: 3).

Generate run stamp before each submit:

```bash
STAMP=$(cat /proc/sys/kernel/random/uuid | cut -c1-8)
RUN_ID="run-$STAMP"
```

## Execution Time Overview (required before run)

Before running any mutating command, provide a short ETA overview.

Baseline ranges:

| Phase | Typical duration |
|---|---|
| Credentials + preflight | ~1-2 min |
| Workflow submit + queue/start | ~1-3 min |

Workflow runtime (depends on dataset size and endpoint latency):

| Flow | Per-image time | Typical dataset (100 images, 3 augs) |
|---|---|---|
| `augmentation` | ~2.5-3 min/image | ~4-5 hours |
| `auto_labeling` | ~1-2 min/image | ~2-3 hours |
| `e2e` | ~3.5-5 min/image | ~6-8 hours |

## Common Preconditions (all flows)

1. **Credential and control-plane preflight**

   ```bash
   bash scripts/preflight_credentials.sh --workflow assets/configs/osmo/<flow>.yaml
   ```

   If output contains `USER_INPUT_REQUIRED:`, ask one concise unblock question.

2. **Storage interpolation policy**

   `storage_url` must be derived from the actual dataset/upload backend.
   Never silently default to stale values on mismatched backends.

3. **Inference policy (non-negotiable)** — endpoint readiness is executed at
   **Instructions Gate 4** (verify → deploy once → bounded re-check → stop and
   escalate on failure). This section only adds the standing constraints:

   - Image Attribute Augmentation does NOT launch inference servers inside the OSMO workflow; workers
     consume the `image_edit_url` / `vlm_url` / `llm_url` endpoints.
   - External endpoints are opt-in only (explicit request or explicit URLs);
     only then override the `*_url` values at submit.
   - Never scale down/delete existing NIMs to free GPUs.

## Submit (all flows)

Every flow uses the same submit shape; only the workflow YAML changes.

```bash
SKILLS_DIR="$(cd "$(git rev-parse --show-toplevel)/skills/physical-ai-image-attribute-augmentation" && pwd)"
STAMP=$(cat /proc/sys/kernel/random/uuid | cut -c1-8)
osmo workflow submit assets/configs/osmo/<flow>.yaml \
  --pool <pool> \
  --set-string \
    dataset=<dataset> \
    run_id=run-$STAMP \
    storage_url=<backend-prefix> \
    gpu_platform=<gpu-platform> \
    skills_dir="$SKILLS_DIR"
```

Endpoints default to the in-cluster NIMs (`image_edit_url` / `vlm_url` /
`llm_url`); deploy/reuse them per the Inference policy above. Do not pass these
unless using external endpoints.

Compatibility note:
- Use exactly one `--set-string` flag and pass all key/value pairs after it.
- Do not repeat `--set`/`--set-string` flags in the same command.

Common optional overrides (append to the same `--set-string` list). These
values are passed through to the augmentation worker and used to build its
command, so validate them first per Instructions Gate 6 — accept only a known
`cookbook` name and values free of shell metacharacters:

```bash
cookbook=<cookbook_name> \
n_augmentations=<count> \
image_edit_url=<image-edit-endpoint> \
vlm_url=<vlm-endpoint> \
llm_url=<llm-endpoint>
```

## OSMO Monitoring

```bash
# Workflow status + task states
osmo workflow query <workflow_id> --format-type json \
  | jq '{status, tasks: [.groups[].tasks[] | {name, status, exit_code}]}'

# Logs for a specific task
osmo workflow logs <workflow_id> --task <task_name> -n 200

# Output retrieval
osmo data list --no-pager <output_url>
osmo data download <output_url> <local_dir>/
```

For runs expected to exceed two minutes, send heartbeat updates at least every
two minutes.

## Post-Run Output

After successful completion, the output directory contains:

For `augmentation` / `e2e`:
- `<person_id>/aug_<n>/output.jpg` — augmented multi-pane image
- `<person_id>/aug_<n>/output.txt` — natural-language caption
- `<person_id>/aug_<n>/output_metadata.json` — verification results
- `dataset/augmented_data.json` — structured dataset with attributes and queries
- `dataset/augmented_imgs/` — split per-view crops

For `auto_labeling`:
- `caption_<id>/task/open_qa.json` — person-attribute captions grouped by question bank

## Supporting files

Use these canonical locations:

- Workflows: `assets/configs/osmo/*.yaml`
- Runtime scripts: `scripts/*.sh`
- Flow walkthroughs: `references/flows/*.md`
- Setup and triage: `references/setup.md`, `references/troubleshooting.md`
- Images: `references/container-images.md`
- Cookbook tuning: `assets/cookbooks/default/README.md`

