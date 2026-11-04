# Component Skills — Minimal Excerpt

**What this is.** Component-level expertise for every CDF component lives in that component's own
repository and skill. `write-dag` task groups are thin Airflow wrappers around those components, so
the component skill is the authority for config schema, prompts, model options, and component
errors. This file links each component skill and keeps a **small, frozen excerpt** of the parts
`write-dag` has actually needed so far, so the skill still works when a component repo is not
reachable.

| Component | Component skill (pinned to latest `main`, 2026-10-20) | Used by |
|---|---|---|
| CDF Simulation | [`cdf-simulation`](https://github.com/NVIDIA/cdf-simulation/tree/a0af274e6352347a1979be40036efdd33b447272/skills/cdf-simulation) | DIG `image_generation` |
| CDF Augmentation | [`cdf-augmentation`](https://github.com/NVIDIA/cdf-augmentation/tree/73c861c43a4952b6060c87265bf39f4abc48199a/skills/cdf-augmentation) | IAA, EVG, VDA, AV, DIG `augmentation` |
| CDF Auto-Labeling | [`cdf-auto-labeling`](https://github.com/NVIDIA/cdf-auto-labeling/tree/6c75c76c52a6c76c3866a38d30885e11123b0981/skills/cdf-auto-labeling) | IAA, EVG, VDA |
| CDF AnomalyGen | [`cdf-anomalygen`](https://github.com/NVIDIA/cdf-anomalygen/tree/63c888cb479e22f6abe343421eaba2ae1847b8f5/skills/cdf-anomalygen) | DIG `anomaly_generation` |
| CDF Curation and Retrieval | [`cdf-curation-and-retrieval`](https://github.com/NVIDIA/cdf-curation-and-retrieval/tree/b4d553d00cc00c1293ad997fa8693af809489c71/skills/cdf-curation-and-retrieval) | AV Cosmos-Curator and Cosmos Dataset Search stages |

**Staleness warning.** The excerpts below are a snapshot, not a live reference. They will drift
from the real repos. If a component skill *is* reachable in your session, prefer it over this
file — it's authoritative. Only fall back to this file when it's unavailable, and if what you need
isn't covered here, say so and ask rather than extrapolating beyond it.

---

## CDF Simulation

> Component-owned guidance for the cdf-simulation half of the Airflow-triggered PM prompt
> templates (DIG Flow Catalog) — golden THT board, THT pin-geometry variation, good/clean SMT
> images, structural pose defects, and solder texture defects.

**Use from a DAG when:** authoring or changing the DIG `image_generation` override (the YAML
JobConfig that drives the CDF Simulation render).

- Turns one matched catalog row or reference flow into one validated `cdf.CDFJob` build plus its
  compiled `JobOrderPlan` (`job.compile_orders(...)`, serialized with `JobOrderCodec.dumps(plan)`).
  It does not render; rendering happens in the Airflow DAG.
- One job per variant: sequential `job.capture.render(...)` calls in one process crash the second
  Kit subprocess, so the DAG must fan out one process per variant.
- Airflow supplies the attachment path and the output root; both are required with no fallback.
- Structural-defect magnitudes have no API defaults and must be set explicitly. Catalog defaults:
  shift `translate_range=0.2` (mm) and `rotate_z_range=15` (degrees); tombstone and sideflip
  `angle_range=(70.0, 90.0)` (degrees).
- Coverage `ratio` applies to structural and solder defects alike: `1.0` for full-board coverage,
  otherwise `0.2`. At least one component is always selected (`max(1, int(pool * ratio))`).
- Crop `class_filter` is not `component_types`.
- Day 1 real-photo texture defects and AnomalyGen fine-tuning are out of scope for this component.

---

## CDF AnomalyGen

> Use when running the CDF AnomalyGen pipeline over a defect dataset — mask placement,
> fine-tuning, synthetic defect-image generation (SDG), evaluation, quality refinement, and
> pseudo-labeling — even when the user names only one stage.

**Use from a DAG when:** authoring or changing the DIG `anomaly_generation` override.

- AnomalyGen fine-tunes the frozen Cosmos base into an anomaly-generation model: given a clean
  image and a defect mask, it paints a realistic defect into that region.
- Inputs follow the `{texture}/{defect}` clean-image plus defect-mask contract; it is not for
  generic image generation.
- Requires a CUDA GPU, the built environment (native `.venv` or the Docker image), and Hugging Face
  auth with network access to download checkpoints.

---

## CDF Curation and Retrieval

> NVIDIA CDF curation and retrieval. Cosmos Curator video and image pipelines (split, dedup,
> shard, annotate) with filtering, captioning, and embedding. Cosmos Dataset Search ingest,
> search, and capability qualification.

**Use from a DAG when:** authoring or changing Cosmos-Curator or Cosmos Dataset Search (CDS) tasks,
as in the AV workflow.

### Curation (Cosmos Curator)

- Video pipelines: `split`, `dedup`, `shard`. Image pipeline: `annotate`
  (load → filter → embed → caption → write).
- Do not emit a Curator pipeline config without context: KPI run output, KPI sample videos, or a
  completed calibration interview covering inputs, domain, goal, hardware, and calibration.
- Workplace safety and employee conduct monitoring configs require confirmed legal review.

### Retrieval (Cosmos Dataset Search)

- Confirm CDS is healthy before any collection, ingest, or search call.
- Public GA image `nvcr.io/nvidia/blueprint/cosmos-dataset-search:1.2.0`. It does not run on
  Blackwell GPUs; use Hopper, Ada, or Ampere with 16GB+ VRAM.
- Search needs a Cosmos-Embed1 service to embed the text query, with weights from
  `nvidia/Cosmos-Embed1-224p` at the pinned revision; the image cannot download its own weights.
- Public GA pipeline id is `cosmos_video_search_milvus`. Create-collection returns an **id**, and
  ingest and search must use that id, not the collection name.
- CDS-bound ingest accepts CE1 embeddings only (`CDS_EMBEDDING_FAMILY=ce1`). Bulk ingest order:
  validate the Curator CE1 export, stage the parquet in CDS-readable object storage, submit one
  non-retried `/insert-data`, poll the job to success, then search.
- `/insert-data` 404 fails closed; it usually means a name was used as the id or a host path was
  sent. On Kubernetes the CDS URL ends in `/api`.

---

## CDF Augmentation

> Use when authoring or validating Cosmos Data Factory (CDF) augmentation YAML configs, or running
> remote Cosmos Transfer (including Cosmos3 WSM controls), Cosmos Predict, image-edit, or
> image-to-video inference.

**Use from a DAG when:** authoring or changing a `CosmosTaskGroup` config (IAA image-edit, EVG
image-to-video, VDA Cosmos Transfer, AV Cosmos3-Nano generation) or the DIG `augmentation`
override.

- Remote-API only: models are HTTP endpoints, the image bundles no weights, and plain remote
  inference needs no GPU or `HF_TOKEN`.
- Model choice follows input type and goal: Cosmos Transfer for video scene changes (Cosmos3 WSM
  via `data[].inputs.controls.wsm`), Cosmos Predict for new or continued video, image edit for
  still edits, image-to-video to animate a first frame.
- Author YAML in this order: `data`, `endpoints`, `captioning`, `augmentation`, `evaluators`,
  `pipeline`, `data_processing`; omit optional sections rather than adding placeholders.
- Config-only requests are not run or preflighted. Run requests validate inside the container
  first, with at most 3 config versions before stopping.
- The candidate loop is bounded to `pipeline.retry + 1` candidates (default 2); retry `n` uses
  `base_seed + n`. An enabled external Cosmos evaluator must pass a `/health`, `/checkers`,
  `/dependency-graph` preflight before generation.
- Out of scope: training or fine-tuning, cluster or NIM deployment.

Configs are validated against a `PipelineConfig` Pydantic model with seven top-level sections:
`data`, `endpoints` (a **list**), `pipeline`, `captioning`, `augmentation`, `data_processing`,
`evaluators`.

### `endpoints` — BYOM registry (a list, not a mapping)

```yaml
endpoints:
  - id: vlm_qwen                 # optional; only needed to disambiguate 2+ same-role endpoints
    role: vlm                    # required
    url: "http://localhost:8000/v1"   # required
    model: "Qwen/Qwen3.6-27B-FP8"     # wire model string, default ""
    # adapter omitted -> role default
    api_key_env: VLM_API_KEY     # optional; name of env var holding the key, never the key itself
    timeout: 120                 # seconds
```

**Roles → default adapter:** `vlm`/`llm` → `openai.chat.completions`; `image_edit` →
`nim`; `video_transfer`/`video_predict` → `nim`; `image2video` → `openai.video.sync`. The
`evaluator` role has no default: an external Cosmos Evaluator endpoint must set
`adapter: cosmos.evaluator.arbitrator`, which is evaluator-only, not a generation adapter.

**Known adapters:** `openai.chat.completions`, `openai.images.edits`, `openai.video.sync`,
`openai.video.async`, `nim`, `passthrough`.

**Model selection:** `augmentation.model.name` resolves to an endpoint by `id`, then by `role`,
then by a model-name→role map (`image-edit`→`image_edit`, `cosmos-transfer2.5`→`video_transfer`,
`cosmos-predict`→`video_predict`, `cosmos3-image2video`→`image2video`). Two endpoints sharing a role
must be disambiguated with `id`.

### Image-edit / IAA-style captioning block

```yaml
captioning:
  llm:
    text: "Change the person's clothing to a {top_outer_color} {top_outer_type}, ..."
    variables: { top_outer_color: ["red"], ... }        # full replace, not merge, per generation
    verification_options: { top_outer_color: [...allowed values...] }
evaluators:
  - attribute_verification:
      enabled: true
      exclude_variables: ["shoe_type", "shoe_color"]     # excluded from MCQ gate, still in prompt
```

`Template has placeholders {x} not found in variables` means every placeholder in `captioning.llm.text`
needs a matching key in `variables` — there is no partial/merge behavior, a caller must supply all of them.

### Serving requirement (IAA image-edit specifically)

`vllm-omni serve Qwen/Qwen-Image-Edit-2511 --omni` — the `--omni` flag is required to expose
`/v1/chat/completions`; without it the endpoint doesn't serve the chat-completions contract that
`adapter: openai.chat.completions` expects.

### Troubleshooting — top entries actually hit so far

| Error | Cause | Fix |
|---|---|---|
| `no endpoint matched selector '<name>'` | `model.name` doesn't match any endpoint `id`/`role` | Add a matching endpoint, or set `model.name` to a specific `id` |
| `selector '<name>' matches multiple endpoints with role '<role>'` | Two endpoints share a role | Give them `id`s, target one by `id` |
| `endpoint <id> resolves to unknown adapter '<x>'` | Typo'd/unsupported adapter | Use one of the known adapters listed above |
| `Template has placeholders {x} not found in variables` | Missing key in `captioning.llm.variables` | Supply every placeholder referenced in the text template |
| `401 Unauthorized` | Missing/wrong API key | Set the env var named by `api_key_env`; local endpoints need none |
| `404` on a chat/video route | Wrong adapter for the server's actual route | Match adapter to route: `/v1/chat/completions` vs `/v1/images/edits` vs `/v1/videos/sync` vs NIM `/v1/infer` |
| Hosted NVCF NIM times out / returns 202 | Generation outlives the hold-open window | `adapter: nim`, raise endpoint `timeout`; poller uses `NVCF-POLL-SECONDS`/`NVCF-REQID` automatically |

---

## CDF Auto-Labeling

> Use when a user needs to get started with CDF Auto-Labeling, plan a scenario, run or debug a
> shipped cookbook, author prompts or cookbooks, migrate a pipeline, or configure a stage.

**Use from a DAG when:** authoring or changing detection and tracking, captioning, visual QA,
reasoning, or person attribute search task groups (IAA, EVG, VDA), or their prompts and question
banks.

- A router: read the matching reference instead of inventing a workflow.
- Confirm critical inputs before executing: input and output paths, VLM/LLM endpoint URLs and
  model names, model cache path, GPU ids, and the `max_tokens` cap for reasoning-capable models.
- Prove the stack on a shipped cookbook (video data augmentation, event and person attribute
  search, or event-verification reasoning) before customizing, then adapt the closest cookbook
  rather than authoring from scratch.
- Keep the first custom pipeline to the stages the requested annotations need, for example
  `detection_and_tracking -> captioning -> visual_qa -> reasoning -> training_export`.
- Runs go through `workflow-runner:main` (dry-run with `--container-dry-run` first). No secrets or
  absolute home paths in committed cookbooks.

### Prompt authoring workflow

1. Classify the prompt type: image caption, video dense caption, VQA evidence, or LLM evidence
   aggregation.
2. Write evidence-first — describe what's visible before any classification; never infer identity,
   intent, fault, or protected attributes. Explicit uncertainty handling required.
3. Put prompt files under the cookbook's `prompts/` tree, referenced via
   `workflow.nodes.<node>.args` / `question_bank_file`, never inlined into shell commands.
   `stage_args` is still accepted for compatibility only.
4. For a **reasoning-capable model** (e.g. Gemini 3 Flash) on the `reasoning`/`visual_qa` stages,
   raise `max_tokens` generously (e.g. `32768`) — reasoning models spend part of the budget on
   internal thinking, so a low cap silently truncates/empties output.

### Question-bank JSON shape (real schema, shared by `visual_qa` and `reasoning`)

```json
{
  "name": "scene_qa",
  "version": "1.0",
  "questions": [
    {
      "id": "q_activity",
      "question": "What is the primary activity in the scene?",
      "options": ["A. Loading", "B. Unloading", "C. Idle", "D. Transit"],
      "aggregation": "any"
    }
  ]
}
```

Wired in via config, not inline:

```yaml
workflow:
  nodes:
    captioning:
      stage: captioning
      args: [--prompt-file, ../prompts/dense_caption/scene_prompt.md]
visual_qa:
  question_bank_file: question_bank.json
```

### Guardrails

- Evidence first, labels second — the model observes and describes before classification.
- Don't ask for a specific output format in prompts; the pipeline stages produce the structured
  output. Request structured captioning output only when the user or service contract requires it.
- Explicit rules against inferring identity, intent, fault, protected attributes, or hidden cause.
- Video prompts require timestamp-aware reasoning and track-ID grounding only when overlays are
  actually readable — don't assume track IDs are visible.
