# Pipeline Operations

> Configs live under `configs/cookbook/<use-case>/`. See the [cookbook index](../../../configs/cookbook/README.md) for the folder layout.

Runtime flow, common editing tasks, a worked example, storage, and security
practices for the `augmentation` skill. Read this when running the pipeline or
editing an existing config; `SKILL.md` keeps only the summary.

## Pipeline Flow

```text
1. Load & validate config (Pydantic PipelineConfig)
2. Initialize captioner (factory dispatch from config)
3. Initialize evaluators (hallucination, attribute verification, vlm verification)
4. Resolve augmentation.model.name -> endpoint -> adapter -> BaseExecutor
5. For each data sample:
   a. Run captioning -> produce prompt
   b. Run generation (with seed) via the endpoint's adapter -> produce output media
   c. Run hallucination check (if configured)
   d. Run attribute verification (if configured)
      - LLM generates MCQ questions from variables
      - VLM answers from frames sampled evenly across the output video
   e. On failure: retry with incremented seed (up to `pipeline.retry`)
      - If `pipeline.regenerate_caption_on_retry: true` and a captioner is
        configured, rerun captioning and overwrite output.caption before
        retrying generation
   f. Write metadata JSON
```

## Worked example: re-render a video as a rainy night

1. **Model:** input is a video and the goal is a scene-attribute change → Cosmos
   Transfer 2.5. Start from `config_video_transfer_CT25_nim.yaml`
   (`augmentation.model.name: cosmos-transfer2.5`; role `video_transfer`, `nim` adapter).
2. **Endpoint:** point the `video_transfer` endpoint's `url` at your CT2.5 NIM;
   keep the `vlm`/`llm` endpoints for captioning + verification.
3. **Inputs/outputs:** point `data.0.inputs.rgb` at the source video and set
   `data.0.output.{video,caption,metadata}`.
4. **Target attributes:** set `captioning.llm.variables.weather_condition: ["raining"]`
   and `lighting_condition: ["night"]`.
5. **Run** (inside the container): `uv run --no-sync modules/cli.py --config configs/cookbook/video-data-augmentation/config_video_transfer_CT25_nim.yaml`.

## Common Tasks

**Add a data sample** — append to the `data:` list: `inputs.rgb` (video for
transfer/predict, image for edit/image2video) plus `output.{video,caption,metadata}`.

### Register a New BYOM Endpoint / Model

Add an entry to the `endpoints:` list and point `augmentation.model.name` at it:

```yaml
endpoints:
  - id: my-edit
    role: image_edit
    url: "http://localhost:8005"
    adapter: nim                 # omit to take the role default
    # api_key_env: BUILD_NVIDIA_API_KEY   # only if the endpoint needs auth
augmentation:
  model:
    name: my-edit                # resolves to the endpoint id above
```

Serving the same model over a different contract is a one-field change
(`adapter:`). A genuinely new wire contract needs a one-time adapter class
registered in `modules/generation/factory.py` (`ADAPTERS`) and
`modules/aug_utils/schema/adapters.py` (`KNOWN_ADAPTERS`).

### Change Target Attributes

Edit `captioning.llm.variables` — each key maps to a list of values (first value
is used by the text captioner; all values are available to LLM generation):

```yaml
captioning:
  llm:
    variables:
      weather_condition: ["snowy"]
      lighting_condition: ["dusk"]
```

For attribute verification, optionally set `verification_options` with the MCQ
answer pool (or set `question_generation.generate_options: true` to let the LLM
invent distractors). When unset, the option pool falls back to `variables`.

For deterministic VLM-template prompting, define the allowed text under
`captioning.template.attributes` and select one `event_type`, `motion_level`, and
`aftermath` ID under each sample's `data[].inputs.prompt_attributes`. See
`captioning-strategy-guide.md`.

### Switch models, batch-generate, tune quality

- **Switch model**: change `augmentation.model.name` and ensure a matching
  `endpoints:` entry (right `role`; `adapter` if not the role default). CLI
  overrides work too.
- **Batch configs**: a workflow YAML with `conditional_variables` for dependent
  attributes (e.g. snowy weather never pairs with dry roads). See
  `config-decision-tree.md`.
- **Tune quality**: `augmentation.parameters` is pass-through (`extra="allow"`) —
  only set knobs are sent. Per-model knobs (Cosmos `sigma`/`guidance`/`num_steps`
  + `modalities`; image-edit `num_inference_steps`/`guidance_scale`/`negative_prompt`;
  image→video model-native) are in `configuration-schema.md`.
- **Run tests** (in-container): `uv run --no-sync pytest tests/`.

## Storage

All file I/O uses `multistorageclient` (aliased as `msc`) which transparently
handles local paths, S3 (`s3://`), GCS (`gs://`), Azure (`az://`), and HTTP URLs.
Configure cloud-storage access with a `multistorageclient` config file referenced
by the `MSC_CONFIG` environment variable (see the multistorageclient docs); keep
any credentials it holds out of version control.

## Security Notes

This skill runs remote inference in Docker with credentials and host networking.
Apply these practices, especially on shared or production hosts:

- **Credentials via environment variables.** API keys are passed as env vars or
  env-files referenced by each endpoint's `api_key_env`. Never hardcode them in
  config YAML, logs, or scripts, and never commit them. Restrict any local
  env-file to owner-only read/write permissions, keep it out of version control,
  and avoid echoing or logging secret values. For production, prefer a secrets
  manager or Docker secrets over an on-disk env-file.
- **Data-directory permissions.** `data/` must be writable by the container user
  (uid 10000). Prefer running the container as your own user
  (`--user "$(id -u):$(id -g)"`) or matching the host directory's ownership.
  Making the directory world-writable is a last resort and must never be used on
  shared or production systems.
- **Host networking.** Prefer Docker's default bridge network, which is sufficient
  whenever your endpoints are remote URLs. Host networking mode removes network
  isolation between the container and the host, so treat it as a last resort
  reserved for the case where an endpoint genuinely listens on the host's own
  `localhost`, and do not use it on shared or production hosts.
