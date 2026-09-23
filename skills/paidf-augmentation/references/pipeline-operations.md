# Pipeline Operations

> Configs live under `configs/cookbook/<use-case>/`. See the [cookbook index](../../../configs/cookbook/README.md) for the folder layout.

Runtime flow, common editing tasks, a worked example, storage, and security
practices for the `augmentation` skill. Read this when running the pipeline or
editing an existing config; `SKILL.md` keeps only the summary.

## Pipeline Flow

```text
1. Load & validate config (Pydantic PipelineConfig)
2. Initialize captioner (factory dispatch from config)
3. Initialize the evaluators enabled by the config
4. Resolve augmentation.model.name -> endpoint -> adapter -> BaseExecutor
5. For each data sample:
   a. Run captioning -> produce prompt
   b. Run generation (with seed) via the endpoint's adapter -> produce output media
   c. Run the configured evaluator sequence and apply its gating results
   d. Branch on the evaluator outcome:
      - gates passed -> accept the candidate and write outputs
      - quality gate failed and attempts remain -> retry with incremented seed
      - quality gate failed after `pipeline.retry + 1` total candidates -> fail
        the evaluation and apply `strict` / `retain_failures`
      - service/contract failure -> stop on the current candidate without
        generating a replacement; surface the failure and apply
        `strict` / `retain_failures`
      - If `pipeline.regenerate_caption_on_retry: true` and a captioner is
        configured, rerun captioning and overwrite output.caption before
        retrying generation
   e. Write metadata and the optional evaluation sidecar
```

This loop is finite: one initial candidate plus at most `pipeline.retry`
replacement candidates; `pipeline.retry` defaults to `1`. For a replacement,
increment the seed first, optionally regenerate the caption when configured,
then run generation. Observe-only external checks do not affect the branch.
For ambiguous external POST recovery, follow
[evaluator-setup-guide.md](evaluator-setup-guide.md#gating-and-failures).

Config authoring has a separate bound: validate inside the launched container,
fix the concrete schema error, and retry at most 3 times total. If the third
validation fails, stop before endpoint preflight or inference and report the
remaining errors rather than repeating the edit loop.

## External-evaluator example: re-render a video as a rainy night

This example is specific to the CT3 Omni config that enables external Cosmos
evaluation. Other configs use only the endpoints, media locations, and
evaluators declared in their own YAML.

1. **Model:** input is a video and the goal is a scene-attribute change → Cosmos
   3 Super. Start from `config_video_transfer_CT3_omni.yaml`
   (`augmentation.model.name: cosmos3-transfer-omni`; role `video_transfer`,
   `openai.video.sync` adapter).
2. **Endpoints:** keep the four shared-network roles: VLM and LLM for captioning
   and checker-side MCQs, `video_transfer` for Cosmos3-Super, and `evaluator` for
   Arbitrator. Arbitrator's checker containers must also resolve the VLM/LLM
   service names.
3. **Inputs/outputs:** set one externally readable source URI at
   `data.0.inputs.rgb`;
   `cosmos_evaluator.inputs.original_video_urls.camera_front_wide_120fov`
   references it automatically. The map key matches `stream_key`.
   Set remote video/caption/metadata destinations and a unique relative
   `output_storage_prefix`.
4. **Target attributes:** set
   `captioning.llm.variables.weather_condition: ["raining"]` and
   `lighting_condition: ["night"]`; set the external normalized selections to
   `weather: rainy` and `time_of_day: night`.
5. **Credentials/preflight:** inject protected multistorage credentials, then
   require Arbitrator health/discovery to report both external checks healthy
   and scheduled.
6. **Run** (inside the container): `uv run --no-sync modules/cli.py --config configs/cookbook/video-data-augmentation/config_video_transfer_CT3_omni.yaml`.

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

This skill runs remote inference in Docker with credentials and outbound network
access.
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
- **Networking.** Prefer a user-defined bridge. Attach local service containers
  to it and replace host `localhost` endpoint URLs with their container DNS
  names. The default bridge is sufficient when every endpoint is remote. Never
  use host networking on shared, multi-tenant, or production hosts. On an
  isolated legacy host, use it only after an explicit user decision and a clear
  warning that it removes network namespace isolation.
