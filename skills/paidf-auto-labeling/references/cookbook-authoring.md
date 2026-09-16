# PAIDF Auto-Labeling Cookbook Authoring

Use this skill when the user asks for a reusable agentic auto-labeling recipe,
domain-specific pipeline cookbook, or production experiment template.

## Workflow

Linear authoring steps. There are no execution branches - this skill produces a
cookbook, it does not run one. Handoffs to other skills (prompts, detector
settings) are delegations, not branches, and *Guardrails* are constraints applied
while authoring, not flow steps.

1. **Start from the closest existing cookbook** under repo-root `cookbooks/`.
2. **Read the schema.** Read [cookbook-schema.md](cookbook-authoring/cookbook-schema.md)
   before adding or changing cookbook files; when authoring a brand-new scenario,
   also read [scenario-authoring.md](cookbook-authoring/scenario-authoring.md).
3. **Write the recipe (single step).** Populate the cookbook's declarative fields
   using the *Examples* skeleton and
   [cookbook-schema.md](cookbook-authoring/cookbook-schema.md) as the field reference;
   prefer additive overrides over copying whole service configs. The individual
   fields are configuration, not separate flow steps.
4. **Delegate specialized content.** Author VLM/LLM prompt files with the
   prompt-authoring skill and RF-DETR/SAM3 settings with the
   detection-and-tracking stage skill, then reference those files from the
   cookbook. This is a delegation, not a branch - return here when done.
5. **Validate.** Confirm path references with the asset checks (workflow-stage
   skill) and dry-run the cookbook:
   `make run SCRIPT=workflow-runner:main ARGS='--cookbook-file <config> --container-dry-run'`.
   If the dry-run or an asset check fails, fix the cookbook and re-run this step
   (bounded: until it passes, or STOP and report a blocker you cannot resolve).

## Examples

Minimal video cookbook skeleton (canonical stages, shared question bank):

```yaml
pipeline: video                       # or image
data:
  - inputs: { media_path: data/video.mp4 }
    output: { out_dir: output/auto_labeling/<scenario> }
runtime:
  model_cache_path: <model-cache>
  gpu_ids: [0]
container:
  env:
    # Bare name only: forwards the host NVIDIA_API_KEY into the container.
    # Never commit NVIDIA_API_KEY=<value>.
    - NVIDIA_API_KEY
workflow:
  nodes:
    super_resolution: { stage: super_resolution }
    detection_and_tracking: { stage: detection_and_tracking, needs: [super_resolution] }
    captioning:
      stage: captioning
      needs: [detection_and_tracking]
      args: [--input-source, tracking, --prompt-file, ../prompts/dense_caption/scene_prompt.md]
    visual_qa:
      stage: visual_qa
      needs: [captioning]
      args: [--generation-mode, window-vlm-llm, --include-reasoning]
    reasoning: { stage: reasoning, needs: [visual_qa] }
    training_export: { stage: training_export, needs: [reasoning] }
endpoints:
  vlm: { url: http://host.docker.internal:18002/v1, model: qwen-vl }
  llm: { url: http://host.docker.internal:18003/v1, model: qwen-llm }
visual_qa:
  question_bank_file: question_bank.json    # shared with reasoning substages
reasoning:
  reasoning: { targets: [scene_description, open_qa, mcq_openended, bcq_openended] }
training_export:
  formats: [tao-vl-reason-v1.0]
  output_dir: output/auto_labeling/<scenario>/dataset
```

Validate the new cookbook with a dry-run before any real run:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/<scenario>/configs/pipeline.yaml --container-dry-run'
```

## References

- [cookbook-schema.md](cookbook-authoring/cookbook-schema.md) - cookbook fields and shape.
- [scenario-authoring.md](cookbook-authoring/scenario-authoring.md) - authoring or
  adapting a domain scenario cookbook.
- [VLM/LLM endpoints](../../../docs/user-guide/vlm-llm-endpoints.md) - how to
  supply `NVIDIA_API_KEY` from the host or CI without writing the value into YAML.

## Guardrails

- Do not include secrets, tokens, absolute user home paths, or production-only
  host paths in committed cookbooks.
- A bare `NVIDIA_API_KEY` (and any other bare name) in `container.env` is a
  runtime secret: the runner copies the host environment value into every stage
  container. Never write `NVIDIA_API_KEY=<value>` or any other `NAME=value`
  secret. Do not print these values in logs, dry-run output, pipeline
  artifacts, or committed files.
- Write `container.env` as a list. Use bare variable names for host-provided
  secrets and `NAME=value` entries only for fixed non-secret configuration.
- Inject secrets at run time from the host (`export NVIDIA_API_KEY=...` with
  `--container-env NVIDIA_API_KEY`) or from CI/CD masked variables / a secrets
  manager. Do not bake values into YAML. See
  [vlm-llm-endpoints.md](../../../docs/user-guide/vlm-llm-endpoints.md).
- Keep placeholder paths explicit, for example `<experiment-root>` and
  `<model-cache>`.
- Do not introduce a second cookbook schema when the runner's YAML/JSON config
  can express the workflow.
- Do not depend on non-PAIDF pipelines, modules, commands, or file locations. A
  committed PAIDF cookbook must be self-contained within this repo and runnable
  through `workflow-runner:main`.
- Every cookbook should be dry-runnable with
  `make run SCRIPT=workflow-runner:main ARGS='--cookbook-file <config> --container-dry-run'`.
- Question-bank files should be shared by the `visual_qa` and `reasoning` stages
  when they describe the same auto-label evidence.
- The current canonical stages are `super_resolution`, `detection_and_tracking`,
  `captioning`, `visual_qa`, `reasoning`, `training_export`, and
  `person_attribute_search` (the Visual Attribute Search product stage). Do
  not reference the retired single `daft_export` stage; DAFT `task/` writing now
  belongs to `reasoning`, and dataset aggregation belongs to `training_export`.
- Use `workflow.nodes` when a recipe needs dependency ordering or repeated stage
  executions. Give every node a stable ID and keep its service CLI flags,
  question banks, and output namespaces together in that node's `args`.
- When a cookbook targets a **reasoning-capable model** (for example Gemini 3
  Flash), raise `max_tokens` for the `reasoning` and `visual_qa` LLM substages.
  Reasoning models spend part of the budget on internal thinking, so a low cap
  truncates or empties outputs. A generous positive value (e.g. `32768`, within
  the model's output limit) avoids the thinking-token tax without code changes.
