# PAIDF Auto-Labeling Cookbook Schema

Cookbooks are runner configs under repo-root `cookbooks/<scenario>/`.
The PAIDF `workflow-runner` service accepts YAML or JSON through `--cookbook-file`.

- [Top-Level Fields](#top-level-fields)
- [Runner Stage Model](#runner-stage-model)
- [Directly Mapped Fields](#directly-mapped-fields)
- [Workflow Nodes](#workflow-nodes)
- [Node Args](#node-args)
- [Runtime Section](#runtime-section)
- [Container Environment](#container-environment)
- [Validation Checks To Document](#validation-checks-to-document)

## Top-Level Fields

| Field | Type | Purpose |
|---|---|---|
| `pipeline` | string or object | `video` or `image`; object form may also carry runtime defaults for compatibility. |
| `runtime` | object | Shared knobs such as `model_cache_path` and `gpu_ids`. |
| `container` | object | Container user, images, environment variables, and mounts. |
| `data` | array | Input media and output root entries. |
| `endpoints` | object | VLM/LLM endpoint URL and model defaults. |
| stage sections | object, optional / conditional | `super_resolution`, `detection_and_tracking`, `captioning`, `grounding_2d`, `referring_expressions`, `visual_qa`, `reasoning`, `training_export`, `person_attribute_search`. |
| `stage_args` | object, legacy | Compatibility input for shared stage CLI args; committed cookbooks use node `args`. |
| `workflow.nodes` | mapping or array, optional | Dependency-aware nodes with `stage`, optional `needs`, and optional node-specific `args`. |

`stage sections` optional — use `stages:` or per-stage `enabled:` flags to
control sequence. Prefer `workflow.nodes` and colocated node `args` in committed
cookbooks; `stage_args` remains accepted only for compatibility.

## Runner Stage Model

The runner maps cookbook sections to canonical container stages:

```text
video: super_resolution -> detection_and_tracking -> captioning -> grounding_2d ->
       referring_expressions -> visual_qa -> reasoning -> training_export ->
       person_attribute_search
image: detection_and_tracking -> captioning -> grounding_2d ->
       referring_expressions -> visual_qa -> reasoning -> training_export ->
       person_attribute_search
```

`grounding_2d` and `referring_expressions` may be absent until their MRs merge;
verify the local runner stage list. Plain `stages:` selections execute in this
fixed relative order. A `workflow.nodes` recipe instead executes in stable
topological node order. Nodes may repeat a stage; keep their IDs and output
namespaces distinct. If neither form is present, optional per-stage `enabled:`
flags adjust the default sequence.

`visual_qa` is optional and disabled by default. To include it, list
`visual_qa` in the `stages:` production sequence or set
`visual_qa.enabled: true` with the per-stage `enabled:` flag.

## Directly Mapped Fields

The runner currently maps these cookbook values directly:

- `runtime.model_cache_path` -> model-cache path mounted and forwarded.
- `runtime.gpu_ids` -> stage model GPU ids.
- `endpoints.vlm.url/model` -> captioning/visual_qa VLM endpoint/model.
- `endpoints.llm.url/model` -> visual_qa/reasoning LLM endpoint/model.
- `super_resolution.resolver`, `super_resolution.variant`.
- `detection_and_tracking.model`, `detection_and_tracking.tracker`,
  `detection_and_tracking.classes`.
- `visual_qa.question_bank_file` (the shared VQA/reasoning question bank).
- `reasoning` and `training_export` by passing the full cookbook config to those
  stages (DAFT `task/` writing and dataset aggregation, respectively).

All other service-specific flags belong in the corresponding workflow node's
`args`.

## Workflow Nodes

Use node arguments when two occurrences of one stage need different inputs or
output namespaces:

```yaml
workflow:
  nodes:
    anomaly_qa:
      stage: visual_qa
      needs: [captioning]
      args: [--question-bank-file, ../anomaly.json]
    person_qa:
      stage: visual_qa
      needs: [anomaly_qa]
      args: [--question-bank-file, ../person.json]
```

Path-valued node arguments are resolved relative to the cookbook config and
mounted. When a stage appears more than once, keep each execution's complete
argument list with that node so its configuration is self-contained.

## Node Args

Use node `args` for prompt files, captioning budgets, detector thresholds, SAM3
knobs, service timeouts, and feature flags:

```yaml
workflow:
  nodes:
    captioning:
      stage: captioning
      args:
        - --prompt-file
        - ../prompts/dense_caption/scene_prompt.md
        - --window-seconds
        - "4.0"
        - --sampling-fps
        - "2.0"
        - --max-frames
        - "24"
        - --resolution
        - "480"
    detection_and_tracking:
      stage: detection_and_tracking
      args:
        - --threshold
        - "0.25"
        - --save-red-id-overlay
```

Path-valued node args such as `--prompt-file`, `--image-prompt-file`, and
`--question-bank-file` are resolved relative to the cookbook config/root.

## Runtime Section

```yaml
runtime:
  model_cache_path: <model-cache>
  gpu_ids: "0"
```

The runner expects `gpu_ids` as a string (`"0"`, `"0,1"`, or `all`), not a YAML
list. A list such as `[0]` is stringified to `"[0]"` and is not a valid device
selector.

Container runtime flags that are not cookbook-specific stay on the runner CLI,
for example `--container-dry-run`, `--container-build-images`,
`--container-mount`, and `--container-env`.

## Container Environment

Write `container.env` as a list. A bare variable name such as `NVIDIA_API_KEY`
forwards the host environment value into every stage container at runtime. That
value is a secret: never commit `NVIDIA_API_KEY=<value>` (or any other
`NAME=value` secret), and do not echo it in logs, dry-run output, or pipeline
artifacts. Use `NAME=value` only for fixed, non-secret configuration. Supply
the key from the host shell or CI/CD masked variables / a secrets manager; see
[vlm-llm-endpoints.md](../../../../docs/user-guide/vlm-llm-endpoints.md).

```yaml
container:
  env:
    - NVIDIA_API_KEY
    - SAM3_MODEL_PATH=/models/sam3
```

## Validation Checks To Document

Prefer checks that a shell script or agent can perform deterministically:

- Required output path exists.
- Required output path is nonempty.
- `pipeline_state.json` exists after media-transforming stages.
- Active media exists before annotation-only stages.
- Stage log exists under `logs/` or `sidecars/logs/`.
- Dry-run output includes every intended pass-through argument.
