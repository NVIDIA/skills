# Service Reuse Matrix

Use this reference before proposing a new UPA task or service.

## Decision Matrix

| Need | Prefer | Reason |
|---|---|---|
| New prompt, schema, question bank, or domain wording | Cookbook asset | No code change needed |
| New frame/window budget or model endpoint | Cookbook `stage_args` or stable stage field | Runtime recipe concern |
| New captioning mode over existing media inputs | Extend `captioning` | Captioning remains reusable IR generation |
| New detector/tracker backend option | Extend `detection_and_tracking` | Same tracking primitive |
| Track crop export or annotated media option | Extend `detection_and_tracking` | Same owned evidence |
| VQA over another prompt/question bank | Extend `visual_qa` | Same question-answer primitive |
| New DAFT `task/` artifact family | Extend `reasoning` | DAFT task writing belongs to reasoning |
| New training dataset aggregation | Extend `training_export` | Dataset aggregation belongs to the export stage |
| Video/image chunk extraction | New generic `media_chunking` service | Reusable primitive |
| Retrieval query generation from annotations | New generic `query_generation` service | Reusable primitive |
| Multi-model classification/voting | New generic `classification` or `anomaly_vote` service | Reusable primitive |
| Fleet scheduling, retries, worker pools | OSMO/Airflow/platform layer | Not workflow-runner's role |

## Existing Service Intent

- `super_resolution`: transform media only when resolution policy requires it.
- `detection_and_tracking`: produce object tracks, masks, crops, overlays, and
  tracking sidecars.
- `captioning`: produce reusable visual-language intermediate representations
  such as dense captions or window metadata.
- `visual_qa`: answer configured questions with evidence over visible content
  and available sidecars.
- `reasoning`: reason over prior evidence and write final DAFT `task/` artifacts
  from reusable sidecars.
- `training_export`: aggregate reusable sidecars and DAFT artifacts into training
  datasets.
- `person_attribute_search`: the Visual Attribute Search product stage
  (image/build target `event-and-person-attribute-search-service`).
- `workflow_runner`: compile cookbooks into container plans; do not put
  perception, annotation, or export behavior here.

## New Service Criteria

Approve a new service when all are true:

- The operation is reusable across multiple future recipes.
- Existing services would become less generic if they owned it.
- Inputs and outputs can be described as a stable sidecar contract.
- It can accept `--input-file` and operate per `DataEntry`.
- It has a focused test surface and Docker image.
- The cookbook can configure it without hard-coded experiment paths.

Reject or defer a new service when:

- It only wraps one dataset's folder layout.
- It only changes a prompt or question bank.
- It mostly renames an existing stage.
- It needs distributed scheduling semantics better owned by platform tooling.
- It would duplicate a sidecar already owned by another stage.

## Naming Guidance

Use capability names:

- `media_chunking`
- `query_generation`
- `anomaly_vote`
- `classification`
- `artifact_export` stays an illustrative generic name; `training_export` is the
  real shipped export stage and `reasoning` now writes DAFT `task/` artifacts
  (the single `daft_export` stage is retired).

Avoid pipeline names:

- `example_search_generation`
- `agentic_captioning`
- `agentic_tracking`
- `rwf_pipeline`
- `warehouse_annotation`
