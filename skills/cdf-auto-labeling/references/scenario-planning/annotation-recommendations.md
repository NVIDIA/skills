# Annotation Recommendations

Use this matrix to recommend annotation outputs. Keep the recommendation tied to
modality, domain, and downstream consumer.

## By Modality

### Image

Usually useful:

- Image caption: visible scene/object/activity evidence.
- VQA evidence: question-bank answers for domain-specific attributes.
- `reasoning`: when downstream expects DAFT `task/`/contextual files.
- `training_export`: when outputs feed a training/export dataset.
- `person_attribute_search`: for per-person attribute search labels.
- Detection: only if object boxes are needed; skip for scene-level captions.

Avoid by default:

- Temporal localization, events, causal linkage, and video summaries unless the
  image task explicitly defines a non-temporal equivalent.
- Super-resolution unless the task needs improved readability and the SR service
  supports the input modality.

### Video

Usually useful:

- Dense window captions: temporal visual evidence and activity summaries.
- Detection/tracking: if object identity, counts, trajectories, overlays, masks,
  or object-level grounding are needed.
- VQA evidence: if the task has reusable questions or decision criteria.
- `reasoning`: if final artifacts should be DAFT `task/`/contextual outputs.
- `training_export`: if per-scene outputs should be aggregated into a dataset.
- Temporal localization: if answers need start/end times and services in the
  current branch support it.
- Causal linkage: only when event chains are visible enough to support it and
  services in the current branch support it.

Optional:

- Super-resolution for low-resolution clips before downstream stages.
- SAM3 when promptable masks/tracks or annotated videos are needed.

## Detector Choice

- Choose RF-DETR when COCO boxes and stable track IDs are enough. It is the
  default for traffic/person/vehicle inventory and is cheaper than SAM3.
- Choose SAM3 when the target objects are not COCO classes, when masks/contours
  are required, or when a VLM should reason over an annotated ID video.
- Request tracking overlays only when downstream prompts need visible IDs. For
  captioning to consume those overlays, set `--input-source tracking` or rely on
  `auto` only after verifying the overlay path exists.
- Do not let detector limitations define the annotation goal. Use VLM/VQA for
  open-vocabulary evidence and use `reasoning`/`training_export` for final
  schemas and dataset aggregation.

## By Domain

### Traffic / Roadway Safety

Recommended outputs: dense captions, vehicle/person tracks, event summaries,
temporal localization when supported by services in the current branch, VQA for
roadway conditions and incidents, `reasoning` verdicts, and `training_export`.
Use SR when clips are low resolution. Use RF-DETR for COCO traffic classes. Use
SAM3 for promptable incident actors, road debris, emergency vehicles, masks, or
annotated ID videos used by captioning/VQA.

### Warehouse / Operational Liability

Recommended outputs: dense captions, worker/equipment tracks when available,
PPE/hazard VQA, temporal localization for near-misses/incidents when supported
by services in the current branch, `reasoning`, and `training_export`.
Use RF-DETR for `person` and coarse COCO proxies such as `truck`; use SAM3 for
forklifts, pallets, pallet jacks, boxes, tools, and other promptable objects.

### Security Surveillance

Recommended outputs: person tracks, dense captions, VQA for entry/exit,
loitering, object handling, confrontation, visible weapons, and response cues.
Avoid identity or demographic inference. Use RF-DETR for person/vehicle/bag COCO
boxes; use SAM3 only for concrete visible promptable objects, not intent labels.

### Employee Conduct

Recommended outputs: person tracks when full-frame video is used, dense
captions, VQA for on-task/off-task activity, station presence, PPE/badge/uniform
visibility, and interaction tone. Avoid intent and employment-status inference.
Use SAM3 only when tool/workcell object IDs or masks materially improve evidence.

### Robotics

Recommended outputs: dense captions, task-phase VQA, success/failure evidence,
object/tool interaction evidence, safety proximity, and optional SAM3 prompts for
specific objects/tools. Avoid inferring robot intent or hidden success criteria.
Use RF-DETR for people and COCO objects; use SAM3 for robot arms, grippers,
workpieces, tools, hands, and task-specific promptable objects.

### Person Attributes

Recommended outputs: image caption and VQA/open QA about visible clothing,
accessories, carried items, pose, crop quality, and occlusion. For the Event and
Attribute Search product, run `detection_and_tracking` then
`person_attribute_search` for per-track attributes. Avoid protected or identity
attributes.

### Generic Image Annotation

Recommended outputs: image caption, scene-level VQA, optional detection for
object inventory, and `reasoning`/`training_export` when required by downstream
systems.

## Minimality Rule

Recommend the smallest stage set that produces the requested annotations. Add
expensive stages only when they improve the evidence needed by downstream tasks.
