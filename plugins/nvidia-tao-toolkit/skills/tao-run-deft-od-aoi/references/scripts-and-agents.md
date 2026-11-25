# Bundled scripts and delegated skills

Use this map to locate the owner of each stage. Application scripts prepare or
validate contracts; model and data skills own their container actions.

## Application scripts

| Script | Purpose |
|---|---|
| `prepare_deft_od_aoi_sources.py` | Canonicalize customer shards, delegate role merging to Data Services `annotations merge`, validate KPI/test/real/clean roles, and emit `sources.json`. |
| `init_deft_od_aoi.py` | Validate normalized roles, freeze the policy, and initialize durable state. |
| `prepare_deft_od_aoi_retrieval.py` | Build candidate crops/specs or per-iteration gap-query/mining specs. |
| `deft_od_aoi_round_robin_selection.py` | Materialize deterministic per-query selections into the standard retrieval-stage artifacts. |
| `admit_deft_od_aoi_coco.py` | Admit unique real/clean/synthetic sources into cumulative binary COCO. |
| `prepare_deft_od_aoi_measurement.py` | Emit KPI/test RT-DETR inference and loose/strict gap specs. |
| `prepare_deft_od_aoi_training.py` | Emit direct main training or deterministic probe specs. |
| `select_deft_od_aoi_training.py` | Select the best probe or KPI-best checkpoint and optional extension. |
| `commit_deft_od_aoi_stage.py` | Verify artifacts and atomically advance application state. |
| `prepare_deft_od_aoi_synthesis.py` | Normalize exact strict FNs into AnomalyGenNext preparation input. |
| `resolve_deft_od_aoi_synthesis.py` | Resolve task weights or emit one-time fine-tuning requests. |

## Delegated stages

| Stage | Owning skill |
|---|---|
| RT-DETR inference and training | `tao-train-rtdetr` |
| Loose and strict OD gap analysis | `tao-analyze-gaps-od-map` |
| Candidate and query embedding | `tao-generate-image-embeddings` |
| Global unique-image mining (`max_similarity`) | `tao-mine-od-images` |
| Per-query selection (`round_robin_similarity`) | Application-owned deterministic selector |
| AnomalyGenNext preparation and AMP | `tao-prepare-anomalygennext-inputs` |
| Missing task-weight training | `tao-finetune-anomalygennext` |
| Synthetic OD generation | `tao-generate-od-defects` |
| Authorization, records, and dispatch | `tao-launch-workflow` plus selected platform |

Read the owning skill and its `references/skill_info.yaml` before dispatch.
Do not guess container commands or inline a replacement implementation.
