# Video Lake Curation

Use this playbook for large-scale search and targeted curation over raw video lakes or Cosmos Curator output lakes. It is instruction-only: do not add helper scripts or modify upstream `cosmos-curator` unless the user explicitly asks for code changes.

## Decision Flow

Choose the workflow before launching expensive jobs:

1. **Search before curation** when the raw lake is large, the target is narrow, or GPU cost matters. Build a lightweight discovery index first, search broadly, then curate candidate videos or time windows.
2. **Search after curation** when Cosmos outputs already exist or the curated lake will be reused for many future queries. Search over clip captions, embeddings, SAM3-verified events, classifier labels, and metadata.
3. **Hybrid** for production data lakes. Run cheap discovery across everything, fully curate candidates, then search the curated subset again.

For large datasets, prefer the hybrid workflow unless the user explicitly wants to fully curate the entire lake first.

## Search Before Curation

Use this for large raw collections such as 20k videos.

1. Inventory source videos: path, duration, resolution, source/camera, size, and any existing labels.
2. Sample sparse frames or short windows at low FPS.
3. Generate cheap discovery metadata: road type, weather, lighting, camera view, traffic density, and incident hints.
4. Search the discovery metadata with high recall.
5. Build a candidate manifest with source video, optional start/end time, match reason, and confidence.
6. Run full Cosmos Curator only on selected candidates.

Keep discovery broad. Missing relevant source videos is worse than admitting extra candidates that can be filtered later.

## Search After Curation

Use this when a Cosmos output folder exists.

Inspect and search:

- `summary.json` for processed video and clip counts.
- `metas/v0/*.json` for source video, clip path, duration span, captions, validity, token counts, and errors.
- `v0/all_window_captions.json` when present for aggregate caption search.
- `sam3_events/*.json` for grounded event verification, event categories, event captions, timestamps, and instance IDs.
- `sam3_objects/`, `sam3_instances/`, and `sam3_tracked/` when object-level filtering matters.
- `iv2_embd/` or `iv2_embd_parquet/` when semantic video retrieval is available.

Search in stages:

1. Retrieve candidates using captions, source names, classifier labels, and SAM3 event text.
2. Apply hard filters for duration, source, scene type, event type, and negative concepts.
3. Use SAM3 event verification as the second source of truth for target events when available.
4. Use embeddings for semantic variants that lexical search may miss.
5. Rerank with a VLM or LLM judge when precision matters.
6. Emit a final manifest with clip path, source video, start/end time, reason, confidence, and curation decision.

## SAM3 As Event Verification

SAM3 should support curation reasoning, not just create annotations. Use
SAM3 tracks and per-event captions as an evidence layer that verifies or
rejects VLM/classifier candidate events.

Verification flow:

1. The VLM captioner or classifier proposes a candidate event.
2. SAM3 tracks the specific actors needed to prove or disprove it.
3. The event-captioning stage checks the event against tracked instance
   IDs, trajectories, temporal order, contact/proximity, occlusion, and
   counter-evidence.
4. The curation decision uses the verified judgment: `keep`, `drop`, or
   `review`.

The event-captioning output should answer:

- Did the target event happen: `present`, `absent`, or `uncertain`?
- Which tracked instances support the judgment?
- What time span contains the evidence?
- What visible evidence supports the judgment?
- What counter-evidence or ambiguity remains?
- What concise reasoning summary connects the evidence to the judgment?
- Should this clip be kept, dropped, or reviewed for the requested dataset?

For strict Cosmos schemas, encode this reasoning compactly in the
accepted `event_caption` field. For schemas that allow richer JSON, use
fields such as `verification`, `reasoning_summary`, `decision_basis`,
`evidence`, `counter_evidence`, `confidence`, and `curation_decision`.
Do not ask the VLM to output long free-form chain-of-thought. Ask it to
reason internally and emit concise auditable reasoning fields: cap
`reasoning_summary` at 25 words, then use `decision_basis` for track
association, temporal relation, visible change, category choice, and
counter-check.
For SAM3 event captions, require dominant-incident selection from object
boxes before category selection. Add an `incident_objects` audit for directly
involved ids, exclude ambient traffic, and use most-specific category
precedence so a motorcycle crash is not labeled `collision_aftermath` or
`normal_traffic_flow`.
For `person_on_ground_in_roadway`, require posture and zone evidence: lying,
sitting, slumped, or motionless in a lane/crosswalk/intersection. A standing or
walking pedestrian on a shoulder, sidewalk, median, or roadside is
counter-evidence, not a positive event.

Use the recommended traffic-event SAM3 verification profile in
`references/sam3-config.md` (canonical YAML keys, the `sam3:` vs `enable_sam3:`
gotcha, and tuning rationale). For lake search, leave `sam3_prompts` out of the
profile and derive the tracked objects from the query (positive/negative
concepts below) instead of a fixed list — see the `sam3_prompts` note in that
reference.

## Highway Query Pattern

For a request like "find highway videos instead of intersections," translate the request into positive and negative evidence.

Positive concepts:

- `highway`, `freeway`, `expressway`
- `multi-lane road`, `divided roadway`, `mainline traffic`
- `ramp`, `overpass`, `barrier`, `median`
- steady vehicle flow, lane-following traffic, high-speed roadway geometry

Negative concepts:

- `intersection`, `traffic light`, `crosswalk`
- `stop line`, `turning across traffic`, `pedestrian crossing`
- `parking lot`, `driveway`, `urban street`

If the user does not define "highway," use this operational default: multi-lane through-road footage without visible intersection controls such as traffic lights, crosswalks, or stop lines.

## Prompt Pattern

For discovery or captioning prompts, ask the VLM to explicitly classify scene type:

```text
Describe this traffic video for data curation. Include:
- road scene type: highway/freeway, intersection, urban street, parking lot, ramp, tunnel, bridge, or other
- evidence for the scene type
- visible road controls: traffic lights, stop signs, crosswalks, lane markings, ramps, dividers
- traffic density and dominant actors
- weather, lighting, and camera viewpoint
- unusual events or safety-critical behavior
```

For highway retrieval, require the model to state why the scene is highway-like and whether intersection evidence is absent or present.

## Manifest Requirements

A useful curation manifest should include:

- `source_video`
- `clip_path` if already curated
- `start_time` and `end_time`
- `scene_type`
- `positive_evidence`
- `negative_evidence_checked`
- `confidence`
- `reason`
- `sam3_verification`: `present`, `absent`, `uncertain`, or `not_run`
- `curation_decision`: `keep`, `drop`, or `review`
- `cosmos_output_root` or run ID

Use JSONL for machine workflows and CSV for quick human review.

## Validation

Before saying a large curation/search task is done:

1. Compare input video count with `summary.json` processed counts.
2. Check clip counts: transcoded, passed, captioned, embedded.
3. Inspect `metas/v0` for nonempty `errors` arrays or invalid clips.
4. Confirm SAM3 event files exist and are nonempty when event search is part of the claim.
5. For event-specific datasets, verify SAM3 evidence supports the keep/drop decisions and is not merely descriptive annotation.
6. Sample-review positive results and hard negatives.
7. Report residual risk, especially if using only lexical search without embeddings, SAM3 verification, or visual reranking.

## Guardrails

- Do not modify upstream `cosmos-curator` unless the user explicitly asks for code changes there.
- Prefer cookbook/YAML configuration over source patches.
- Do not run full SAM3/Qwen curation over a huge lake for an exploratory query without explaining cost and offering a discovery pass.
- Keep raw data, Cosmos outputs, derived indexes, and final manifests separate.
- Treat search results as candidates until sample review or reranking validates them.
