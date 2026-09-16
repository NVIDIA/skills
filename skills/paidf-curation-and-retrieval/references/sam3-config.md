# SAM3 Tracking — Canonical Configuration

Shared SAM3 configuration for the split pipeline, referenced by
`references/video-curation.md` and `references/video-lake-curation.md`. Treat
SAM3 as a second source of truth: it grounds and verifies candidate events for
the VLM rather than acting as a broad annotation phase. It adds a separate
`sam3` Pixi environment plus extra GPU/disk cost, so enable it only when
track-level evidence is needed.

## Canonical YAML keys (gotcha)

Use `sam3:` and `event_captioning:` — **not** `enable_sam3:` /
`enable_event_captioning:`, which are silently ignored so SAM3 never runs while
the pipeline still reports success. See
`references/gotchas.md` §SAM3 / event captioning canonical keys for the full
explanation and the unit-test coverage in `tests/unit/test_pipeline_config.py`.

## Recommended traffic-event verification profile

```yaml
sam3: true
sam3_prompts:                  # objects to track; set per domain (see note below)
  - "a car"
  - "a motorcycle"
  - "a pedestrian"
sam3_target_fps: 3.0
sam3_max_clip_duration_s: 45.0
sam3_session_reset_s: 5.0
sam3_score_threshold_detection: 0.6
sam3_det_nms_thresh: 0.1       # pin SAM3 native NMS; null uses the same ~0.1
sam3_new_det_thresh: 0.8
sam3_fill_hole_area: 256
sam3_recondition_every_nth_frame: 8
sam3_recondition_on_trk_masks: true
sam3_high_conf_thresh: 0.5
sam3_high_iou_thresh: 0.4
sam3_region: contour           # contour = mask outlines; box = rectangles
sam3_output_format: native     # native | coco | mot
sam3_write_annotated_video: false  # forced on when event captioning is enabled
sam3_annotated_video_trails: false
sam3_annotated_video_label_style: "id"  # id | name | none
sam3_annotated_video_mask_opacity: 0    # 0-100 fill; 0 is outline/box only
```

**Rationale**: run SAM3 at modest FPS, allow clips up to 45s, recondition every
8 sampled frames, use tracker masks during reconditioning, require 0.6 detection
confidence and 0.8 new-track confidence, and use 0.5/0.4 confidence/IoU
association thresholds. If split windows are shorter than 45s,
`sam3_max_clip_duration_s` is only a guardrail; if the split duration is raised,
VRAM scales with `sam3_target_fps * clip_duration`.

**On `sam3_prompts`**: this lists the objects to track and should be set per
domain. The video-lake-curation verification profile omits it because that
workflow derives the tracked objects from the search query rather than a fixed
list; add `sam3_prompts` explicitly when the target objects are known up front.

**On `sam3_region`**: Curator 2.3.0 default is `contour` (mask outlines).
`box` draws axis-aligned rectangles from `box_xyxy` and skips contour
extraction. Use `box` when overlays should look like bounding boxes (warehouse
cookbook). `sam3_annotated_video_mask_opacity` only controls mask fill; it does
not switch contour vs rectangle.

**On `sam3_output_format`**: `native` writes Curator per-frame tracks.
`coco` and `mot` are export formats.
