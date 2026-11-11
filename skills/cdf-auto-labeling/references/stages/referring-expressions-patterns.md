# Referring Expressions patterns

## Prerequisites

`contextual/objects.json` must exist with per-frame instances / boxes
(`bounding_box_2d_tight`, `object_id`). Typical producers:

- `detection_and_tracking` (SAM3 or RF-DETR)
- `grounding_2d` (after caption → SAM3)

## Step 0 behavior

1. Draw numbered SoM-lite overlays (optional `--draw-box-overlay`)
2. VLM returns `mark`, `bbox_2d`, `type`, `color`, `description`
3. Link by mark id, then greedy IoU (`--min-match-iou`)
4. Keep authoritative DAFT boxes on output regions

`type` is open vocabulary (slug-normalized). Soft spelling aliases only; no
closed domain allowlist.

## Mounts (product)

- Detect→refer: SAM3 weights mount on the cookbook for the detection image only.
- Refer-only: no weight mounts; runner bind-mounts `media_path` / `out_dir`.
- Do not mount source trees or set `PYTHONPATH` for product runs.

## Validation checks

- Dry-run shows `referring-expressions-service` with VLM flags and no product
  PYTHONPATH binds.
- After a run: `sidecars/referring_expressions/referring_expressions.json` has `regions` with
  `description`; overlay path present when `--draw-box-overlay` is set.
- Failures: missing objects.json, unreachable VLM — report the concrete cause;
  do not invent phrases.
