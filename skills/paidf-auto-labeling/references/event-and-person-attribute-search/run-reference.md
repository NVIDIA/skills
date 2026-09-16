# Visual Attribute Search run reference

## Cookbook matrix

| Cookbook | Required input | Ordered nodes | Endpoints | Important mounts |
|---|---|---|---|---|
| `pipeline_image_attributes_pas.yaml` | one attribute JSON | PAS | LLM | attribute parent, query prompt |
| `pipeline_image_multiview_pas.yaml` | representative image + local image group | captioning -> Visual QA -> PAS | VLM + LLM | image group, person bank |
| `pipeline_video_pas_reasoning.yaml` | video(s) | tracking -> captioning -> event VQA -> reasoning -> person VQA -> PAS -> export | VLM + LLM | media/output, cache, SAM3, both banks, config |
| `pipeline_video_epas.yaml` | video(s) | tracking -> captioning -> anomaly VQA -> person VQA -> PAS | VLM + LLM | media/output, cache, SAM3, both banks, config |

The PAS container is `event-and-person-attribute-search-service`. It uses the LLM
endpoint but no VLM endpoint. Captioning and Visual QA use their own containers.

## Dry-run and execution

Replace `<config>` with one of the four names and use a gitignored local copy for
real deployment values:

```bash
make run SCRIPT=workflow-runner:main \
  ARGS='--cookbook-file cookbooks/visual_attribute_search/configs/<config> --container-dry-run'
```

After a successful preflight and explicit approval, remove
`--container-dry-run`. Add `--container-ensure-images build-if-missing` only when
preflight found missing images.

## Expected artifacts

- Attribute-only: `sidecars/person_attribute_search/bundle_attributes.json` and
  `bundle_queries.json` (plus optional `bundle_hitl.json`).
- Multi-view: `sidecars/captioning/image_caption.json`,
  `sidecars/visual_qa/{items.json,windows.normalized.json}`, then PAS
  `attributes.json` and `queries.json`.
- Video reasoning: detection tracks/crops, caption windows,
  `visual_qa_event_verification/` evidence and DAFT QA, reasoning outputs,
  `visual_qa_per_track/{items.json,windows.normalized.json}`, PAS `pas.json` and
  `chunk_queries.json`, the configured contextual mirror, and training export.
- Video EPAS: detection tracks/crops, caption windows,
  `visual_qa_anomaly/items.json`,
  `visual_qa_per_track/{items.json,windows.normalized.json}`, PAS `pas.json`,
  `chunk_queries.json`, `pas_anomaly.json`, query buckets, and
  `contextual/person_attributes.json`.

For video, mount SAM3 weights read-only at `/models/sam3` and set
`SAM3_MODEL_PATH=/models/sam3`. If host model servers must be reached from a
non-host container network, use a host name reachable from that network.
