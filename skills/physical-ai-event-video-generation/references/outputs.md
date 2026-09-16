# Event Video Generation output layout

All run artifacts are written under `<output_directory>/<run_id>/`, where `<output_directory>` comes
from the payload and `<run_id>` is the Airflow `dag_run_id`.

```text
cosmos/<video_key>/<augmentation_index>/
  config.yaml                     # generated per augmentation
  output.mp4                      # generated anomaly video (required by output validation)
  caption.txt                     # caption
  metadata.json                   # generation metadata
auto_labeling/<video_key>/<augmentation_index>/
  contextual/
    objects.json
    instances.json
    person_attributes.json        # only when the scene has crop-eligible person tracks
    pas_queries.json               # only when the scene has crop-eligible person tracks
  sidecars/
    detection_and_tracking/tracks.json   # absent means the scene had no person tracks
    captioning/video_captions.json
    visual_qa_anomaly/items.json
    visual_qa_per_track/items.json
    visual_qa_per_track/windows.normalized.json
    person_attribute_search/       # only when the scene has crop-eligible person tracks
      pas.json
      chunk_queries.json
      pas_anomaly.json
anomaly_dataset/
  dataset.json                    # canonical summary manifest
  <video_key>_aug<augmentation_index>/
    raw/video.mp4
    contextual/                   # copied wholesale from auto_labeling/<video_key>/<augmentation_index>/contextual
    sidecars/                     # copied wholesale from auto_labeling/<video_key>/<augmentation_index>/sidecars,
      cosmos/                     #   plus a cosmos/ sidecar with the generation config, caption, and metadata
        config.yaml
        caption.txt
        metadata.json
reports/                          # only when enable_performance_reporting is true
  paidf_orchestration_stats.yaml
  paidf_orchestration_stats.html
```

`<video_key>` is derived from the seed image's filename (or storage key, for a single-file
`input_path`) — see `derive_video_key` in `dags/shared/utils/video_input_utils.py`. It is not a
person ID; Event Video Generation has no person-ID input hierarchy.

`anomaly_dataset/dataset.json` is the canonical summary manifest. Its `metadata` reports
`total_scenes` and `original_inputs`; each entry identifies `scene_id`, `input_key`,
`augmentation_index`, `scene_path`, `auto_labeling_source_path`, and a `paths` object with
`video`, `config`, `caption`, `metadata`, `contextual`, and `sidecars` locations. A Cosmos
augmentation that exhausts its Airflow retries is excluded from the dataset entirely rather than
appearing with missing fields — cross-check `total_scenes` against
`processed images * num_augmentation` if you expect every requested augmentation to be present.

Use:

```bash
python scripts/summarize_results.py --results-dir /tmp/evg-results
python scripts/summarize_results.py --results-dir /tmp/evg-results --json
```

A successful DAG performs its own output validation twice: `cosmos_augmentation.validate_outputs`
requires `config.yaml`, `output.mp4`, `caption.txt`, and `metadata.json` for every expected
`<video_key>/<augmentation_index>` pair, and `validated_output.validate_outputs` requires the
`contextual/` and `sidecars/` annotation files under `auto_labeling/` for every scene (person
attribute search sidecars are required only for scenes with at least one crop-eligible person
track; a scene with no eligible person track is logged, not treated as a failure).
