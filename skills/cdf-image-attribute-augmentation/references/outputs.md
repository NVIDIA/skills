# Image Attribute Augmentation output layout

All run artifacts are written under `<output_directory>/<run_id>/`, where `<output_directory>` comes
from the payload and `<run_id>` is the Airflow `dag_run_id`.

```text
preprocessing/<person_key>/
  <person_key>.json               # combined-pane metadata
cosmos/<person_key>/<augmentation_index>/
  config.yaml                     # generated per augmentation
  output.jpg                      # augmented image (required by output validation)
  output.txt                      # caption
  output_metadata.json            # selected attributes and edit prompt
  postprocessing/
    augmented_data.json
    augmented_imgs/<seed>_aug<index>/
auto_labeling/<person_key>/<augmentation_index>/
  ...                             # attribute-search scene outputs
augmented_dataset/
  augmented_data.json             # canonical summary manifest
  <person_key>_aug<index>/
    raw/
    contextual/
    task/
    sidecars/
      augmented_data.json
reports/                          # only when enable_performance_reporting is true
  cdf_orchestration_stats.yaml
  cdf_orchestration_stats.html
cosmos_skipped.json               # present when some augmentations were skipped
```

The `auto_labeling/` directory name is retained in code even though the payload section is now
`event_and_person_attribute_search`.

`augmented_dataset/augmented_data.json` is the canonical summary manifest. Its `metadata` reports
total IDs/scenes, and each entry identifies the source person, augmentation, selected attributes,
easy/medium/hard queries, verification data, and scene paths.

Sync `<output_directory>/<run_id>/augmented_dataset/` into `/tmp/iaa-results/`, then summarize it.
`--results-dir` may be the dataset directory or the run directory above it; the summarizer reads
the canonical manifest and ignores the per-scene `sidecars/augmented_data.json` copies:

```bash
python scripts/summarize_results.py --results-dir /tmp/iaa-results
python scripts/summarize_results.py --results-dir /tmp/iaa-results --json
```

Compare expected scenes (`processed person IDs * num_augmentation`) with manifest entries. A
successful DAG performs its own output validation: it requires `output.jpg` for each cosmos
augmentation, and requires each augmented-dataset scene directory to contain objects under `raw`,
`contextual`, `task`, or `sidecars`.

That validation does not cover attribute metadata. If a candidate's `output_metadata.json` is
missing, dataset generation logs `Missing optional output metadata` and still writes the entry,
with empty `selected_attributes`, no easy queries, and the generic `person` / `Person.` medium and
hard queries. The summarizer's `Quality warnings` section (`quality_warnings` in `--json` output)
lists these entries, entries missing any clothing attribute, and skipped augmentations. Treat any
warning as a quality problem even when the run state is `success`.

If you used `scripts/workflow.py` against the webserver API, `workflow.py download` recreates
relative paths locally from time-limited signed URLs and rejects absolute or traversal paths.
