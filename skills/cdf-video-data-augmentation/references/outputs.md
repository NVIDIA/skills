# Video Data Augmentation outputs

All artifacts are written below `<output_directory>/<dag_run_id>/`.

```text
cosmos/<video_key>/<augmentation_index>/
  config.yaml
  output.mp4
  caption.txt
  metadata.json

auto_labeling_input/<video_key>/...
auto_labeling_output/<video_key>/<augmentation_index>/...

training_export/auto_labeling_input/...
training_export/auto_labeling_output/...

output/report.json
```

`auto_labeling_input` contains labels for source videos. `auto_labeling_output` contains labels for
complete Cosmos candidates that passed enabled evaluator gates. Their training exports remain
separate; there is no combined final dataset manifest.

The final validator expects every requested input/augmentation pair to have a Cosmos candidate
directory, requires `output.mp4`, `caption.txt`, and `metadata.json` for successful candidates,
requires at least one successful candidate overall, and checks that both auto-labeling and both
training-export prefixes contain objects. Evaluator-rejected or incomplete candidates should be
reported separately from operationally successful candidates.

Download one run, not the entire shared output prefix:

```bash
aws s3 sync \
  "<output_directory>/<dag_run_id>/" \
  /tmp/vda-results/

python3 scripts/summarize_results.py --results-dir /tmp/vda-results
python3 scripts/summarize_results.py --results-dir /tmp/vda-results --json
```

The summarizer counts candidate completeness, evaluator states, and files under both labeling and
training-export branches. It does not decode media. For acceptance, also use `ffprobe` or another
trusted decoder on generated `output.mp4` files and compare the observed candidates with
`processed input videos × num_augmentation` and the recorded evaluator outcomes.

`output/report.json` is useful reporting evidence when present, but it does not replace Airflow's
terminal DagRun state or final artifact validation.
