<!-- miagent-template-source: assets/challenge_template/shared/visualization/README.md -->
<!-- miagent-template-contract: active; keep the miagent-template-source loaded for creation/material updates; reload after resume or compaction -->
# Shared Visualization

Challenge-level debugging visualization used by every baseline. Replace the
stub in `visualizer.py` before the first baseline smoke run, then reuse that
implementation for all experiments. Do not fork separate renderers inside
baseline folders.

| Field | Value |
|---|---|
| task type | `<segmentation / registration / generation / representation / VLM / other>` |
| implementation | `shared/visualization/visualizer.py` plus any challenge-wide helpers |
| canonical command | `python shared/visualization/visualizer.py --manifest <run>/visualization_manifest.json --output <run>/result_viewer.html` |
| normalized inputs | `<case IDs, image paths, reference, prediction, intermediates, metrics, and provenance>` |
| output | `baselines/<baseline>/runs/<exp>/result_viewer.html` plus relative assets |
| validated by | `<smoke-run Outcome and viewer link>` |

Follow the relevant `references/<task>-analysis.md` contract. The shared
renderer owns orientation, display ranges, colors/legends, synchronized views,
case selection, and HTML layout so evidence is comparable across baselines. A
baseline may add a thin exporter for its native output format, but it must feed
the shared renderer rather than create a separate gallery.

Run this renderer after every experiment that produces predictions. Before an
Important Completed Run drives the next training decision, expand its viewer to
satisfy the relevant task-specific result-analysis skill. An
execution-only repair may reuse an existing viewer when its job evidence
explains why no new predictions exist.

## Stable API

Keep these two callable entry points in `visualizer.py`:

```python
visualize_case(
    inputs,
    prediction,
    output_dir,
    *,
    ground_truth=None,
    case_id,
    intermediates=None,
    metrics=None,
    metadata=None,
)

visualize_run(manifest, output_html)
```

`inputs`, `prediction`, and `ground_truth` may be a path, a list of paths, or a
named path mapping. For example, segmentation can use `inputs={"image": ...}`
and `prediction={"mask": ...}`; registration can use
`inputs={"moving": ..., "fixed": ...}` and
`prediction={"warped": ..., "displacement": ...}`. The direct function is
for quick debugging. The run function consumes a JSON manifest with the same
fields for every case and builds the all-case `result_viewer.html`. Also keep
the canonical CLI so trainers and evaluation jobs can call the renderer without
importing challenge code.

Keep saved predictions and metrics as source artifacts; visualization should be
cheap to rerun without retraining. Extend this implementation when a result
reveals a missing diagnostic view, then regenerate affected run viewers. Treat
renderer interfaces as challenge-wide APIs: preserve existing inputs or update
all affected baseline adapters together.
