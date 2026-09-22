# Bare OK/NG AOI Annotation

## Record contract

```json
[
  {
    "id": "aoi_001_93c3e56d",
    "images": ["boards/aoi_001.png", "boards/golden_001.png"],
    "conversations": [
      {
        "from": "human",
        "value": "Compare the AOI image with the golden reference. Return OK or NG."
      },
      {"from": "gpt", "value": "NG"}
    ]
  }
]
```

The annotation file root is one JSON array, not JSONL. `images` is always
`[AOI, golden_reference]`. The final assistant value is exactly `OK` or `NG`.
This skill does not accept rich answers, reasoning tags, captions,
multiple-choice fan-out, or label-derived prose.

`id` is **required on the Proxy and Benchmark splits**: `cosmos-rl-evaluate`
hard-indexes `item["id"]` and reuses it as the per-sample output filename, so a
record without one fails the evaluator with `KeyError: 'id'` — the first GPU job
the workflow submits. It must be unique within the file and filesystem-safe. A
stable derivation such as `f"{stem(images[0])}_{sha1(images[0])[:8]}"` works.
Validate it with `validate_sharegpt.py --require-id`.

Mining Pool and generated Train files do not need an `id`; the training loader
reads media with `.get()`. The bundled emitters therefore do not produce one.
Carrying one anyway is valid and is validated the same way, which is useful for
tracing a mined record back to its source row.

`"$PYTHON" scripts/check_annotations.py --print-contract` prints the authoritative
per-role field list; its `ROLE_CONTRACT` table is the source of truth if this
document ever disagrees.

## Align mined paths

Mining returns paths, not full training records. Align them to the recorded
Mining pool:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/emit_mined_sharegpt.py" \
  --mined-parquet "$RESULTS_DIR/$LABEL/mining/mined_filtered.parquet" \
  --source-annotations "$MINING_ANNOTATIONS" \
  --media-root "$MEDIA_ROOT" \
  --emit-relative \
  --output "$RESULTS_DIR/$LABEL/assemble/mined_sharegpt.json" \
  --summary "$RESULTS_DIR/$LABEL/assemble/emit_mined_summary.json"
```

The join tries resolved/exact paths and then a unique basename. Missing or
ambiguous matches hard-stop. Each emitted record inherits the source prompt,
golden reference, and exact label.

## Emit synthetic records

AnomalyGen writes paired `reconstructed_image/` (generated defect) and
`original_image/` (clean source) files, which is already the
`[AOI, golden_reference]` shape. Each generated sample becomes one exact `NG`
record:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/emit_sdg_sharegpt.py" \
  --sdg-csv "$RESULTS_DIR/$LABEL/anomalygen/sdg/SDG_result.csv" \
  --media-root "$MEDIA_ROOT" \
  --prompt-from "$MINING_ANNOTATIONS" \
  --emit-relative \
  --output "$RESULTS_DIR/$LABEL/anomalygen/sdg_sharegpt.json"
```

The prompt is inherited from the Mining pool so synthetic and mined records ask
the same question. A missing or empty image on either side of a pair
hard-stops. See `references/tao-generate-anomalies.md`.

## Assemble monotonically

```bash
"$PYTHON" "$SKILL_ROOT/scripts/assemble_training_json.py" \
  --new-json "$RESULTS_DIR/$LABEL/assemble/mined_sharegpt.json" \
  --new-json "$RESULTS_DIR/$LABEL/anomalygen/sdg_sharegpt.json" \
  --validation-json "$PROXY_ANNOTATIONS" \
  --validation-json "$BENCHMARK_ANNOTATIONS" \
  --dedupe \
  --output "$RESULTS_DIR/$LABEL/assemble/train_iter_${ITERATION}.json" \
  --summary "$RESULTS_DIR/$LABEL/assemble/assemble_summary.json"
```

Repeat `--new-json` once per producer; omit the AnomalyGen input when the
stage was skipped. For `iter1`, omit `--previous-json`; the mined and
synthetic records become the first training set. For later iterations add
`--previous-json "$PREVIOUS_TRAIN_JSON"` using the preceding iteration's
combined training JSON.

## Validate

```bash
"$PYTHON" "$SKILL_ROOT/scripts/validate_sharegpt.py" \
  --annotations "$RESULTS_DIR/$LABEL/assemble/train_iter_${ITERATION}.json" \
  --media-root "$MEDIA_ROOT" --require-files \
  --summary "$RESULTS_DIR/$LABEL/validate/validation_report.json"

"$PYTHON" "$SKILL_ROOT/scripts/validate_split_contract.py" \
  --workspace "$WORKSPACE" \
  --synthetic "$RESULTS_DIR/$LABEL/anomalygen/sdg_sharegpt.json" \
  --train "$RESULTS_DIR/$LABEL/assemble/train_iter_${ITERATION}.json"
```

For N>1, also pass
`--previous-train "$RESULTS_DIR/iter$((ITERATION - 1))/assemble/train_iter_$((ITERATION - 1)).json"`.
Omit `--previous-train` only for `iter1`, and omit `--synthetic` when the
current iteration skipped AnomalyGen.

`--validation-report` is the `validate_sharegpt.py --summary` output; keep the
`validate_split_contract.py --summary` beside it as a sibling artifact. Their
shapes differ — only the first has a top-level `mode` and an integer `records`,
which is what the committed validation stage records.

The iteration cannot train until this report records
`mode=bare_okng`, a positive record count, exact labels, unique targets, and
existing files. The split validator must also prove that every generated Train
target comes from Mining, the immediate `--previous-train` seed, or the current
iteration's `--synthetic` output. It verifies that the new Train retains every
preceding record and that none of these targets overlap Proxy or Benchmark.
