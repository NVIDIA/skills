# Inference Phase Commands

Exact command invocations for Phase 5 (per-sample search round) and Phase 7
(filter + regen + eval).

## Phase 5 — run a search round

For `r` in `1..NUM_SEARCH_RUN`, after writing
`${ROUNDS}/round_${r}/draws.json`:

```bash
${ANOMALYGEN_SCRIPTS}/run_round.sh \
    --base-jsonl ${JSONL} \
    --draws ${ROUNDS}/round_${r}/draws.json \
    --output-dir ${ROUNDS}/round_${r} \
    --real-path ${DATASET_DIR} --anomaly-types ${DEFECTS[@]} \
    --checkpoint-dir ${CKPT} --step ${STEP} \
    [--model-size ${MODEL_SIZE}]
```

## Phase 7 — filter + regen + eval

```bash
python3 -m scripts.utilities.filter_with_regen \
    --searched-dir ${SEARCHED} \
    --per-sample-csv ${SEARCHED}/per_sample.csv \
    --threshold ${NN_THRESHOLD} \
    --num-sdg ${NUM_SDG} \
    --rounds-dir ${ROUNDS} \
    --regens-dir ${REGENS} \
    --dataset-dir ${DATASET_DIR} \
    --clean-dir ${CLEAN_DIR} \
    --defect-spec ${DEFECT_DESC} \
    --real-path ${DATASET_DIR} \
    --anomaly-types ${DEFECTS[@]} \
    --checkpoint-dir ${CKPT} --step ${STEP} \
    --model-size ${MODEL_SIZE} --num-gpus ${NUM_GPUS}
```
