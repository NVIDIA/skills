# Guard Preflight & Custom Per-Defect Counts

## Guard preflight (product mode only)

```bash
if [[ "${ANOMALYGEN_PRODUCT_MODE:-}" == "1" ]]; then
    python3 .agents/skills/anomalygen-guard/scripts/preflight.py \
        --mode ${MODE} \
        --name ${NAME} \
        --dataset-dir ${DATASET_DIR} \
        --defect-spec ${DEFECT_DESC} \
        --num-search-run ${NUM_SEARCH_RUN} \
        --model-size ${MODEL_SIZE} \
        ${CLEAN_DIR:+--clean-dir ${CLEAN_DIR}} \
        ${NUM_SDG:+--num-sdg ${NUM_SDG}} \
        ${CKPT:+--checkpoint-dir ${CKPT}} \
        ${STEP:+--step ${STEP}} \
        ${VALIDATION_JSONL:+--validation-jsonl ${VALIDATION_JSONL}}
fi
```

`--validation-jsonl` is forwarded only when the user supplied one; preflight
then verifies every `defect_spec` type appears in the file and that
`image_filename` / `mask_filename` paths exist. Auto-generated validation
JSONLs are caught upstream by `allocate_samples.py`, which refuses to
allocate 0 entries to any defect.

For `MODE=finetune_only`, omit `--num-sdg` if the user did not supply one.

## Phase 2 — custom per-defect counts

**Custom per-defect counts (skill-driven from user intent):** when the user's
natural-language request specifies counts per defect type (e.g. "give me 5
IC+bridge and 10 passive_component+missing"), translate to `--num-sdg` +
`--per-defect-counts`:

```bash
${ANOMALYGEN_SCRIPTS}/prep_testcase.sh \
    --name ${NAME} --num-sdg 15 \
    --dataset-dir ${DATASET_DIR} \
    --clean-dir ${CLEAN_DIR} \
    --defect-spec ${DEFECT_DESC} \
    --amp-output-dir ag_inference/${NAME}/amp \
    --output-jsonl ${JSONL} \
    --per-defect-counts '{"IC+bridge":5,"passive_component+missing":10}'
# passive_component+excess_solder gets 0 (not in dict).
```

Sum of `--per-defect-counts` should equal `--num-sdg`. If they disagree, the
script prints a stderr warning and uses the override sum. Confirm the
allocation with the user before invoking when the user's intent is
ambiguous (e.g. "each defect 10" + "total only 1" — must clarify).
