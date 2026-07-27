# Phase 1 Fine-Tune Commands

The exact command sequence for Phase 1 (skip when `MODE=inference_only`). Step 3
generates the training config — show it to the user and confirm before writing.

```bash
# Step 1: Validate dataset structure — derive anomaly types
python3 -m scripts.utilities.validate_dataset ${DATASET_DIR}

# Step 2: Generate validation JSONL (skip if user provided VALIDATION_JSONL)
# num_SDG = total training mask count from Step 1 output
# --mode validation is required (prep_testcase.sh default is inference).
${ANOMALYGEN_SCRIPTS}/prep_testcase.sh \
    --name validation_${NAME} \
    --num-sdg <total_mask_count> \
    --dataset-dir ${DATASET_DIR} \
    --clean-dir ${CLEAN_DIR} \
    --defect-spec ${DEFECT_DESC} \
    --amp-output-dir ag_inference/validation_${NAME}/amp \
    --output-jsonl ag_inference/validation_${NAME}/testcase.jsonl \
    --mode validation
VALIDATION_JSONL=ag_inference/validation_${NAME}/testcase.jsonl

# Step 3: Generate training config — show to user and confirm before writing
python3 -m scripts.utilities.generate_config \
    --name ${NAME} --dataset-dir ${DATASET_DIR} \
    --defect-spec ${DEFECT_DESC} --validation-jsonl ${VALIDATION_JSONL} \
    --output ag_configs/${NAME}.yaml \
    --model-size ${MODEL_SIZE} --max-iter ${MAX_ITER} \
    --save-iter ${SAVE_ITER} --validation-iter ${VALIDATION_ITER} \
    --lr ${LR} --batch-size ${BATCH_SIZE} \
    --image-size ${IMAGE_SIZE}

# Step 4: Launch training (run in background)
${ANOMALYGEN_SCRIPTS}/launch_training.sh \
    --ag-config ag_configs/${NAME}.yaml \
    --num-gpus ${NUM_GPUS} \
    --model-size ${MODEL_SIZE}
```

After training, derive `CKPT` and `STEP` (uppercase model_size in path):

```bash
MODEL_SIZE_UPPER="${MODEL_SIZE^^}"
CKPT="./results/anomaly_gen/${NAME}/${NAME}_training_FP32_lr${LR}_bs=${BATCH_SIZE}_${MODEL_SIZE_UPPER}_${IMAGE_SIZE}x${IMAGE_SIZE}"
# STEP = highest nn_score step from validation logs (see references/finetune.md)
```

If `MODE=finetune_only`: stop here.
