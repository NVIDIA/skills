# SOP Inference Microservice Evaluation Prompt

Evaluate `@ds_sop_microservice` from scratch using the `deepstream-sop` skill.

Read first:

- `deepstream-sop/SKILL.md`
- `deepstream-sop/references/skill_12_evaluation_workflow.md`

Use the environment settings provided by the user for:

- `DDM_MODEL_PATH`
- `VLLM_MODEL_PATH`
- `MODEL_ROOT_DIR`
- `HOST_CACHE`
- `MEDIA_STORAGE_DIR`
- `ACTION_CONFIG_PATH`
- `VLM_PROMPT_PATH`
- `VLM_FPS`
- `TEST_VIDEO_PATH`
- optional Basler camera serial and live-camera chunk settings

Required outcome:

1. Run static validation before building.
2. Build the Docker image.
3. Launch the service with `deploy/.env`.
4. Run the API unit tests.
5. Run file-video API evaluation.
6. Run requested latency, camera, or Kafka evaluation.
7. Fix issues found during evaluation.
8. Record commands, results, timings, and fixes in a Markdown report.
