# Common Gotchas

## Single-GPU hosts fail before the profile starts

The stock Alerts profile `.env` reserves GPU 0 and assigns GPU 1 to RT-VLM,
LLM, and VLM. On a GPU-0-only host, changing only those three IDs to `0` still
fails because GPU 0 remains reserved. Set the Step 7 final values in the source
`${VSS_PROFILE_DIR}/.env` so it has `RESERVED_DEVICE_IDS=''`,
`FIXED_SHARED_DEVICE_IDS='0'`, and all four workload IDs set to `0` (the stock
`RT_CV_DEVICE_ID='0'` is retained). This selects the `local_shared` NIM compose
path. Do not patch `generated.env`, because `dev-profile.sh` regenerates it.

## `dev-profile.sh up` wipes the ONNX model

Every full `dev-profile.sh up` run deletes `$VSS_DATA_DIR/models/` and re-downloads default models. Re-stage your ONNX model under `${VSS_DATA_DIR}/models/yolo/` after each full redeployment.

## Docker creates a ghost directory for file mounts

If you specify a volume mount at a file path that doesn't exist yet, Docker creates a directory there. Mount the parent directory instead and reference the file by name inside the container.

## Custom parser `.so` reports `dlsym failed`

The function name in `parse-bbox-func-name` (nvinfer config), the `extern "C" bool NvDsInferParseCustomYoloE(...)` declaration in your `.cpp`, and the `CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloE)` macro must all match exactly.

## Redis stream contamination when running multiple VST stacks

If multiple VST stacks share a Redis instance, the SDR can pick up stale stream events from the wrong stack, hit `NUM_SENSORS` limit, and leave DeepStream with `Active sources: 0`.

Diagnose:
```bash
docker exec mdx-redis redis-cli hkeys sdr-deepstream
docker exec mdx-redis redis-cli xread COUNT 5 STREAMS vst.event 0
```

Fix: remove stale entries and re-publish the correct event:
```bash
docker exec mdx-redis redis-cli hdel sdr-deepstream <stale-key>
docker exec mdx-redis redis-cli xadd vst.event '*' sensor.id "<correct-event-json>"
```
