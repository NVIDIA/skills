# PAS Troubleshooting

## Common Issues

### Endpoint connectivity

**Symptom**: Worker logs show `ERROR: <endpoint> endpoint not ready after <timeout>s`

**Resolution**:
1. Verify the endpoint URL is reachable from within the OSMO cluster.
2. Check that the model is fully loaded: `curl -sS <endpoint_url>/v1/models | jq .`
3. Increase timeout via `ENDPOINT_WAIT_TIMEOUT_SECONDS` environment variable.
4. For in-cluster NIMs, verify the service is running: `kubectl get pods -n osmo-nims`
5. The NIMService name must match the configured endpoint host (default
   `qwen-image-edit-2511`, `qwen3-vl`, `qwen25-14b`). If you deploy under a
   different name, override `image_edit_url`/`vlm_url`/`llm_url` at submit time.
6. Triton-based Visual GenAI NIMs do not expose `/v1/models`; `endpoint_common.sh`
   falls back to probing `/v1/health/ready` for readiness.

### Image-edit endpoint returns 404 on requests

**Symptom**: Augmentation worker requests to the image-edit endpoint fail with
`404 Not Found` on `/v1/chat/completions` even though the service is "ready".

**Resolution**:
1. The image-edit endpoint must be **OpenAI-compatible** (expose
   `/v1/chat/completions` and `/v1/models`). The PAS augmentation worker calls
   the image-edit model via `client.chat.completions.create(...)` and reads the
   edited image from `choices[0].message.content[0].image_url.url`.
2. Do **not** use the Triton-based NVIDIA NIM
   `nvcr.io/nim/qwen/qwen-image-edit:1.0.0-variant`. It passes health checks and
   serves `/v1/infer` + `/v1/images/edits`, but not `/v1/chat/completions`, so
   every image-edit call returns `404` and no images are produced (captions and
   metadata still generate, then auto-labeling fails with no images to caption).
3. Deploy the checked-in `references/nim/qwen-image-edit-2511.yaml` manifest,
   which runs `vllm serve Qwen/Qwen-Image-Edit-2511 --omni`. The `--omni` flag
   is **required** — vLLM-Omni only enables the Qwen-Image-Edit diffusion serving
   path (and the chat/completions image response) when started with it. The
   plain `vllm.entrypoints.openai.api_server` entrypoint without `--omni` will
   not serve image editing.

### No pane images produced

**Symptom**: `ERROR: No pane images produced from input data`

**Resolution**:
1. Verify the input dataset follows the expected structure:
   `<person_id>/<view>.jpg` subdirectories.
2. Check that images are readable JPEG or PNG files.
3. Ensure the OSMO data URL points to the correct dataset path.

### Augmentation verification failures

**Symptom**: High `failed` count in augmentation summary

**Resolution**:
1. Check `output_metadata.json` for specific verification failures.
2. The MCQ verification may be too strict for certain attribute combinations.
3. Increase retry count in `augmentation_config.yaml` (`pipeline.retry`).
4. Verify the VLM endpoint is responsive and returning valid answers.

### Auto-labeling produces no open_qa.json

**Symptom**: `WARNING: No open_qa.json produced`

**Resolution**:
1. Verify the VLM endpoint is healthy.
2. Check that the `person_attributes` question bank exists in the container
   at `/workspace/cookbooks/person_attributes/question_bank.json`.
3. Check worker logs for VLM timeout or connection errors.

### Permission errors on output

**Symptom**: `Permission denied` when writing to output directory

**Resolution**:
1. The paidf-augmentation container runs as `nvidia` (UID 10000).
2. Ensure OSMO output paths have correct write permissions.
3. For local testing, use `chmod -R o+rwX` on the output directory.

### OSMO credential issues

**Symptom**: `USER_INPUT_REQUIRED: Provide missing secrets`

**Resolution**:
1. Set `HF_TOKEN` in your environment before running preflight.
2. Run `osmo credential list` to verify existing credentials.
3. Use `--refresh` flag on preflight to force credential update.

## Log Inspection

```bash
# View augmentation worker logs
osmo workflow logs <workflow_id> --task augmentation_worker_0 -n 500

# View auto-labeling worker logs
osmo workflow logs <workflow_id> --task auto_labeling_worker_0 -n 500

# View setup logs
osmo workflow logs <workflow_id> --task setup -n 200
```

## Performance Tuning

- **Reduce augmentation time**: Lower `n_augmentations` (default: 3).
- **Reduce verification strictness**: Add attributes to
  `evaluators[0].attribute_verification.exclude_variables` in
  `augmentation_config.yaml`.
- **Speed up captioning**: Use a faster VLM model or increase endpoint
  concurrency.
