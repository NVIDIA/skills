# Troubleshooting

> Configs live under `configs/cookbook/<use-case>/`. See the [cookbook index](../../../configs/cookbook/README.md) for the folder layout.

Run inference and schema validation **inside the `paidf-augmentation:1.1.0` Docker container** (see SKILL.md → *Before You Start, Step 2*) for a consistent environment. All inference is remote (no local model weights), so most failures are config-resolution or endpoint/auth issues, not missing local dependencies.

## Config Validation Errors

All configs are validated by the `PipelineConfig` Pydantic model. Common issues:

| Error | Cause | Fix |
|-------|-------|-----|
| `no endpoint matched selector '<name>'` | `augmentation.model.name` doesn't match any endpoint `id` or `role` | Add an endpoint whose `role` matches the model (or set `model.name` to an endpoint `id`) |
| `selector '<name>' matches multiple endpoints with role '<role>'` | Two endpoints share the model's role | Give the endpoints `id`s and set `augmentation.model.name` to the specific `id` |
| `endpoint <id> resolves to unknown adapter '<x>'` | `adapter:` isn't one of the known contracts | Use one of: `openai.chat.completions`, `openai.images.edits`, `openai.video.sync`, `openai.video.async`, `nim`, `passthrough` |
| captioning with VLM but no `vlm`-role endpoint | VLM captioner configured without a matching endpoint | Add an endpoint with `role: vlm` |
| `'text' and 'file_path' are mutually exclusive` | Both set in `captioning.llm` | Use one or the other |
| `captioning.llm.text / file_path cannot be combined with captioning.vlm` | Invalid captioner combination | Use text/file alone, or VLM+LLM without text/file |
| `Template has placeholders {x} not found in variables` | Text template references an undefined variable | Add the missing variable to `captioning.llm.variables` |

## Runtime / Endpoint Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `401 Unauthorized` from an endpoint | Missing/wrong API key | Set the env var named by the endpoint's `api_key_env` (e.g. `VEO_API_KEY`) and forward only that name with `docker run -e VEO_API_KEY`. Local endpoints need no key. |
| `404` on a chat/video route | Wrong adapter or `url` for the contract | Match the adapter to the server's route (`/v1/chat/completions` vs `/v1/images/edits` vs `/v1/videos/sync` vs NIM `/v1/infer`); a hosted async video model (Veo) needs `adapter: openai.video.async`. base_url should end at `/v1` for chat (the SDK appends `/chat/completions`). |
| `Connection refused` | Endpoint not reachable from the container | With `--network host`, `curl <url>` from the host; for remote URLs use the default bridge network; on macOS/Windows use `host.docker.internal`. |
| Request hangs for a long time | The endpoint is wedged/slow and the adapter timeout is high | Set a saner `timeout:` on the endpoint; verify the endpoint responds (`curl`). |
| Hosted NVCF NIM returns `202` or times out at the gateway | Generation outlives the NVCF hold-open window | Use the `nim` adapter and set endpoint `timeout:` above queue plus generation latency. The adapter sends `NVCF-POLL-SECONDS` and polls `NVCF-REQID` automatically; `NVCF_POLL_SECONDS` configures each poll window (default/max 300). |
| Verification fails repeatedly | Generated output doesn't match target attributes, or the VLM can't see a mid-video event | Increase `pipeline.retry`, adjust `augmentation.parameters.guidance`/`sigma`, or raise `vlm_verification.frames` so the VLM samples more frames |
| Hallucination check fails | Excessive motion artifacts in output | When artifacts are real, reduce them (lower `augmentation.parameters.sigma` or improve generation). Passing needs score ≥ threshold, so **raising** the threshold makes it harder — only **lower** `evaluators[].hallucination_check.threshold` for a false positive. |
| `data_processing.alignment` errors / `No module named 'cupy'` | Running the alignment post-processor without a GPU/cupy | Alignment is GPU-only; add `--gpus all` (or a specific device) to `docker run` — the host driver is provided at runtime and the CUDA runtime wheels are baked into the image. Plain remote inference does not need this. |
| Output not written / permission denied on `data/` | Container runs as UID 10000 and can't write the mounted directory | Run as your own user so ownership matches: `--user "$(id -u):$(id -g)"`. If that's not possible, grant just UID 10000 (e.g. `setfacl -R -m u:10000:rwX data/`) rather than making `data/` world-writable. |

## Typical Inference Timeline

Approximate per-sample timing (remote endpoints):

| Stage | image edit | Cosmos Transfer / image-to-video |
|-------|------------|----------------------------------|
| Config validation + captioner init | <2s | <2s |
| VLM+LLM captioning | ~3s | ~5s |
| Generation | ~25–35s | ~2–2.5 min (video) |
| Attribute verification (multi-frame) | ~10s | ~12s |
| **Total (no retries)** | **~1 min** | **~3 min** |

Defect Image Generation image-edit runs (`config_image_edit_defect_chat_api.yaml` / `config_image_edit_defect_images_api.yaml`) add ~30–40s per sample for the GPU `data_processing.alignment` post-processor (single-level MI search at the generator's output resolution).
