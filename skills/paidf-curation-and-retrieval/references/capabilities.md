# Capability → Config Key Reference

Maps each cosmos-curator capability to its config keys. Keys are flat
`snake_case` argparse `dest` names; see `configs/*.yaml` for the authoritative
full-key reference with inline comments.

## Video

| Capability | Config keys | Notes |
|-----------|-------------|-------|
| TransNetV2 scene detection | `splitting_algorithm: transnetv2`, `transnetv2_*` | GPU fractional |
| Fixed-stride splitting | `splitting_algorithm: fixed-stride`, `fixed_stride_*` | Uniform clips |
| Qwen3 captioning | `captioning_algorithm: qwen3_vl_30b_fp8` or `qwen3_5_27b` | FP8 quantized, 1 GPU |
| Cosmos-R1/R2 captioning | `captioning_algorithm: cosmos_r1` or `cosmos_r2` | NVIDIA models |
| Gemini captioning | `captioning_algorithm: gemini`, `gemini_*` | API-based |
| OpenAI captioning | `captioning_algorithm: openai`, `openai_*` | API-based |
| vLLM async captioning | `captioning_algorithm: vllm_async` (auto-configured; explicit `vllm_async_*` knobs removed in v2.0.0) | Multi-GPU via `qwen_num_gpus_per_worker` + `vllm_performance_mode: throughput` |
| SAM3 tracking | `sam3` (NOT `enable_sam3`), `sam3_prompts`, `sam3_*` quality knobs | Object tracking + per-clip JSON. See `references/sam3-config.md` for the verification profile. **Wrong YAML key (`enable_sam3:`) is silently ignored - SAM3 will not run.** |
| Per-event captions | `event_captioning` (NOT `enable_event_captioning`), `event_caption_*` | VLM event JSON from SAM3 tracks; dominant-incident object audit, most-specific category precedence, concise reasoning fields. Same canonical-name gotcha as `sam3:`. |
| Caption enhancement | `enhance_captions: true`, `enhance_captions_*` | MCQ/structured output |
| Prompt variants | `captioning_prompt_variant: [default, av]` | Domain prompts |
| Custom prompts | `captioning_prompt_text: "..."` | Inline prompt string |
| Super-resolution | `super_resolution: true`, `sr_*` | SeedVR2 3B/7B/7B-sharp |
| Motion filter | `motion_filter: enable`, `motion_*` | Optical flow scoring |
| Aesthetic filter | `aesthetic_threshold: 4.5` | CLIP aesthetic model |
| Artificial text filter | `artificial_text_filter: enable`, `artificial_text_*` | OCR detection |
| VLM semantic filter | `vlm_filter: enable`, `vlm_filter_*` | VLM quality gate |
| Video classifier | `video_classifier: true`, `video_classifier_*` | Category tagging |
| InternVideo2 embedding | `embedding_algorithm: internvideo2` | Default |
| Cosmos-Embed1 embedding | `embedding_algorithm: cosmos-embed1-*` | 224p, 336p, 448p |
| OpenAI embedding | `embedding_algorithm: openai`, `openai_embedding_*` | API-based |
| Semantic dedup | Separate `dedup` pipeline, `n_clusters`, `eps_to_extract` | K-Means + cosine |
| WebDataset sharding | Separate `shard` pipeline, `target_tar_size_mb` | Training-ready tars |
| Multi-camera mode | `multi_cam: true`, `primary_camera_keyword` | Sync multi-angle |
| Presigned URLs | `input_presigned_s3_url`, `output_presigned_s3_url` | Credentialless S3 |
| Stage replay/compare | `stage_save`, `stage_replay`, `stage_compare_*` | Debug individual stages |
| Cosmos-Predict2 dataset | `generate_cosmos_predict_dataset: predict2` | Training dataset |

## Image (annotate)

| Capability | Config keys | Notes |
|-----------|-------------|-------|
| Image annotate pipeline | `pipeline: "annotate"` in `configs/image.yaml` | Still-image curation; see `references/image-curation.md` |
| Image semantic filter | `semantic_filter: enable`, `semantic_filter_*` | VLM-based image rejection |
| Image classifier | `image_classifier: enable`, `image_classifier_*` | Allow/block image taxonomy |
| Image embeddings | `embedding_algorithm: [internvideo2, clip, cosmos-embed1-*, openai]` | One vector per image |
| Image captioning | `captioning_algorithm`, `caption_prompt_variant: "image"` | Per-image VLM caption |
