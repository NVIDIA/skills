# Recorded inference entrypoints

Source: `scripts/README.md` at revision `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Full-source SHA256: `c34d4126346c93556c98110eeadca98c2dcbf0935b0c6b13781226ac18cc406a`.

## scripts/README.md:1-14

# Scripts

Inference and evaluation scripts for FastGen.

## Inference

Generate images and videos using pretrained or distilled models.

| Script | Modality | Modes |
|--------|----------|-------|
| [`image_model_inference.py`](inference/image_model_inference.py) | Image | Unconditional, class-conditional, T2I, image-editing (I2I) |
| [`video_model_inference.py`](inference/video_model_inference.py) | Video, Video+Audio | T2V, I2V, V2V, Video2World |
| [`flashdreams_inference.py`](inference/flashdreams_inference.py) | Video | FlashDreams Self-Forcing Wan 2.1 T2V example |
| [`vlm_infer.py`](inference/vlm_infer.py) | Image, Video | VLM-as-judge scores for existing media listed in a JSONL file, using the endpoint and token of [`VLMLogger`](../fastgen/callbacks/README.md#log-modules) |

## scripts/README.md:34-39

#### Checkpoint selection

`--ckpt_path` loads student `net` and configured EMA weights, not the teacher. Student sampling uses the first
configured EMA when present; teacher sampling uses its constructor/pretrained configuration. To evaluate a
fine-tuned SFT checkpoint, select student sampling. Without `--ckpt_path`, the scripts use constructor/pretrained
weights and do not discover the latest training checkpoint. Check startup's loaded keys and selected weight source.
