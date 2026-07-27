# Setup Reference — Checkpoint Download and Verification

Full detail for Phase 0. Read when troubleshooting checkpoint issues or
running setup for the first time.

---

## Prerequisites

- `cosmos-predict2` conda env active. The scripts do **not** create it.
- `HF_TOKEN` exported (Hugging Face access token — required for the
  `nvidia/Cosmos-Predict2-*` repos, which are gated).
- `huggingface_hub` and `huggingface-cli` installed (already in the env per
  the tutorial). If missing: `pip install -U huggingface_hub`.

---

## What gets downloaded (~140 GB total)

| Path | Source | Size | Used by |
|---|---|---|---|
| `checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/` | HF | ~18 GB | FT, SDG |
| `checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/` | HF | ~64 GB | FT, SDG |
| `checkpoints/google-t5/t5-large/` | HF | ~3 GB | FT, SDG (configurable via `ag_config.t5_model_name`) |
| `checkpoints/google-t5/t5-11b/` | HF | ~45 GB | FT, SDG (alternative to t5-large) |
| `checkpoints/nvidia/C-RADIO-V3/model.safetensors` | HF | ~375 MB | FT, eval |
| `checkpoints/NVDINOV2/nv_dinov2_classification_model.ckpt` | NGC (anonymous) | ~1.2 GB | FT, SDG |
| `checkpoints/facebook/dinov2-large/` | HF | ~1.2 GB | training-validation + eval |
| `checkpoints/sam2/sam2.1_hiera_large.pt` | facebook public | ~857 MB | AMP |
| `checkpoints/Qwen/Qwen3-VL-4B-Instruct/` | HF | ~9 GB | AMP |

---

## Step 1: Download

```bash
${ANOMALYGEN_SCRIPTS}/download_checkpoints.sh \
    [--checkpoint-dir checkpoints]
```

What the script does:
- Refuses to start if `HF_TOKEN` is unset.
- Runs `huggingface-cli login --token $HF_TOKEN --add-to-git-credential` once
  before invoking the in-repo `scripts.download_checkpoints` module.
- Skips the upstream module entirely when every artifact it would produce is
  already on disk (avoids redownloading NVDINOV2 via wget).
- Skips SAM2 and Qwen3-VL when already present.
- Idempotent — safe to re-run after an interrupted download.

## Step 2: Verify

```bash
${ANOMALYGEN_SCRIPTS}/check.sh \
    [--checkpoint-dir checkpoints]
```

- Exits `0` when every required artifact is present.
- Exits `1` otherwise — lists each missing path with the remediation command.
- Run this before any training or SDG job to catch missing files early.

---

## Error handling

| Symptom | Fix |
|---|---|
| "HF_TOKEN unset" on start | `export HF_TOKEN=<your_token>` |
| HF 401 Unauthorized | Re-issue token at https://huggingface.co/settings/tokens with read access; accept license on each `nvidia/Cosmos-Predict2-*` model page |
| Disk full mid-download | ~140 GB required; free space or use `--checkpoint-dir` on a larger volume |
| `huggingface-cli: command not found` | `pip install -U huggingface_hub` |
| NVDINOV2 redownloading every run | Confirm `check.sh` exits 0; skip logic checks artifact presence before invoking the module |
| SAM2 / Qwen3-VL downloading again | Delete the partial file and re-run |
