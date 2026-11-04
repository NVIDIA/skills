# Setup Reference — Checkpoint Download and Verification

Full detail for Phase 0. Read when troubleshooting checkpoint issues or
running setup for the first time.

---

## Prerequisites

- `cosmos-predict2` conda env active. The scripts do **not** create it.
- `HF_TOKEN` (Hugging Face access token — required for the
  `nvidia/Cosmos-Predict2-*` repos, which are gated) in the environment —
  exported in your shell or sourced from a user-approved env file.
- `huggingface_hub >= 1.x` installed, which provides the `hf` CLI (already in
  the env per the tutorial). If missing: `pip install -U huggingface_hub`.

---

## What gets downloaded

Defaults fetch the **2B** base + t5-large + the shared encoders/guardrail
(**~40 GB**). The **14B** base and **t5-11b** are opt-in (`--model-sizes`
incl. `14B` / `--with-t5-11b`); pulling both brings the full set to **~150 GB**.

| Path | Source | Size | Default? | Used by |
|---|---|---|---|---|
| `checkpoints/nvidia/Cosmos-Predict2-2B-Text2Image/` | HF | ~18 GB | ✓ | FT, SDG |
| `checkpoints/nvidia/Cosmos-Predict2-14B-Text2Image/` | HF | ~64 GB | `--model-sizes 14B` | FT, SDG |
| `checkpoints/google-t5/t5-large/` | HF | ~3 GB | ✓ | FT, SDG (configurable via `ag_config.t5_model_name`) |
| `checkpoints/google-t5/t5-11b/` | HF | ~45 GB | `--with-t5-11b` | FT, SDG (alternative to t5-large) |
| `checkpoints/nvidia/Cosmos-Guardrail1/` | HF | ~7 GB | ✓ | SDG image guardrail |
| `checkpoints/nvidia/C-RADIO-V3/model.safetensors` | HF | ~375 MB | ✓ | FT, eval |
| `checkpoints/NVDINOV2/nv_dinov2_classification_model.ckpt` | NGC (anonymous) | ~1.2 GB | ✓ | FT, SDG |
| `checkpoints/facebook/dinov2-large/` | HF | ~1.2 GB | ✓ | training-validation + eval |
| `checkpoints/sam2/sam2.1_hiera_large.pt` | facebook public | ~857 MB | ✓ | AMP |
| `checkpoints/Qwen/Qwen3-VL-4B-Instruct/` | HF | ~9 GB | ✓ | AMP |

---

## Step 1: Download

```bash
${ANOMALYGEN_SCRIPTS}/download_checkpoints.sh \
    [--checkpoint-dir checkpoints] [--model-sizes "2B"] [--with-t5-11b]
```

What the script does:
- Refuses to start if `HF_TOKEN` is unset.
- Defaults to the **2B** base + t5-large. Pass `--model-sizes "2B 14B"` (or
  `--model-sizes 14B`) for the 14B base, and `--with-t5-11b` to also fetch
  T5-XXL (~45 GB); both are off by default. `--model-sizes` accepts a
  space-separated subset of `{2B, 14B}`.
- Runs `hf auth login --token $HF_TOKEN --add-to-git-credential` once
  before invoking the in-repo `scripts.download_checkpoints` module.
- Skips the upstream module entirely when every artifact it would produce
  (for the requested sizes) is already on disk (avoids redownloading NVDINOV2
  via wget).
- Skips SAM2 and Qwen3-VL when already present.
- Idempotent — safe to re-run after an interrupted download.

## Step 2: Verify

```bash
${ANOMALYGEN_SCRIPTS}/check.sh \
    [--checkpoint-dir checkpoints] [--model-sizes "2B"]
```

- Exits `0` when every required artifact is present.
- Exits `1` otherwise — lists each missing path with the remediation command.
- Verifies the requested base size(s), t5-large **or** t5-11b (either
  satisfies training), the Cosmos-Guardrail1 image guardrail, and the shared
  encoders. `--model-sizes` must match what you downloaded (default `2B`).
- Run this before any training or SDG job to catch missing files early.

---

## Error handling

| Symptom | Fix |
|---|---|
| "HF_TOKEN unset" on start | `export HF_TOKEN=<your_token>`, or `set -a; source /path/to/.env; set +a` |
| HF 401 Unauthorized | Re-issue token at https://huggingface.co/settings/tokens with read access; accept license on each `nvidia/Cosmos-Predict2-*` model page |
| Disk full mid-download | ~40 GB for the default set (~150 GB with 14B + t5-11b); free space or use `--checkpoint-dir` on a larger volume |
| `hf: command not found` | `pip install -U huggingface_hub` (needs `>= 1.x`) |
| NVDINOV2 redownloading every run | Confirm `check.sh` exits 0; skip logic checks artifact presence before invoking the module |
| SAM2 / Qwen3-VL downloading again | Delete the partial file and re-run |
