# Pre-Flight Checks and Summary

## Pre-Flight

Resolve everything possible before asking the user. In order:

1. Locate workspace root, specs, CSVs, checkpoints, augmentation assets. Derive a timestamped run directory: `RESULTS_DIR=<workspace>/results/run_$(date +%Y%m%d_%H%M%S)`. If resuming an existing run, set `RESULTS_DIR` to the existing run directory instead (detect by checking for `results/run_*/deft_state.json`). All references to `results/` throughout this skill mean `${RESULTS_DIR}/`.

   **Host Python deps.** The DEFT loop needs `pandas`, `numpy`, `matplotlib` (KPI analysis), `pyarrow` (parquet I/O for routing and mining), `huggingface_hub` (backbone staging), and `boto3` (S3 ops). Verify with `python3 -c "import pandas, numpy, matplotlib, pyarrow, huggingface_hub, boto3"`. If any are missing, set up a venv:
   ```bash
   python3 -m venv ~/.venvs/deft
   ~/.venvs/deft/bin/pip install pandas numpy matplotlib pyarrow huggingface_hub boto3
   ```
   Invoke scripts via that interpreter — on Ubuntu 24.04+ / fresh Brev boxes a bare `pip3 install --user` hits PEP 668. Alternatively run analysis inside the TAO toolkit image. Do not silently skip — KPI plots and parquet I/O are part of every loop's output.
2. Read the relevant `references/*.md` files for command syntax and output contracts. See `## Stage Reference Modules` in `references/scripts-and-agents.md` for the stage→skill mapping.
3. Source `<workspace>/.env` if it exists (`set -a; source <workspace>/.env; set +a`). Then verify the credentials the workflow actually consumes:

   | Variable | Required for | Image prefix it gates |
   |---|---|---|
   | `NGC_KEY` | All nvcr.io image pulls — TAO toolkit (train/infer/deploy/data services) and the paidf-anomalygen SDG container | `nvcr.io/nvstaging/tao/*`, `nvcr.io/nv-metropolis-dev/*` |
   | `HF_TOKEN` | Pre-Flight HuggingFace model downloads (ChangeNet backbone, Cosmos diffusion, T5, C-RADIO-V3, DINOv2, SAM2, Qwen-VL, SigLIP) — cached under `augmentation/anomalygen/base_checkpoints/`. Also gates the PCB reference dataset auto-fetch. | huggingface.co |

   Both variables must be non-empty. The single `NGC_KEY` must have read access to both `nvstaging/tao` (TAO Toolkit images) and `nv-metropolis-dev` (paidf-anomalygen). If either is missing, show the user `.env.example` (next to this skill), ask them to copy it to `<workspace>/.env` and fill in values, and do not proceed until set.
4. `docker login nvcr.io` once with `NGC_KEY` (username `$oauthtoken`, password = the key). nvcr.io stores one credential per host. Do not fall back to host-side TAO wrappers.
5. **Resolve container image refs from `versions.yaml`.** The rest of this skill — including the Pre-Flight Summary's `docker image inspect` line, every stage launch, and the `references/*.md` files — references three env vars. They are **not** defined elsewhere; resolve them here using `scripts/resolve_versions_key.py` (the single owner of `versions.yaml` schema knowledge) and `export` them so all downstream commands see them:

   ```bash
   SB=${TAO_SKILL_BANK_PATH:-~/tao-skills-external}
   export TAO_PYT_IMAGE=$($SB/scripts/resolve_versions_key.py images.tao_toolkit.pyt)
   export TAO_DS_IMAGE=$($SB/scripts/resolve_versions_key.py  images.tao_toolkit.data_services)
   export AG_IMAGE=$($SB/scripts/resolve_versions_key.py      images.metropolis_sdg.paidf_anomalygen)
   ```

   | Env var | `versions.yaml` key | Used by |
   |---|---|---|
   | `TAO_PYT_IMAGE` | `images.tao_toolkit.pyt` | `train`, `evaluate`, `rca` (TAO toolkit pyt container) |
   | `TAO_DS_IMAGE` | `images.tao_toolkit.data_services` | `data_mining` (TAO data services container) |
   | `AG_IMAGE` | `images.metropolis_sdg.paidf_anomalygen` | `anomalygen` (paidf-anomalygen container) |

   The script exits non-zero (with a diagnostic on stderr) if a key is missing or empty. Hard stop here — without the export, bash silently substitutes `""`, the next step's `docker image inspect` reports `0` MISSING for every image, and the failure mode points at the wrong root cause.
6. Verify every image resolved in step 5 is present locally (`docker image inspect "$TAO_PYT_IMAGE" "$AG_IMAGE" "$TAO_DS_IMAGE"`).

   **Architecture compatibility check.** The AnomalyGen (`$AG_IMAGE`) container is published as amd64-only and will fail silently on arm64 hosts (e.g. DGX Spark). This surfaces only after schema generation, credential injection, and a 24 GB download — so check it now:

   ```bash
   HOST_ARCH=$(uname -m)   # x86_64 on amd64, aarch64 on arm64
   AG_ARCHS=$(docker manifest inspect "$AG_IMAGE" 2>/dev/null \
     | python3 -c "import sys,json; [print(p['platform']['architecture']) for p in json.load(sys.stdin).get('manifests',[])]" \
     2>/dev/null || echo "unknown")
   echo "Host arch: $HOST_ARCH  |  AG image platforms: $AG_ARCHS"
   ```

   Map `x86_64` → `amd64` and `aarch64` → `arm64` before comparing. Hard stop with a clear message if the host architecture is not in the image's platform list — there is no emulation path for GPU workloads.

   **GPU-arch runnability probe.** Matching CPU arch isn't sufficient — the image's CUDA build must also support the host GPU's compute capability (e.g. DGX Spark `sm_121` vs a `cu128` build passes the manifest check but fails at the first CUDA call). Probe it directly: `docker run --rm --gpus all "$TAO_PYT_IMAGE" python3 -c "import torch; torch.zeros(1).cuda()"` — a non-zero exit or `no kernel image is available` means the build can't target this GPU; hard stop.

7. Apply the path rule: pre-create iter dirs under `${RESULTS_DIR}/iter${ITER}/` and mount `<workspace>` into containers at the same absolute path. Workflows enforce their own container-level invariants (entrypoints, env vars); the loop just supplies the workspace mount and the resolved image URI.
8. Verify GPU count. Probe the three AnomalyGen override slots under `augmentation/anomalygen/` (`checkpoints/<project>/`, `base_checkpoints/`, `datasets/<project>/`) and report their status in the Summary. **Empty slots are not missing — auto-fetch from HuggingFace is the default and requires no user action.** NVIDIA publishes the PCB fine-tuned checkpoint (`nvidia/Cosmos-AnomalyGen-PCB-2B`) and the PCB reference dataset (`nvidia/Cosmos-AnomalyGen-PCB-Dataset`) publicly on HuggingFace; paidf-anomalygen downloads them automatically on first use. Users who want to provide their own fine-tuned checkpoint or custom dataset can pre-stage the directory to override. Do not ask the user about missing AnomalyGen assets — treat empty slots as `will auto-fetch from HF (default)` and proceed. If `base_checkpoints/` is pre-staged, export its host path as `COSMOS_MODELS_DIR` for downstream mounts. Stage the ChangeNet pretrained backbone by running `scripts/stage_backbone.py --workspace <workspace>`, then set `specs/baseline_spec.yaml::model.backbone.pretrained_backbone_path` to the staged file and bind-mount it per `references/visual-changenet.md` → *Pre-Flight responsibility*. Staging is mandatory — hard-stop if the script exits non-zero; there is no URL fallback. See `references/paidf-anomalygen.md` for invocation and mount layout.
9. **GPU memory sanity check.** ChangeNet classify with C-RADIOv2-B (ViT-B) at the spec defaults (`batch_size: 64`, `image_width/height: 224`, `cls_weight: [1.0, 10.0]`, learnable difference modules) OOMs on a single 48GB-class GPU. Inspect `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` and warn if the assembled spec's `dataset.classify.batch_size` is too large for the available memory: as a rule of thumb, **≤ 16 on 48GB GPUs, ≤ 8 on 24GB GPUs**. Surface the recommendation in the Pre-Flight Summary's `GPUs` row — let the user accept or override before launch rather than failing 30 seconds into training.
10. Run train/validation leakage check before resuming any prior run.

Ask one consolidated question only for missing required inputs. Never ask about a parameter with a default.

**Required input — `max_iterations`.** No default; ask the user if not supplied and do not proceed past Pre-Flight without it. If the user gives a time limit instead, convert it to an estimated `max_iterations` using the §Runtime Estimate per-iteration figure and surface the estimate for confirmation.

**Defaults:**

- `training_epochs`: `num_epochs` from `specs/baseline_spec.yaml`. For a small seed set (~200 rows) use **10** — ChangeNet on the 150M-param C-RADIOv2-B backbone overfits a few-hundred-row set past ~10 epochs (val_loss climbs, FAR@recall=100% degrades). The bundled `references/baseline_spec.yaml` template ships `num_epochs: 10` for this reason. Raise toward 20 only once the combined CSV grows into the low thousands of rows across iterations.
- `num_SDG`: 20 (per-iteration AnomalyGen output budget; raise explicitly when more synthetic coverage is needed)
- `min_similarity` (mining cosine cutoff): 0.9 — read from `config.mining_filter.min_similarity` in `deft_state.json`; the literal `0.9` referenced in Pipeline step 4 is just the fallback default.
- workspace root: user prompt, else `~/workspace`
- pretrained backbone: first `*.pth`/`*.ckpt`/`*.safetensors` under `augmentation/backbone/`; if absent, stage it from `nvidia/C-RADIOv2-B` via the recipe in `references/visual-changenet.md` (HF_TOKEN required). Mandatory — a URL is not a valid value; hard-stop if it cannot be staged.
- AnomalyGen checkpoint: pre-staged `augmentation/anomalygen/checkpoints/<project>/`; if absent, auto-download from `nvidia/Cosmos-AnomalyGen-PCB-2B` on HF (HF_TOKEN required)
- AnomalyGen dataset: pre-staged `augmentation/anomalygen/datasets/<project>/`; if absent, auto-fetch from `nvidia/Cosmos-AnomalyGen-PCB-Dataset` on HF (HF_TOKEN required)
- Cosmos base models: pre-staged `augmentation/anomalygen/base_checkpoints/`; if absent, container downloads on first run (~22 GB for 2B-only, ~140 GB with 14B + T5-11b)

## Pre-Flight Summary

Once all checks pass, print this summary and **STOP — wait for explicit user approval before launching anything**. This is the one user gate in the entire workflow (see `## Agent Behavior` in SKILL.md); the loop is autonomous *after* this point, never before.

```
## DEFT Loop — Pre-Flight Summary

### Run config
| Field                          | Value                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------ |
| KPI Target                     | FAR < X% at Recall=100%                                                        |
| Max Iterations                 | N                                                                              |
| Stop condition                 | KPI met **or** max_iterations reached — reaching the KPI is not guaranteed; FAR may regress between iterations |
| Training Epochs                | N per iteration                                                                |
| Num SDG                        | N synthetic samples per iteration                                              |
| Mining cutoff                  | cosine ≥ <min_similarity> (default 0.9)                                        |
| GPUs                           | N                                                                              |
| Resuming                       | yes — iter N complete / no                                                     |
| Est. runtime                   | ~max_iterations × 33 min on RTX 6000 Ada — estimate only (+~Yh downloads if MISSING) |

### Dataset
| Field                          | Value                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------ |
| Training CSV                   | <path> (N rows)                                                                |
| Validation CSV                 | <path> (N rows)                                                                |
| KPI test CSV                   | <path> (N rows, X defect types)                                                |
| Images dir                     | <path>                                                                         |

### Augmentation
For all AnomalyGen assets, **auto-fetch from HuggingFace is the default** — no pre-staging required.
Users may override any asset by pre-staging the directory before launch.

| Field              | Value                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------ |
| AnomalyGen ckpt    | `<path>` (FOUND, step N) **or** will auto-fetch from HF (`nvidia/Cosmos-AnomalyGen-PCB-2B`, ~5 GB) **[default]** |
| Defect spec        | `<N types: type1, type2, ...>` (from staged dataset) **or** will auto-fetch from HF **[default]**                 |
| Cosmos base models | `<path>` (FOUND) **or** will auto-download on first container run (~22 GB for 2B, ~140 GB with 14B + T5-11b) **[default]** |
| SigLIP model       | `<cached / download / local path>`                                                                                 |
| Backbone           | `<path>` (FOUND) **or** will auto-download from HF (`nvidia/C-RADIOv2-B`, ~393 MB) **[default]**                  |

### Docker Images
Fill the `Image` column with the actual URI resolved in Pre-Flight step 5
(i.e. the value of the env var), not the literal `${VAR}` placeholder.
Print one row per env var so the audit trail shows exactly which tag will run.

| Env var          | Image (resolved from `versions.yaml`)                                          | Status     |
| ---------------- | ------------------------------------------------------------------------------ | ---------- |
| `TAO_PYT_IMAGE`  | `<$TAO_PYT_IMAGE>` (key: `images.tao_toolkit.pyt`)                             | OK/MISSING |
| `AG_IMAGE`       | `<$AG_IMAGE>` (key: `images.metropolis_sdg.paidf_anomalygen`)                 | OK/MISSING |
| `TAO_DS_IMAGE`   | `<$TAO_DS_IMAGE>` (key: `images.tao_toolkit.data_services`)                    | OK/MISSING |
```

To populate the summary, run:
```bash
wc -l <training_csv> <validation_csv> <kpi_testing_csv>
python3 -c "import pandas as pd; df=pd.read_csv('<kpi_testing_csv>'); print(df['label'].value_counts().to_string())"
cat <workspace>/augmentation/anomalygen/checkpoints/<project>/checkpoints/latest_checkpoint.txt
cat <workspace>/augmentation/anomalygen/datasets/<project>/defect_spec.jsonl | python3 -c "import sys,json; [print(json.loads(l)['defect_type']) for l in sys.stdin]"
nvidia-smi --list-gpus | wc -l
# ${TAO_PYT_IMAGE}, ${AG_IMAGE}, ${TAO_DS_IMAGE} are exported by Pre-Flight step 5
# from versions.yaml via scripts/resolve_versions_key.py. Loop per-image so the
# output maps 1:1 to the Docker Images table rows above (you can't fill a
# per-row Status column from a single aggregate "grep -c sha256" count).
for var in TAO_PYT_IMAGE AG_IMAGE TAO_DS_IMAGE; do
  ref="${!var:?$var unset — re-run Pre-Flight step 5}"
  if docker image inspect "$ref" --format '{{.Id}}' >/dev/null 2>&1; then
    printf '%-14s OK       %s\n' "$var" "$ref"
  else
    printf '%-14s MISSING  %s\n' "$var" "$ref"
  fi
done
```

### Runtime Estimate
**Estimate only** — heuristic from a measured **RTX 6000 Ada (48 GB)** run at **~200 train rows**, default epochs; scales with rows/epochs/num_SDG. Per-iteration reference ≈ 33 min:

| Stage | Time | Scales with |
|---|---|---|
| rca | ~2 min | KPI-test rows |
| routing | <1 min | — |
| anomalygen | ~15 min + 5–10 min ckpt load | # images |
| data_mining | ~4 min | pool size |
| train | ~11 min | train rows × epochs |
| evaluate | ~2 min | KPI-test rows |

`total ≈ baseline + max_iterations × ~33 min` + overhead (10 iters ≈ ~6.5h wall). Add the one-time ~22–140 GB base-checkpoint/image pull separately when image/Cosmos rows are `MISSING`.

**Ask the user to confirm before proceeding.** Wait for explicit approval ("looks good", "go", "yes"). Do not start the loop until the user confirms.
