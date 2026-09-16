# Pre-Flight Checks and Summary

## Pre-Flight

Resolve everything possible before asking the user. In order:

Resolve `network_mode` first, without probing the network. Read exactly one
branch: `references/air-gap.md` for `airgap`, or
`references/network-bootstrap.md` for `network-enabled`. Record the mode and
activation source; never load or execute the network bootstrap in air-gap mode.

1. Locate workspace root, specs, CSVs, checkpoints, augmentation assets. Derive a timestamped run directory: `RESULTS_DIR=<workspace>/results/run_$(date +%Y%m%d_%H%M%S)`. If resuming an existing run, set `RESULTS_DIR` to the existing run directory instead (detect by checking for `results/run_*/deft_state.json`). All references to `results/` throughout this skill mean `${RESULTS_DIR}/`.

   **Resolve the real-image root before any Docker launch.** Prefer the
   canonical `<workspace>/images`; accept `<workspace>/kpi/images` only as a
   legacy fallback. Resolve symlinks, export the absolute result as
   `IMAGES_DIR`, and hard-stop if neither directory exists. Run
   `"$PYTHON" scripts/validate_training_csv.py` against each base CSV so at least the
   CSV-declared input and golden paths are proven to resolve on disk:

   ```bash
   for BASE_CSV in \
     "$WORKSPACE/train/base/training_set.csv" \
     "$WORKSPACE/train/base/validation_set.csv" \
     "$WORKSPACE/kpi/testing_set.csv"
   do
     PYTHON=$(bash "$TAO_SKILL_BANK_PATH/skills/applications/tao-run-deft-aoi/scripts/deft_python.sh")
     "$PYTHON" "$TAO_SKILL_BANK_PATH/skills/applications/tao-run-deft-aoi/scripts/validate_training_csv.py" \
       --csv "$BASE_CSV" \
       --workspace-root "$IMAGES_DIR"
   done
   ```

   `init_deft_state.py` records the same resolved directory as
   `config.images_dir`; all ChangeNet containers must mount that state value
   rather than reconstructing a path from `kpi/`.

   **Resolve the mining source independently from its images.** A common staged
   layout places the CSV at
   `<workspace>/augmentation/mining_pool/mining_pool.csv` and the referenced
   images under the shared `<workspace>/images` root. Treat these as discovery
   hints, not proof: prefer an explicit user/harness path, inspect the CSV path
   fields, verify the files on disk, and record the resolved paths in state.
   Do not assume an `augmentation/mining_pool/images/` directory exists merely
   because the CSV is under `augmentation/mining_pool/`.

   **Host Python deps.** The DEFT loop needs `pandas`, `numpy`, `matplotlib` (KPI analysis), `pyarrow` (parquet I/O for routing and mining), `huggingface_hub` (backbone staging), and `boto3` (S3 ops). Select the interpreter with `bash scripts/deft_python.sh`; do not probe a different bare interpreter first:
   ```bash
   PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
   "$PYTHON" -c \
     "import pandas, numpy, matplotlib, pyarrow, huggingface_hub, boto3"
   ```
   If imports are missing in **air-gap mode**, hard-stop and report the missing
   modules. Do not invoke `pip`, `pip3`, `uv`, `conda`, `apt`, or any other
   package manager; even an attempted install invalidates the air-gap run.

   In network-enabled mode only, follow `references/network-bootstrap.md`.
   Alternatively run analysis inside the TAO toolkit image. Do not silently
   skip — KPI plots and parquet I/O are part of every loop's output.
2. Read the relevant `references/*.md` files for command syntax and output contracts. See `## Stage Reference Modules` in `references/scripts-and-agents.md` for the stage→skill mapping.
3. In network-enabled mode, credentials reach the session from the user's shell
   or from a user-approved env file — `~/.tao/secrets.env`,
   `~/.config/tao/.env`, or one the user points at — loaded with
   `set -a; source /path/to/.env; set +a`, which prints nothing. Verify presence
   only and never print a credential value. In air-gap mode record
   credentials as `N/A (offline)`:

   | Variable | Required for | Image prefix it gates |
   |---|---|---|
   | `NGC_KEY` | All nvcr.io image pulls — TAO toolkit (train/infer/deploy/data services) and the paidf-anomalygen SDG container | the registry orgs of the manifest-resolved image URIs in step 5 |
   | `HF_TOKEN` | Pre-Flight HuggingFace model downloads (ChangeNet backbone, Cosmos diffusion, T5, C-RADIO-V3, DINOv2, SAM2, Qwen-VL, SigLIP) — cached under `augmentation/anomalygen/base_checkpoints/`. Also gates the PCB fine-tuned checkpoint and reference dataset auto-fetch. | huggingface.co |

   For planned network actions both variables must be non-empty in the process
   environment. The single `NGC_KEY` must have read access to every registry
   org referenced by the resolved image URIs. If either is missing, ask the
   user or harness to inject it without revealing the value, or point the run
   at a user-approved env file to source — some runtimes (notably Codex) do not
   reliably inherit shell exports. The run never creates that file, writes a
   credential value into it, or prints its contents.
4. Network-enabled mode may perform the approved registry login after the user
   gate. Air-gap mode must not log in or pull; local image inspection is the
   only permitted registry-related check. Do not expose credential values.
5. **Resolve and export the version-managed container image env vars.** The rest of this skill — including the Pre-Flight Summary's `docker image inspect` line, every stage launch, and the `references/*.md` files — references three env vars. Resolve every value from the installed skill bank's `versions.yaml`; never copy a tag into this document or preserve a tag from an earlier run:

   ```bash
   PYTHON=$(bash <skill_root>/scripts/deft_python.sh)
   TAO_PYT_IMAGE=$(
     "$PYTHON" \
       "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" \
       images.tao_toolkit.pyt --skill-bank "$TAO_SKILL_BANK_PATH"
   )
   TAO_DS_IMAGE=$(
     "$PYTHON" \
       "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" \
       images.tao_toolkit.data_services --skill-bank "$TAO_SKILL_BANK_PATH"
   )
   AG_IMAGE=$(
     "$PYTHON" \
       "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" \
       images.metropolis_sdg.paidf_anomalygen --skill-bank "$TAO_SKILL_BANK_PATH"
   )
   : "${TAO_PYT_IMAGE:?versions key images.tao_toolkit.pyt did not resolve}"
   : "${TAO_DS_IMAGE:?versions key images.tao_toolkit.data_services did not resolve}"
   : "${AG_IMAGE:?versions key images.metropolis_sdg.paidf_anomalygen did not resolve}"
   export TAO_PYT_IMAGE TAO_DS_IMAGE AG_IMAGE
   ```

   Hard-stop on any resolver error. `versions.yaml` is authoritative even when
   a reference, cached transcript, or previously installed plugin mentions a
   different tag.

   | Env var | versions-key | Used by |
   |---|---|---|
   | `TAO_PYT_IMAGE` | `images.tao_toolkit.pyt` | `train`, `evaluate`, `rca` (TAO toolkit pyt container) |
   | `TAO_DS_IMAGE` | `images.tao_toolkit.data_services` | `data_mining` (TAO data services container) |
   | `AG_IMAGE` | `images.metropolis_sdg.paidf_anomalygen` | `anomalygen` (paidf-anomalygen container) |

   Hard stop here if any export is missing — without it, bash silently substitutes `""`, the next step's `docker image inspect` reports `0` MISSING for every image, and the failure mode points at the wrong root cause.
6. Verify every image resolved in step 5 is present locally (`docker image inspect "$TAO_PYT_IMAGE" "$AG_IMAGE" "$TAO_DS_IMAGE"`).

   **Architecture compatibility check.** The AnomalyGen (`$AG_IMAGE`) container is published as amd64-only and will fail silently on arm64 hosts (e.g. DGX Spark). This surfaces only after schema generation, credential injection, and a 24 GB download — so check it now:

   ```bash
   HOST_ARCH=$(uname -m)   # x86_64 on amd64, aarch64 on arm64
   AG_ARCHS=$(docker manifest inspect "$AG_IMAGE" 2>/dev/null \
     | python3 -c "import sys,json; [print(p['platform']['architecture']) for p in json.load(sys.stdin).get('manifests',[])]" \
     2>/dev/null || echo "unknown")
   echo "Host arch: $HOST_ARCH  |  AG image platforms: $AG_ARCHS"
   ```

   Run that remote manifest check only in network-enabled mode. In air-gap
   mode use `docker image inspect --format '{{.Architecture}}' "$AG_IMAGE"`
   and fail if the local image is absent. Do not query a registry manifest.

   Map `x86_64` → `amd64` and `aarch64` → `arm64` before comparing. Hard stop with a clear message if the host architecture is not in the image's platform list — there is no emulation path for GPU workloads.

   **GPU-arch runnability probe.** Matching CPU arch isn't sufficient — the image's CUDA build must also support the host GPU's compute capability. In air-gap mode launch with `docker run --pull=never`; after state initialization all Docker commands go through `deft_exec.py`. A non-zero exit or `no kernel image is available` is a hard stop.

7. Apply the path rule: pre-create iter dirs under `${RESULTS_DIR}/iter${ITER}/` and mount `<workspace>` into containers at the same absolute path. Workflows enforce their own container-level invariants (entrypoints, env vars); the loop just supplies the workspace mount and the resolved image URI.
8. Verify GPU count and record the exact GPU model plus memory reported by the
   selected platform (for local Docker:
   `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`). Preserve
   that string for `init_deft_state.py --gpu-model`; never substitute a local
   GPU when the selected backend is remote. Probe the three AnomalyGen override
   slots under `augmentation/anomalygen/` (`checkpoints/<project>/`,
   `base_checkpoints/`, `datasets/<project>/`) and report their status in the
   Summary. In network-enabled mode, empty slots may be a post-approval fetch
   plan. In air-gap mode every required slot must already be non-empty and a
   missing asset is a hard stop. NVIDIA publishes the PCB fine-tuned
   checkpoint (`nvidia/Cosmos-AnomalyGen-PCB-2B`) and the PCB reference dataset
   (`nvidia/Cosmos-AnomalyGen-PCB-Dataset`) publicly on Hugging Face; the
   `tao-generate-anomalies` skill downloads them automatically on first use.
   Users can pre-stage their own fine-tuned checkpoint or dataset to override
   the default; a checkpoint override must match the pinned container's PAIDF
   major.minor. Do not ask about missing AnomalyGen assets in network-enabled
   mode: report `will auto-fetch from HF (default)` and proceed. If
   `base_checkpoints/` is pre-staged, export its host path as
   `COSMOS_MODELS_DIR` for downstream mounts. Before SDG, use the
   executable file check in `references/tao-generate-anomalies.md` to gate the
   AnomalyGen Guardrail safety model; a missing file is a hard stop.
   Stage the ChangeNet pretrained
   backbone by running `"$PYTHON" scripts/stage_backbone.py --workspace <workspace>`,
   then set `specs/baseline_spec.yaml::model.backbone.pretrained_backbone_path`
   to the staged file and bind-mount it per `references/visual-changenet.md` →
   *Pre-Flight responsibility*. Staging is mandatory — hard-stop if the script
   exits non-zero; there is no URL fallback. See
   `references/tao-generate-anomalies.md` for invocation and mount layout.
9. **GPU memory sanity check.** ChangeNet classify with C-RADIOv2-B (ViT-B) at the spec defaults (`batch_size: 64`, `image_width/height: 224`, `cls_weight: [1.0, 10.0]`, learnable difference modules) OOMs on a single 48GB-class GPU. Inspect `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` and warn if the assembled spec's `dataset.classify.batch_size` is too large for the available memory: as a rule of thumb, **≤ 16 on 48GB GPUs, ≤ 8 on 24GB GPUs**. Surface the recommendation in the Pre-Flight Summary's `GPUs` row — let the user accept or override before launch rather than failing 30 seconds into training.
10. Run train/validation leakage check before resuming any prior run.

Ask one consolidated question only for missing required inputs. Never ask about a parameter with a default.

**Required input — `max_iterations`.** No default; ask the user if not supplied and do not proceed past Pre-Flight without it. If the user gives a time limit instead, convert it to an estimated `max_iterations` using the §Runtime Estimate per-iteration figure and surface the estimate for confirmation.

**Defaults:**

- `training_epochs`: `num_epochs` from `specs/baseline_spec.yaml`. For a small seed set (~200 rows) use **10** — ChangeNet on the 150M-param C-RADIOv2-B backbone overfits a few-hundred-row set past ~10 epochs (val_loss climbs, FAR@recall=100% degrades). The bundled `references/baseline_spec.yaml` template ships `num_epochs: 10` for this reason. Raise toward 20 only once the combined CSV grows into the low thousands of rows across iterations.
- `num_SDG`: 20 (per-iteration AnomalyGen output budget; raise explicitly when more synthetic coverage is needed)
- `min_similarity` (mining cosine cutoff): 0.9 — read from `config.mining_filter.min_similarity` in `deft_state.json`; the literal `0.9` referenced in Pipeline step 4 is just the fallback default.
- `top_k_per_target`: 5 — preserve an explicit user value. Raise it only when
  the history summary shows that prior selections dominate the current narrow
  neighborhood.
- workspace root: user prompt, else `~/workspace`
- pretrained backbone: first staged weight under `augmentation/backbone/`;
  network-enabled mode may plan the documented post-approval fetch, while
  air-gap mode hard-stops when absent.
- AnomalyGen checkpoint, dataset, and Cosmos base models: prefer the staged
  `augmentation/anomalygen/` paths. Missing assets are fetch plans only in
  network-enabled mode; in air-gap mode they are hard stops.

## Pre-Flight Summary

Once all checks pass, print this summary and **STOP — wait for explicit user approval before launching anything**. This is the one user gate in the entire workflow (see `## Agent Behavior` in SKILL.md); the loop is autonomous *after* this point, never before.

```
## DEFT Loop — Pre-Flight Summary

### Run config
| Field                          | Value                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------ |
| KPI Target                     | FAR < X% at Recall=100%                                                        |
| Network mode / source          | airgap or network-enabled / <activation source>                               |
| Selected Python                | <absolute dependency-complete executable>                                     |
| Max DEFT Iterations            | N                                                                              |
| Stop condition                 | KPI met **or** max_iterations reached — reaching the KPI is not guaranteed; FAR may regress between iterations |
| Training Epochs                | N per iteration                                                                |
| Num SDG                        | N synthetic samples per iteration                                              |
| Mining top-K                  | N neighbours per target (default 5)                                             |
| Mining cutoff                  | cosine ≥ <min_similarity> (default 0.9)                                        |
| Compute / GPUs                 | N GPU(s) · <exact model> (<memory>)                                             |
| Resuming                       | yes — iter N complete / no                                                     |
| Est. runtime                   | ~max_iterations × 33 min on RTX 6000 Ada — estimate only (+~Yh downloads if MISSING) |

### Dataset
| Field                          | Value                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------ |
| Training CSV                   | <path> (N rows)                                                                |
| Validation CSV                 | <path> (N rows)                                                                |
| KPI test CSV                   | <path> (N rows, X defect types)                                                |
| Images dir                     | <path>                                                                         |
| Mining CSV / image root        | <independent absolute paths; resolver status>                                  |

### Augmentation
Show `WILL_FETCH` only in network-enabled mode. In air-gap mode every row must
be a staged local path and no download fallback may appear.

| Field              | Value                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------ |
| AnomalyGen ckpt    | `<path>` (FOUND, step N) **or** will auto-fetch from HF (`nvidia/Cosmos-AnomalyGen-PCB-2B`, ~5 GB) **[default]** |
| Defect spec        | `<N types: type1, type2, ...>` (from staged dataset) **or** will auto-fetch from HF **[default]**                 |
| Cosmos base models | `<path>` (container check FOUND) **or** will auto-download the 2B workflow set on first container run (~22 GB) **[default]** |
| SigLIP model       | `<cached / download / local path>`                                                                                 |
| Backbone           | `<path>` (FOUND) **or** will auto-download from HF (`nvidia/C-RADIOv2-B`, ~393 MB) **[default]**                  |

### Docker Images
Fill the `Image` column with the actual URI resolved in Pre-Flight step 5
(i.e. the value of the env var), not the literal `${VAR}` placeholder.
Print one row per env var so the audit trail shows exactly which tag will run.

| Env var          | Image (resolved in Pre-Flight step 5)                                          | Status     |
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
# (URIs resolved from the installed versions.yaml). Loop per-image so the
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

`total ≈ baseline + max_iterations × ~33 min` + overhead (10 iters ≈ ~6.5h wall). Add the one-time ~22 GB base-checkpoint/image pull separately when image/Cosmos rows are `MISSING`.

**Ask the user to confirm before proceeding.** Wait for explicit approval ("looks good", "go", "yes"). Do not start the loop until the user confirms.
