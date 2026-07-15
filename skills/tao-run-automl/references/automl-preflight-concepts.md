# TAO AutoML Preflight And Concepts

Preflight, platform prerequisites, support checks, and core AutoML concepts from the pre-refactor guide.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Preflight
- Prerequisites
- Verify LLM features (optional)
- Verify WandB (optional)
- Concepts: What is TAO AutoML?
- Quick Support Queries

# TAO AutoML Skill

This is a skill-bank **workflow** skill at `skills/applications/tao-run-automl/`. The agent
discovers it by reading this file directly (or via the `tao-skills` plugin).

Run automated hyperparameter optimization (HPO) for any TAO network. The agent uses `AutoMLRunner` — a single interface that manages the full loop: generate hyperparameter recommendations, launch training jobs, extract metrics, and feed results back to the optimizer.

The runner is **platform-agnostic** — it takes any object implementing the standard SDK shape (`create_job`, `get_job_status`, `get_job_logs`, `get_failure_analysis`) and calls those methods. Pick whichever SDK matches where you want jobs to run; the runner doesn't care:

| SDK | Best for AutoML |
|---|---|
| `BrevSDK` | Cost-tuned sweeps on Brev instances (single-instance per rec, multi-GPU OK). Multi-credential / multi-workspace accounts must pass `cloud_cred_id=` and `workspace_group_id=` to `create_job` — see `skills/platform/tao-run-on-brev/SKILL.md`. |
| `SlurmSDK` | Large sweeps on shared HPC clusters with queue/quota |
| `KubernetesSDK` | Sweeps on EKS / GKE / AKS / on-prem clusters with the NVIDIA GPU Operator |
| `DockerSDK` | Local debugging or single-host sweeps with a few recs |

Multi-node per rec works on SLURM and K8s (each rec is an N-node distributed training job). Brev and local Docker are single-host per rec — multi-GPU within one host still works (`gpu_count > 1`), but you can't parallelize one rec across multiple hosts.

## Preflight

This skill needs `nvidia-tao-automl`, which pulls `nvidia-tao-sdk` as a
transitive dependency. Both packages are pinned in `versions.yaml` under
`wheels:`. Resolve the selected platform extra with
`scripts/resolve_versions_key.py`:

```bash
python -c "import tao_automl" 2>/dev/null || {
  SB="${TAO_SKILL_BANK_PATH:-~/tao-skills-external}"
  echo "MISSING: nvidia-tao-automl not installed. Pick the platform extra you need:"
  echo "  pip install \"$($SB/scripts/resolve_versions_key.py wheels.tao_automl_slurm)\"       # on-prem SLURM cluster"
  echo "  pip install \"$($SB/scripts/resolve_versions_key.py wheels.tao_automl_kubernetes)\"  # K8s (EKS / GKE / on-prem)"
  echo "  pip install \"$($SB/scripts/resolve_versions_key.py wheels.tao_automl_docker)\"      # local Docker daemon"
  echo "  pip install \"$($SB/scripts/resolve_versions_key.py wheels.tao_automl_brev)\"        # Brev GPU instances"
  echo "  pip install \"$($SB/scripts/resolve_versions_key.py wheels.tao_automl_all)\"         # all platforms"
  echo "  (append ,llm or ,wandb to the extra only when needed)"
  exit 1
}
```

(For local development against a checkout: `pip install -e '~/tao-run-automl[brev]'` from the cloned repo.)

If missing, the agent prompts the user to authorize the install via Bash, then re-runs the preflight before continuing.

## Prerequisites

Before running AutoML:

1. **Shared launch preflight**: Run the `tao-launch-workflow` intake pattern first. AutoML must not create runner files, workspaces, state files, logs, compatibility shims, or install dependencies until the selected platform's credentials, access check, dataset visibility, model credentials, container image confirmation, and compute shape are satisfied. This prevents wasting the AutoML budget on fake recommendation failures caused by SSH, storage, image, or credential setup.
2. **SDK credentials**: env vars read from the session environment (export them in your shell before launching). Required env vars depend on which SDK you choose — see each platform's SKILL.md (`skills/platform/tao-run-on-brev`, `skills/platform/tao-run-on-slurm`, `skills/platform/tao-run-on-kubernetes`, `skills/platform/tao-run-on-local-docker`). Before asking for credentials, run:
   ```bash
   ${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
     --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
     --platform <platform> --format text
   ```
   Ask only for credentials from that output. For example, SLURM needs SLURM credentials and not Brev or S3 credentials; Kubernetes and local Docker do not need SLURM or Brev credentials. Ask S3 credentials only when the selected platform and dataset/result URIs use `s3://`. For container pulls: `NGC_KEY`. The agent never reads values — only checks presence with `[ -n "$VAR_NAME" ]`. Construct the SDK with no arguments — e.g., `BrevSDK()`, `SlurmSDK()`, `KubernetesSDK()`, or `DockerSDK()`.
2. **Dataset**: Training data accessible from the compute backend. URI format depends on the SDK's platform:
   - Brev / cloud: `s3://bucket/path` (S3-compatible; do not generate `aws://...`)
   - Slurm / internal shared storage: an absolute shared filesystem path visible to the Slurm job, e.g. `/lustre/fsw/tao_datasets/<model>/train` and `/lustre/fsw/tao_datasets/<model>/eval`
   - Azure: `azure://container/path`
   - Local / Docker: local filesystem path
   Accept either dataset roots or exact spec-key paths. For exact spec paths,
   preserve user-supplied keys such as
   `custom.train_dataset.annotation_path=/lustre/.../annotations.json` and
   `custom.train_dataset.media_path=/lustre/.../videos.tar.gz`; do not force
   both files to share one parent directory.
3. **Skill bank available**: the runner takes an explicit `skill_dir` — the **absolute path to a model directory** inside the skill bank, e.g. `<bank-root>/skills/models/tao-train-dino`. No global env var; pass per run. The agent already knows the bank root (it loaded this SKILL.md from there) — use that same root. Resolve user model aliases to a packaged skill directory before constructing this path; do not assume `network_arch` equals the directory name. Common locations:
   - cloned standalone: `~/tao-skills-external/` (or wherever the user cloned).
   - Installed skill-bank cache: `<agent-cache>/tao-skill-bank/<version>/`.
   - Codex plugin: `~/.codex/plugins/cache/<marketplace>/tao-skill-bank/<version>/`.
   - submodule inside a cloned SDK: `<sdk>/tao-skills-external/`.
   ```python
   from pathlib import Path
   SKILL_BANK = Path("<bank-root>")        # substitute the actual path
   skill_dir  = SKILL_BANK / "skills" / "models" / model_skill
   ```
   The bank structure is:
   ```
   tao-skills-external/
   └── skills/
       ├── applications/         # workflow configs (this skill)
       ├── models/               # per-network skill packages
       │   ├── <model_skill>/
       │   │   ├── SKILL.md
       │   │   ├── schemas/
       │   │   │   └── train.schema.json          # REQUIRED AutoML gate
       │   │   └── references/
       │   │       ├── skill_info.yaml             # actions, data_sources, container image
       │   │       └── spec_template_train.yaml    # default training spec (recommended)
       │   └── ...
       ├── data/
       └── platform/
   ```
   **CRITICAL**: AutoML requires a packaged generated train dataclass schema at `<bank-root>/skills/models/<model_skill>/schemas/train.schema.json`. The schema must exist and parse as JSON — it's the AutoML support gate because it defines `automl_enabled` parameters, defaults, ranges, options, weights, and popular metadata. Schemas are generated during skill-bank maintenance and shipped with the plugin; the runtime must not expect `~/tao-core` to exist. If the packaged train schema is missing, do not run AutoML for that model.

   `references/spec_template_<action>.yaml` is required for **non-TAO-Core models** (cosmos-rl, clip, etc.) — without it the runner has no defaults and the trial spec will be missing keys. For **TAO Core / Hydra-based models** (DINO, BEVFusion, etc.) the template is optional; Hydra fills container-side defaults at runtime.
4. **`nvidia-tao-automl` installed** with the platform extra you want. Resolve
   the pinned install command from `versions.yaml`:
   ```bash
   SB="${TAO_SKILL_BANK_PATH:-~/tao-skills-external}"
   pip install "$($SB/scripts/resolve_versions_key.py wheels.tao_automl_brev)"   # or _slurm, _kubernetes, _docker, _all
   # With LLM/agentic algorithms, append ,llm to the resolved extra:
   pip install "$($SB/scripts/resolve_versions_key.py wheels.tao_automl_brev | sed 's/]/,llm]/')"
   ```
   For local development against a checkout: `pip install -e '~/tao-run-automl[brev]'`.

Verify setup:
```bash
python3 -c "from tao_automl.runner import AutoMLRunner; print('OK')"

# Verify LLM features (optional)
python3 -c "from tao_automl.brain.llm_brain import LLMBrain; print('LLM OK')"

# Verify WandB (optional)
python3 -c "import wandb; print('WandB OK')"
```

---

## Concepts: What is TAO AutoML?

TAO AutoML automates the "try different hyperparameter values → train → compare results → repeat" cycle. Instead of manually tweaking training settings, you tell AutoML:

- **What network** to train (`network_arch`)
- **Which hyperparameters** to search over (from the model skill and schema)
- **What metric** to optimize (from the model skill or user request)
- **How many trials** (budget)

AutoML then:
1. Picks hyperparameter values using a search algorithm (Bayesian, Hyperband, LLM, etc.)
2. Launches a real training job on whichever backend the SDK targets (Brev, SLURM, Kubernetes, or local Docker)
3. Reads the result metric from training logs
4. Feeds the result back to the algorithm so it learns what works
5. Repeats until budget is exhausted
6. Returns the best configuration found

Each "trial" is called a **recommendation** (rec). One rec = one full training run with a specific set of hyperparameters.

---

## Quick Support Queries

When the user asks what models/networks are supported for AutoML, run the
packaged model-list helper in AutoML mode. AutoML enablement is **model-level**
metadata (`skills/models/<network>/references/skill_info.yaml` has
`automl_enabled: true`), not workflow-level metadata. The helper reads that
model metadata, then validates whether the model also has a packaged,
parseable train dataclass schema:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_models.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --scope automl --format text
```

The compatibility wrapper below is also valid and delegates to the same logic:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_automl_support.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --format text
```

Return both sections from that output: runnable AutoML models and
AutoML-enabled models still blocked on schema packaging. The support rule is:
AutoML is enabled at model level; runnable AutoML also requires
`skills/models/<network>/schemas/train.schema.json` to be packaged and valid.

---
