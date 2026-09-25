# Upstream CT examples

Use the volumes in [NV-Reason-CT/examples](https://github.com/NVIDIA-Medtech/NV-Reason-CT/tree/main/examples) for end-to-end inference. The upstream README uses `example_1.nii.gz` for both chest and abdomen inference. This skill does not assume the anatomy coverage of the other two volumes or provide reference diagnoses.

Run these commands from the skills repository root using a fresh Python environment configured according to [upstream installation guidance](https://github.com/NVIDIA-Medtech/NV-Reason-CT#installation). Dependency requirements are maintained upstream, not in this skill; the recorded environment in `SKILL.md` is historical evidence only, with a known security finding. Git/Git LFS, authorized GitHub/model access, and a compatible CUDA GPU must already be available. No scan, weight, or generated medical response belongs in a commit.

## Get the actual example volumes

Use a separate disposable checkout; do not reset an existing development checkout or run a global `git lfs install`:

```bash
nv_reason_ct_demo="$(mktemp -d)"
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA-Medtech/NV-Reason-CT.git \
  "$nv_reason_ct_demo/upstream"
git -C "$nv_reason_ct_demo/upstream" lfs install --local
git -C "$nv_reason_ct_demo/upstream" lfs pull --include="examples/*.nii.gz"
git -C "$nv_reason_ct_demo/upstream" lfs fsck
git -C "$nv_reason_ct_demo/upstream" rev-parse HEAD
git -C "$nv_reason_ct_demo/upstream" lfs ls-files --long
```

Keep the Git commit and LFS content hashes with local evidence. The default checkout follows upstream `main`; retain an immutable reviewed commit when replaying an earlier run. GitHub archive downloads or clones with LFS smudging disabled may contain only pointer files. Missing authorization, missing volumes, or pointers block inference until resolved.

## Run chest and abdomen end to end

Keep downloaded model assets and custom-code caches in the disposable directory, not the user's shared cache:

```bash
export HF_HOME="$nv_reason_ct_demo/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$nv_reason_ct_demo/cache/modules"
export XDG_CACHE_HOME="$nv_reason_ct_demo/cache/xdg"
export TORCH_HOME="$nv_reason_ct_demo/cache/torch"
export NV_REASON_CT_REVISION=REVIEWED_IMMUTABLE_MODEL_COMMIT
```

Replace the revision placeholder with the reviewed 40-character Hugging Face commit. With any required `HF_TOKEN` already supplied securely, stage the assets once and inspect the read-only setup report:

```bash
hf download nvidia/NV-Reason-CT --revision "$NV_REASON_CT_REVISION"
python skills/nv-reason-ct/scripts/run_nv_reason_ct.py \
  --check-setup --fail-on-not-ready --revision "$NV_REASON_CT_REVISION"
```

Proceed only when setup exits `0` and reports `ready_for_live_cuda_inference`; exit `1` means the reported setup blocker must be resolved first. Separately review the selected custom model/processor code and obtain explicit caller authorization for its unsandboxed execution. Only then use `--trust-model-code` in the commands below; cached assets and setup success are not consent. Enable offline mode only after the download completes. Retain stdout JSON and stderr (including the code-execution warning) through the calling tool or harness:

```bash
python skills/nv-reason-ct/scripts/run_nv_reason_ct.py \
  "$nv_reason_ct_demo/upstream/examples/example_1.nii.gz" \
  --anatomy-region chest --prompt "write a structured chest CT report" \
  --thinking --max-new-tokens 2048 --local-files-only --trust-model-code \
  --revision "$NV_REASON_CT_REVISION" --out-dir runs/nv_reason_ct_example_1_chest

python skills/nv-reason-ct/scripts/run_nv_reason_ct.py \
  "$nv_reason_ct_demo/upstream/examples/example_1.nii.gz" \
  --anatomy-region abdomen --prompt "write a structured abdominal CT report" \
  --thinking --max-new-tokens 2048 --local-files-only --trust-model-code \
  --revision "$NV_REASON_CT_REVISION" --out-dir runs/nv_reason_ct_example_1_abdomen
```

Require exit `0`, `runtime.mode == "hf_transformers"`, `runtime.mock == false`, the selected `runtime.resolved_revision`, nonempty text, and `runtime.truncated_by_max_new_tokens == false`. Verify the recorded volume hash and crop selection. Exit `3` is incomplete generation, not a successful end-to-end run. These are engineering checks, not checks of the generated medical content.

## Optional verification

The offline unit suite is self-contained: it uses test doubles at the loader and inference boundaries, requires no fixture plugin, and does not generate CT scans. Install `pytest`, `jsonschema`, `PyYAML`, `numpy`, and `nibabel` in a disposable test environment (no PyTorch or Transformers installation is needed for these tests), then run:

```bash
python -m pytest skills/nv-reason-ct/tests/test_nv_reason_ct.py
```

Real input checks use all three downloaded volumes. After the same explicit custom-code authorization, add live parity checks to compare the wrapper with upstream `inference.py` for example_1 in both regions. `--nv-reason-ct-trust-model-code` acknowledges that both paths load the selected model code; live tests reject its absence:

```bash
python -m pytest skills/nv-reason-ct/tests \
  --nv-reason-ct-upstream "$nv_reason_ct_demo/upstream" \
  --nv-reason-ct-live --nv-reason-ct-trust-model-code \
  --nv-reason-ct-revision "$NV_REASON_CT_REVISION"
```

Without these options only the three real-volume and two GPU integration cases skip; every offline contract test runs. When requested, missing examples, invalid revisions, or unavailable model assets fail rather than silently skipping or substituting test data. Offline test doubles do not establish live inference success. The local-export loading contract has offline regression coverage; the recorded live baseline in `SKILL.md` used the released-model snapshot, not a newly trained checkpoint.

Passing local tests or reproducing the historical GPU baseline does not establish managed evaluation or signing status. A signed-skill claim requires verification of `skill.oms.sig` against the exact skill contents and the catalog's trusted certificate. That signature covers the skill, not separately downloaded model code or weights, and does not replace the caller's explicit `--trust-model-code` authorization.

## Cleanup

After inference and when cleanup is authorized, remove only the task-created checkout, environment, and caches. Preserve result JSON, source/content hashes, and setup versions separately; do not purge shared caches or delete a caller-owned input scan.
