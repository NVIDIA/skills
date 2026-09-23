# Cosmos3 DEFT AOI Pre-Flight

Run these checks in order. Resolve network mode through `references/air-gap.md`
before checking Python dependencies. They are read-only until the single
approval gate. Read exactly one branch after resolving the mode:
`references/air-gap.md` or `references/network-bootstrap.md`. In air-gap mode,
use only an interpreter selected by `bash scripts/deft_python.sh`; if none is
complete, report the missing imports and stop without invoking a package
manager. Do not
create `${RESULTS_DIR}`, write specs, pull images, or submit jobs before
approval.

## 1. Select and preflight a platform

Discover installed platform skills and ask the user to choose. Read the chosen
`SKILL.md` completely and run its Preflight section. Record the native CLI,
storage tier, compute-frame paths, GPU/node shape, monitoring preference, and
poll interval. Never default to a platform.

## 2. Resolve inputs

The minimum a user has to provide is the annotations, the images they
reference, and the decisions below. Everything else — both spec files, the
prepared checkpoint path, output paths — is produced by this workflow. Do not
ask for a spec file.

Require:

- absolute workspace and media-root paths as seen from the selected platform;
- Proxy KPI, frozen Benchmark KPI, and Mining Pool annotation JSON arrays;
- `max_iterations`;
- the requested KPI (`recall_ng >= 1.0` by default, or an explicit accuracy /
  precision / F1 gate);
- GPU and node count.

Check only required environment-variable presence.

## 3. Validate bare annotations

Use the same selected host Python for every bundled script:

```bash
SKILL_ROOT=<absolute-skill-root>
PYTHON=$(bash "$SKILL_ROOT/scripts/deft_python.sh")

# MEDIA_ROOT is the base that annotation relative paths resolve against, and
# that is the WORKSPACE ROOT — not workspace/images. The paths already begin
# with "images/", so pointing it at the images directory yields
# .../images/images/... and every file appears missing. check_annotations.py
# detects that specific doubling and names the directory to use instead.
: "${MEDIA_ROOT:=$WORKSPACE}"

# Checks all three inputs against their per-role field contract in one pass.
"$PYTHON" "$SKILL_ROOT/scripts/check_annotations.py" \
  --workspace "$WORKSPACE" --media-root "$MEDIA_ROOT" --require-files \
  --summary "$WORKSPACE/annotations/contract_check.json"

# Which fields each role needs, and why:
"$PYTHON" "$SKILL_ROOT/scripts/check_annotations.py" --print-contract
```

`scripts/check_annotations.py` owns the field contract; `ROLE_CONTRACT` in that
file is the single source of truth, not this document. It wraps
`validate_sharegpt.py`, which remains available per-file with `--require-id`.

Any assistant response other than exact `OK` or `NG` is a hard stop. Each
file root must be a non-empty JSON array, and each record must contain exactly
`[AOI, golden_reference]`. JSONL input is invalid even when it contains the same
records. A missing `id` on Proxy or Benchmark is a hard stop here — see
`references/aoi-annotation.md`. Adding ids changes the Benchmark SHA-256, so do
it before `init_deft_state.py` freezes the hash.

## 4. Validate split isolation

```bash
"$PYTHON" "$SKILL_ROOT/scripts/validate_split_contract.py" \
  --workspace "$WORKSPACE"
```

Target AOI images must be disjoint across Proxy, Benchmark, and Mining. Golden
references may be shared. There is no input Train annotation. Record the
printed Benchmark SHA-256 in the summary; `init_deft_state.py` freezes it after
approval.

## 5. Resolve current images

Resolve, do not guess:

```bash
: "${TAO_SKILL_BANK_PATH:?set TAO_SKILL_BANK_PATH to the installed TAO skill-bank root containing versions.yaml and scripts/resolve_versions_key.py}"
test -f "$TAO_SKILL_BANK_PATH/versions.yaml" || { echo "missing $TAO_SKILL_BANK_PATH/versions.yaml" >&2; exit 2; }
test -f "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" || { echo "missing $TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" >&2; exit 2; }
test -f "$TAO_SKILL_BANK_PATH/scripts/resolve_tao_image.py" || { echo "missing $TAO_SKILL_BANK_PATH/scripts/resolve_tao_image.py" >&2; exit 2; }
COSMOS_MODEL_ID="${COSMOS_MODEL_ID:-nvidia/Cosmos3-Nano}"
COSMOS_RL_IMAGE=$(
  "$PYTHON" "$TAO_SKILL_BANK_PATH/scripts/resolve_tao_image.py" \
    --skill-bank "$TAO_SKILL_BANK_PATH" \
    --model "$COSMOS_MODEL_ID" \
    --action train \
    --backend cosmos-rl \
    --format json \
  | "$PYTHON" -c 'import json, sys; image = json.load(sys.stdin).get("image"); assert image; print(image)'
)
TAO_DS_IMAGE=$(
  "$PYTHON" "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" \
    --skill-bank "$TAO_SKILL_BANK_PATH" \
    images.tao_toolkit.data_services
)
AG_IMAGE=$(
  "$PYTHON" "$TAO_SKILL_BANK_PATH/scripts/resolve_versions_key.py" \
    --skill-bank "$TAO_SKILL_BANK_PATH" \
    images.metropolis_sdg.paidf_anomalygen
)
```

Use image inspection through the chosen platform. If an image is absent and
pulling requires a credential, report the missing variable name only. A
`DENIED` error on a public image usually means a stale `nvcr.io` entry in
`~/.docker/config.json`, not a missing key — retry once with a throwaway empty
`DOCKER_CONFIG` before reporting a credential problem.

Run these evaluator compatibility checks against the model skill's selected
`cosmos-rl` image and report them in the Pre-Flight Summary rather than
discovering them on the first GPU job. Training is unaffected:

- `cosmos_rl/evaluation/evaluator.py` hard-indexes `item["id"]` on the
  conversations branch, so evaluation annotations need an `id` (step 3).
- `scripts/patch_eval_image_cap.py` reads `cosmos_rl/evaluation/base.py` from
  the selected image and reports `patch_required`, `already_sufficient`, or
  `cap_absent` from the source rather than parsing its tag. Use `--probe` here:
  it reports `classification`, `cap_in_image`, and `patch_needed` without
  writing. After approval, run it with `--output-dir` and mount the emitted
  file only for `patch_required`. If it reports `classification=unknown`, the
  evaluator still references a changed cap/vLLM shape; hard-stop and verify it.

Probe the AnomalyGen assets read-only and report each as present or
`WILL_AUTO_FETCH`: the fine-tuned checkpoint directory holding `ag_config.yaml`,
the dataset directory with `defect_spec.jsonl` and
`semantic_segmentation_labels.json`, and the Cosmos base-checkpoints cache.
The default fine-tuned checkpoint is `nvidia/Cosmos-AnomalyGen-PCB-2B`; a BYO
override must match the pinned container's PAIDF major.minor. Probing only —
the bootstrap that populates missing assets is post-gate and is owned by
`references/tao-generate-anomalies.md`. In air-gapped runs every asset must be
pre-staged; report a missing one instead of planning a download. Before Phase
3, use the executable file check in that reference to gate the AnomalyGen
Guardrail safety model; a missing file is a hard stop.

## 6. Model and evaluator contract

Read:

- `skills/models/tao-finetune-cosmos-reason/SKILL.md`;
- its `references/skill_info.yaml`;
- its `references/spec_template_train.yaml` and
  `references/spec_template_evaluate.yaml`;
- `references/cosmos-reason.md`.

Reuse an existing spec, generate an absent one. When
`workspace/specs/<name>.toml` is already present it is the operator's own
tuning — validate it against the invariants below and keep it; never overwrite
it with a freshly generated file. Generate only what is missing.

Build separate Proxy and Benchmark nested spec dictionaries in memory from the
model skill's same evaluate template. Validate the model skill's Train template
plus the workspace's initial Mining Pool annotation path and requested
hyperparameters. Show every override and its source, and report per spec
whether it was reused or generated. Do not write staged TOML files until
approval. Required invariants hold for both paths — a reused spec that violates
one is a stop, not a silent regeneration:

- normalize `nano`, `edge`, or `super` to `nvidia/Cosmos3-Nano`,
  `nvidia/Cosmos3-Edge`, or `nvidia/Cosmos3-Super`;
- preserve the variant selected in the prompt and use Nano when none is
  selected;
- recommend hardware for the selected variant and report insufficient
  resources; when the prompt asks for a hardware- or workload-based model
  recommendation, explain the tradeoff and obtain an explicit variant choice
  before state initialization;
- never silently change or fall back from the selected variant;
- identify the published Cosmos Reason 3 checkpoint as the conversion source
  (recognizable by `model_type="cosmos3_omni"`) and plan a prepared Qwen3-VL
  PTM path;
- prove the selected Cosmos-RL image can load the prepared PTM and train the
  selected variant, and derive its conversion and compute requirements
  independently;
- `automl_policy=off` is a workflow argument, never a spec key;
- `policy.model_name_or_path` uses the prepared Qwen3-VL PTM path while the
  selected Cosmos3 ID remains the source-model lineage;
- `policy.model_max_length >= 40960`;
- `train.train_policy.type="sft"`;
- the workspace Train template initially points to `mining_pool.json`;
- baseline evaluates the frozen Benchmark gate first, and no Train job runs
  before the gate is shown unmet and zero-shot Proxy plus Proxy RCA follow;
- after RCA and Mining selection, the staged Train annotation path is replaced
  with the current `train_iter_<N>.json`;
- Proxy and Benchmark both use the model skill's current `evaluate` action,
  the same checkpoint, and separate job-records;
- Proxy and Benchmark annotation and output paths are distinct;
- Proxy is not gate eligible; Benchmark alone drives loop stopping;
- output paths land under the stage job-record's bound results directory.

## 7. Plan checkpoint preparation

Read the `Nano checkpoint model-type choice` section in the model skill, run
`prepare_cosmos3_vlm_checkpoint.py --help`, and review the command in
`references/cosmos-reason.md`. Before approval, perform read-only checks and
show:

- the selected reasoner by name and canonical ID, e.g.
  `Cosmos3 Nano Reasoner (nvidia/Cosmos3-Nano)`;
- host output path and selected-platform compute-frame path for the prepared
  Qwen3-VL checkpoint;
- whether a complete prepared output can be reused;
- the exact `prepare_cosmos3_vlm_checkpoint.py` command planned after approval;
- the selected backend image passed through `--runtime-image` and its resolved
  image ID or digest passed through `--runtime-image-digest`;
- the variant-matched `--vlm-architecture-model-path-or-uri` and, for a model
  ID or URI, its immutable `--vlm-architecture-model-revision`.

Nano may use the model helper's packaged Qwen3-VL 8B default. Edge and Super
must have their own validated mapping; never inherit Nano's default. The
preparation helper may clone, pull, and write files, so run it only after the
single approval gate and before the baseline frozen Benchmark evaluation.

## 8. Credentials

Check only variables required for confirmed missing assets and the chosen
platform. Neither token is required by default: the Cosmos-RL, data-services,
and AnomalyGen images this workflow uses are public on `nvcr.io`, so `NGC_KEY`
matters only for a registry that actually rejects an anonymous pull. Report
`UNSET` as the normal case, not as a finding.

A required variable is either already exported in the session or comes from a
user-approved env file (see AGENTS.md). Load such a file in the same command as
its consumer, never read it back: these checks report presence only.

```bash
set -a; source /path/to/.env; set +a   # omit if already exported

# HF_TOKEN and HUGGING_FACE_HUB_TOKEN are two names for the same HuggingFace
# access token, not two credentials. huggingface_hub honors both (HF_TOKEN
# wins), and prepare_cosmos3_vlm_checkpoint.py forwards whichever is set. Treat
# either one as satisfying the HuggingFace requirement.
if [ -n "$HF_TOKEN" ] || [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
  echo HF_TOKEN=SET
else
  echo HF_TOKEN=UNSET
fi
# Only meaningful if an image pull is actually refused; the images used here
# are public.
[ -n "$NGC_KEY" ] && echo NGC_KEY=SET || echo NGC_KEY=UNSET
```

An unset token is never a blocker on its own. Escalate only from an observed
failure — a refused pull, or a gated HuggingFace download — and then name the
single variable the user must supply, exported in their shell or added to a
user-approved env file the run sources. Never ask the user to paste a value,
never report a missing HuggingFace token when only the legacy name is exported,
and never present an unset `NGC_KEY` as a problem when the pull succeeds
anyway.

## 9. Compute and runtime

Record:

- GPUs per node, node count, and the exact GPU model plus memory reported by
  the selected platform. Preserve that exact string for
  `init_deft_state.py --gpu-model`; never report the local host's accelerator
  for a remote Docker, Kubernetes, SLURM, Brev, or external-platform run;
- `policy.parallelism.dp_shard_size` and `dp_replicate_size`;
- LoRA rank/alpha/target modules;
- epochs, batch size, learning rate;
- Proxy and Benchmark sample counts;
- mining top-K (default 5), cosine floor,
  and run-level filepath history ledger;
- estimated baseline and per-iteration runtime.

Do not invent a runtime estimate when no comparable run exists; label it
`unknown (measure baseline)`.

## 10. Launch review

State the container user mapping explicitly. Every job that writes into
`${RESULTS_DIR}` runs as the invoking user, never root — root-owned artifacts
inside an operator-owned results tree cannot be cleaned up or re-rendered
without `sudo`. See `references/cosmos-reason.md`.


Show the platform-native `submit/status/logs/cancel` mapping, storage tier,
container images, annotation JSON paths, prepared checkpoint host/compute-frame
paths, input/output compute-frame paths, and job-record root. The
record-then-launch ordering must be explicit.

## 11. Pre-Flight Summary

```text
## Cosmos3 DEFT AOI — Pre-Flight Summary

| Field | Effective value | Source |
|---|---|---|
| Platform | <selected platform> | user |
| Network mode / source | <airgap or network-enabled; activation source> | environment/user/harness/default |
| Selected Python | <absolute dependency-complete executable> | preflight |
| Workspace / media root | <absolute compute-frame paths> | user/default |
| Base model | <Cosmos3 <variant> Reasoner (canonical ID); Nano by default> | default/user |
| Prepared PTM | <that reasoner -> Qwen3-VL PTM path; reuse/prepare> | workflow/model skill |
| Annotation mode | bare_okng | workflow |
| Specs | <per file: reused from workspace, or generated> | workspace/workflow |
| Proxy KPI | <path; OK/NG counts; RCCA only> | user/default |
| Benchmark KPI | <path; counts; SHA-256; stop gate only> | user/default |
| Mining Pool | <path; OK/NG counts; commercial-training eligible> | user/default |
| Next Train | `train_iter_<N>.json`, created after Proxy RCA and Mining selection | workflow |
| KPI | <metric operator target> + unknown_predictions <= 0 | user/default |
| Iterations | <N> | user |
| Train shape | <nodes x GPUs; exact GPU model/memory; epochs; batch; LR; LoRA> | user/spec/platform |
| Mining | <top-K, default 5; cosine floor; history-aware filepath dedup> | user/default |
| AnomalyGen | <project; num_SDG; each asset path or will auto-fetch from HF (default)> | user/default |
| Cosmos-RL image | <resolved URI> | versions.yaml |
| Data-services image | <resolved URI> | versions.yaml |
| AnomalyGen image | <resolved URI> | versions.yaml |
| Credential status | <names with SET/UNSET only> | environment |
| Job tracking | record before every native launch | workflow |
| Monitoring | <yes/no and interval> | user/default |
```

Name the model, do not print a bare repo id. The Base model and Prepared PTM
rows must read as e.g. `Cosmos3 Nano Reasoner (nvidia/Cosmos3-Nano)` and
`Cosmos3 Nano Reasoner -> /abs/path/Cosmos3-Nano-VLM`. A Summary showing only
`Cosmos3-Nano -> Qwen3-VL` never tells the reader which model is being
fine-tuned; the id is the locator, the reasoner is the model.

Keep every row on one line — this block is reproduced verbatim, so a cell long
enough to wrap destroys the alignment. Put anything that does not fit in the
lines below the table instead:

- the variant-matched VLM base for the prepared PTM (Edge and Super never
  inherit Nano's default; see step 7);
- which AnomalyGen assets are staged; `WILL_AUTO_FETCH` is legal only in
  network-enabled mode, plus their commercial-training approval.

Then stop. Remind the user that approval permits checkpoint preparation,
post-gate spec/state creation, any network-enabled AnomalyGen bootstrap, and GPU
submissions. After approval, prepare and
validate the Qwen3-VL checkpoint, write the staged specs with concrete nested
values, initialize state once, re-read it, then begin baseline frozen
Benchmark evaluation.
