---
name: nemotron-customize
description: "Plan Nemotron customization pipelines from repo steps: SFT, PEFT/LoRA, AutoModel vs Megatron-Bridge, DPO/RLVR/GRPO/RLHF, curate-then-translate, BYOB/MCQ benchmark prep or translation, checkpoint conversion, ModelOpt optimization, and endpoint or checkpoint evaluation."
version: 0.1.1
license: Apache-2.0
author: NVIDIA Nemotron Team
tags:
  - nemotron
  - customization
  - training
  - pipelines
metadata:
  author: NVIDIA Nemotron Team 
  tags:
    - nemotron
    - customization
    - training
    - pipelines
tools:
  - Read
  - Write
  - Bash
  - Search
---

# nemotron-customize

## Purpose

Use this skill to turn a model-customization request into a repo-native Nemotron step pipeline. It plans the step DAG, validates artifact wiring, and creates only the YAML configs needed to run existing steps.

Use it only for inspecting, configuring, validating, running, or submitting
existing Nemotron steps or multi-step training/customization pipelines. If the
request is a frontend, dashboard, visualization, generic ML-advice,
billing/access, or unrelated coding task, stop with a short scope note and do
not inspect the step catalog or edit files in that turn.

## Security Notes

This skill may use `Write` to create or modify YAML/README files and `Bash` to
run repository commands. Confirm with the user before file writes or shell
execution. Keep Bash usage scoped to repo-safe commands such as `uv run
nemotron steps ...`, `python -m pytest ...`, `git status/diff`, and targeted
validation commands. Never run environment dumps (`env`, `printenv`, broad
`export`) or commands that expose secret values.

## Requirements

- Checkout of this Nemotron repo with `src/nemotron/steps/` present.
- **Invoke from the repo root.** All paths in this document are repo-root-relative.
- User-provided model, data, hardware, backend, and output constraints before writing configs.
- Backend credentials only when the selected step needs them (translation, W&B, hosted endpoints).

## Limitations

- Does not invent new catalog steps when an existing one fits.
- New Python/shell code only in Explorer mode after the gap is explicit.
- Post-training deployment-only requests are out of scope.

Invocation: `/nemotron-customize`. The repo under `src/nemotron/steps/` is the
source of truth; this skill orchestrates and does not duplicate per-step
knowledge.

Priority order: (1) reuse existing repo code, CLIs, recipes, steps, runners,
and configs; (2) add YAML configs for the user's request; (3) generate new
Python/shell only when the repo cannot satisfy the request, and name the gap
first.

For a command request: verify repo root, read the step catalog, read the
selected `step.toml`, verify the requested config exists, read the active env
TOML for any remote profile, then emit the complete command. Do not guess
`--batch` profiles from examples or naming conventions.

## Quick Decision Tree

- **AutoModel vs Megatron-Bridge**: small GPU count, Hugging Face model,
  LoRA/PEFT, or OpenAI-style chat JSONL → AutoModel path (`sft/automodel`
  or the matching PEFT AutoModel step). Large distributed training, packed
  Parquet/binidx data, or full fine-tuning → Megatron-Bridge, but verify
  against `hardware.md` and the category README first.
- **BYOB / MCQ benchmark inputs route to `byob/mcq`, NOT
  `translate/nemo_curator`**. BYOB preserves the multiple-choice schema
  (question, choices, answer); the translate path would flatten or strip
  those fields. Trigger on phrases like "BYOB benchmark", "MCQ", "evaluation
  benchmark Parquet", "multiple-choice prep".
- **Curate then translate**: when the user says "curate and translate",
  "filter then translate", or "prep data before translating", chain
  `curate/nemo_curator` (filter raw JSONL) → `translate/nemo_curator`
  (translate curated JSONL). Do not skip the curate stage.
- **Checkpoint conversion**: route "Megatron to HF", "HF export", "convert
  checkpoint", or "iter_* to safetensors" to `convert/megatron_to_hf`; route
  "HF to Megatron" imports to `convert/hf_to_megatron`. Use a concrete
  `iter_*` source for Megatron exports.
- **Existing endpoint or checkpoint eval**: route hosted endpoint smoke tests
  and benchmark requests to `eval/model_eval`; use `tiny_chat` for hosted chat
  smoke and `default` for Megatron checkpoint evaluation.
- **No env TOML profile present**: do not invent Lepton or `--batch`
  profiles; ask the user or fall back to local execution.

Required inputs before finalizing configs or commands:

- `model`, `input_path`, `output_dir`, hardware/GPU count, backend/env profile,
  and any needed API key environment variable name such as `HF_TOKEN` or an
  evaluator key.
- For translation commands, also collect `server.url`, target/source languages,
  and the runtime-visible input/output paths.
- For BYOB, collect benchmark/source document path, stage (`prepare`,
  `generate`, `translate`, or `all`), target/source languages when translating,
  and output directory.
- For conversion, collect source checkpoint path, output path, model/config
  source, and whether the source is HF, Megatron `iter_*`, or LoRA adapter.
- For eval, collect endpoint URL/model ID or checkpoint path, task IDs,
  endpoint type, API-key environment variable name, and sample limit.

Response shape for recommendations: `Decision`, `Why`, `Required inputs`,
`Config/command`, `Avoid`, and `Next step`. Always call out the stack to avoid
when the user's constraints make it a poor fit.

## How information is split (and where to find it)

| Question | Look here |
|---|---|
| What does step X consume / produce / parameterize? | `src/nemotron/steps/<cat>/<X>/step.toml` |
| When/why pick step X over its siblings? | `src/nemotron/steps/<cat>/<X>/README.md` |
| Which step in category C should I pick? | `src/nemotron/steps/<cat>/README.md` |
| What runner code does step X use? | `src/nemotron/steps/<cat>/<X>/step.py` → `src/nemotron/steps/_runners/` |
| Cross-step constraint (tokenizer lock, sequence packing, data quality, ...) | `src/nemotron/steps/patterns/<id>.md` |
| Artifact compatibility / `is_a` hierarchy | `src/nemotron/steps/types.toml` |
| GPU memory / parallelism heuristics | `src/nemotron/steps/hardware.md` |
| Library API extracts for exceptional code generation | `references/context/index.toml` → `references/context/<pack>.txt` |
| Project scaffold rules, only when repo code cannot support the request | `references/act/PROJECT.md` |
| Per-stage code rules, only when repo code cannot support the request | `references/act/STAGE.md` |

If two sources say the same thing, the **deeper, more specific** one wins
(`step.toml` > category `README.md` > this file).

---

## Instructions

**Pipeline workflow (≥2 stages)**: Orient → Plan → Act → Verify. Discover
candidate steps, propose a DAG with validated artifact wiring, wait for
approval, create the minimal YAML configs, and re-check before reporting done.
Not general ML advice — `src/nemotron/steps/` is the source of truth.

**Single-step command flow**:

1. Confirm the repo root has `pyproject.toml` and `src/nemotron/steps/`.
2. Run `uv run nemotron steps list --json` when available; otherwise read
   `src/nemotron/steps/STEPS.md`.
3. Read the selected step's `step.toml` and the requested checked-in config.
4. For remote execution, read `NEMOTRON_ENV_FILE` or a repo-root `env*.toml`
   and pick an actual section whose profile matches the step.
5. Emit the full command in one reply; then add brief rationale for the
   config/profile choices. For translation, also read
   `src/nemotron/steps/translate/README.md` and return `Decision`, `Config`,
   `Run`, `Output`, `Env`.

**Source tiers** for command answers — Verified (CLI + manifest + config +
env + dry-run all succeeded), Repo-grounded (manifest/config/env read, no
dry-run), Blocked (a required repo file or env TOML is missing — name it and
stop before guessing).

**Canonical commands**:

```bash
uv run nemotron steps run <step_id> -c <config-or-path> --dry-run
uv run nemotron steps run <step_id> -c <config-or-path> --dry-run --batch <profile>
uv run nemotron steps run <step_id> -c <config-or-path> --batch <profile>
```

---

## Workflow

Four phases, in order: **Orient → Plan → Act → Verify.** Never skip Verify.
For detailed phase checklists and Explorer-mode implementation rules, read
`references/WORKFLOW.md`.

---

## Operational Nuances

- Smoke configs (`tiny.yaml`, `tiny_chat.yaml`) are wiring tests, not quality evidence.
- `${art:...}` references belong in recipe-backed configs; standalone YAML uses plain paths.
- Keep pretraining `bin/idx` data and `blend.json` from the same Nemotron release.

## Examples

- **Single step**: read manifest + config + env profile, then return a complete
  `uv run nemotron steps run <step_id> -c <config> --dry-run` command.
- **Translate (one-shot command)**: for "translate EN → <lang>" requests,
  collect `server.url`, `model`, source/target language, `api_key_env`, and
  runtime-visible input/output paths first, then emit the full command in one
  reply (do not split across turns):

  ```bash
  uv run nemotron steps run translate/nemo_curator \
    -c <translate-config.yaml> \
    --batch <env-profile-from-env.toml>
  ```

- **Curate then translate**: chain `curate/nemo_curator` →
  `translate/nemo_curator`. The curate stage produces filtered JSONL that
  becomes the translate stage input. Both steps need YAML overlays; wire
  curate's `output_dir` to translate's `input_glob`.
- **BYOB benchmark prep**: route MCQ Parquet inputs through `byob/mcq`, not
  `translate/nemo_curator`, so the multiple-choice schema is preserved.
- **SFT pipeline**: plan the DAG (`data_prep` → `sft/megatron_bridge` or
  `sft/automodel`), validate artifact edges via `types.toml`, then create the
  YAML overlays.

---

## Two modes

### Catalog mode — a step exists

Fast path: `STEPS.md → category/README.md → step.toml → step.py → adapt YAML
config`. Use whenever the user's request maps to a step in the catalog.

### Explorer mode — no repo path supports it

Use only after confirming no existing step, runner, recipe, CLI, or YAML config
surface can satisfy the request. Follow
`references/WORKFLOW.md`.

### Choosing a mode

| User says | Mode |
|---|---|
| "SFT with Megatron-Bridge / AutoModel" | Catalog |
| "DPO / RLVR / GRPO / RLHF" | Catalog: `rl/nemo_rl/*` |
| "Synthesize preference / SFT data" | Catalog: `sdg/data_designer` |
| "Translate EN → \<lang\> for training data" | Catalog: `translate/nemo_curator` |
| "Curate and translate" / "filter then translate" | Catalog chain: `curate/nemo_curator` → `translate/nemo_curator` |
| "Curate web text" | Catalog: `curate/nemo_curator` |
| "BYOB benchmark" / "MCQ benchmark prep" | Catalog: `byob/mcq` (preserves MCQ schema) |
| "Train with X exotic backend" | Explorer or **ask** |
| Post-training-only request | Out of scope; redirect to a more appropriate workflow. |
| Ambiguous | **Ask** |

---

## Boundaries

**Do**: build pipelines from existing steps and cite `step.toml` directly;
reuse repo CLIs/runners/recipes first; adapt configs (don't copy
`default.yaml` blindly); ask about hardware/data/backend/output path; surface
tradeoffs (Megatron-Bridge vs AutoModel, full FT vs LoRA); present the plan
and wait for approval.

**Don't**: invent steps; skip Plan for pipelines ≥2 stages; generate Python or
shell when YAML suffices; import modules outside the step's reference code;
add monitoring/W&B unless asked; tune parallelism beyond `hardware.md` and
`[[strategies]]`; assume GPU count; generate Slurm/Airflow/Kubeflow wrappers;
handle non-training requests in this skill; modify `src/nemotron/steps/`;
restate per-step rules here — link the step's `README.md`.

## Troubleshooting

| Situation | Action |
|---|---|
| Artifact types do not chain | Recheck `types.toml`; change the DAG before writing configs. |
| Remote profile unclear / `--batch` ambiguous | Read the active env TOML; do not guess. |
| Config key unclear | Read the step config, `step.py`, and shared runner before editing. |
| Strategy points to a missing skill file | Skip the load; use the `then:` text and flag the plan with `WARNING: <topic> docs unavailable`. |
| Hardware too small | Show `[[models]]` `min_gpus`; suggest smaller model → AutoModel → LoRA. |
| Two failed Act attempts | Stop, explain what was tried and what failed, ask the user how to proceed. |
| No existing repo path matches | Check libraries cited in `step.toml [reference]`. If supported, use Explorer mode; otherwise ask. |
