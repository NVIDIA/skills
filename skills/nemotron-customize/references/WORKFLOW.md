# Nemotron Customize Workflow

Use this reference when `SKILL.md` says to run the full pipeline workflow or
Explorer mode.

## Phase 1: Orient

Goal: enumerate candidate steps and gather constraints in one pass.

Discover via the CLI when available:

```bash
nemotron steps list --json
nemotron steps list --json --category sft
nemotron steps list --json --consumes training_jsonl
nemotron steps list --json --produces checkpoint_megatron
nemotron steps show <step_id>
```

Read these first:

- `src/nemotron/steps/STEPS.md`
- `src/nemotron/steps/PATTERNS.md`
- `src/nemotron/steps/types.toml`
- `src/nemotron/steps/hardware.md` when hardware is in scope

For each candidate category, read `src/nemotron/steps/<cat>/README.md`. For
each candidate step, read its `step.toml` end-to-end, especially `[[consumes]]`,
`[[produces]]`, `[[parameters]]`, `[[strategies]]`, `[[errors]]`, and
`[reference]`.

Ask for missing constraints before planning: model, data path/source, data size,
GPU count/type/nodes, backend or env profile, W&B preference, output directory,
and required API key environment variable names.

## Phase 2: Plan

Produce a markdown plan the user reviews before code or config changes.

Include:

- `Intent`
- `Stages`
- `Validation`
- `Infrastructure`

For each stage, list the step id, input source, output artifact, 2-3 key
parameters, matched `step.toml` strategies, and matched patterns. Use a Mermaid
graph for artifact flow.

Hard checks:

- Artifact types chain via `types.toml`.
- Tokenizer, chat template, and sequence length align across prep and train.
- RL stages warm-start from an SFT-compatible checkpoint.
- GPU count satisfies the selected model and training stack.
- Applicable patterns are cited.

If a check fails, surface it as `WARNING:` and propose a fix. For too-small
hardware, suggest smaller model, then AutoModel, then LoRA, before full
Megatron-Bridge fine-tuning.

Wait for user approval before Act. If new code is necessary, name the missing
repo capability and get approval for Explorer mode.

## Phase 3: Act

Prefer YAML-only changes for existing steps. No placeholders or TODOs.

Before creating code, identify the existing execution path:

- CLI commands under `src/nemotron/cli/`
- Step entrypoints in `src/nemotron/steps/<cat>/<step>/step.py`
- Shared runners in `src/nemotron/steps/_runners/`
- Existing configs under the selected step, recipe, or runner directory

Generate configs under:

```text
<project-name>/
├── configs/
│   └── <stage-name>.yaml
└── README.md
```

Only add `README.md` when the user asks for run docs. YAML must match keys read
by the existing `step.py` and runner, adapt checked-in configs rather than
inventing schemas, use user-provided paths and environment choices, and preserve
artifact compatibility from the approved plan.

## Explorer Mode

Use Explorer mode only when no existing callable step, runner, CLI, recipe, or
YAML config surface can satisfy the request.

Load:

- `skills/nemotron-customize/references/act/PROJECT.md`
- `skills/nemotron-customize/references/act/STAGE.md`
- The relevant context pack from `references/context/index.toml`, if mapped
- The closest `src/nemotron/steps/<cat>/<step>/step.py`
- The relevant shared runner, if the step imports one

Implement the narrowest missing stage. Mirror existing `step.py` shape, type
consumes/produces with `types.toml`, and report files written, exposed knobs,
UPSTREAM notes, and followed strategies. If the same Explorer build keeps
appearing, suggest contributing a catalog step under `src/nemotron/steps/`.

## Phase 4: Verify

Check before reporting completion:

- Every generated YAML file parses and uses keys supported by the step/runner.
- Stage output artifact types match the next stage's input types.
- Existing CLI or runner commands can consume the generated configs.
- Exceptional code has valid Python syntax and imports real repo modules.
- README commands, if written, match actual configs.
- Smoke configs use reduced iters, batch sizes, or max steps.
- Tokenizer and sequence length align across prep and training configs.
- Standalone YAML does not leak `${art:...}` references unless a recipe path
  explicitly requires them.

Fix verification issues before reporting completion.
