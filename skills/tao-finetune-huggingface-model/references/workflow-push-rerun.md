# HuggingFace Fine-Tune Push And Rerun

Hub push, rerun-skill emission, error playbook usage, and communication expectations from the pre-refactor guide.

Load this file only when the compact `SKILL.md` points here for the current task. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Contents

- Error playbook
- Communication style

### Step 6 — Push & emit rerun skill

**Goal:** publish the run and ensure it can be reproduced without re-research.

**6a. Push to HF Hub** — use the script in `references/hub-push.md`. Pushes:
- model weights (merged or final)
- model card (`README.md`) generated from `config.yaml` + eval JSONs
- `results/{eval,baseline}_results.json`, `config.yaml`, `Dockerfile`,
  `requirements.txt`, `inference_samples/*.jpg`
- `report.{pdf,html}` if `emit_report: true`

Skip iff `push_to_hub: false` is explicit in `config.yaml`.

**6b. Emit rerun skill** at `<output_dir>/skills/run-<short>/SKILL.md` per the
template in `references/pipeline-skill-template.md`. Every `<placeholder>` must
be substituted with a real value. Literal placeholders in the output are a bug.
Include full YAML (`license`, `compatibility`, `metadata`, `allowed-tools`) and
the NVIDIA copyright notice in an HTML comment (`<!--` … `-->`) immediately after
the closing `---`, as in that template. If you generate an emitter script, make it fail unless the emitted `SKILL.md` contains those fields and the HTML copyright comment.

**Gate (Done criteria):** all of:
- Step 5 gate met
- HF Hub repo exists at the resolved URL with weights + card + `results/`
  (unless `push_to_hub: false`)
- `<output_dir>/skills/run-<short>/SKILL.md` exists, no `<placeholder>` left,
  with metadata + copyright HTML comment per `pipeline-skill-template.md`

**Final message to user** — terse, with direct URLs:
- wandb URL
- HF Hub URL
- primary metric: baseline → fine-tuned (Δ)
- path to `reports/inference_samples/`
- path to `<output_dir>/skills/run-<short>/SKILL.md`

---

## Error playbook

When you hit a known runtime error, consult `references/error-playbook.md`
before redesigning anything — it carries the symptom → minimal-fix table
(NGC ENTRYPOINT, PyTorch 2.5 SDPA+GQA bug, `transformers>=4.51`
`@check_model_inputs` regression, numpy 2.x ABI break, Albumentations
degenerate bbox, PEFT + gradient_checkpointing, Idefics3 / SmolVLM SDPA,
LoRA target-regex breadth, missing CV augmentation, OOM at step 0, …).

When a row in that table fires twice across runs, lift it into
`compat-workarounds.md` with a `detect` rule — that registry is the
durable form, auto-applied in Step 1d before the error has a chance to fire.

---

## Communication style

- Terse. No filler, no restating the request. One-word answers when appropriate.
- Always include direct Hub and wandb URLs when referencing artifacts.
- On error: state what went wrong, why, what you changed. No menus.
- Never present "Option A/B/C" for a request that has a clear answer. Act.
