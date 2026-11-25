# Orientation Routing Reference

`nvflare-orient` is the lead skill for ambiguous NVFLARE requests. It should
turn project evidence and user intent into one narrow next action.

## Evidence Sources

- `nvflare agent inspect source <path> --format json` for ownership,
  integration, scan findings, and source routing.
- `nvflare agent inspect data <path> --format json` for dataset and data
  routing.
- User-provided target files, job folders, logs, or stated deployment context.

## Routing Rules

- Existing PyTorch training loop needing FLARE conversion:
  `nvflare-convert-pytorch`.
- Hugging Face `Trainer`, `Seq2SeqTrainer`, or TRL `SFTTrainer` owning
  training: `nvflare-convert-huggingface`.
- Lightning `Trainer` owning training: `nvflare-convert-lightning`.
- Active Hugging Face and Lightning Trainers in the same inspected project:
  ask the user to select one training-loop owner or split them into separate
  entrypoints/jobs before handing off to a converter.
- Hugging Face Trainer/configuration evidence whose `train()` ownership cannot
  be resolved statically: ask the user to name the training entrypoint or
  identify the Trainer construction that owns the call, then route to the
  matching converter.
- Statistics, data summaries, histograms, or quantiles across sites, or an
  inspect result with `dataset.modality` `tabular`/`image`
  (data-only targets recommend `nvflare-fed-stats` directly):
  `nvflare-fed-stats`.
- Generic "help me use FLARE here" with no clear workflow: inspect first, then
  recommend the narrowest skill.
- Existing FLARE job that fails or produces suspicious logs:
  `nvflare-diagnose-job`, not conversion.
- Existing FLARE job or project to optimize or improve — low accuracy, an
  underperforming metric, or hyperparameter/algorithm exploration:
  `nvflare-autofl`, not diagnosis.
- POC startup, production submission, Kubernetes deployment, or identity setup:
  route to the corresponding operations or deployment skill when available.
- Non-FLARE Python, web, data science, or generic ML questions: no FLARE skill.

## Output Shape

Summaries should name:

- target path inspected;
- strongest evidence found;
- recommended next skill or no-skill decision;
- unresolved semantic prerequisites and the validation expected in the next
  workflow.

Do not turn routing into implementation. Once the next skill is clear, hand off
instead of continuing with broad advice.
