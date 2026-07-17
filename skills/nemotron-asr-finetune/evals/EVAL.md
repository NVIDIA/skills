# Nemotron ASR Orchestration Eval Guidance

Use `evals/evals.json` to verify activation, scoping, path selection, and sub-skill delegation for the orchestration
skill `nemotron-asr-finetune`.

## What to grade

- The skill should activate for any goal to make a Nemotron Speech / Riva ASR model work better on a domain or
  language: improving accuracy, reducing WER, adding a language, or planning a fine-tune.
- It is an **orchestration** skill. The central behaviors are:
  1. **Scope first** — run the discovery questions (real audio hours, domain text, eval set/target, latency/HW budget,
     deployment target) before recommending a technique.
  2. **Choose the cheapest sufficient path** — word boosting → custom vocab/pronunciation / n-gram LM → fine-tune →
     (last resort) from scratch — and escalate only when the target is missed.
  3. **Delegate to sub-skills** — data (`data-designer` for synthetic **text**; TTS audio via `nemotron-speech` if
     needed), training (`nemo-speech-asr-finetune`), evaluation (`nemo-speech-asr-finetune`'s eval stage), deployment
     (`nemotron-speech`). The remaining placeholder is `asr-data-profiling` (name it as such with interim guidance). Do
     not route ASR work to LLM-only skills (`nemotron-customize`) or to the platform eval CLI (`nemo-evaluator-plugin`).
  4. **Answer cost/time/data questions** with ranges + assumptions, preferring a measured pilot over a fabricated
     hours-to-WER number.
- Evaluation contract: normalized WER on the domain set **plus** an A/B forgetting check on a general set; error-driven
  analysis to choose the next lever; consult the user before extra tuning cycles.
- Grade down: recommending fine-tuning for a problem boosting/LM would solve; failing to escalate when the residual
  error is acoustic; skipping scoping; or reimplementing sub-skill execution instead of delegating.
- Positive cases should load `SKILL.md` and the relevant reference (`workflow.md`, `path-selection.md`,
  `sub-skills.md`, or `planning-answers.md`). `scripts/main.py` is harness-only.
- Negative cases stay silent for pure export/deployment of an existing model (defer to `nemotron-speech`), OpenAI
  Whisper, and text-LLM fine-tuning.

## Harness-only script

`scripts/main.py` exists only because the evaluation harness requires a script entry point. It is a deterministic
prompt-to-reference router whose default is the orchestration workflow. It does not invoke sub-skills and is not part of
the agent-facing workflow; do not use it as grading evidence for positive cases.
