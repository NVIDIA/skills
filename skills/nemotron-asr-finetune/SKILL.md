---
name: "nemotron-asr-finetune"
description: Orchestration skill for NVIDIA Nemotron Speech (Riva) / NeMo ASR domain and language adaptation. Given a goal like "improve/fine-tune ASR for my domain or language", it scopes the task, picks the cheapest sufficient path (word boosting → n-gram LM → fine-tuning), delegates each stage to the right sub-skill (data generation, training, evaluation, deployment), and answers cost/time/data questions along the way.
triggers:
  - fine-tune ASR for my domain
  - adapt ASR to my language
  - improve ASR accuracy
  - customize ASR
  - ASR domain adaptation
  - new language ASR
  - reduce WER
  - my ASR gets jargon wrong
  - noisy call-center ASR
  - how much data to fine-tune ASR
  - cheapest way to improve ASR
  - orchestrate ASR fine-tuning
  - ASR customization pipeline
version: "1.0.0"
license: Apache-2.0
metadata:
  author: "Nemotron Speech Team"
  team: riva
  tags:
    - nvidia
    - nemotron-speech
    - riva
    - nemo
    - asr
    - speech-to-text
    - orchestration
    - customization
    - domain-adaptation
    - fine-tuning
    - word-boosting
    - language-model
    - synthetic-data
    - evaluation
    - deployment
  domain: ml
---

# Nemotron Speech ASR Customization — Orchestration Skill

> **Note:** "Nemotron Speech" is the public-facing name for what NVIDIA documents today as **Riva** / **Riva NIM**; the acoustic models are trained and fine-tuned with **NVIDIA NeMo**. Commands, config paths, imports, and doc URLs still use **"Riva"** / **"NeMo"** — the rename is brand-only. Do not rename them.

## What This Skill Is

This is a **high-level orchestration skill**, not a step-by-step training manual. Its job, given a goal such as *"I want to fine-tune ASR for my domain/language"*, is to:

1. **Scope** the problem (how much real audio, target eval set, latency/hardware budget, language/domain).
2. **Choose the cheapest sufficient path** — word boosting, n-gram LM fusion, or fine-tuning — and escalate only when quality falls short.
3. **Delegate each stage to the right sub-skill** (data generation, training, evaluation, deployment/optimization).
4. **Answer cost/time/data questions along the way** (how many hours to hit X% WER, synthetic vs real, L40S vs H100, expected cost).

It owns the plan and the routing; the sub-skills own the execution. When a needed sub-skill does not exist yet, this skill names it as a **placeholder** and gives interim guidance.

## When to Use

Use for any request to make a Nemotron Speech / Riva ASR model work better on a specific domain or language — improving accuracy, reducing WER, adding a language, or planning a fine-tune. Start here even when the user names a specific technique, so the cheapest sufficient path is chosen and the right sub-skills are sequenced.

## Orchestration Workflow

Run the loop below; each stage names the sub-skill it invokes. Full detail in [`references/workflow.md`](references/workflow.md).

| # | Stage | What happens | Sub-skill |
|---|---|---|---|
| 1 | **State the goal** | Capture the target: domain/language, the errors, the metric. | Orchestration (this skill) |
| 2 | **Clarify & scope** | Ask the discovery questions: how much real audio? target eval set? latency/HW budget? deployment target? | Orchestration |
| 3 | **Choose the path** | Pick the cheapest sufficient rung (boosting → n-gram LM → fine-tune). Escalate only if quality is short; experiment while proposing the full plan. | Orchestration → Research/Training |
| 4 | **Get the data right** | If data is scarce/noisy: synthetic (TTS), TTS-friendly formatting, noise profiling/harvest, blend, score vendor samples; align customer data to training format; flag missing real data. | SDG / Data |
| 5 | **Train** | Apply the recipe (configs, hyperparameters, replay/curriculum, GPU/OOM preflight) and run. | Research / Training |
| 6 | **Evaluate** | Normalized WER on the domain set + A/B forgetting check on a general set; error-driven analysis to find the next lever. | Evaluation |
| 7 | **Loop or ship** | If short of target, loop to 4/5 with targeted data; else select/average checkpoints. Consult the user before more cycles. | Orchestration |
| 8 | **Deploy** | Export to NIM/HF, hot-swap the checkpoint, serve. | Deployment / Optimization |

**Stages 4–8 are the fine-tune path** (`data → NeMo train → NeMo eval → Riva deploy`). Cheaper rungs (boosting, custom vocab, n-gram LM) take a **shorter branch owned by a single sub-skill** — don't force them through the full loop. See the branch-by-rung table in [`references/workflow.md`](references/workflow.md) (§3b).

Throughout, answer the **"along the way"** questions (data volume, synthetic vs real, hours to reach a WER target, cost, GPU choice) — see [`references/planning-answers.md`](references/planning-answers.md).

## Sub-Skills This Skill Calls

Detailed registry, invocation, and handoff contracts in [`references/sub-skills.md`](references/sub-skills.md).

| Role (per the architecture) | Purpose | Sub-skill to invoke |
|---|---|---|
| **Research / Training** | NeMo configs, recipes, fine-tuning, checkpoint averaging | `nemo-speech-asr-finetune` |
| **SDG / Data Designer** | Synthetic transcripts/text, noise profiling, vendor-data impact, blends | `data-designer` (synthetic **text**; audio via TTS in `nemotron-speech`); *placeholder:* `asr-data-profiling` |
| **Evaluation** | Normalized WER, A/B forgetting, error analysis | Offline file WER → `nemo-speech-asr-finetune`; **served-endpoint WER → `nemotron-speech`** |
| **Deployment / Optimization** | NIM/Riva export, checkpoint swap, NIM-build optimization, serving | `nemotron-speech` |

If a sub-skill is unavailable, say so, give the interim guidance from the reference, and continue the plan.

## Choosing The Path (cheapest first)

The scoping in Stage 3 selects the lowest-cost rung that can meet the target. Summary; full docs-grounded ladder in [`references/path-selection.md`](references/path-selection.md).

- **Word boosting** — a bounded set of known words/names/jargon. Runtime, no training. → Deployment sub-skill.
- **Custom vocabulary / pronunciation** — OOV or consistently mispronounced terms. Deploy-time. → Deployment sub-skill.
- **N-gram (KenLM) LM** — domain phrasing/word-sequences when you have text but little audio. **Two realizations that are different artifacts:** *pilot (NeMo)* to prove lift offline (`nemo-speech-asr-finetune`), or *deploy (Riva)* to ship it (`nemotron-speech`). Don't ship the pilot LM — rebuild it in Riva word-level format. See [`references/path-selection.md`](references/path-selection.md).
- **Fine-tune** — real acoustic gaps (accents, noise, channel) with enough transcribed audio (NIM guide: 100+ h; ~10 h floor only if mixed to avoid catastrophic forgetting). → Research/Training.
- **Train from scratch / cross-language transfer** — a new language with no suitable checkpoint (last resort). → Research/Training.

Ordering and per-model support follow the NVIDIA Speech NIM ASR customization guide:
<https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html>.

## Key Principles

- **Scope before you pick.** Don't recommend fine-tuning before the discovery questions and a measured baseline.
- **Cheapest sufficient path.** Escalate rungs only when the current one provably can't hit the target; you may experiment on a cheap rung while presenting the full fine-tuning plan.
- **Measure with a contract.** Report normalized WER on the domain set plus an A/B forgetting check on a general set — never in-training logs alone.
- **Delegate, don't reimplement.** Route execution to the sub-skills; keep this skill focused on the plan, sequencing, and cost/time/data answers.
- **Real target-domain audio is the usual bottleneck.** Prefer real data; use synthetic to fill measured gaps, kept separately weighted so it can be ablated.
- **Consult the user before extra tuning cycles**, and when a needed sub-skill is a placeholder.

## Source of Truth

| Topic | Location |
|---|---|
| NIM Speech docs home | https://docs.nvidia.com/nim/speech/latest/index.html |
| ASR customization guide (methods, per-model support) | https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html |
| ASR support matrix (models & features) | https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html |
| NeMo fine-tuning (flags/config) | `docs/source/asr/fine_tuning.rst`, and the `nemo-speech-asr-finetune` sub-skill |
| Riva ASR tutorials (boosting, LM, fine-tune) | https://github.com/nvidia-riva/tutorials |
| Tokenizer extension to new language + acoustic fine-tune | https://github.com/nvidia-riva/tutorials/blob/main/asr-extend-tokenizer-to-newlang-ft-acoustic-model.ipynb |

## Limitations

- Orchestration only — execution happens in the sub-skills. Where a sub-skill is a placeholder, guidance is interim until it exists.
- GPU required for the training rungs; deployment/serving is owned by the `nemotron-speech` sub-skill.
- Model names, config paths, flags, and per-model feature support drift across NeMo/Riva releases — verify against the support matrix and the current checkout.
- Public branding is **"Nemotron Speech"**; commands, imports, config paths, and doc URLs still use **"Riva"** / **"NeMo"** — do not rename.
