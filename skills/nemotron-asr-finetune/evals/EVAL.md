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
     (last resort) from scratch — and escalate only when the target is missed. **Hard floor:** below ~10 hours of
     real transcribed target-domain audio, do not recommend fine-tuning at all — catastrophic forgetting and
     overfitting risk are both severe at that volume — recommend word boosting instead.
     - **Word boosting has a NeMo pilot and a Riva deploy realization** — different artifacts with different,
       non-interchangeable score systems. Pilot: GPU-PB context biasing via the `boosting_tree` config
       (CTC: `ctc_decoding.*.boosting_tree`; RNN-T/TDT: `rnnt_decoding.*.boosting_tree`, both through
       `speech_to_text_eval.py`, weighted by `boosting_tree_alpha`) — owned by `nemo-speech-asr-finetune`, entirely
       self-contained, no Riva required. Deploy: Riva's per-request `boosted_lm_score` — owned by `nemotron-speech`.
       Do not conflate the two score systems, and do not force a customer working the NeMo pilot to also touch
       Riva.
     - **The n-gram LM also has a NeMo pilot (NGPU-LM) and a Riva deploy realization, but carry-over to Riva is
       architecture-dependent, not a blanket rule.** Pilot: build with `train_kenlm.py` (`save_nemo=True`), score
       via `speech_to_text_eval.py`'s `ctc_decoding.*.ngram_lm_model`/`rnnt_decoding.*.ngram_lm_model` — CTC also
       has a second, older NeMo-side mechanism (its own Flashlight decoder, word-level KenLM + lexicon).
       **Deploy — CTC needs a rebuild** into a word-level LM for Riva's Flashlight decoder. **Deploy — RNN-T/TDT
       reuses the pilot `.nemo` artifact unchanged** — verified against NVIDIA's own RNNT tutorial. Exact
       `riva-build` flags/syntax for either case are owned by `nemotron-speech` (`references/pipelines.md`) — the
       orchestrator should state the routing/carry-over fact and defer to that skill for flags, not recite them
       from its own memory. Do not apply the CTC rebuild rule to RNNT, and do not assume the RNNT exception
       applies to CTC.
  3. **Delegate to sub-skills** — data (`data-designer` for synthetic **text**; TTS audio via `nemotron-speech` if
     needed), training (`nemo-speech-asr-finetune`), evaluation (`nemo-speech-asr-finetune`'s eval stage), deployment
     (`nemotron-speech`). The remaining placeholder is `asr-data-profiling` (name it as such with interim guidance). Do
     not route ASR work to LLM-only skills (`nemotron-customize`) or to the platform eval CLI (`nemo-evaluator-plugin`).
  4. **Answer cost/time/data questions** with ranges + assumptions, preferring a measured pilot over a fabricated
     hours-to-WER number.
  5. **Gate on data quality before training (Stage 4a)** — verify/convert audio to 16 kHz offline before handing a
     manifest to Stage 5, and, when the user asks, audit transcript quality by running a reference pretrained
     checkpoint (not the fine-tune target) and comparing its WER against the provided ground truth, surfacing flagged
     utterances rather than dropping them silently. If the user insists on feeding 8 kHz files as-is instead of
     offline-converting, confirm that choice explicitly (it is not recommended — online resampling during training is
     slow) and, once confirmed, keep `model.sample_rate` at the checkpoint's expected rate (typically 16000) so
     NeMo's loader upsamples online; do not set `model.sample_rate=8000`, which trains a different, 8 kHz-native model
     rather than fine-tuning the target checkpoint.
  6. **Gate on the training environment before training (Stage 4b)** — three-way branch:
     - **Nothing provided/found:** pull the latest `main` from the current canonical repo (`NVIDIA-NeMo/Speech`) for
       local execution, or the latest published container tag for container execution. Mandatory, not optional.
     - **Something is provided:** check its version (`nemo.__version__`, git commit/branch for a source checkout),
       but still recommend switching to latest `main` (staleness risk — a provided checkout can predate a feature the
       chosen technique needs) and **ask the user** rather than silently picking either side.
     - **User insists on keeping the provided checkout:** proceed with it, do not pull preemptively. Only
       recommend/pull latest `main` again if a concrete version-specific issue is actually hit later (e.g. an
       unsupported config key or missing feature) — treat that as new evidence, not as overriding the user's choice.
     Record what was resolved (and any later escalation) in the ledger.
  7. **Make `nemo-speech-asr-finetune` reachable before Stage 5** — it is project-local (lives inside the NeMo/Speech
     repo's own `.claude/skills/`), not catalog-published, so it is not invocable just because it was named. Make it
     discoverable (try `npx skills add NVIDIA-NeMo/Speech --skill nemo-speech-asr-finetune ...`, fall back to
     `scripts/link-nemo-subskill.sh <checkout_path>`), then change the actual working context (cd, or a scoped
     sub-agent) into that checkout before running any of its commands — they are paths relative to the repo root,
     and linking alone only makes the skill discoverable, not runnable.
- Evaluation contract: normalized WER on the domain set **plus** an A/B forgetting check on a general set; error-driven
  analysis to choose the next lever; consult the user before extra tuning cycles.
- Grade down: recommending fine-tuning for a problem boosting/LM would solve; failing to escalate when the residual
  error is acoustic; skipping scoping; reimplementing sub-skill execution instead of delegating; training on audio
  that was never verified as 16 kHz; silently dropping/ignoring flagged bad-transcript utterances instead of
  surfacing them to the user; for a user who wants to skip offline conversion of 8kHz audio, either silently
  offline-converting without asking or setting `model.sample_rate=8000` (which changes the model, not just the
  input pipeline) instead of keeping `model.sample_rate` at the checkpoint's expected rate; or recommending/proceeding
  with fine-tuning when real transcribed target-domain audio is under ~10 hours without flagging the severe
  catastrophic-forgetting/overfitting risk and recommending word boosting instead; proceeding to Stage 5 without
  resolving and recording a NeMo version (silently assuming one exists, or silently training against an unverified/
  stale checkout instead of pulling latest main or checking the provided one); silently accepting a provided
  checkout without recommending latest main and asking the user; or, once the user has explicitly chosen to keep a
  provided checkout, second-guessing that choice / pulling latest main anyway without a concrete version issue
  having actually surfaced; assuming `nemo-speech-asr-finetune` is invocable at Stage 5 without first making it
  discoverable and changing the working context into its NeMo checkout; for a NeMo-only word-boosting request,
  routing to `nemotron-speech`/Riva instead of the NeMo pilot, or presenting NeMo's `context_score`/
  `boosting_tree_alpha` as if it were Riva's `boosted_lm_score` (or vice versa); or, for the n-gram LM, applying a
  single blanket rule across CTC and RNNT instead of checking which architecture applies — e.g. telling an RNNT
  user to rebuild a word-level KenLM/`decoding_vocab` they don't need, or telling a CTC user their pilot artifact
  ships to Riva unchanged when it doesn't.
- Positive cases should load `SKILL.md` and the relevant reference (`workflow.md`, `path-selection.md`,
  `sub-skills.md`, or `planning-answers.md`). `scripts/main.py` is harness-only.
- Negative cases stay silent for pure export/deployment of an existing model (defer to `nemotron-speech`), OpenAI
  Whisper, and text-LLM fine-tuning.

## Harness-only script

`scripts/main.py` exists only because the evaluation harness requires a script entry point. It is a deterministic
prompt-to-reference router whose default is the orchestration workflow. It does not invoke sub-skills and is not part of
the agent-facing workflow; do not use it as grading evidence for positive cases.
