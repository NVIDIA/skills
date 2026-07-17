# Sub-Skill Registry

The orchestrator delegates execution to these sub-skills. Each entry gives the role (from the ASR customization
architecture), the skill to invoke, what to hand it, what it returns, and the fallback when it is a **placeholder**
(not yet available). Invoke a sub-skill with the Skill tool by its `name`; if unavailable, tell the user, give the
interim guidance, and continue.

## Research / Training

- **Invoke:** `nemo-speech-asr-finetune` (NeMo ASR fine-tuning: container setup, checkpoint selection, Lhotse data,
  tokenizer, training, checkpoint averaging, and the eval stage).
- **Hand it:** the chosen path, checkpoint/family, style-audited train/val/test manifests, constraints
  (streaming/offline, precision, GPU budget), and any blend/replay requirement.
- **Returns:** trained/averaged `.nemo` checkpoint(s) and in-training `val_wer`.
- **Notes:** owns all exact NeMo flags/config paths and model-family recipes (CTC/RNNT/TDT/hybrid/AED, plain vs prompted
  cache-aware streaming). Also holds the "recipes / best-practice knowledge base." (Do **not** route ASR training to
  `nemotron-customize` — that skill is for Nemotron **LLM** customization, not speech.)
- **Also owns the n-gram LM *pilot* (offline only):** build a subword KenLM (`train_kenlm.py`) and score greedy-vs-LM
  with the GPU `beam_batch` decoder (`eval_beamsearch_ngram_ctc.py`) to prove lift cheaply. This pilot LM is a
  **NeMo-only** artifact — to ship, hand the *corpus* (not the pilot `.bin`) to `nemotron-speech`, which rebuilds the
  Riva-format LM. See the n-gram Rung Note in [`path-selection.md`](path-selection.md).

## SDG / Data Designer

- **Invoke:** `data-designer` (or the NeMo-platform variant `nemo-data-designer-plugin`) to build **synthetic text
  datasets** — domain term lists, target-style transcripts, and prompts. It generates text/tabular data, **not audio**.
- **For synthetic audio:** turn that text into speech with a TTS model. `nemotron-speech` can run TTS inference
  (e.g. Magpie) to synthesize audio, but it is a NIM deploy/run skill, so this is a heavier step than text generation —
  scope whether synthetic audio is actually needed before committing.
- **Hand it:** the domain/language, target transcript style, required volume, and formats of any vendor/customer data.
- **Returns:** synthetic transcripts/text (data-designer) and, if used, synthesized audio (via TTS).
- **Placeholder — `asr-data-profiling`:** noise profiling, in-domain noise harvest, vendor-data impact analysis, and
  assembling `(audio, transcript)` manifests are not yet a dedicated skill. Interim: profile audio (sample rate, SNR,
  duration/tps distributions), harvest realistic in-domain noise, score vendor samples with the current model, align
  format, and keep synthetic sources separately weighted. Flag missing real target-domain data explicitly.

## Evaluation

Evaluation has **two surfaces** — route to the one that matches the branch (see
[`workflow.md`](workflow.md) §3b/§6):

- **Offline file WER** (a `.nemo`/`.riva` checkpoint, before serving) — **Invoke:** `nemo-speech-asr-finetune`'s
  **evaluation stage**: standalone WER/CER via `speech_to_text_eval.py` (default contract: lowercased, punctuation
  removed) and cache-aware streaming eval; the n-gram **pilot** uses `beam_batch` greedy-vs-LM here.
  - **Hand it:** the checkpoint(s), the domain eval set, and a general guardrail set.
  - **Returns:** normalized domain WER and per-variant comparisons; the orchestrator computes the general-set forgetting
    delta and the error-category breakdown from these to pick the next lever.
- **Served-endpoint WER** (a client scoring a **running NIM**, for any served rung: boosting, custom vocab, n-gram LM
  deploy, or a fine-tune after deploy) — **owned by `nemotron-speech`** (it serves the NIM; score it with the riva
  client). Use this for the deployed before/after — the served decoder can differ from the in-NeMo result.
- **Note:** `nemo-evaluator-plugin` is a NeMo Platform eval CLI for served endpoints/LLM-style metrics, **not** ASR WER —
  do not route ASR accuracy evaluation there.

## Deployment / Optimization

- **Invoke:** `nemotron-speech` (Riva NIM: export `nemo2riva`, `riva-build`/`riva-deploy`, pipeline config, and serving).
  This is also where **runtime/decoding customizations** (word boosting, custom vocab/pronunciation, n-gram LM at decode,
  ITN, VAD, diarization) and any NIM build-time optimization live. It also **owns the whole Riva n-gram LM realization
  end-to-end** (build the Riva-format LM → `riva-build` flashlight → serve) and the **served-endpoint WER** for it.
- **Hand it:** the source model (a deployable `.riva` when only doing decode-time changes, else the evaluated `.nemo`),
  the target hardware/latency, and any runtime customization list (boost words, LM, vocab) recommended in path selection.
- **Artifact typing (n-gram LM):** type every LM handoff as `{format: nemo-subword | riva-word-level, vocab,
  tokenizer-clean?}`. `nemotron-speech` consumes **only** the `riva-word-level` LM plus a `decoding_vocab` whose
  characters are all in the model tokenizer's alphabet. **Never** hand it the NeMo subword *pilot* LM — it will not
  decode correctly (rebuild for Riva from the same corpus). Deploy-side prerequisites `nemotron-speech` owns:
  `.nemo`→`.riva` via a **version-matched `nemo2riva`** (or start from a deployable `.riva`), and NIM images ship
  **different `riva-build` CLIs** (classic rejects `.nemo`; Hydra accepts it) — verify before choosing the ingestion path.
- **Returns:** a deployed/served NIM, runtime configuration, and served-endpoint WER.
- **Note:** ASR export/serving optimization is part of the NIM build in `nemotron-speech`; do **not** route it to
  `nemotron-customize` (Nemotron LLM customization/ModelOpt), which does not apply to speech models.

## Ownership / provenance (for maintainers)

The architecture assigns owners per box — Research/Training (Research), SDG/Data (Data Designer + Eng), Evaluation
(Research/Eng), Deployment/Optimization (Skills/NIM). Keep this registry updated as placeholder skills are published so
routing points at the real `name` rather than interim guidance. A working analog of this orchestration pattern is the
Clinical ASR Flywheel (`digital-health-clinical-asr-*`: setup → build → eval → finetune), read from those skills'
frontmatter.
