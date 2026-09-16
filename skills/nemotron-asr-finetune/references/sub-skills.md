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
- **Also owns the n-gram LM *pilot* (offline only, NGPU-LM):** build a subword KenLM with `train_kenlm.py`
  (`save_nemo=True`) and score via `speech_to_text_eval.py`'s GPU shallow fusion — CTC uses `ctc_decoding.*`,
  RNN-T/TDT uses `rnnt_decoding.*` (no CTC-only script substitutes for RNNT; `eval_beamsearch_ngram_ctc.py` is a
  CTC-only grid-search convenience tool, not the general path). CTC also has a second, older NeMo-side mechanism —
  its own Flashlight decoder (word-level KenLM + lexicon). **Whether a pilot artifact ships to `nemotron-speech`
  unchanged depends on architecture and which pilot mechanism was used, not a blanket rule:** the NGPU-LM `.nemo`
  artifact is subword-tokenized, so it **definitely** needs the corpus rebuilt into a Riva word-level LM for CTC —
  confirmed. Whether the Flashlight mechanism's own word-level KenLM + lexicon can be reused as-is for CTC is
  **unverified**, not a confirmed rebuild either way. RNNT hands the exact same NGPU-LM pilot `.nemo` file straight
  to `nemotron-speech`, confirmed. See the n-gram Rung Note in [`path-selection.md`](path-selection.md).
- **Also owns the word-boosting *pilot* (offline only, GPU-PB context biasing):** decode-time shallow fusion via a
  GPU-accelerated boosting tree — no retraining. CTC uses `ctc_decoding.*.boosting_tree`, RNN-T/TDT uses
  `rnnt_decoding.*.boosting_tree`, both via `speech_to_text_eval.py`, weighted by `boosting_tree_alpha`; optionally
  pre-built with `scripts/asr_context_biasing/build_gpu_boosting_tree.py`. This uses a different score system
  (`context_score`/`depth_scaling`/`boosting_tree_alpha`) than Riva's `boosted_lm_score` — a NeMo pilot result does
  not hand off a reusable score to `nemotron-speech`, only a validated phrase list and confidence that boosting
  helps. See the word-boosting Rung Note in [`path-selection.md`](path-selection.md).
- **Setup — this sub-skill is not catalog-published.** Unlike the other sub-skills, `nemo-speech-asr-finetune` is a
  *project-local* Claude Code skill living inside the NeMo/Speech repo itself (`.claude/skills/nemo-speech-asr-finetune/`),
  not the `nvidia-skills` catalog — every command in it is a path relative to that repo's root, so it only works when
  reachable from inside (or linked to) an actual checkout. Do not assume it is invocable just because it was named
  here. Before handing off to it:
  1. Try the standard installer first: `npx skills add NVIDIA-NeMo/Speech --skill nemo-speech-asr-finetune --agent
     claude-code --global --yes`. Unverified whether the installer supports arbitrary repos beyond the `nvidia/skills`
     catalog — test it before relying on it.
  2. If that doesn't work, fall back to the bundled [`../scripts/link-nemo-subskill.sh`](../scripts/link-nemo-subskill.sh)
     `<checkout_path>` — validates the source, refuses to clobber an existing real directory, and records the real
     repo root so it can be found reliably afterward.
  3. Either way, see [`workflow.md`](workflow.md) §4b for resolving the checkout in the first place, and for the
     required step of changing the working context into that checkout before Stage 5 actually runs any of this
     sub-skill's commands — linking it only makes it *discoverable*, not *runnable*.

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
- **Owns the pre-flight sample-rate audit/conversion** (see [`workflow.md`](workflow.md) §4a) as part of format
  alignment: verify every file is 16 kHz and resample offline (`sox`/`ffmpeg`) any file that is not, before handing
  the manifest to training.

## Evaluation

Evaluation has **two surfaces** — route to the one that matches the branch (see
[`workflow.md`](workflow.md) §3b/§6):

- **Offline file WER** (a `.nemo`/`.riva` checkpoint, before serving) — **Invoke:** `nemo-speech-asr-finetune`'s
  **evaluation stage**: standalone WER/CER via `speech_to_text_eval.py` (default contract: lowercased, punctuation
  removed) and cache-aware streaming eval; the n-gram **pilot** (NGPU-LM) uses this same script's
  `ctc_decoding.*.ngram_lm_model`/`rnnt_decoding.*.ngram_lm_model` overrides, and the word-boosting **pilot** uses
  its `ctc_decoding.*.boosting_tree`/`rnnt_decoding.*.boosting_tree` overrides.
  - **Hand it:** the checkpoint(s), the domain eval set, and a general guardrail set.
  - **Returns:** normalized domain WER and per-variant comparisons; the orchestrator computes the general-set forgetting
    delta and the error-category breakdown from these to pick the next lever.
- **Served-endpoint WER** (a client scoring a **running NIM**, for any served rung: boosting, custom vocab, n-gram LM
  deploy, or a fine-tune after deploy) — **owned by `nemotron-speech`** (it serves the NIM; score it with the riva
  client). Use this for the deployed before/after — the served decoder can differ from the in-NeMo result.
- **Note:** `nemo-evaluator-plugin` is a NeMo Platform eval CLI for served endpoints/LLM-style metrics, **not** ASR WER —
  do not route ASR accuracy evaluation there.
- **Also reused pre-training:** the pre-flight transcript-quality audit ([`workflow.md`](workflow.md) §4a) uses this
  same WER tooling — offline via `nemo-speech-asr-finetune` or served via `nemotron-speech` — but scores a stock
  pretrained reference checkpoint against the *provided ground truth*, not a fine-tune checkpoint against a held-out
  set. Same mechanism, different purpose: flagging bad labels before training, not measuring a trained model.

## Deployment / Optimization

- **Invoke:** `nemotron-speech` (Riva NIM: export `nemo2riva`, `riva-build`/`riva-deploy`, pipeline config, and serving).
  This is also where **runtime/decoding customizations** (word boosting, custom vocab/pronunciation, n-gram LM at decode,
  ITN, VAD, diarization) and any NIM build-time optimization live — exact `riva-build` flags and LM file formats are
  documented there (`references/pipelines.md`'s Language Models section), not duplicated here. It **owns the deploy
  realization of word boosting end-to-end** — per-request, no build step, no dependency on the NeMo pilot — and the
  deploy realization of the n-gram LM, which **differs by architecture**: CTC's NGPU-LM path needs a rebuilt
  Riva-format LM — confirmed, since that pilot artifact is subword-tokenized (whether the Flashlight-pilot's own
  word-level KenLM + lexicon can be reused as-is is unverified — see [`path-selection.md`](path-selection.md));
  RNNT reuses the NeMo pilot's `.nemo` artifact directly, no rebuild step, confirmed. Owns the **served-endpoint
  WER** for both.
- **Hand it:** the source model (a deployable `.riva` when only doing decode-time changes, else the evaluated `.nemo`),
  the target hardware/latency, and any runtime customization list (boost words, LM, vocab) recommended in path selection.
- **Artifact typing (n-gram LM) — differs by architecture, confirm which applies before handoff:**
  - **CTC:** `nemotron-speech` consumes a Riva word-level LM, not a subword one — see `references/pipelines.md` for
    the exact format/flags. **Never** hand it the NeMo subword *pilot* LM for CTC — it will not decode correctly
    (rebuild for Riva from the same corpus).
  - **RNN-T/TDT:** the opposite — hand `nemotron-speech` the pilot's `.nemo` NGPU-LM artifact (from
    `train_kenlm.py ... save_nemo=True`) **unchanged** — see `references/pipelines.md` for the exact flags. Do not
    rebuild it into a CTC-style word-level form; that's not what the RNNT deploy path expects.
  Deploy-side prerequisites `nemotron-speech` owns:
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
