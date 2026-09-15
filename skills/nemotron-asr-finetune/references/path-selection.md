# Path Selection — Cheapest Sufficient Rung

Stage 3 of the workflow. Pick the **lowest-cost path that can meet the target**, escalate only when quality falls short,
and delegate execution to the sub-skills. Always diagnose the failure mode first and measure a baseline (Evaluation
sub-skill) so each step is justified.

Ordering and per-model support follow the NVIDIA Speech NIM (Riva) ASR customization guide:
<https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html>. Verify parameters and support there.

## The Ladder (cheapest → most expensive)

| Rung | Fixes | Needs | Cost | Runs where → sub-skill |
|---|---|---|---|---|
| **Word boosting — pilot (NeMo)** | Cheaply prove boosting lift *offline* before deploying | A word list + a `.nemo`/pretrained checkpoint | Minimal, no training | `nemo-speech-asr-finetune` (GPU-PB `boosting_tree` via `speech_to_text_eval.py`) |
| **Word boosting — deploy (Riva)** | Known words/names/jargon/commands at serving | A word list | Minimal, no training | `nemotron-speech` (runtime `boosted_lm_score`, no build) |
| **Custom vocabulary / pronunciation** | OOV words, consistent mispronunciations | Vocab/lexicon file | Low, deploy-time | `riva-build` → `nemotron-speech` |
| **N-gram LM — pilot (NeMo)** | Cheaply prove LM lift *offline* before deploying | Domain **text** + a `.nemo` | Low, no serving | `nemo-speech-asr-finetune` (NGPU-LM via `speech_to_text_eval.py`) |
| **N-gram LM — deploy (Riva)** | Domain phrasing / word sequences at serving | Domain text (CTC) or the pilot `.nemo` (RNNT) | Moderate, decode-time | `nemotron-speech` (CTC: rebuild word-level LM; RNNT: reuse pilot `.nemo` directly — exact flags in `pipelines.md`) |
| **Fine-tune** | Real acoustic gaps (accents, noise, channel) | 100+ h transcribed (10 h floor if mixed; **<10 h → don't fine-tune, use word boosting**) + GPU | High | Research/Training (`nemo-speech-asr-finetune`) |
| **Train from scratch / cross-language** | A new language/dialect, no checkpoint | Thousands of h (16+ h for transfer) | Very high | Research/Training (last resort) |

Rungs compose: a fine-tuned model still uses boosting and an LM at serving time.

**One rung, one owner, end-to-end.** Route the *whole* lifecycle of a customization to a single sub-skill rather than
splitting its build → eval → deploy across skills. This applies to word boosting too, not just the n-gram LM: each has
**two realizations that are different artifacts with different score systems** — a NeMo **pilot** (owned by
`nemo-speech-asr-finetune`) and a Riva **deploy** (owned by `nemotron-speech`). Whether the pilot artifact carries
into Riva unchanged **depends on architecture, not on a blanket rule** — for word boosting it never does (different
score systems entirely); for the n-gram LM it depends on CTC vs RNNT (see the Rung Notes — RNNT is the one case
where the pilot artifact ships to Riva verbatim). Pick the realization up front from the goal (validate vs ship); do
not assume portability either way without checking the Rung Note. A customer working purely on a NeMo pilot never
needs to touch Riva — it is a complete, self-contained loop.

## Diagnose First

- **A short, known set of specific words wrong** (names, SKUs, commands, acronyms) → **word boosting**. Decide the
  *realization* up front: **pilot (NeMo)** to prove lift offline for cheap and see which phrases actually help, or
  **deploy (Riva)** to ship it — different, non-interchangeable score systems (see Rung Notes). If truly OOV or
  mispronounced → **custom vocabulary / pronunciation**.
- **Wrong word sequences / phrasing**, with domain text available → **n-gram LM**. Decide the *realization* up front:
  **pilot (NeMo)** to prove lift offline for cheap, or **deploy (Riva)** to ship it. Whether the pilot artifact
  carries into Riva unchanged depends on architecture — CTC needs a rebuild, RNNT reuses it directly (see Rung
  Notes).
- **Model mis-hears audio** (accent, noise, channel) with a real acoustic gap and **≥10 h** of real transcribed
  target-domain audio → **fine-tune**. Below that floor, do not recommend fine-tuning — see the guardrail in Rung
  Notes below.
- **New language/dialect, no suitable checkpoint** → train from scratch / cross-language transfer (rare).
- **Formatting only** (punctuation, casing, numbers) → runtime automatic-punctuation / ITN flags, not accuracy work.

## Rung Notes (docs-grounded)

- **Word boosting — two realizations, different score systems; NOT interchangeable.** Cannot fix acoustics. Decide
  which realization the goal needs before building:
  - **Pilot (NeMo, offline, GPU-PB context biasing)** — owner `nemo-speech-asr-finetune`. Decode-time shallow fusion
    via a GPU-accelerated boosting tree — no retraining, no build step required against any `.nemo`/pretrained
    checkpoint. Supported for CTC, RNN-T/TDT, and AED (Canary). This whole pilot runs entirely inside NeMo; nothing
    here requires Riva. **GPU-PB is the generic method — the only one of NeMo's three word-boosting mechanisms that
    covers plain RNN-T/TDT.** CTC has two other, older trails (see the fallback bullet below); default to GPU-PB
    unless a specific reason points elsewhere.
    - Optional pre-build (reusable across runs): `scripts/asr_context_biasing/build_gpu_boosting_tree.py
      asr_model_path=<.nemo> key_phrases_file=<list> save_to=<btree_path> context_score=1.0 depth_scaling=2.0
      use_triton=True`. Otherwise pass `key_phrases_file`/`key_phrases_list` inline at decode time.
    - **CTC** — via `examples/asr/speech_to_text_eval.py`: `ctc_decoding.strategy=greedy_batch|beam_batch`, keys
      under `ctc_decoding.greedy.boosting_tree.*` / `.beam.boosting_tree.*`, `boosting_tree_alpha=<weight>`.
    - **RNN-T/TDT** — same script: `rnnt_decoding.strategy=greedy_batch|malsd_batch`, keys under
      `rnnt_decoding.greedy.boosting_tree.*` / `.beam.boosting_tree.*`, `boosting_tree_alpha=<weight>`.
    - Shared knobs: `context_score=1.0` recommended; `depth_scaling=2.0` for CTC/RNN-T/TDT (`1.0` for Canary);
      `boosting_tree_alpha` is the overall weight (`0` disables); per-phrase weight via
      `key_phrase_items_list='[{phrase:"x",alpha:2.0},...]'`; `bpe_mode=case_insensitive` avoids manual
      capitalization on models that output case. Score with
      `scripts/asr_context_biasing/compute_key_words_fscore.py --input_manifest=<decoded> --key_words_file=<list>`.
    - **CTC-only alternative trails** (also the fallback when a checkout lacks GPU-PB — `boosting_tree` was added
      to NeMo mid-2025; verify
      `rnnt_decoding.*.boosting_tree`/`ctc_decoding.*.boosting_tree` exists in the target checkout first, e.g.
      `grep boosting_tree nemo/collections/asr/parts/submodules/rnnt_decoding.py`): CTC-WS
      (`scripts/asr_context_biasing/eval_greedy_decoding_with_context_biasing.py decoder_type=ctc`; for RNNT this
      only works via a **Hybrid RNNT/CTC** checkpoint's CTC head, `decoder_type=rnnt`) or Flashlight-based boosting
      (CTC lexicon-decoding only, a `word<TAB>score` file, ±20–100 range — this is the same convention Riva itself
      uses below, not GPU-PB's `context_score`/`boosting_tree_alpha`).
    - **Do not reuse these parameters in Riva.** `context_score`/`depth_scaling`/`boosting_tree_alpha` is a
      graph-based shallow-fusion system with its own score semantics — it does not map onto Riva's
      `boosted_lm_score` below. The NeMo pilot tells you *whether* boosting helps and which phrases matter; it does
      not give you a Riva score to reuse.
  - **Deploy (Riva, runtime)** — owner `nemotron-speech`. Parakeet CTC/RNNT/TDT and Nemotron ASR Streaming. Scores
    ~20–100 (CTC), 0.5–2.0 (RNNT/TDT); per-stream ~500 words (RNNT/TDT) / 5,000+ (CTC); global (deploy) 5,000+.
    Applied per-request via the client (`RecognitionConfig`/`speech_contexts`/`boosted_lm_words`), not a
    `riva-build` flag — no build step, and the deployed model doesn't need to have gone through the NeMo pilot at
    all. Tutorial: <https://github.com/nvidia-riva/tutorials/blob/stable/asr-wordboosting.ipynb>.
- **Custom vocab / pronunciation / speech hints:** deploy-time `riva-build`, primarily Parakeet CTC.
- **N-gram (KenLM) LM — two realizations; Riva carry-over differs by architecture, not a blanket rule.** Text-only;
  can't add acoustic capability. Decide which realization the goal needs before building:
  - **Pilot (NeMo, offline, NGPU-LM)** — owner `nemo-speech-asr-finetune`. GPU-accelerated shallow fusion at decode
    time, no retraining. This whole pilot runs entirely inside NeMo; nothing here requires Riva.
    - **Build (same script, both architectures):** `scripts/asr_language_modeling/ngram_lm/train_kenlm.py
      nemo_model_file=<.nemo> train_paths=<text/manifests> kenlm_bin_path=<kenlm/bin> kenlm_model_file=<out>
      ngram_length=6 save_nemo=True` (6 recommended for BPE models; `save_nemo=True` emits the `.nemo`-format
      artifact — this exact file is what Riva reuses directly for RNNT, see Deploy below).
    - **CTC** — via `examples/asr/speech_to_text_eval.py`: `ctc_decoding.greedy.ngram_lm_model=<.nemo/.ARPA>
      ctc_decoding.greedy.ngram_lm_alpha=<weight> ctc_decoding.strategy=greedy_batch` (greedy), or
      `.beam.ngram_lm_model=...` + `beam_size=...` + `strategy=beam_batch` (beam;
      `final_score = acoustic + alpha*lm_score + beam_beta*seq_length`). For a beam-width/alpha/beta grid search
      instead of one run, `eval_beamsearch_ngram_ctc.py` is the tool for that specific job — CTC-only.
    - **RNN-T/TDT** — same script: `rnnt_decoding.greedy.ngram_lm_model=... rnnt_decoding.greedy.ngram_lm_alpha=...
      rnnt_decoding.strategy=greedy_batch` (greedy), or `.beam.ngram_lm_model=...` + `beam_size=...` +
      `strategy=malsd_batch` (beam; `pruning_mode=late` + `blank_lm_score_mode=lm_weighted_full` recommended;
      `final_score = acoustic + alpha*lm_score`, normalized by sequence length). No RNNT equivalent of
      `eval_beamsearch_ngram_ctc.py` exists — `speech_to_text_eval.py` covers both greedy and beam directly.
    - **CTC-only alternative: NeMo's own Flashlight decoder** — a second, older NeMo-side mechanism, distinct from
      NGPU-LM and not GPU-accelerated the same way. Set `decoding_cfg.strategy="flashlight"`,
      `decoding_cfg.beam.kenlm_path=<word-level .bin from lmplz>`,
      `decoding_cfg.beam.flashlight_cfg.lexicon_path=<lexicon>` (built with
      `scripts/asr_language_modeling/ngram_lm/create_lexicon_from_arpa.py`), tune `beam_alpha`/`beam_beta`/
      `beam_size`. This is the same Flashlight integration word boosting's CTC fallback uses
      (`flashlight_cfg.boost_path`) — the two can be combined in one beam search. Confirmed no RNNT path exists for
      this mechanism (zero Flashlight references anywhere in NeMo's RNNT/transducer decoding modules).
    - Tutorials (source of truth — neither sub-skill documents this): **CTC**
      <https://github.com/nvidia-riva/tutorials/blob/stable/asr-python-advanced-nemo-ngram-training-and-finetuning.ipynb>
      — NeMo-only end to end, uses the Flashlight path above, **no Riva deploy step in it at all**. **RNN-T (NGPU-LM)**
      <https://github.com/nvidia-riva/tutorials/blob/main/asr-train-and-deploy-NGPU-LM-for-parakeet-rnnt.ipynb> —
      covers both the NeMo pilot above and the Riva deploy step below, verified by reading its actual commands.
  - **Deploy (Riva, decode-time)** — owner `nemotron-speech`; exact `riva-build` flags and LM file formats are
    documented there (`references/pipelines.md`'s Language Models section) — do not duplicate them here. **What
    matters at this layer is whether the NeMo pilot artifact carries over, and that differs by architecture:**
    - **CTC — the NGPU-LM pilot artifact definitely needs a rebuild; the NeMo-Flashlight pilot's artifacts are an
      open question, not a confirmed rebuild.** The NGPU-LM pilot's `.nemo`/`.ARPA` is subword-tokenized — a
      certain mismatch with Riva's word-level Flashlight decoder, so that one always needs a fresh word-level LM
      built for Riva from the domain corpus. Whether the NeMo-Flashlight pilot's own word-level KenLM + `.lexicon`
      (from `create_lexicon_from_arpa.py`) can be handed to `nemotron-speech` as-is is **unverified** — checked
      against both `nemotron-speech`'s own docs and the live NVIDIA docs, neither was specific enough to confirm
      either way. Test directly before assuming a rebuild is needed.
    - **RNN-T/TDT — no rebuild; reuse the pilot artifact directly.** Verified against NVIDIA's own RNNT tutorial by
      reading its actual `riva-build` command: the exact `.nemo` NGPU-LM artifact the pilot produced
      (`train_kenlm.py`'s `save_nemo=True` output) is handed to `nemotron-speech` unmodified. RNNT uses a
      genuinely different Riva decoder path from CTC — it never goes through Flashlight either.
    - Deploy-side prerequisites (version-matched `nemo2riva`, `riva-build` CLI differences across NIM images) are
      owned by `nemotron-speech` — see [`sub-skills.md`](sub-skills.md).
- **Fine-tune:** for genuine acoustic gaps. NIM guide: 100+ h recommended; ~10 h floor **only if mixed** with a
  larger dataset to avoid catastrophic forgetting. Lossless audio, ≥16 kHz, noise augmentation. Supported: Parakeet
  CTC/RNNT/TDT and Nemotron ASR Streaming. Verify/convert sample rate and, on request, audit transcript quality via
  the pre-flight dataset quality check ([`workflow.md`](workflow.md) §4a) before training.
- **Below the 10 h floor — do not recommend fine-tuning.** Under ~10 hours of real transcribed target-domain audio,
  the risk is severe on two independent axes: **catastrophic forgetting** (general capability degrades) and
  **overfitting** (the model memorizes the tiny training set rather than generalizing, even within the target
  domain) — mixing with a larger dataset no longer offsets this the way it does at the 10 h floor. Recommend
  **word boosting** instead (the rung above — runtime, no training, needs only a word list). Only revisit
  fine-tuning once real transcribed target-domain audio actually reaches the ~10 h floor, or the user explicitly
  accepts the risk after being told it.
- **Train from scratch / cross-language:** 5,000+ h from scratch; ~16+ h with cross-language transfer. Prefer
  fine-tuning a multilingual checkpoint first.
- **Tokenizer extension to a new language:** when the target language requires new tokens (new script, phonemes, or
  characters not covered by the base tokenizer), extend the tokenizer before fine-tuning the acoustic model. Source of
  truth:
  <https://github.com/nvidia-riva/tutorials/blob/main/asr-extend-tokenizer-to-newlang-ft-acoustic-model.ipynb>.

## Escalation Rule

Start at the lowest matching rung, measure, and escalate only if the target is missed. It is fine to run a cheap-rung
experiment (boosting / LM) immediately while presenting the full fine-tuning plan and its cost — see
[`planning-answers.md`](planning-answers.md). Change one lever per iteration.
