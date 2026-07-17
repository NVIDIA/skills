# Path Selection — Cheapest Sufficient Rung

Stage 3 of the workflow. Pick the **lowest-cost path that can meet the target**, escalate only when quality falls short,
and delegate execution to the sub-skills. Always diagnose the failure mode first and measure a baseline (Evaluation
sub-skill) so each step is justified.

Ordering and per-model support follow the NVIDIA Speech NIM (Riva) ASR customization guide:
<https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html>. Verify parameters and support there.

## The Ladder (cheapest → most expensive)

| Rung | Fixes | Needs | Cost | Runs where → sub-skill |
|---|---|---|---|---|
| **Word boosting** | Known words/names/jargon/commands | A word list | Minimal, no training | Runtime → `nemotron-speech` |
| **Custom vocabulary / pronunciation** | OOV words, consistent mispronunciations | Vocab/lexicon file | Low, deploy-time | `riva-build` → `nemotron-speech` |
| **N-gram LM — pilot (NeMo)** | Cheaply prove LM lift *offline* before deploying | Domain **text** + a `.nemo`/`.riva` | Low, no serving | `nemo-speech-asr-finetune` (build + `beam_batch` eval) |
| **N-gram LM — deploy (Riva)** | Domain phrasing / word sequences at serving | Domain **text** | Moderate, decode-time | `nemotron-speech` (word-level KenLM + vocab → `riva-build` flashlight → serve) |
| **Fine-tune** | Real acoustic gaps (accents, noise, channel) | 100+ h transcribed (10 h floor if mixed) + GPU | High | Research/Training (`nemo-speech-asr-finetune`) |
| **Train from scratch / cross-language** | A new language/dialect, no checkpoint | Thousands of h (16+ h for transfer) | Very high | Research/Training (last resort) |

Rungs compose: a fine-tuned model still uses boosting and an LM at serving time.

**One rung, one owner, end-to-end.** Route the *whole* lifecycle of a customization to a single sub-skill rather than
splitting its build → eval → deploy across skills. In particular the n-gram LM has **two realizations that are different
artifacts** — a NeMo **pilot** LM (owned by `nemo-speech-asr-finetune`) and a Riva **deploy** LM (owned by
`nemotron-speech`) — and the pilot artifact cannot be shipped to Riva unchanged (see the n-gram Rung Note). Pick the
realization up front from the goal (validate vs ship); do not hand a half-built LM across a boundary.

## Diagnose First

- **A short, known set of specific words wrong** (names, SKUs, commands, acronyms) → **word boosting**; if truly OOV or
  mispronounced → **custom vocabulary / pronunciation**.
- **Wrong word sequences / phrasing**, with domain text available → **n-gram LM**. Decide the *realization* up front:
  **pilot (NeMo)** to prove lift offline for cheap, or **deploy (Riva)** to ship it — different, non-interchangeable LM
  formats (see Rung Notes).
- **Model mis-hears audio** (accent, noise, channel) with a real acoustic gap and enough transcribed audio →
  **fine-tune**.
- **New language/dialect, no suitable checkpoint** → train from scratch / cross-language transfer (rare).
- **Formatting only** (punctuation, casing, numbers) → runtime automatic-punctuation / ITN flags, not accuracy work.

## Rung Notes (docs-grounded)

- **Word boosting:** Parakeet CTC/RNNT/TDT and Nemotron ASR Streaming. Scores ~20–100 (CTC), 0.5–2.0 (RNNT/TDT);
  per-stream ~500 words (RNNT/TDT) / 5,000+ (CTC); global (deploy) 5,000+. Cannot fix acoustics. Tutorial:
  <https://github.com/nvidia-riva/tutorials/blob/stable/asr-wordboosting.ipynb>.
- **Custom vocab / pronunciation / speech hints:** deploy-time `riva-build`, primarily Parakeet CTC.
- **N-gram (KenLM) LM — two realizations, one artifact each; NOT interchangeable.** Text-only; can't add acoustic
  capability. Decide which realization the goal needs before building:
  - **Pilot (NeMo, offline)** — owner `nemo-speech-asr-finetune`. Build a subword KenLM with `train_kenlm.py`, then
    score greedy-vs-LM with the GPU `beam_batch` decoder (`eval_beamsearch_ngram_ctc.py`). No Flashlight build, no
    serving. Use it to **prove lift cheaply** before committing to a deploy cycle.
  - **Deploy (Riva, decode-time)** — owner `nemotron-speech`. Build a **word-level** KenLM (`lmplz` on domain text) +
    a `decoding_vocab`, then `riva-build` with `decoder=flashlight` (`--decoding_language_model_binary` /
    `--decoding_vocab`; RNNT uses `--nemo_decoder.language_model_file`) and serve.
  - **Do not carry the pilot LM into Riva.** The NeMo subword KenLM is not the word-level format the Riva flashlight
    decoder consumes — rebuild for Riva from the same corpus. Deploy-side gotchas are owned by `nemotron-speech`:
    `.nemo`→`.riva` needs a **version-matched `nemo2riva`** (or start from a deployable `.riva`); `riva-build` CLIs
    differ across NIM images (classic vs Hydra — some reject `.nemo`); and `decoding_vocab` characters must all be in
    the model tokenizer's alphabet (digits/punctuation fail `riva-deploy` with `Out of vocab character`).
  - Tutorials: CTC
    <https://github.com/nvidia-riva/tutorials/blob/stable/asr-python-advanced-nemo-ngram-training-and-finetuning.ipynb>,
    RNNT (NGPU-LM) <https://github.com/nvidia-riva/tutorials/blob/main/asr-train-and-deploy-NGPU-LM-for-parakeet-rnnt.ipynb>.
- **Fine-tune:** for genuine acoustic gaps. NIM guide: 100+ h recommended; ~10 h floor **only if mixed** with a
  larger dataset to avoid catastrophic forgetting. Lossless audio, ≥16 kHz, noise augmentation. Supported: Parakeet
  CTC/RNNT/TDT and Nemotron ASR Streaming.
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
