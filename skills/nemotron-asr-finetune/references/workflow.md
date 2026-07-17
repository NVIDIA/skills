# Orchestration Workflow

The end-to-end loop for "improve/fine-tune ASR for my domain/language." Each stage says who runs it and which
sub-skill it invokes. Announce each step to the user (**Step N/8 — Title**). Sub-skill details are in
[`sub-skills.md`](sub-skills.md); path choice is in [`path-selection.md`](path-selection.md); cost/data answers are in
[`planning-answers.md`](planning-answers.md).

## 1. State the goal — *Orchestration*

Capture, in the user's words: the domain or target language, concrete examples of what's wrong (e.g. "German medical
jargon wrong", "add Farsi", "noisy call-center audio, improve WER"), and the metric that defines success.

## 2. Clarify & scope — *Orchestration*

Ask the discovery questions before recommending anything:

- **Data:** how many hours of *real, transcribed* target-domain audio exist? Is there domain *text* (for an LM)? Any
  vendor/customer data, and in what format?
- **Target:** what eval set and WER target define "done"? Is there a general set to guard against regressions?
- **Constraints:** latency (streaming vs offline), hardware/serving budget, deployment target (NIM/HF), timeline.
- **Model:** current model/language, and whether it must stay the same for deployment.

Record answers; they drive both the path choice and the cost/time/data answers.

## 3. Choose the path — *Orchestration → Research/Training*

Pick the **cheapest sufficient rung** from [`path-selection.md`](path-selection.md): word boosting → custom
vocab/pronunciation / n-gram LM → fine-tune → (last resort) train from scratch. Escalate the tuning
method only if quality is short of target. You may **run a cheap-rung experiment now while presenting the full
fine-tuning plan** so the user sees the trade-off. State which sub-skills the chosen path will use.

## 3b. Branch by rung — *Orchestration*

**Stages 4–8 below are the fine-tune–shaped loop** (`data → NeMo train → NeMo eval → Riva deploy`). Do **not** force a
cheaper rung through it — each rung has its own shorter shape, owned end-to-end by **one** sub-skill. Pick the branch,
then announce steps for *that* branch (the "Step N/8" count only applies to the full fine-tune path):

| Chosen rung | Flow | Owner(s) | Eval surface |
|---|---|---|---|
| **Word boosting** | runtime word list → serve (no build) | `nemotron-speech` | served-endpoint WER |
| **Custom vocab / pronunciation** | `riva-build` vocab/lexicon → serve | `nemotron-speech` | served-endpoint WER |
| **N-gram LM — pilot (NeMo)** | build KenLM → `beam_batch` greedy-vs-LM eval; **no deploy** | `nemo-speech-asr-finetune` | offline file WER |
| **N-gram LM — deploy (Riva)** | build word-level KenLM+vocab → `riva-build` flashlight → serve | `nemotron-speech` | served-endpoint WER |
| **Fine-tune / from-scratch** | full Stages 4–8 | Data → `nemo-speech-asr-finetune` → `nemotron-speech` | offline WER (train) **and** served WER (post-deploy) |

Key consequences: the **n-gram LM does not follow `train(NeMo) → eval(NeMo) → deploy(Riva)`** — its pilot and deploy
realizations are different artifacts (see [`path-selection.md`](path-selection.md)); if the user wants to ship, route
straight to `nemotron-speech` and build the Riva-format LM there, optionally after a NeMo pilot. Anything **served** is
evaluated on the running NIM (**served-endpoint WER**, owned by `nemotron-speech`), not via the offline NeMo eval.

## 4. Get the data right — *SDG / Data*

Only when the path needs training data and it is scarce or noisy. Delegate to the data sub-skill(s):

- Generate **synthetic text** (domain terms, target-style transcripts, TTS-friendly formatting) with `data-designer`.
- If synthetic **audio** is genuinely needed, synthesize it with a TTS model — `nemotron-speech` can run TTS inference
  (e.g. Magpie). Scope whether audio synthesis is worth it before committing; text-only rungs (LM) may suffice.
- **Profile and harvest in-domain noise**; blend it in at realistic levels.
- **Score vendor/customer samples** and align their format (sample rate, channels, transcript style) to training needs.
- Explore existing customer data; **flag when real target-domain data is missing** and recommend collecting it.
- Keep synthetic/augmented sources separately weighted so they can be ablated.

Invoke: `data-designer` (synthetic **text**, not audio), `nemotron-speech` (TTS audio inference if needed), placeholder
`asr-data-profiling` (noise profiling / vendor-data impact / manifest assembly). Hand the assembled, style-audited
manifests to Stage 5.

## 5. Train — *Research / Training*

Delegate execution to `nemo-speech-asr-finetune`: apply the recipe (checkpoint choice, configs, hyperparameters,
replay/curriculum to limit forgetting, GPU/OOM preflight) and run. The orchestrator supplies the path
decision, data, and constraints; the sub-skill owns the flags/config. Key facts the orchestrator should carry:

- Prefer the generic `speech_to_text_finetune.py`; it covers CTC/RNNT/TDT/hybrid/AED and both cache-aware streaming
  families (plain English vs prompted multilingual — different architectures).
- Data gate: NIM guide recommends 100+ h; ~10 h is a floor only when mixed with a larger set to avoid catastrophic
  forgetting.

## 6. Evaluate — *Evaluation*

Delegate to `nemo-speech-asr-finetune` and use its evaluation stage:

- **Normalized WER** on the domain set (default: lowercased, punctuation removed) — the success metric.
- **A/B forgetting check** on a general set to catch regressions from adaptation.
- **Error-driven analysis** (numbers, entities, jargon, accents, noise, long audio) to identify the next lever.
- For streaming models, evaluate with the cache-aware streaming inference path at a trained `att_context_size`.

**Two eval surfaces — use the one that matches the branch (see [`sub-skills.md`](sub-skills.md)):** offline file WER on
a `.nemo`/`.riva` (`speech_to_text_eval.py` / `beam_batch`) is owned by `nemo-speech-asr-finetune`; **served-endpoint
WER** (a client scoring the running NIM) is owned by `nemotron-speech`. For any served rung (boosting, custom vocab,
n-gram LM deploy, or a fine-tune after deploy), measure on the **served endpoint** — in-NeMo numbers can differ from what
the deployed decoder actually produces.

## 7. Loop or ship — *Orchestration*

Compare against the target. If short: loop to Stage 4/5 with **targeted** data addressing the worst error categories —
change one lever at a time. If met: select/average the best checkpoints (keep the averaged model only if it wins). Do
not train on the eval set; keep a blind holdout for final claims. **Consult the user before starting another tuning
cycle.**

## 8. Deploy — *Deployment / Optimization*

Delegate to `nemotron-speech`: export to NIM/HF, hot-swap the checkpoint, serve, and apply any NIM build-time
optimization there if the latency/HW budget requires it. Re-run the runtime pipeline checks after swap. (Do not route
this to `nemotron-customize` — that is Nemotron LLM customization, not speech.)

## Answer along the way — *Orchestration*

At any stage, expect and answer planning questions: how much data was/will be used, synthetic vs real mix, hours needed
to hit an X% WER target, cost to run, and whether L40S or H100 is more efficient. See
[`planning-answers.md`](planning-answers.md).
