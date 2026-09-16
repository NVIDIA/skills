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

**Hard guardrail on the Stage 2 data answer:** if real transcribed target-domain audio is **below ~10 hours**, do not
present fine-tuning as the plan — the catastrophic-forgetting and overfitting risk is severe at that volume (see
[`path-selection.md`](path-selection.md)). Recommend word boosting instead. Only revisit fine-tuning once more real
data is available, or if the user explicitly accepts the risk after being told it.

## 3b. Branch by rung — *Orchestration*

**Stages 4–8 below are the fine-tune–shaped loop** (`data → NeMo train → NeMo eval → Riva deploy`). Do **not** force a
cheaper rung through it — each rung has its own shorter shape, owned end-to-end by **one** sub-skill. Pick the branch,
then announce steps for *that* branch (the "Step N/8" count only applies to the full fine-tune path):

| Chosen rung | Flow | Owner(s) | Eval surface |
|---|---|---|---|
| **Word boosting — pilot (NeMo)** | GPU-PB `boosting_tree` decode-time eval; **no deploy** | `nemo-speech-asr-finetune` | offline file WER |
| **Word boosting — deploy (Riva)** | runtime word list → serve (no build) | `nemotron-speech` | served-endpoint WER |
| **Custom vocab / pronunciation** | `riva-build` vocab/lexicon → serve | `nemotron-speech` | served-endpoint WER |
| **N-gram LM — pilot (NeMo)** | build KenLM → NGPU-LM shallow-fusion eval; **no deploy** | `nemo-speech-asr-finetune` | offline file WER |
| **N-gram LM — deploy (Riva)** | CTC: rebuild word-level LM → serve. RNNT: reuse pilot `.nemo` directly → serve (exact flags in `pipelines.md`) | `nemotron-speech` | served-endpoint WER |
| **Fine-tune / from-scratch** | full Stages 4–8 | Data → `nemo-speech-asr-finetune` → `nemotron-speech` | offline WER (train) **and** served WER (post-deploy) |

Key consequences: **neither word boosting nor the n-gram LM follows `train(NeMo) → eval(NeMo) → deploy(Riva)`** —
each has pilot and deploy realizations, and whether the pilot artifact carries into Riva unchanged is **not a
blanket rule** (see [`path-selection.md`](path-selection.md)): word boosting never carries over (different score
systems entirely); the n-gram LM depends on architecture — CTC needs a rebuild, RNNT reuses the pilot `.nemo`
verbatim. A customer working a NeMo pilot only never needs to touch Riva — it is a complete, self-contained loop. If
the user wants to ship, route straight to `nemotron-speech` and build (or reuse, for RNNT LM) the Riva-side artifact
there, optionally after a NeMo pilot. Anything **served** is evaluated on the running NIM (**served-endpoint WER**,
owned by `nemotron-speech`), not via the offline NeMo eval.

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
`asr-data-profiling` (noise profiling / vendor-data impact / manifest assembly). Hand the assembled manifests to the
pre-flight dataset quality check (§4a) before Stage 5.

## 4a. Pre-flight dataset quality check — *Orchestration → SDG/Data / Evaluation*

A safeguard gate between data assembly and training. Two independent checks; run whichever the user asks for.
Recommend both whenever new (unaudited) data is about to be trained on for the first time, but only execute a check
once the user confirms — the text-side check spends GPU/API budget on inference. Record results in the run ledger
([`../assets/experiment-ledger-template.md`](../assets/experiment-ledger-template.md), §4a).

**Audio — sample rate.** 16 kHz is the default target — the fine-tune rung's floor (see
[`path-selection.md`](path-selection.md)) and `nemo-speech-asr-finetune`'s own default ("prefer 16 kHz mono audio
unless the model card/config says otherwise"). Confirm the target rate against the chosen checkpoint's model
card/config before assuming 16 kHz — a small number of checkpoints are configured for a different rate. Probe every
file's sample rate (`soxi -r file.wav`, `ffprobe -show_streams`, or read it while building the manifest) and bucket by
rate. Any file that does not match the target rate must be **resampled offline before training** — do not rely on
online resampling inside the training data loader; it is slow at scale, and NeMo's own telephony-audio guidance is to
convert once, offline:

```bash
sox in.wav -r 16000 out_16k.wav
# or
ffmpeg -i in.wav -ar 16000 out_16k.wav
```

(Substitute the checkpoint's actual target rate for `16000` if it differs.) This applies in both directions — upsample
8 kHz telephony audio and downsample 22.05/44.1/48 kHz audio, all to the target rate. Upsampling does not recover
bandwidth the source never had: if the underlying data is native 8 kHz (e.g. telephony), note that provenance in the
ledger — a model trained on upsampled-8kHz audio has a different acoustic gap profile than one trained on
native-16kHz wideband audio, which matters for the Stage 6 error-driven analysis. Re-verify sample rates on the
converted files; do not hand an unresampled manifest to Stage 5.

**If the user insists on feeding the 8 kHz files as-is** (declining the offline conversion step): this is **not
recommended**, but not for an accuracy reason — the fix is not to change `model.sample_rate`. Do not set
`model.sample_rate=8000` to "match" the raw files; that reconfigures the preprocessor's mel-spectrogram frontend into
a genuinely different, 8 kHz-native architecture than the pretrained checkpoint's — a materially different model than
the one being fine-tuned, not a convenience shortcut. Instead:

- **Leave `model.sample_rate` at the value the target checkpoint actually expects — typically `16000` — unchanged.**
  This is "the model's expectation" the checkpoint was built for; do not silently apply the default 16 kHz *audio*
  conversion, but do keep the 16 kHz *config* target.
- NeMo's audio loader resamples every sample to `model.sample_rate` at load time, so leaving the files at 8 kHz still
  produces 16 kHz features — just resampled **online, per sample, every epoch** instead of once offline. This is what
  makes it "not recommended": per NeMo's own telephony guidance, online upsampling during training "is very time
  consuming and may slow down training significantly," even though the end result is the same model.
- Confirm with the user that the trade-off being accepted is **training throughput, not model quality** — ask them to
  confirm before proceeding, and record the decision in the ledger. If training throughput later becomes the
  bottleneck, that's the signal to fall back to the offline `sox`/`ffmpeg` conversion above rather than continuing on
  the slow path — not a signal to switch to `model.sample_rate=8000`.
- Reserve `model.sample_rate=8000` (a genuinely 8 kHz-native model, still worth initializing from a 16 kHz checkpoint
  via `+init_from_pretrained_model` per NeMo's guidance) for the distinct, rarer case where the user explicitly wants a
  model that expects 8 kHz end-to-end, including at deployment/serving — not as a response to "I don't want to
  upsample."

**Text — transcript quality.** Only on user request. This is a different check from `nemo-speech-asr-finetune`'s own
transcript-style preflight (casing/punctuation/ITN/symbol **consistency**, run inside that sub-skill once it has the
data) — this one audits whether the transcript **words are correct**, before the data is even handed off. It does not
evaluate the model being fine-tuned; it sanity-checks whether the *provided ground truth* can be trusted before a
training run is spent on it. Pull a strong
off-the-shelf pretrained checkpoint (not the one being fine-tuned — a clean reference), run inference on a sample of
(or the whole) manifest, and compute per-utterance normalized WER (same contract as Stage 6: lowercased, punctuation
removed) between the reference model's hypothesis and the provided transcript:

- **Cheapest path:** cloud-hosted inference via `nemotron-speech` (build.nvidia.com, no GPU needed) with a
  matching-language Parakeet/Canary checkpoint — transcribe each file and diff against the manifest transcript.
- **If a local GPU/checkpoint is already in play:** offline batch inference + WER via `nemo-speech-asr-finetune`'s
  evaluation stage (`speech_to_text_eval.py`) using a stock pretrained `.nemo`, not the target fine-tune checkpoint.
- There is no universal WER cutoff for "bad transcript" — look at the **distribution** across the sample and flag the
  upper tail (utterances far worse than the bulk, or worse than the reference model's typical WER on a comparable
  public benchmark for that language). A tail spike usually means bad transcription, misalignment, wrong
  language/channel, or truncated audio — not necessarily model error.
- Do not silently drop flagged utterances. Report the count/rate to the user and let them decide (drop, re-transcribe,
  or keep after a manual spot-check shows the reference model — not the transcript — was wrong). A high flagged rate
  (e.g. a clear double-digit percentage of the sample) is itself a finding: it means the vendor/customer data needs a
  transcription-quality fix, not just a training-recipe fix, and should be raised before Stage 5 rather than
  discovered after a wasted training run.

Do not proceed to Stage 5 with unresolved sample-rate mismatches. The transcript check is advisory — the user decides
what to do with flagged utterances — but its results must be surfaced, not skipped, whenever it was requested.

## 4b. Pre-flight environment / dependency check — *Orchestration → Research/Training*

A safeguard gate alongside §4a, run before Stage 5. Confirms the training environment is on a known, checked NeMo
version rather than an unverified or missing one — `nemo-speech-asr-finetune`'s own docs repeatedly flag
version-sensitive breakage (e.g. the deprecated checkpoint-averaging utility fails on PyTorch 2.6+ because
`torch.load` defaults to `weights_only=True`; some checkpoint-averaging scripts are marked deprecated and need
verifying against the current checkout), so this is a real, not hypothetical, risk.

**NeMo:**

- **If no NeMo is provided or found** (no checkout path given, `import nemo` fails, no container tag specified):
  **pull the latest `main`** — mandatory, there is nothing to check otherwise.
  - **Local/source execution:** the canonical repo is `https://github.com/NVIDIA-NeMo/Speech.git` — NeMo's
    ASR/TTS/speech functionality moved there in a 2026 repository split from the original `NVIDIA-NeMo/NeMo`; verify
    this is still current (check the repo's own README "Updates" section) before treating it as permanent, since
    org/repo names can move again.
    ```bash
    git clone https://github.com/NVIDIA-NeMo/Speech.git
    cd Speech && uv sync --extra all --extra cu13   # or --extra cu12 for CUDA 12.x
    ```
  - **Container execution** (`nemo-speech-asr-finetune`'s own default posture): pull the latest published NGC
    container tag rather than hardcoding one — fetch the current tag from the NGC catalog/support docs, the same
    "don't hardcode, verify current" pattern used throughout this skill set for container tags and function IDs.
  - Either way, this gate's job is only to make sure *some* version is actually resolved and recorded before Stage 5,
    not to reimplement `nemo-speech-asr-finetune`'s Stage 1 (`setup-checkpoints.md`), which owns the exact
    setup/checkpoint-selection commands.
- **If a NeMo checkout/install is already provided** (a mounted source checkout, an installed `nemo-toolkit` package,
  or a pinned container tag) — check its version first, don't assume it's current:
  ```bash
  python -c "import nemo; print(nemo.__version__)"
  git -C <checkout_path> log -1 --format='%H %ci'   # if it's a source checkout: exact commit + date
  git -C <checkout_path> branch --show-current
  ```
  **Still recommend pulling latest `main` instead, and ask the user before proceeding** — a provided checkout always
  carries staleness risk: it may predate a feature the chosen technique needs. This is not hypothetical — e.g. NeMo's
  GPU-PB word boosting (`rnnt_decoding.*.boosting_tree`) was added after mid-2025, so a checkout from before then
  silently lacks it with no error until you try to use it. Do not silently proceed on the provided checkout, and do
  not silently switch to latest main either — ask, and follow what the user decides:
  - **User accepts switching to latest main:** pull it as in the "not provided" branch above.
  - **User insists on the provided checkout:** proceed with it as-is — do not pull preemptively. Record the
    version/commit in the ledger and continue to Stage 5. Only recommend pulling latest `main` again if a concrete
    version-specific issue is actually hit later (a config key doesn't exist, a script errors on an unsupported
    feature, a needed capability turns out to be missing) — treat that as new evidence justifying the recommendation,
    not as overriding the user's earlier choice.

**Make the sub-skill reachable, then operate from inside it.** Resolving a NeMo checkout (above) is necessary but not
sufficient — `nemo-speech-asr-finetune` is a *project-local* skill (lives in that repo's own `.claude/skills/`), not a
catalog entry, so it only activates when Claude Code's working context is actually inside that checkout:

1. Make it discoverable: try `npx skills add NVIDIA-NeMo/Speech --skill nemo-speech-asr-finetune --agent claude-code
   --global --yes` first; if that doesn't work, run the bundled
   [`../scripts/link-nemo-subskill.sh`](../scripts/link-nemo-subskill.sh) `<checkout_path>` (validates the source,
   won't clobber a real directory, records the real repo root).
2. Before Stage 5 runs any of its commands, **change the actual working context into that checkout** — `cd` there
   directly, or spawn a sub-agent/session scoped to that path — since every command in that sub-skill
   (`examples/asr/speech_to_text_finetune.py`, `scripts/checkpoint_averaging/checkpoint_averaging.py`, etc.) is a path
   relative to the repo root, not to wherever the orchestrator happened to start. A symlink under `~/.claude/skills/`
   only makes the skill *discoverable*; it does not change what directory a command actually runs in.

Do not assume `nemo-speech-asr-finetune` is invocable just because it was named in Stage 5 — verify it is actually
reachable and that the working context is correct first.

**Other packages (lighter touch):** at minimum record `torch.__version__` and `torch.cuda.is_available()` (already
checked in `nemo-speech-asr-finetune`'s Stage 1) and flag known version-sensitive gotchas before they surface mid-run
— e.g. the PyTorch 2.6+/`weights_only=True` checkpoint-averaging failure above.

## 5. Train — *Research / Training*

Before starting, confirm the pre-flight checks (§4a data quality, §4b environment/dependency) have passed or were
explicitly waived by the user. Delegate execution to `nemo-speech-asr-finetune`: apply the recipe (checkpoint choice, configs, hyperparameters,
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
