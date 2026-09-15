# ASR Customization Orchestration Plan

## 1. Goal
- Domain / language:
- What's wrong (examples):
- Success metric + target WER:

## 2. Scope (discovery answers)
- Real transcribed audio (hours):
- Domain text available (for LM)? :
- Vendor/customer data + format:
- Eval set / general guardrail set:
- Latency (streaming/offline), hardware/serving budget, timeline:
- Current model / must stay same for deploy? :

## 3. Chosen path
- Rung (boosting / vocab-pronunciation / n-gram LM / fine-tune / from-scratch):
- Why this is the cheapest sufficient rung:
- Escalation plan if target missed:
- Real transcribed target-domain audio (hours) vs. the ~10 h fine-tune floor:
- If <10 h and fine-tune was still chosen: user told the catastrophic-forgetting/overfitting risk and explicitly
  accepted it (Y/N + rationale)? :

## 4. Data (SDG / Data sub-skill)
- Sub-skill(s) invoked:
- Synthetic vs real mix + weights:
- Noise profiling / vendor scoring:
- Missing real data flagged? :

## 4a. Pre-flight dataset quality check
- Audio sample-rate audit: rates found / files off 16 kHz:
- Conversion applied (tool + command)? :
- Native sample rate of source audio (flag if originally 8 kHz / telephony):
- Fed 8 kHz files as-is instead of offline-converting? (Y/N — if Y, user confirmed deliberately, accepting slower
  online-resampling throughput; `model.sample_rate` left at the checkpoint's expected rate, e.g. 16000):
- Training a genuinely 8 kHz-native model instead (`model.sample_rate=8000`, rare, distinct from the above)? (Y/N +
  rationale if Y):
- Transcript WER audit run? (Y/N, reason if skipped):
- Reference model used for the audit:
- Flagged-utterance rate + disposition (dropped / re-transcribed / kept after review):

## 4b. Pre-flight environment / dependency check
- NeMo source: provided (checkout/install/container) vs. pulled fresh (none was provided/found):
- NeMo version / commit resolved:
- If pulled fresh: source repo URL used (verify still current — `NVIDIA-NeMo/Speech` as of this writing) + tag/branch:
- If provided: user asked whether to switch to latest `main` instead (Y/N)? User's decision + reason:
- If user insisted on the provided checkout: version issue hit later that justified revisiting (e.g. unsupported
  functionality)? What, and what was done:
- `nemo-speech-asr-finetune` made reachable how: `npx skills add` / `scripts/link-nemo-subskill.sh` / already linked:
- Working context changed into the checkout before Stage 5 (cd / scoped sub-agent)? :
- torch version / `torch.cuda.is_available()`:
- Known version-sensitive gotchas flagged (e.g. PyTorch 2.6+ `weights_only=True` breaking checkpoint averaging):

## 5. Train (Research/Training sub-skill)
- Sub-skill invoked (nemo-speech-asr-finetune):
- Checkpoint / family:
- Recipe notes (config, replay/curriculum, GPU/OOM preflight):
- Output checkpoint(s):

## 6. Evaluate (Evaluation sub-skill)
| Variant | Domain WER (normalized) | General WER (forgetting) | Notes |
| --- | --- | --- | --- |
| baseline | | | |
| chosen rung | | | |

- Worst error categories / next lever:

## 7. Loop or ship
- Met target? loop or ship:
- Checkpoint selected/averaged:
- User consulted before extra cycle? :

## 8. Deploy (Deployment/Optimization sub-skill)
- Handoff to nemotron-speech (export/serve):
- NIM-build optimization (e.g. quantization) needed? (handled in nemotron-speech):

## Planning answers given
- Data volume / hours-to-target basis:
- Synthetic vs real rationale:
- Cost / GPU choice (L40S vs H100):

## Placeholders encountered
- Sub-skills not yet available + interim guidance used:
