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

## 4. Data (SDG / Data sub-skill)
- Sub-skill(s) invoked:
- Synthetic vs real mix + weights:
- Noise profiling / vendor scoring:
- Missing real data flagged? :

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
