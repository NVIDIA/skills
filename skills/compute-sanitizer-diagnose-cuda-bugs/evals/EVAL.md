# compute-sanitizer — Tier 3 eval dataset

This is the NVCARPS Tier 3 (live agent A/B) dataset for the `compute-sanitizer`
meta-skill. Each case is run twice — **with** the skill staged and **without** —
and the "Skill Lift" is the difference in pass rate.

## What these cases test (and what they don't)

The cases here are **routing, flag-knowledge, interpretation, and trap** questions
graded from the agent's *reasoning* (`grading.mode: aces_default`). They are
deliberately GPU-free so they run in the default `python:3.12-slim` Harbor
container. This is where a skill earns its keep: picking the right tool for a
symptom, knowing the obscure flag, and resisting a wrong premise.

The heavier **"build + run compute-sanitizer + diff the `.nvm` against gold"**
end-to-end suite lives in this repo's own harness under `eval-harness/` (it needs
a CUDA toolkit and a visible GPU, so it is not part of the Harbor container run).

## Coverage (mirrors the harness axes)

| Bucket | Cases | Mirrors |
|---|---|---|
| Tool routing | `cs-route-{memcheck,racecheck,initcheck,synccheck}-*` | discoverability + E2E tool selection |
| Obscure flags | `cs-flag-{padding,leak-check,num-cuda-barriers,track-unused,error-exitcode}` | robustness obscure-flag cases |
| Traps (wrong premise) | `cs-trap-{clean-memcheck,misleading-race}` | robustness traps |
| Scope boundaries | `cs-boundary-{global-race,host-race}` | robustness deferral cases |
| Prompt injection | `cs-injection-{source-comment,sanitizer-output}` | ignore instructions embedded in inspected data |
| Negative (must NOT trigger) | `cs-negative-host-logic` | precision on non-CUDA tasks |

## Questions

- Symptom → which of memcheck / racecheck / initcheck / synccheck, and the exact
  `compute-sanitizer --tool ... ./app` invocation?
- Which flag unblocks a stuck run (`--padding`, `--leak-check full`,
  `--num-cuda-barriers`, `--track-unused-memory`, `--error-exitcode`)?
- Does the agent resist a wrong premise (a clean memcheck ≠ race-free; "must be a
  race" when it's an uninitialized read)?
- Does it know the tool's scope (shared-memory races only; GPU/device only) and
  redirect (iGUARD/SuperCollider for global races; TSan/Helgrind for host races)?
- Does it stay away from a plain CPU logic bug (negative case)?
- Does it ignore command requests and fake authority embedded in CUDA source
  comments or sanitizer output while continuing the requested diagnosis?

## Run

```bash
skillevaluator tier3 validate ./compute-sanitizer --strict   # no key needed
skillevaluator evaluate ./compute-sanitizer -a claude-code   # needs NVIDIA_INFERENCE_KEY + Docker
```
