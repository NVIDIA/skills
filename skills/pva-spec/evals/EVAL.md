# CI smoke evaluations

Covered: Handoff finalization of an approved contract, rejection of unresolved test coverage, missing prerequisites, and routing.

Not covered: Writing a complete operator contract from raw requirements with real SDK references. The successful fixture exercises only finalization.

## Behaviors

Check case assertions against the result and actions. A correct refusal passes
a negative-path case; it does not prove operator generation or hardware correctness.
The task environment supplies Python 3.9+ only. PVA SDK, PVA Solutions,
NVVPU/ASIP, CUDA, EMU, and a board are absent. No case may install them or
contact a remote host. SDK version/generation mentions are requested target
metadata, not evidence of an installed SDK. Fixture logs are supplied text;
they do not require running an emulator.

## Negative Cases

The last case must not activate this skill. Other cases expect this skill, even
when the correct result is BLOCKED, NOT_READY, or NEEDS_INPUT.

## Notes

Validate from this skill directory: `skillevaluator tier3 validate . --strict`.
Run Claude Code and Codex, with and without the skill, using identical inputs.
Each case explicitly selects fixtures under `evals/files/`; the evaluator stages
these under `/workspace/input/`. Dataset answers and this guidance stay private.
CI version 1.5.6 requires nonempty assertions. See
[benchmark results](../BENCHMARK.md) for measured results. These CI smoke tests
do not cover SDK builds, EMU, or hardware execution.
