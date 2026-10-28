# CI smoke evaluations

Check missing installation sources, release mismatch, platform scope, the
default PVA Solutions source plan, and negative triggering.
These cases need no SDK, emulator, board, or network.
They must not install software or contact remote machines.

Validate from this skill directory with `skillevaluator tier3 validate . --strict`.
Compare Claude Code and Codex with and without the skill using identical inputs.

This dataset does not verify package installation, compiler or EMU operation,
or target runtime compatibility. Test those on a development host and L4T board.
See [benchmark results](../BENCHMARK.md) for the recorded CI smoke-test results.
