# Evaluating physical-ai-physics-selection

`evals.json` adapts Neelakantan Mani's original cases to the NVSkills dataset
contract. Run paired with/without-skill evaluations on Codex and Claude Code
with the same tools, runtime, and input evidence. Use skill-load traces for
routing checks and grade observable results, not matching prose.

Retain the user's engine, runtime, and physical requirements. Distinguish
metadata/source checks from executed simulation and measured improvement.
If a runtime or input asset is unavailable, report that limitation rather than
inventing a passing result. Negative cases must not load this skill.

Helper unit tests use synthetic inputs to check metadata handling; they do not
validate simulation behavior or physics fidelity.
SkillEvaluator performs the separate live-agent evaluation. Record its actual
runtime and results; do not author `BENCHMARK.md`, `skill-card.md`, or
`skill.oms.sig` by hand. CI generates those artifacts after validation.
