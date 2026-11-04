# Evaluation guidance

## Catalog CI smoke suite

The publication-facing `evals.json` contains three short P0 cases:

- `nva-ambient-explicit-welcome` checks explicit activation, the exact welcome, and the no-action boundary.
- `nva-ambient-scenario-disclosure` checks the three workflow choices, endpoint disclosure, fictional-data disclosure, and the wait-for-selection boundary.
- `nva-ambient-public-endpoint-env-disclosure` checks that runtime choices include the public default and give the user the checkout `.env` path and `NVIDIA_API_KEY` instruction in the same message.
- `nva-ambient-negative-ordinary-deploy` checks that ordinary NVA deployment guidance does not activate this healthcare customization skill.

This set covers output quality and positive/negative triggering and requires no staged input files. It is the only dataset intended for the NVCARPS per-PR evaluation gate and must remain within the one-hour runner limit.

## Harnesses and model selection

Evaluate every selected case with and without the skill on both harnesses:

| Harness | Agent model selection |
|---|---|
| Codex | SkillEvaluator-maintained default |
| Claude Code | SkillEvaluator-maintained default |

Do not pin harness models in `evals/config.yml`. The publication CI selects supported defaults for its current environment.

Use Docker isolation for both harnesses and keep provider credentials only in the evaluator process environment. Set `SKILL_EVAL_JUDGE_MODEL=openai/gpt-oss-20b` for text judging.

Use `default_plus_custom` grading for the CI smoke suite.

## Metrics

Collect SkillEvaluator's security, correctness, discoverability, effectiveness, and efficiency dimensions. Retain the underlying `skill_execution`, `skill_efficiency`, `accuracy`, `goal_accuracy`, `behavior_check`, and `nva_workflow_contract` results.

## Acceptance

- `skillevaluator tier3 validate --strict --harbor-contract` passes.
- Every expected Codex and Claude Code with-skill and baseline arm is scored.
- The CI smoke suite finishes within one hour on the publication runner.
- Positive cases satisfy their required behavior and negative cases do not activate the skill.
- No credential, secret, or unapproved healthcare record appears in trajectories or reports.
- `BENCHMARK.md` names the harnesses, model versions, tasks, metrics, with-skill and baseline results, token use, wall-clock time, and limitations.

Keep `BENCHMARK.md` limited to the current supported result. Do not include execution identifiers, local report paths, chronological logs, or superseded experiments.
