# Evaluation inputs

These ten cases test workflow behavior in isolated evaluator sandboxes. They do
not replace independent qualification on customer assets.

- `files/factory/` is a synthetic USD stage in metres with repeated robots and
  clamps, shared materials and animated equipment. Each stage-based case explicitly
  stages this directory under `/workspace/input/factory/`. It contains no boundary
  decisions or expected optimized output.
- `files/completed-run/` supplies reporting inputs. Native USD Optimize 1.3.1
  triangulated an already triangulated panel with an authored extent. The before
  and after layers are identical; the receipt records that result and labels
  unmeasured performance fields. It is an intentional no-change fixture, not
  evidence of performance improvement.
- The sandbox has no OpenUSD or USD Optimize runtime. The prompts state this and
  forbid installation, so the cases test the decisions the skill owns around
  execution: runtime preflight, existing authorization and missing loss budgets,
  scoping of expensive checks, source protection, structural-only and no-change
  reporting, and routing. Optimization cases ask for a plan in
  `/workspace/output/plan.md`. Native execution is qualified separately.
- The viewer case tests routing only. No application implementation or USD input
  is required for that negative case.

The evaluator stages the same case inputs for the baseline and skill-assisted runs.
Expected answers and assertions stay in evaluator metadata, outside the agent's
input directory. All prompts are at most 50 words.

Validate the dataset with `skillevaluator tier3 validate <skill-directory> --json`.
Missing fixture files are test failures, not successful skill behavior. Retain
failed evaluations when correcting their setup; do not adjust scores, thresholds
or agent-produced outputs.
