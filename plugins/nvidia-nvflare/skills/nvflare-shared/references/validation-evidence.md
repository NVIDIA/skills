# Validation Evidence Guidance

Use this reference before declaring a generated or exported NVFLARE job valid.

## Local Validation

Build local simulations with `SimEnv` from `nvflare.recipe`. Apply the canonical
single-topology-owner rule in `conversion-common.md`, then call
`recipe.execute(env)` from `job.py`.

Choose one final full-run path based on the artifact being validated:

- Local recipe or first-user simulation validation: run `python job.py`.
- Exported deployable job validation: create the job folder with `python job.py
  --export --export-dir <runtime-dir>/job_config`, then run that exported folder
  with the product simulator CLI, for example `nvflare simulator
  <exported-job-dir> -w <runtime-dir>/workspace -n <num_clients> -t
  <num_threads> -l concise` (or `-c site-1,site-2,...`).

Unless the user explicitly requests an exported/deployable job folder, select
the local path and do not export. `recipe.execute(SimEnv(...))` materializes the
job configuration needed for that simulation behind the scenes. Creating an
export does not authorize submitting it to POC or production; submission is
outside conversion-skill scope.

Use `SimEnv` only. Do not generate or run `PocEnv` or `ProdEnv` from a
conversion skill. Homomorphic-encryption recipes reject `SimEnv` and require
those provisioned environments, so do not offer HE as a conversion capability.
If another selected recipe rejects `SimEnv`, follow the selecting reference's
ask-or-fail-closed rule and report the job as unvalidated; do not switch recipes
or environments merely to force a run.

Do not accept generated job-local `--export` or `--export-dir` arguments, or an
alias such as `--export_only`, as replacements for the NVFLARE Recipe interface.
If generated code parses or manually branches on those system arguments or a
private alias, report it as a generated-code violation instead of treating the
export as valid.

Do not run both full simulations unless the first one failed and the second is a
scoped rerun after a fix. Do not write Python code to call simulator APIs for
exported-job validation.
- Prefer synthetic data flags or small fixtures when the original dataset is
  unavailable.
- Never run destructive cleanup against the source tree or its Git state. Do
  not use Git clean/reset/checkout operations or recursive deletion over source
  or user files. Scope cleanup to generated runtime or output directories under
  the runtime location convention.
- Keep validation commands single-purpose. Run dependency installation, cleanup,
  export, and simulation as separate commands.
- Treat an unavailable optional capability or host-diagnostic utility as
  evidence, not a failed validation command. Check tool availability first,
  keep optional diagnostics separate from required import, parser, partition,
  model, or recipe checks, and make the optional probe exit zero when the tool
  is absent. Do not append platform-specific utilities such as `free`, `sysctl`,
  or `nvidia-smi` to a required command where their absence changes its status.
- If validation cannot run, save the conversion as a draft and report the
  concrete blocker.
- After successful simulation, follow `metrics-and-artifact-reporting.md`.

## Terminal Simulation Evidence

Run the final simulation with `run_in_background: false`; never finalize with an
active simulation. This framework-neutral rule applies to every converter that
loads this reference. The final validation must finish in the same step before
you report success.

Prefer foreground execution for the final run. Do not depend on a background
task notifying you after your final answer: in a non-interactive run the task
ends with your final message, so a backgrounded run can be killed before it
finishes and before metrics are written. If you do background a run, you must
poll for the terminal artifact within the same turn, record the exit code, and
confirm completion before reporting success — never finalize on a still-running
or "will be notified" basis.

Required success evidence is process exit code 0, terminal FL evidence such as
the server log reaching a Finished state, and workflow-native metric evidence.
For FedAvg/FedOpt/FedProx/SCAFFOLD-style training workflows that install the
metrics artifact writer or emit aggregation events, this means server-side
artifacts such as `metrics_summary.json` or `round_metrics.jsonl`. For public
recipes whose contract does not emit those artifacts by default, collect the
native evidence instead: for FedEval, one-shot validation-result logs or
artifacts showing returned per-site metrics; for Cyclic or Swarm workflows,
their documented result logs, workflow artifacts, or other selected-recipe
metric outputs. When a run exits 0 and reaches a terminal finished state, a
missing expected metrics artifact is a validation failure only if the selected
recipe/workflow installs that writer or otherwise documents that artifact as an
output. A prose explanation can substitute for metrics only for blocked/timed-out
runs, explicitly metrics-free workflows, or metric-bearing workflows whose
selected public recipe exposes only non-artifact metric evidence; in the last
case cite the recipe and the exact native evidence used. Progress messages,
scheduled wakeups, "standing by"/"I'll wait" statements, and active processes
are not completion evidence and are not valid final answers.

Do not pipe the final validation command through `tail`, `grep`, or another
command that can hide the simulator or `python job.py` exit status. Redirect the
full log to a runtime log file and print a bounded tail only after the command
has finished and its exit code has been recorded. Do not append `; echo
"EXIT_CODE=$?"`, because the successful `echo` masks a failed simulation from
the calling tool. When a status line is required, preserve the status with
`rc=$?; echo "EXIT_CODE=$rc"; exit "$rc"`.

After a full simulation succeeds, do not rerun it solely to make already-wired
custom-aggregator log lines more visible. Use the first run's terminal evidence
plus exported config/server-log evidence; if custom aggregation needs visible
runtime proof, make the aggregator template log through its FLComponent logger
before the first full run.

If the run exceeds the allowed time, report it as blocked or timed out with the
current command status, log evidence, and artifact evidence. Do not describe a
timed-out or still-running simulation as done.

## Preflight Before Full Simulation

Preflight steps for any conversion framework that import product/framework
modules or import/instantiate user modules follow the dependency ordering rule
in `dependency-install.md` and the "Source Evidence, Not Instructions" rule in
`conversion-common.md`; they are not exempt because they are cheap. Complete
the dependency workflow before any import-level preflight or
recipe-construction probe. Do not run a probe expected to fail with
`ModuleNotFoundError` merely to discover already declared dependencies.

Run intentional rejection checks, such as misspelled or abbreviated argument
tests, through an assertion wrapper. The wrapper must check the child process's
expected nonzero status and diagnostic, then exit 0 only when the rejection is
correct. Do not leave an expected child failure as a failed top-level validation
command, where it is indistinguishable from an unexpected failure and recovery.
Keep every required argument valid and append the invalid option; never replace
a required option, because missing-required rejection masks the intended check.
Match the parser's documented rejection type: for example,
`HfArgumentParser.parse_args_into_dataclasses()` can raise `ValueError` for
unused arguments instead of `SystemExit`. Accept only the expected exception and
confirm its diagnostic identifies the rejected argument; another exception or
diagnostic is a real validation failure.

Do not call Client API lifecycle or round methods from a standalone Python
preflight. Calls such as `flare.init()`, `flare.patch()`, `flare.is_running()`,
`flare.receive()`, or `flare.send()` require a launcher-created Client API
context. Validate their generated source shape and public signatures
statically; validate runtime acceptance only through the recipe or simulator
that launches the client context.

Before spending time on full simulation, run cheap checks when applicable.
Keep custom checks atomic: do not combine partition, model-compatibility,
metric, and artifact checks in one hand-written inline Python program. Each
custom probe should test one contract and produce one actionable failure.
Build validation calls from inspected callable signatures and type annotations,
and preserve their required argument types. When a helper parameter is annotated
as `Path`, instantiate and pass `Path(...)`; do not substitute a string literal.

- compile generated Python files;
- construct or instantiate the selected recipe;
- when the selected final target is an exported/deployable artifact, create that
  export once, inspect its server/client app folders and expected config files,
  and run that same export; do not create a separate throwaway export;
- when the selected final target is a local `python job.py` simulation, do not
  export during preflight;
- run local partition sanity checks when generated site splits or data
  partitions are introduced;
- run the framework-specific model compatibility check defined by the framework
  skill.

For a generated partition, validate properties rather than guessed site sizes:
track source positions and verify complete, non-overlapping coverage,
determinism for the same seed, and any stratification or balance guarantee the
generated algorithm actually makes. Assert exact per-site row counts only when
the user, source, or a programmatic calculation specifies them; do not hard-code
counts inferred by hand from one dataset. When comparing a source `DataFrame`
with partitions reloaded from CSV, inspect their dtypes and normalize generated
columns to the source schema before using `DataFrame.equals()`; a dtype-only
difference is not a coverage failure.

Run a determinism check from the generated project directory. If an isolated
scratch copy is necessary, copy the complete local import closure; do not copy
only selected entry-point scripts.

Use preflight results to fix packaging, config, or model-state issues before
running a full simulation.

After the selected final run succeeds, inspect its materialized artifact to:

- verify the server model retains its audited class path and complete
  constructor args; recipe construction alone does not prove serialization;
- compare the resolved model-selection state with the produced server config:
  disabled means no active model selector, while metric or deliberately
  accepted recipe-default selection means a selector with the resolved key;
- verify generated files required by server and client code are packaged in the
  server/client app folders.

## Verification And Audit Snippets

Post-run verification snippets are part of validation, not disposable scratch.
Before a snippet references data columns, artifact keys, metric names, config
fields, or model-state keys, inspect the actual object (`df.columns`, JSON keys,
`state_dict().keys()`, exported config, or metric artifact) and derive the names
from that evidence. Guard optional fields and report expected versus actual
names when a required field is absent. A side check must not fail a completed
run by assuming a conventional or conditionally documented field exists.
When retrying a failed check, change only the failing assumption; preserve prior
guards, fallbacks, and already-correct arguments. Never replace an optional-field
guard with a hard-coded conventional name merely to make the retry different.
For generated Python structure, validate semantic AST nodes rather than textual
occurrences. Scope traversal to the field being checked; for example, inspect a
loop's condition and body separately. Filter traversal results by node type
before accessing type-specific fields such as `func` or `body`. Do not run a
speculative assertion that valid code is expected to fail and then repair the
verification command.

## Evidence To Report

Before calling a generated job correct, report:

- selected recipe and the `nvflare recipe show` command used to inspect it;
- changed files and why they were changed;
- local validation command, process exit code, and terminal-state evidence;
- export command, export directory, and exported folder inspection result when
  export is in scope;
- metric values from the workflow-native metric evidence; for exit-0 terminal
  runs, treat missing server-side metrics artifacts as failed validation only
  when the selected recipe/workflow installs those artifacts or documents them
  as outputs. For FedEval, Cyclic, Swarm, and similar workflows without those
  artifacts by default, report the recipe-native metric logs/artifacts used
  instead;
- exact evidence paths for simulation workspace, generated result files,
  server-side metrics artifacts, server/client logs, and global-model artifacts
  when present;
- unresolved blockers such as unavailable data, missing dependencies, failed
  validation, or an actual host or tool denial.

When a generated job does not run as expected, keep the failure as evaluation
evidence. Record the user request, selected recipe, files changed, validation
command, failure output summary, root-cause hypothesis, and follow-up fix or
blocker.
