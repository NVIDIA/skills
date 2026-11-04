## Set up or repair the standalone training package

Use the [official setup guide](https://nvidia-isaac-ros.github.io/manipulation/training/setup_training.html)
and the selected manipulation-repository ref. This workflow installs Isaac Lab as a
pinned dependency; no user-managed Isaac Lab source checkout is needed.
Do not edit the training project's source, dependency declarations, lockfile, or
installed package metadata. All fixes here are skill-managed installation choices
inside `.venv`; record project/lockfile hashes before and after setup.

### 1. Discover access, destination, and capacity

Honor a user-selected path or fresh-install request. Otherwise check recorded
setup/session paths for a compatible installation, including non-default locations.
Before reuse, inspect origin/ref/dirty status and run preflight with its recorded
[launch environment](environment.md). Preserve edits and reuse a verified `.venv`
and matching smoke evidence without reinstalling. Continue partial installs from
the failed step under the repair rules below.

If none is suitable, choose and announce an unused `~/isaac_ros_manipulation`
(with a numeric suffix if occupied). Existing setup authorization is sufficient;
ask only for missing required choices and never repurpose another checkout.

Check OS/architecture, GPU/driver/VRAM, active GPU jobs, RAM, disk, Git, Python, and
uv. The release requires Linux and Python 3.12. Compare simulator requirements
against the version pinned in `pyproject.toml`/`uv.lock`. Verify device/network
failures from the authorized host context before declaring a blocker. Do not
upgrade drivers or kill unrelated jobs.

The repository is public; GitHub login and `gh` are not prerequisites. Check
branch access with:

```bash
git ls-remote --heads https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulation.git release-r1
```

Report unresolved access errors without substituting a personal fork or requesting tokens.

Read-only discovery, EULA status, repository cloning/entry, and initial uv
installation are bootstrap exceptions to the package-root and launch-helper rules. Create a timestamped
skill-local `logs/<timestamp>__setup/setup.log` before the first setup mutation.
Record paths, refs, commands, outputs, exit codes, and repairs; never credentials.

### 2. Prepare the manipulation repository and consent

For a new destination, clone the public release over HTTPS:

```bash
git clone --branch release-r1 --single-branch \
  https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulation.git "$REPO"
cd "$REPO/isaaclab_training"
```

`REPO` is the chosen absolute clone destination. Set `training_root` to the
absolute package directory reached by `cd`. For an existing checkout, inspect
origin, branch/tag/commit, and dirty status. Preserve local edits and environments;
do not reset, switch refs, pull, or reclone over them. Honor an explicitly selected
ref, inspecting its compatibility instead of assuming it matches the default.

Read `pyproject.toml`, `uv.lock`, and repository instructions. Confirm the editable
project, `sim` group, and official Isaac Lab Git wheel-builder pin. Git metadata
belongs to the parent repository.

Keep the default checkout on `release-r1`; no checkout commit is
pinned by this skill. Record the actual HEAD and inspect the selected files on
each fresh setup. As inspected on 2026-10-31, the branch's
[package configuration](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulation/blob/release-r1/isaaclab_training/pyproject.toml)
requires Python 3.12, Isaac Sim 6.1.0.0, and optional LEAPP 0.6.1.
The task uses native RSL-RL 5
`agent.actor` and `agent.critic` configuration. If these dependencies or interfaces
change on the branch, review compatibility before applying the setup below;
do not change upstream pins to bypass a compatibility failure.

Follow [EULA acceptance](eula.md) before installing or importing Isaac Sim. Reuse
valid consent; otherwise ask the exact consent question there and require Yes.
Run receipt commands with system Python from the selected package directory.
The receipt covers Isaac Sim 6.1. Stop if a selected ref pins a different simulator
scope; do not silently widen the acceptance or change the dependency pin.

### 3. Install the locked environment and skill-managed supplements

First locate uv. If missing, install it at user level using the official setup
guide's method, directly from the host shell before defining `training_cmd`.
This bootstrap step is exempt from the launch-helper requirement because the
helper requires an existing uv executable. Keep shell startup files unchanged,
record the installation command in `setup.log`, and verify the installed uv with
its absolute path and `--version`.

Set `environment.uv_executable` in the plan to that absolute path, then prepare
`training_cmd` from [Process environment](environment.md). Run the environment
installation commands below through that helper so setup and runtime use the
same selected uv, Python/cache locations and library/import overrides. Do not
inherit an active environment or old checkout through shell startup files.
Let uv provision the required Python version.
For `release-r1` with Isaac Sim 6.1.0.0, use GPU ONNX
Runtime only. The locked plan includes CPU `onnxruntime`, but the resolved
`isaacsim-robot` wheel metadata requires `onnxruntime-gpu`; omit the CPU package
from installation and supply the GPU package separately. Open3D's missing
`ipywidgets` requirement is also supplied by the skill. The exact supplemental
pins live in `assets/standalone-supplements.txt`.

These supplements deviate from the upstream lock. Revalidate them if the selected
release changes; [ONNX Runtime permits only one distribution per environment](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime).
Training does not require LEAPP. For export, follow the companion
[deployment setup](../../isaac-cable-insertion-policy-deployment/references/setup.md), including
its version-bound GPU-provider exception and numerical validation.

From `training_root`, after acceptance, with `SKILL` set to this skill's absolute path:

```bash
training_cmd uv sync --locked --group sim --no-install-package onnxruntime --dry-run
training_cmd uv sync --locked --group sim --no-install-package onnxruntime
```

This installs the editable training package, pinned Isaac Lab dependency, and
Isaac Sim in `training_root/.venv`.
`--locked` prevents silently regenerating the release lockfile. If it reports a
stale lock, stop and report the mismatch; do not run `uv add`, regenerate the lock,
or edit project files. A working verified environment needs no sync. For a repair
that must retain supplemental packages, use
`uv sync --locked --inexact --group sim --no-install-package onnxruntime` and
review the intended changes. Never repair an environment used by an active trainer.
An already active source environment must not become the destination: do not use
`--active`, and unset a conflicting `UV_PROJECT_ENVIRONMENT` before syncing.
Verify the dry run targets `training_root/.venv`.

Export constraints to a temporary file outside the training project, then install
the skill's supplements. The CPU ONNX pin can remain in this constraints file:
constraints do not request installation of that package.

```bash
CONSTRAINTS=$(mktemp /tmp/cable-training-constraints.XXXXXX)
training_cmd uv export --locked --group sim --no-hashes --no-emit-project --no-emit-package isaaclab \
  --output-file "$CONSTRAINTS"
training_cmd uv pip install --python .venv/bin/python --constraint "$CONSTRAINTS" --dry-run \
  --requirement "$SKILL/assets/standalone-supplements.txt"
# After reviewing the constrained dry run:
training_cmd uv pip install --python .venv/bin/python --constraint "$CONSTRAINTS" \
  --requirement "$SKILL/assets/standalone-supplements.txt"
```

If CPU `onnxruntime` is already installed in an existing idle environment, uninstall
that distribution before installing the supplements. If both distributions were
present, removing CPU files can damage the shared import: reinstall GPU ONNX
Runtime afterward with `--reinstall-package onnxruntime-gpu`. Use the same
constraints and supplement file, and log the repair. Confirm that only the GPU
distribution remains installed and that `import onnxruntime` succeeds.

For requested video, include `'moviepy==1.0.3' imageio imageio-ffmpeg` in both
supplemental install commands above so they resolve together; record the resolved
versions and pass the encoder/decoder probe. This keeps MoviePy's decorator
requirement compatible with the IPython dependencies added by `ipywidgets`.

Do not replace locked NumPy, simulator, Isaac Lab, or RL packages to satisfy a
supplemental dependency. Constraints are mandatory for supplemental installs and
repairs. Report an unresolved conflict instead of dropping constraints or upgrading.
Afterward use `uv run --no-sync` for every trainer, evaluator, and probe, so ordinary
launches cannot change the environment. Exact `uv sync` removes supplemental
packages outside the lockfile; repairs must retain `--inexact --group sim` and
`--no-install-package onnxruntime`, then repeat preflight. Recheck the unchanged
project/lockfile hashes and record the supplement file hash and installed versions
in the setup log.

Run `uv pip check --python .venv/bin/python`. Record all conflicts; accept only
exceptions explicitly declared by this training project’s `tool.uv.override-dependencies`
whose replacement constraint is satisfied by the installed locked version, plus
the version-bound LEAPP GPU-provider substitution in the deployment setup.
An override is not a blanket waiver for other conflicts or absent packages.
The supervisor's `dependencies` probe enforces this and records accepted exceptions.
Stop on missing packages or unexplained conflicts. Verify editable task imports,
locked Isaac Lab provenance, registration, and video again after any repair.

### 4. Verify before the campaign

From `training_root`, prepare a plan and run:

```bash
training_cmd .venv/bin/python "$SKILL/scripts/supervisor.py" preflight --plan "$PLAN"
```

See [Preflight](preflight.md) for the checks; this command installs nothing and
creates no campaign.

After a fresh install or simulator/task/video repair, run a separate bounded smoke
session with the committed supervisor: one run, `decision_mode=preplanned`,
evaluation disabled (`evaluation_budget_seconds=null`), `purpose=smoke`,
`num_envs=16`, `max_iterations=3`, `num_steps_per_env=64`, video length 32 and
`video_schedule=start_end` when requested. Keep task physics and reset IK at their
defaults. The supervisor aligns the curriculum step divisor.

Reserve at most 900 seconds for GPU verification, reduced to fit any smaller user
budget. Verify task assets, PPO progress, exit status, final checkpoint, resolved
YAMLs, and decodable clips. First-launch downloads can consume this budget; report
a timeout truthfully. Start the requested campaign only after verification passes.
Reuse recent matching smoke evidence for an unchanged installation. Report setup,
verification, and campaign time separately; subtract verification from any explicit
total GPU budget. A smoke test is not policy-success evidence.

### Recovery and delivery

For a known dependency/network failure, log it, apply one documented bounded repair,
and retry once. Stop on unresolved version, access, driver, or asset errors. Keep
partial downloads/environments and continue from the failed step on re-invocation.
Never patch Isaac Lab or task sources to make installation pass.

Link setup and smoke evidence in the campaign’s single `report.md` using supervisor
notes at an idle point. Preserve real exit codes when using `tee` (`pipefail` and
`PIPESTATUS`). If setup fails, deliver its log and precise blocker, and do not claim
training started. Generated records stay in the skill’s ignored `logs/` directory.
