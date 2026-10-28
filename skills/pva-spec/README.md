<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# PVA development skills

Agent skills for creating, changing, or repairing NVIDIA PVA operators
in your PVA SDK application or PVA Solutions source tree. They turn
requirements into code, verify it on Tegra hardware, and repair failures within
a debug budget.

## Skills

| Skill | Purpose |
| --- | --- |
| [Doctor](https://github.com/NVIDIA/skills/blob/main/skills/pva-doctor/SKILL.md) | Check the SDK, toolchain, project, target, and SDK references. |
| [Setup](https://github.com/NVIDIA/skills/blob/main/skills/pva-setup/SKILL.md) | Install missing SDK, NVVPU, and L4T runtime components, then rerun Doctor. |
| [Spec](https://github.com/NVIDIA/skills/blob/main/skills/pva-spec/SKILL.md) | Define operator behavior, correctness tests, and the OpGen handoff. |
| [OpGen](https://github.com/NVIDIA/skills/blob/main/skills/pva-opgen/SKILL.md) | Generate and integrate the operator, then invoke Build-Test and Debug. |
| [Build-Test](https://github.com/NVIDIA/skills/blob/main/skills/pva-build-test/SKILL.md) | Build, verify correctness on hardware, and measure requested performance cases. |
| [Debug](https://github.com/NVIDIA/skills/blob/main/skills/pva-debug/SKILL.md) | Diagnose and repair a reproduced failure. |

## Prerequisites

- An authenticated coding agent with local skill support and shell/file access.
  See [agent support](#agent-support) below.
- A development host running regular x86-64 Ubuntu supported by the requested SDK.
- An Arm64 Jetson board running compatible Jetson Linux (L4T), with working
  key-based SSH and a trusted host key.
- A matching SDK installation guide (file or URL), access to its installer,
  and administrative permissions for required installation steps.
- A writable project: your PVA SDK application or a PVA Solutions source package
  or checkout. This is where OpGen adds or changes code. Install Git LFS if the
  project uses it.

When PVA Solutions is selected, Setup uses
[NVIDIA/PVA-Solutions](https://github.com/nvidia/pva-solutions) by default and
clones into the supplied project path, or `$WORK_DIR/source` when omitted.
It reuses an existing source tree and accepts another repository or archive.
Doctor checks the selected source and reports missing required files.

Run the agent and build tools on the development host. `PVA_BUILD_MODE=L4T`
cross-compiles there for execution on the board. `PVA_BUILD_MODE=EMU` builds
for emulation on the development host.

Setup installs or reuses the SDK, NVVPU, required build and reference tools,
and board runtime. EMU is optional. Setup calls Doctor before
making changes and again afterward; no existing Doctor report is required.
Setup reads the installer URL from the guide. Supply a separate installer path
or URL only when the guide does not provide one. The guide and installer must
match the requested SDK release.
Setup reads the board's installed BSP and checks SDK compatibility. Supply an
expected BSP release only when you want to require a specific release.

## Set up a workspace

Copy all six PVA skill folders from the public
[NVIDIA skills catalog](https://github.com/NVIDIA/skills/tree/main/skills)
into `$HOME/pva-sdk-skills`. Keep each folder's scripts, references, and assets.
The resulting layout is `$HOME/pva-sdk-skills/pva-setup/SKILL.md`, and the
same layout for the other five skills.

```bash
git clone --depth 1 https://github.com/NVIDIA/skills.git "$HOME/nvidia-skills"
mkdir -p "$HOME/pva-sdk-skills"
for pva_skill in pva-setup pva-doctor pva-spec pva-opgen pva-build-test pva-debug; do
  cp -R "$HOME/nvidia-skills/skills/$pva_skill" "$HOME/pva-sdk-skills/"
done
```

Use a separate workspace for each operator task. Replace the project path below
with your writable PVA SDK application or PVA Solutions source directory.

```bash
mkdir -p "$HOME/operator_test/my_operator"
cd "$HOME/operator_test/my_operator"
ln -s "/absolute/path/to/your/project" source

pva_skill_dir=.claude/skills # For Codex, use .agents/skills
mkdir -p "$pva_skill_dir"
for pva_skill in "$HOME/pva-sdk-skills"/pva-*; do
  ln -s "$pva_skill" "$pva_skill_dir/"
done
```

Reload skills or restart the agent session after creating the links.
`source/` points to the project OpGen will modify; the surrounding workspace
holds your inputs, generated configuration, and reports. Each skill carries its
required scripts and templates. Install all six for the full workflow;
OpGen invokes Build-Test and Debug by their installed skill names.
Generated SDK references belong in `references/sdk/` in this workspace.
Doctor records their verified absolute location in its report.

## Agent support

- **Claude Code:** Used for the recorded operator runs and the commands below.
- **Codex:** Supports [local skills](https://learn.chatgpt.com/docs/build-skills#where-to-save-skills)
  in `.agents/skills`, invoked with `$pva-spec` or another skill name.
  These skills include `agents/openai.yaml` metadata. OpGen uses the agent's
  available skill mechanism for Build-Test and Debug. Without a callable skill
  tool, it loads the child skill's `SKILL.md` and executes its full workflow
  before resuming OpGen with the reports and verdict.

## Run Setup → Doctor → Spec → OpGen

Create `requirements.md` in the workspace: operator name/path, task mode (`new`,
`change`, or `repair`), required behavior, supported inputs, API, and performance
requirements. For an existing operator, include relevant source and test paths.

Run these **headless Claude Code commands** one at a time from the development
host's workspace. Replace the guide and board settings below. Setup resolves
the SDK release, installer, package names, and installation paths from the
guide and package metadata. Package paths are not fixed in the skill.
You can provide an explicit SDK release or path override when needed.

```bash
export PVA_OPERATOR_MAX_DEBUG_ATTEMPTS=3

# 1. Setup: check and prepare the host and board
claude -p "Run pva-setup for spec, opgen, and build-test.
Installation guide: http://docs.nvidia.com/pva/sdk/installation.html.
Target: user@<board-IP>; PVA generation: <generation for your board>.
Project: $PWD/source; work directory: $PWD."

# 2. Doctor: verify readiness after Setup
claude -p "Run pva-doctor for spec, opgen, and build-test using
$PWD/reports/pva-setup-report.md.
Project: $PWD/source; work directory: $PWD."

# 3. Spec: run after Doctor is ready
# Spec runs its finalizer to create and validate configure/opgen-inputs.json.
claude -p "Invoke pva-spec through the Skill tool using
$PWD/requirements.md and $PWD/reports/doctor-report.md.
Project: $PWD/source; work directory: $PWD."

# 4. OpGen: run after Spec returns CONFIGURED
claude -p "Invoke pva-opgen through the Skill tool using $PWD/configure/opgen-inputs.json."
```

Doctor writes `$WORK_DIR/reports/doctor-report.md` automatically. The examples
use `$PWD` as the work directory so Spec and OpGen can use the resulting files.

OpGen runs Build-Test and invokes Debug as needed. Read the final result in
`reports/opgen-report.md`.

See [reference maintenance](https://github.com/NVIDIA/skills/blob/main/skills/pva-doctor/references/reference-maintenance.md) for SDK-reference
regeneration and upgrades.

## Maintaining the helpers

Spec owns [finalize_opgen_inputs.py](https://github.com/NVIDIA/skills/blob/main/skills/pva-spec/scripts/finalize_opgen_inputs.py).
OpGen ships [an identical copy](https://github.com/NVIDIA/skills/blob/main/skills/pva-opgen/scripts/finalize_opgen_inputs.py)
so handoff validation works when the skills are installed separately.
After changing the Spec helper, update the OpGen copy too.
Generated SDK references and operator reports belong in the operator workspace.
