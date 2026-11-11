---
name: nvshmem-install
description: Plan and validate NVSHMEM and NVSHMEM4Py installations. Use for package, container, or source deployments.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA NVSHMEM Team <nvshmem@nvidia.com>
  tags:
    - nvshmem
    - install
    - deployment
---

# NVSHMEM Installation

## Purpose

Guide safe, version-aware NVSHMEM and NVSHMEM4Py installations on workstations, clusters, multi-node systems, and containers. Select an installation method from documented requirements and observed system facts, then provide exact commands and validation steps. Execute installation actions only after separate explicit approval.

## Prerequisites

- `$nvshmem-docs` must be available in the skill registry or readable at `../nvshmem-docs/SKILL.md` to resolve official, version-specific requirements. If neither route is available, exact requirements and installation commands are blocked; return the mapped official links and identify the unresolved claims. No API keys are required.
- Use registered `$nvshmem-troubleshoot-and-report-bugs` for post-install launch, initialization, runtime, or validation failures. If it is unavailable, preserve the diagnostic evidence and provide official support guidance instead of assuming that an unreadable local fallback exists.
- Confirm network access for remote artifacts and write authority for the chosen installation prefix before execution.
- Do not inspect the target automatically. First complete the installation intake, then ask whether the user wants to run the read-only probe themselves or wants the agent to run it. Run it only after the user explicitly selects the latter.
- Resolve NVSHMEM, CUDA, driver, compiler, launcher, bootstrap, and transport dependencies from the exact-version documentation and target inspection instead of treating them as fixed prerequisites.
- Inspect the target only through an already available unprivileged shell. If direct inspection is unavailable, provide the probe for the user to run and return its output. Never request or use `sudo`, root access, or privilege escalation for inspection.

## Inputs

Use the detailed intake under `Resolve the Request`.

- **Required:** API surface, target type, administrator and target-scope authority, installation scope and prefix, launcher/bootstrap selection or `recommend`, and enough target-system facts to assess documented requirements.
- **Optional:** A complete NVSHMEM version, defaulting to `latest`, and a preferred installation method.
- **Conditional:** Source-build configuration when source is selected.
- **Probe choice:** Required before direct target inspection: user-run probe or agent-run probe.
- **Execution approval:** Optional and not approved by default; approval applies only to the exact command block presented.

Use the first non-conflicting value in this source order:

1. State file.
2. Explicit prompt arguments.
3. Agent context.
4. Free-form user prompt.

Report conflicts instead of silently merging them. Execution always requires current explicit approval for the exact command block presented; a missing response is not approval.

## Limitations

- Depend on live official documentation for exact version requirements and commands. If retrieval fails, provide mapped official links and identify unverified claims instead of synthesizing an answer.
- Treat the system probe as a snapshot of the current shell and node; it cannot establish cluster-wide consistency, site policy, container-host configuration, or an untested network path.

## Instructions

### Safety and Defaults

- Default to providing commands, not executing them. An install request expresses intent, not approval.
- Never request or use `sudo`, root access, or privilege escalation for inspection. Use only an existing unprivileged shell; otherwise give the user the probe to run.
- Treat downloads, repository changes, package operations, container pulls/builds, source builds, and install-prefix writes as installation actions. Before executing any such action, present the exact command block, effects, and paths, then obtain explicit approval for that block. Silence is not approval; revised commands require fresh approval.
- Do not change drivers, kernel modules, system configuration, modulefiles, or shared cluster state unless separately requested and approved.
- Stop on the first failed install command and preserve its output. Permit at most three separately approved revised attempts before handing off.

### Resolve the Request

Before running any target probe or other system-inspection command, ask for and resolve the installation intake. Do not infer an answer from the current shell when the user has not yet chosen the intended target or installation scope.

Ask the following questions verbatim in one numbered message. Omit only a question whose answer is already explicit, unambiguous, and non-conflicting in the request or state file. Do not probe, retrieve version-specific documentation, or propose an installation method until the user answers Question 8.

1. **API:** Which NVSHMEM interface do you need: `C/C++`, `Python (NVSHMEM4Py)`, or `both`?
2. **Version:** Which complete NVSHMEM version do you want? Reply `latest` to use the newest fully documented release.
3. **Target:** Where should NVSHMEM run: `HPC cluster / multi-node system` or `single machine`?
4. **Authority:** Do you have administrator rights **and** authority to install system-wide software in the target scope? Reply `administrator` or `unprivileged`.
5. **Scope:** Where should NVSHMEM be installed: `user-owned prefix`, `system packages`, `container`, `virtual environment`, ? Include the exact prefix or environment path when it is not a container or system packages.
6. **Method:** Do you prefer a method: `binary archive`, `system repository/package`,  `source build`, `HPC SDK container`, `PyPI`, `Conda`,  ? Reply `recommend` if you have no preference.
7. **Probe:** May I run the bundled read-only system probe on the target shell? Reply `agent-run`, or reply `user-run` and I will give you the command to run and paste back.
8. **Launcher:** Which launcher/bootstrap should the application and two-PE validation use: `mpirun/mpiexec`, `nvshmrun`, `srun`, `oshrun`, another launcher, or `recommend`?

Interpret `administrator`, `unprivileged` exactly as stated. If Question 4 is unanswered after one prompt, record `unknown` and apply the unprivileged planning default from `Safety and Defaults`. Reject a scope that the recorded authority cannot modify and offer feasible alternatives. If Question 2 supplies an incomplete version such as `major.minor`, treat it as a prefix query: use `$nvshmem-docs` to resolve the newest fully documented matching release, show the complete resolved version, and obtain confirmation before executing version-specific commands. Otherwise preserve the literal requested version or `latest`. Treat Question 8 as launcher/bootstrap intent, not proof that the launcher is installed or compatible. After the probe, verify an explicit selection against the observed commands and versioned documentation. For `recommend`, select the compatible launcher implied by the execution environment when unambiguous; ask when multiple compatible choices remain. If none is available, record `none` rather than inventing a launcher, continue install planning when a launcher is not an installation prerequisite, and mark multi-PE runtime validation as blocked until a compatible launcher is available.

After Question 8 is answered, run or provide the bundled probe as selected. Use the chosen unprivileged probe for every fact it can collect, and request only the remaining facts. Ask the following conditional questions only when their condition applies:

- **Source build:** “What GPU and network topology will the build support (nodes, GPUs per node, and interconnect)? Which RDMA NICs/link layer or fabric provider are required? Which CUDA architectures must be built? Should the build include Python bindings, tests, examples, Hydra, or packages?” Read [source-build.md](references/source-build.md).
- **Managed Slurm/HPC cluster:** “May the resulting install be visible from every intended compute node, and does site policy permit the selected user prefix or container runtime?”
- **Shared cluster prefix:** “Who owns this prefix, and do all participating compute nodes mount the same path?”
- **Existing or new container:** “Which runtime will be used (`Docker`, `Apptainer/Singularity`, or another approved runtime), and is GPU and required network-device passthrough permitted by site policy?”
- **Python or both:** “Which Python interpreter or existing environment should contain NVSHMEM4Py? May I create a new virtual environment if needed?”

If the requested API is `Python (NVSHMEM4Py)` and the requested method is `binary archive` or `system repository/package`, reject that method combination before selecting artifacts or presenting commands. NVSHMEM4Py must be installed through a supported Python package channel (`PyPI` or `Conda`); select `source build` only when the user explicitly needs to build the bindings. Do not treat a native NVSHMEM archive or system NVSHMEM package as an NVSHMEM4Py installer or request one from the user. For `both`, an archive or system package may supply the C/C++ runtime, but NVSHMEM4Py still requires its own PyPI, Conda, or explicitly requested source-build installation.

Record every answer in `Target Summary`, marking supplied answers as observed user intent and unanswered optional values as `unknown`.

Record the resolved launcher and applicable bootstrap in `Target Summary` or `Requirements Check`, using `none` or `not applicable` explicitly when appropriate. A Python import, native-library load, or single-PE run does not replace the two-PE same-node runtime test. Never silently omit that test when launcher, GPU, scheduler, or allocation prerequisites are unavailable.

After the user selects a probe path, do not ask for information that can be obtained through the chosen unprivileged inspection.

### Consult Live Documentation

Use `$nvshmem-docs` before stating requirements, compatibility, supported versions, package names, repository URLs, image tags, dependency versions, build variables, or exact commands. When working from this source tree and the sibling skill is not registered, read and follow `../nvshmem-docs/SKILL.md` directly.

- Pass the normalized complete version or the literal `latest`.
- Resolve `latest` to the newest NVSHMEM release whose version, requirements, and compatibility are established by the live NVSHMEM documentation. Treat a Python binding's PyPI/Conda version as an independent package version, not as an NVSHMEM release candidate: select its current compatible released artifact from its package metadata and report the two versions separately. Apply the separate-candidate rule only when an artifact represents a newer NVSHMEM runtime/library release than the latest matching documentation. Do not apply older-release compatibility claims to such a runtime candidate.
- Request the installation guide, release compatibility and limitations, package or source procedure, and NVSHMEM4Py page when relevant.
- For NVIDIA HPC SDK containers, also request the NGC image and container-guide routes. Keep the HPC SDK/image version separate from the bundled NVSHMEM version.
- Hardcode only the stable method families and decision process in this skill. Never assume that a package name, CMake option, numeric requirement, or container tag from a previous release is still valid.
- For binary archives and Linux package downloads (DEB, RPM, or Ubuntu packages), start from the official NVIDIA NVSHMEM downloads page: `https://developer.nvidia.com/nvshmem-downloads`. When its linked official redistribution manifest provides the exact artifact `relative_path`, resolve it against that manifest's official download base and show a copyable command for the resulting URL. Likewise, show a command when the live official page exposes a full artifact URL. Selecting `binary archive` is sufficient to present the command, but never to execute it. Include this notice with the command: "By downloading and using the software, you agree to fully comply with the terms and conditions of the NVIDIA Software License Agreement." If NVIDIA requires authentication or interactive license acceptance and neither an exact URL nor an official manifest path is available to the current session, report that external limitation and provide the official download page. Never construct an artifact filename or URL from a remembered naming pattern.
- For every source build, obtain the selected release from the official NVIDIA NVSHMEM GitHub releases page: `https://github.com/NVIDIA/nvshmem/releases`. Use the release's published source archive or release tag, verify that its version matches the resolved NVSHMEM version, and report the release URL as provenance.
- If live documentation cannot be retrieved, return the official routed links and identify the blocked claims. Do not invent installation commands from memory.
- If official pages conflict, report the conflict and avoid execution until it is resolved.

### Inspect and Assess the Target

Only after the installation intake is complete and the user explicitly asks the agent to do so, run the bundled read-only probe without arguments:

```text
run_script("scripts/collect-system-info.sh")
```

If the user elects to run it themselves, provide that command and wait for its output. Keep the probe unprivileged. On a cluster, inspect representative compute nodes only within an existing allocation; otherwise provide the script or equivalent commands for the user to run there.

Compare the collected facts with the exact-version documentation and classify each relevant requirement:

- `pass`: the target satisfies the documented requirement.
- `fail`: the target contradicts the documented requirement. Stop and provide remediation choices.
- `unknown`: the available evidence is insufficient. Ask for the smallest missing fact and do not execute installation commands.
- `not applicable`: the requirement belongs only to an unselected optional feature.

For a missing optional transport dependency, do not enable that transport and explain the capability loss. If it is the NIC-matched baseline remote transport, ask before proceeding without remote transport support. Do not weaken a mandatory requirement.

When several features or transports are enabled, require the union of their documented prerequisites. Never disable a documented prerequisite while leaving the dependent feature enabled.

For every source build, select a remote transport before generating the configuration. When the probe finds a usable RDMA NIC or a supported fabric provider, choose the matching baseline remote transport from [source-build.md](references/source-build.md) and assess its prerequisites. Do not omit remote support merely because the first validation is same-node.

Only a `pass` verdict permits a transport to be enabled by default. An `unknown` prerequisite blocks generation of an exact source configuration until the missing fact is resolved; do not enable the transport or silently omit it. A `fail` verdict disables that transport and requires the capability loss to be reported, with user acceptance when it removes the NIC-matched baseline remote path.

When no remote transport is selected—because the user limits the scope to local GPUs, no matching NIC/provider is present, or the user accepts a documented prerequisite failure—leave every remote-transport support option off. If a matching transport has an unknown prerequisite, obtain the missing fact or ask the user before generating the installation command.

### Choose, Execute, and Validate

After the requirements check, read [installation-matrix.md](references/installation-matrix.md) to select a method. For a source build, also read [source-build.md](references/source-build.md). Before presenting commands, requesting execution approval, or validating an installation, read [execution-and-validation.md](references/execution-and-validation.md). On any planning, installation, or validation failure, read [failure-handoff.md](references/failure-handoff.md).

## Troubleshooting

For any planning, installation, or validation failure, preserve the command, output, target summary, selected transport/bootstrap, and documentation version, then follow [failure-handoff.md](references/failure-handoff.md). For post-install launch, initialization, runtime, or validation failures, use `$nvshmem-troubleshoot-and-report-bugs` when available; otherwise provide official support guidance with the preserved evidence.

## Available Scripts

| Script | Purpose | Arguments |
| --- | --- | --- |
| [`scripts/collect-system-info.sh`](scripts/collect-system-info.sh) | Collect read-only, unprivileged system, GPU, CUDA, launcher, RDMA, and package-manager information for installation planning. | None |

## Examples

Request: “Install the latest NVSHMEM C/C++ development package on this Ubuntu system.”

Resolve `latest`, ask about administrator authority, inspect the system, and recommend APT only when the user confirms authority over the target. Present exact commands and validation before requesting execution approval.

Request: “Build NVSHMEM from source with IBGDA for these Infiniband-connected GPUs.”

Inspect the GPU, NIC, launcher, and dependency environment; verify the requested release and IBGDA prerequisites through `$nvshmem-docs`; then present a complete CMake configuration, installation sequence, and same-node and two-node validation plan before requesting approval.

<!-- ci trigger -->
