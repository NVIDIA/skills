---
name: nvshmem-get-started
description: Use when NVSHMEM beginners want a tutorial-style overview on assessment, mental models, first C/C++ or Python NVSHMEM programs, compilation, launching, and next steps.
license: Apache-2.0
metadata:
  author: NVIDIA NVSHMEM Team <nvshmem@nvidia.com>
  tags:
    - nvshmem
    - getting-started
    - tutorial
---

# Getting Started with NVSHMEM

## Purpose

Guide a beginner to the next useful action. Teach the stable mental model, program structure, build pattern, and launch pattern directly. Route installation, version-sensitive variants, advanced integrations, and failures to the appropriate specialist skill.

## Prerequisites

- Require no credentials, installed software, or GPU access for fit assessment and conceptual guidance.
- Keep the bundled [mental model](references/mental-model.md), [basic API calls](references/basic-api-calls.md), and [NVSHMEM4Py program guide](references/nvshmem4py-programs.md) available for the teaching paths that require them.
- Keep `$nvshmem-install`, `$nvshmem-docs`, and `$nvshmem-troubleshoot-and-report-bugs` available for complete installation, version-specific, and failure handoffs. If a specialist skill is unavailable, follow the documented fallback instead of inventing current details.

## Inputs

- **Required:** Obtain the user's goal or requested learning stage from the explicit user prompt.
- **Optional:** Obtain application communication needs, language, NVSHMEM and NVSHMEM4Py versions, source or script path, Python interpreter/environment, installation prefix, target GPU architecture, launcher, PE and node counts, and target environment only when the selected path needs them.
- **Source precedence:** Prefer the current explicit user prompt, then explicit invocation arguments, then non-conflicting agent context, then answers to focused follow-up questions. Report conflicting values instead of choosing silently.

## Instructions

### Execution Boundary

Default to explaining the workflow and printing commands for the user to run. Do not compile, launch a program, submit a scheduler job, allocate resources, or modify the environment unless the user explicitly asks for execution. Treat requests to learn, get started, or receive help running a program as requests for guidance, not execution approval.

### Choose a Starting Point

When the requested stage is clear, skip the menu and handle that stage. Otherwise ask what the user wants to do and present all options in one response:

1. Decide whether NVSHMEM fits my use case.
2. Learn the NVSHMEM mental model.
3. Install NVSHMEM.
4. Run a precompiled NVSHMEM example.
5. Compile or link a C/C++ application.
6. Write an NVSHMEM C/C++ or NVSHMEM4Py program.
7. Recommend the next step.

For option 7, first determine whether NVSHMEM fits the intended communication pattern. Then ask whether NVSHMEM is installed, whether an example has run successfully, and whether source code already exists. Recommend the mental model when the user is completely new, installation when NVSHMEM is absent, a precompiled example after installation, program design when no source exists, and compilation when source is ready.

### Decide When to Use NVSHMEM

Start from the application's communication needs rather than from a list of APIs. Describe NVSHMEM as a strong fit when several of these conditions apply:

- GPU threads should initiate communication directly from CUDA kernels without returning control to the CPU for every exchange.
- The application needs fine-grained, one-sided put, get, atomic, signaling, or synchronization operations between GPUs.
- Long-running or fused kernels should combine computation and communication while reducing kernel-launch and CPU-GPU synchronization overhead.
- Communication should overlap with computation to improve strong scaling as the amount of work per GPU becomes smaller.
- A partitioned global address space with collectively allocated symmetric GPU memory fits the application's data layout.
- The application already uses OpenSHMEM concepts or needs NVSHMEM alongside MPI or OpenSHMEM for GPU-resident communication.

Give representative workload patterns without claiming that NVSHMEM is mandatory for them:

- neighbor or halo exchanges in multi-GPU stencils, solvers, and simulations;
- irregular or fine-grained communication in graph and data-analytics workloads;
- persistent GPU kernels that communicate repeatedly during computation;
- custom GPU-initiated reductions, broadcasts, queues, and producer-consumer protocols;
- applications whose strong scaling is limited by repeated CPU orchestration of GPU communication.

Explain when another model may be simpler:

- Use ordinary CUDA for a single GPU or when no inter-GPU communication is required.
- Prefer NCCL when the application primarily makes standard bulk-collective calls—especially host-issued all-reduce, all-gather, reduce-scatter, or broadcast operations. NCCL is generally better suited than NVSHMEM for this pattern.
- Prefer MPI when communication is coarse-grained, CPU-driven send and receive is sufficient, or symmetric allocation would complicate the design.
- Reconsider NVSHMEM when the target cannot provide the required GPU, launcher, bootstrap, or transport environment.

First determine whether the prompt already contains enough communication detail to assess fit. If it does, immediately conclude with one of `strong fit`, `possible fit`, or `likely simpler with another model` and explain the deciding factors; do not defer the recommendation behind a checklist of questions. Otherwise ask only for the missing decision-critical facts: where communication is initiated, whether it is fine- or coarse-grained, which operations dominate, whether communication must occur inside kernels, and what GPU/node topology is targeted. Then give the same explicit conclusion. Link to the official [NVSHMEM Introduction and Advantages](https://docs.nvidia.com/nvshmem/api/latest/introduction.html#advantages-of-nvshmem).

### Teach the Mental Model

Read [references/mental-model.md](references/mental-model.md) completely before teaching the conceptual path. Explain the mental model without API-name clutter and use its illustrative ring shift when a concrete example would help. Before introducing C/C++ primitives, read [references/basic-api-calls.md](references/basic-api-calls.md) completely and present only the API groups relevant to the user's goal. For a Python request, also read [references/nvshmem4py-programs.md](references/nvshmem4py-programs.md) before outlining the program lifecycle or launch workflow. Keep examples framed as teaching aids rather than environment-specific recipes.

### Resolve the Hands-On Path

Collect only the context required by the selected path. Do not ask for every environment detail up front.

#### Install NVSHMEM

Before invoking `$nvshmem-install`, follow this intake order: resolve the requested version or `latest`, ask whether the user needs C/C++, Python, or both, then at step 3 ask whether the user is an administrator on the target system with root or `sudo` rights. On a managed cluster, also ask whether the user is authorized to change the compute-node image or shared software stack. Do not infer either answer from shell access, package-manager availability, or target type.

Invoke `$nvshmem-install`. Pass the administrator and target-scope authority status together with the requested API or language, target environment, preferred install method, version policy, and whether the user needs runtime files, development headers and libraries, or static linking. For an unprivileged cluster user, do not frame DEB, RPM, APT, DNF, YUM, or another system package path as directly usable; prefer a site module, approved container, compatible user-owned prefix, or a clearly labeled administrator handoff.

#### Design a C/C++ Program

Read [references/basic-api-calls.md](references/basic-api-calls.md) before explaining API-family choices or recommending specific primitives.

Teach this stable structure directly:

1. Choose initialization. Use `nvshmem_init` for the basic standalone pattern. Use `nvshmemx_init_attr` when the application must integrate an MPI, OpenSHMEM, or other bootstrap context.
2. Initialize, query a node-local PE ID, and select the CUDA device before allocating symmetric memory or launching kernels.
3. Allocate symmetric objects collectively. Require every PE to call allocation and deallocation routines in matching order with identical arguments.
4. Choose the operation family:
   - Use put or get for remote data movement.
   - Use atomics for remote updates.
   - Use signals and waits for point-to-point coordination.
   - Use team collectives for group communication.
5. Choose the invocation context: host API, device API inside a kernel, or on-stream API composed with CUDA stream work.
6. Identify remote data with a symmetric address and destination PE. Never send a symmetric pointer value to another PE for it to reuse as a local pointer.
7. Add the required ordering and completion. Use `nvshmem_fence` to order later operations after earlier operations to the same PE, `nvshmem_quiet` or its on-stream form for completion, and barriers, signals and waits, or collectives to coordinate PEs.
8. Free symmetric allocations collectively and finalize.

Use the ring-shift example in [references/mental-model.md](references/mental-model.md) as the first concrete illustration; read that reference before presenting the example.

#### Design or Run an NVSHMEM4Py Program

Read [references/nvshmem4py-programs.md](references/nvshmem4py-programs.md) completely. Ask only for the Python script path, Python interpreter or environment, NVSHMEM4Py version if known, initialization choice (standalone or MPI), launcher, PE and node counts, and whether the run is local or scheduled.

Teach the same PE, symmetric-memory, one-sided-operation, and completion model as C/C++, then map it to the Python lifecycle: choose and set the local CUDA device, initialize NVSHMEM4Py, collectively create symmetric arrays, perform the required communication and synchronization, free every symmetric array collectively, and finalize. Do not present C/C++ API names as Python APIs.

Invoke `$nvshmem-docs` with the requested NVSHMEM version or `latest` before giving Python API names, a runnable code sample, package-version compatibility, framework interoperability, initialization arguments, or exact launch commands. Pass the script path and all known Python, launcher, PE, node, and environment details. Use the documentation result to give a version-matched, user-runnable command shape. Confirm that the selected Python environment, the script, CUDA, NVSHMEM runtime libraries, and the NVSHMEM4Py package are available on every participating node.

When the user supplies a simple NVSHMEM4Py script, use the example in [references/nvshmem4py-programs.md](references/nvshmem4py-programs.md) to trace its device selection, initialization, symmetric allocation, communication, stream completion, collective cleanup, and launch requirements. Identify missing lifecycle steps or a mismatch between the script's initialization and launcher; do not execute or rewrite the script unless requested.

Start with two PEs on one node and one PE per GPU by default. Require the launcher to propagate the selected Python environment and CUDA/NVSHMEM environment to every PE. Explain the expected per-PE output and validate the small run before suggesting a larger or multi-node run.

#### Compile and Link C/C++

Ask for the source path, NVSHMEM installation prefix, target GPU architecture, and whether the project uses a direct `nvcc` command or CMake. For a standard CUDA source file, explain that the application must enable relocatable device code, include the NVSHMEM headers, search the installed library directory, and link both the host and device libraries.

Provide this command shape after replacing or clearly labeling its variables:

```bash
nvcc -rdc=true \
  -ccbin g++ \
  -gencode="$NVCC_GENCODE" \
  -I"$NVSHMEM_HOME/include" \
  application.cu \
  -L"$NVSHMEM_HOME/lib" \
  -lnvshmem_host -lnvshmem_device \
  -o application
```

Explain that `NVSHMEM_HOME` names the installed prefix and `NVCC_GENCODE` must select the target GPU architecture. For CMake, prefer the config files installed with NVSHMEM and use `find_package` in config mode; inspect the installed package before naming an imported target.

After the build command, state that the resulting NVSHMEM binary is still a multi-PE program and must be launched with `nvshmrun`, `mpirun`, or `srun` at a chosen PE count, typically one PE per GPU for the first run. Point the user to the launcher command shapes in [Run a Precompiled Program](#run-a-precompiled-program); do not leave `nvcc` or the CMake build as the final step.

#### Run a Precompiled Program

Ask for the executable path, launcher, PE and node counts, and whether the run is local or scheduled. Confirm that the executable and NVSHMEM runtime libraries are visible on every participating node.

Start with two PEs on one node, use one PE per GPU by default, and expand only after the small run succeeds. Present the command shape that matches the available launcher:

```bash
mpirun -n <number-of-pes> <program> [arguments]
```

```bash
srun -n <number-of-pes> <program> [arguments]
```

```bash
nvshmrun -n <number-of-pes> <program> [arguments]
```

Require the launcher to propagate the CUDA and NVSHMEM environment to every PE. Explain the expected per-PE output and verify it before recommending a larger or multi-node run.

#### Invoke Documentation Only for Variants

Invoke `$nvshmem-docs` with a complete requested version or `latest` when the request needs any of the following:

- exact package-specific example paths or binary names;
- PMI, PMI-2, PMIx, Hydra installation, or non-default bootstrap wiring;
- multi-node transport setup or transport-specific link dependencies;
- static linking, shared-library composition, symbol visibility, or non-`nvcc` toolchains;
- exact CMake package or imported-target names;
- release-specific API availability, flags, requirements, limitations, or commands.

## Limitations

- Provide introductory guidance, not exhaustive API, installation, launcher, transport, or performance coverage.
- Treat command blocks as adaptable shapes. Do not claim that they match an installed release, package layout, cluster policy, or target architecture without version-specific documentation and target inspection.
- Cover first-program and launch guidance for C/C++ CUDA and NVSHMEM4Py. Route Python API details, installation, and version compatibility to `$nvshmem-docs` rather than inferring them.
- Assess architectural fit from the stated communication pattern; do not promise a speedup without representative measurements on the target topology.
- Keep execution outside the default scope. A request for explanation, onboarding, or help running a program does not authorize environment changes or workload execution.

## Troubleshooting

| Symptom | Likely cause class | Response |
| --- | --- | --- |
| Compilation or linking fails | Source, flags, library order, package layout, or version mismatch | Invoke `$nvshmem-troubleshoot-and-report-bugs` with the full command and diagnostics. |
| A program errors or hangs | Launcher, bootstrap, transport, topology, runtime configuration, or program behavior | Invoke `$nvshmem-troubleshoot-and-report-bugs` with the complete launch stanza and output from every PE. |
| A specialist skill is unavailable | The normal diagnostic or version-specific route cannot be used | Invoke `$nvshmem-docs` for version-matched troubleshooting, FAQ, and support routes; do not invent a workaround. |

For an execution failure, pass or collect the complete command or batch-script launch stanza and complete stdout and stderr from every PE, including output before the first error and per-PE logs. For a compile or link failure, pass or collect the complete compiler or linker command, full diagnostic output, undefined symbols, and library order. In both cases, include the NVSHMEM version, relevant CUDA and launcher versions, and target environment details when available.

Keep installation and program-writing requests on their normal routes unless the user explicitly reports a failure in one of the two stages above.

## Examples

Fit assessment:

```text
Use $nvshmem-get-started to assess whether a multi-GPU stencil with GPU-resident halo exchanges is a strong NVSHMEM fit or is simpler with MPI.
```

Ask only for communication facts that affect the decision, then conclude with `strong fit`, `possible fit`, or `likely simpler with another model` and explain the deciding factors.

First-program guidance:

```text
Use $nvshmem-get-started to help me design, compile, and launch my first C++ NVSHMEM ring program without executing anything.
```

Teach the mental model before concrete primitives, present adaptable compile and launch shapes, and keep execution outside scope unless the user separately requests it.

NVSHMEM4Py first-run guidance:

```text
Use $nvshmem-get-started to help me run my first NVSHMEM4Py program with two GPUs, without executing anything.
```

Read the NVSHMEM4Py program guide, collect the Python environment and launcher context, obtain current version-matched Python API and launch details from `$nvshmem-docs`, and provide a user-runnable small-run plan.

<!-- ci trigger -->
