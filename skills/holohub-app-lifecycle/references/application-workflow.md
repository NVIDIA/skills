# HoloHub application workflow

Use this reference for a non-failing application task. Validate every command
against the selected checkout's local help.

## Resolve the workspace

Keep the initial project/data workspace distinct from the HoloHub checkout.
Look only at:

1. a path explicitly named by the user;
2. the initial directory's Git root when it is HoloHub;
3. an exact `holohub/` child of the initial directory;
4. an exact conventional location already known to the agent.

Do not recursively scan the home directory or execute a wrapper merely to test
a candidate. A valid root is one Git checkout containing the executable root
wrapper, `AGENTS.md`, `CONTRIBUTING.md`, application registration, metadata
schemas, and CLI documentation. Record its canonical path, remotes, branch,
full HEAD, and concise status.

Use a current or user-explicit checkout at its existing revision after
reporting provenance and state. Use one auto-discovered checkout only when it
is unambiguous, clean, structurally compatible, and directly official or a
fork with an official upstream. Never fetch, switch, stash, reset, or clean it
without authorization.

For a new app in a clean selected checkout, create an unused task branch from
its current HEAD before editing. A dirty checkout may contain required work;
do not touch overlapping paths or silently substitute another revision.

If no checkout is usable, stop and request explicit authorization to clone,
naming the official repository URL and exact proposed destination. Propose
`<initial-workspace>/holohub` only when that path is absent, the workspace is
writable, and it is not another Git worktree; do not create it before the user
authorizes both the clone and destination. Never clone into the workspace root,
overwrite an existing destination, or follow a symlink outside the workspace.
After authorization, follow the rolling-main fresh-checkout policy in the CLI
contract unless the user requested an immutable revision. Verify the official
remote and resolved full SHA, then create the task branch.

## Define inputs and evidence

Confirm:

- contribution type, language, platform, GPU/driver/SDK, and display/device
  requirements;
- input origin, license or use terms, relevant hashes, and redistribution
  limits;
- finite success condition and machine-readable, visual, or recorded evidence;
- performance question and measurement boundary when benchmarking;
- accuracy, clinical, safety, regulatory, and performance claims that are not
  supported.

Keep public reproducible inputs under the checkout's conventional `data/` only
when repository policy and `git check-ignore -v` prove the exact target is
ignored. Keep private, sensitive, restricted, shared, or irreplaceable input
outside the checkout.

For external input:

- use distinct canonical input and output directories outside the checkout;
- stage to simple shell-safe paths if the wrapper/Docker option cannot
  represent the originals unambiguously;
- mount input read-only and output writable, never mounting `/`, the home
  directory, or the whole workspace;
- preserve required mode Docker options, because a CLI override can replace
  them;
- preview and inspect mounts, devices, capabilities, IPC, display, network,
  credentials, and output paths before execution;
- use a finite headless mode and the narrowest runtime exposure possible.

A read-only mount is not a complete sandbox. Stop or use a disposable approved
environment when residual exposure exceeds the user's authorization.

## Establish the baseline

Read `AGENTS.md`, root and project documentation, `CONTRIBUTING.md`, CLI
documentation, schemas, metadata, and nearby source. Capture:

```bash
git rev-parse HEAD
git status --short --branch
./holohub version --json
./holohub env-info --json
./holohub env-check --json
./holohub status --json
```

Review environment output before sharing it. `env-check --json` exits nonzero
when any capability check fails; preserve and parse the document, but stop the
application workflow only when the failed GPU, Docker, display, device, or
other capability is required by the selected project's documentation or the
requested proof.

## Select local patterns

Inspect two or three strong local applications:

- the closest graph, domain, input, tensor, and operator contract;
- a same-language example with current metadata, CMake, modes, and tests;
- a focused data, Holoviz, recording, or benchmark example when needed.

Use maturity metadata as a filter, not proof. Prefer relevant level 0 or 1
projects, exclude deprecated level 5 projects, and verify source, README,
schema, tests, data/license handling, and registered CMake behavior. Record the
selected paths and exact patterns to adapt. Reuse an existing capability
instead of duplicating it.

## Scaffold only new applications

Use current contribution guidance to select application, operator-plus-demo,
tutorial, or fix. Holoscan CLI 4.5.0 removes the `workflow` project type, and
public HoloHub no longer accepts new `workflows/`.

In one selected wrapper environment, preview template dependency setup before
any create invocation:

```bash
./holohub setup --scripts template --dryrun
```

Inspect the previewed `python3 -m pip install` target and requirements, then
obtain explicit user authorization for the host dependency installation
before running:

```bash
./holohub setup --scripts template
```

If setup fails, stop and hand off that exact failure. After successful setup,
preview the scaffold, review its destination and parent registration, obtain
any repository-required approval, then run the identical create command:

```bash
./holohub create <app> --language <cpp-or-python> -i False --dryrun
./holohub create <app> --language <cpp-or-python> -i False
```

Treat generated README commands as drafts and reconcile them with local help.
Do not scaffold over an existing app. Keep reusable operator code separate from
demo policy.

## Implement and prove the app

Validate current schemas and nearby examples. Use top-level `run` or `modes`,
never both; define a default when multiple modes exist. Inspect discovery:

```bash
./holohub list --json
./holohub modes <app> --language <cpp-or-python> --json
```

Give automated modes a finite frame/message count and nonzero exit on
mismatch. Count at the acceptance boundary and emit a final verdict containing
mode, expected and observed counts, artifact paths, and pass/fail state.
Document graph ports, tensor names/shapes/types/memory domains, CUDA streams,
resources, scheduling conditions, and termination.

Reuse public HoloHub or SDK operators when they meet the brief. Keep data,
models, engines, recordings, build/install output, and generated evidence out
of Git. Update only the needed source, metadata, README, local CMake,
registration, and deterministic tests. Ask before changing shared schemas,
Dockerfiles, or registration macros when repository guidance requires it.

Preview and run matching explicit-language commands:

```bash
./holohub build <app> <mode> --language <cpp-or-python> --dryrun --verbose
./holohub build <app> <mode> --language <cpp-or-python> --verbose
./holohub run <app> <mode> --language <cpp-or-python> --dryrun --verbose
./holohub run <app> <mode> --language <cpp-or-python> --verbose
./holohub test <app> --language <cpp-or-python> --dryrun --verbose
./holohub test <app> --language <cpp-or-python> --verbose
```

Run the shortest finite smoke mode first. Inspect representative frames or a
short clip for visual/recording modes; exit zero alone proves neither pixels
nor encoding.

After a full proof, reuse an unchanged image with `--no-docker-build`. Use
`--no-local-build` only after a matching build or proof that mounted-source
execution needs no regenerated artifacts. Rebuild the image after Dockerfile,
base image, build argument, or setup-script changes.

## Review

Run focused tests and the full wrapper test. Then run `git diff --check` and
final status. In an explicitly selected dirty checkout, run auto-fixing lint
only on task paths and inspect every change. When a commit is requested, apply
the exact candidate diff to a clean disposable checkout at the same revision
and run the full lint workflow required by `AGENTS.md` there before committing;
do not run repository-wide auto-fixes over the user's unrelated dirty work.
For benchmark work, also restore source, remove instrumentation, rebuild
normally, and rerun the finite smoke case. Do not commit or push unless
requested.
