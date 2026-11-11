# Holoscan CLI debugging workflow

Use this after capturing one exact failing command.

## Baseline and reproduction

Read the applicable repository guidance (`README.md`, `CONTRIBUTING.md`, and
`docs/ci.md` in holoscan-applications, or a generated Module's `README.md`,
`DEVELOPER.md`, and `AGENTS.md`), plus the project's metadata, schemas, and
local help. Record project, mode, language, build type, base image, SDK root,
input, display/headless state, devices, output path, full HEAD, and concise
status.

If the request is planning-only or forbids execution, return only the proposed
diagnostic command order, inspection points, approval boundaries, and proof
requirements. Do not invoke `holoscan` or any other command, edit files, clear
caches or artifacts, install or repair an environment, or perform setup.
Continue below only for an execution-authorized request.

The command runs whichever `holoscan` comes first on `PATH`: a checkout
`.venv`, a `uv tool` installation, a module's `uv run` environment, or, inside
a container, the image's own installation. Use `version --json` and
`env-info --json` to identify it and compare it with the committed pin,
`env-check --json` for host/GPU/Docker/display/device/SDK health, and
`status --json` for image/build facts. Compare outer and container
`env-info --json` when their behavior differs.

## Sharing diagnostics

Prepare the shareable excerpt from diagnostic output already captured for the
reproduction; this step does not authorize any additional file or environment
inspection. Keep the original evidence local and unchanged.

Start with an empty excerpt. Include only the CLI/Python versions, exit status,
configuration key names, error class, and the minimal error text needed to
explain the failure. Environment values and authentication material are excluded
by default. In supplied JSON or log text, replace values of fields such as
`TOKEN`, `SECRET`, `PASSWORD`, `API_KEY`, `AUTHORIZATION`, and `COOKIE` with
`[REDACTED]`; exclude their contents even when they appear in nested objects or
command arguments. Share URLs only without userinfo or query values. A
nonsecret field name does not establish that its value is safe to disclose.

Replace private paths, hostnames, registry routes, and patient/media identifiers
with consistent placeholders unless disclosure is authorized for the destination.
Review every retained value in the excerpt; omit uncertain values and report
only their names and roles. Do not publish raw diagnostic output or send it to
an external service for redaction. If this minimal excerpt cannot explain the
failure safely, request an approved private destination before sharing more.

## Controlled reproduction

Preview the original command without changing its effect-bearing arguments.
Inspect the selected root, project/mode/language, base image, SDK, mounts,
devices, environment, workdir, child commands, and output; dry runs hide
`--docker-opts` and CMake option values, so compare those with the supplied
arguments and metadata. If the command performs cache cleanup, including
`clear-cache` or `test --clear-cache`, review the resolved affected paths and
obtain explicit user authorization before reproduction; the original failure
report is not approval for that cleanup. Reproduce once without edits and
capture the first causal error rather than only shutdown noise. Bound a hanging
reproduction with an external timeout derived from the recorded hang boundary.
Use the normal runtime or last observed progress time plus a stated grace
period, subject to the user's runtime limit. If neither is known, agree on a
limit before rerunning a potentially hanging mutation. On Linux, a disposable
reproduction can use `timeout --signal=TERM --kill-after=30s <limit> <command>`;
record the limit, signal, and exit status (124 normally means deadline expiry).
A timeout does not guarantee rollback or stop a detached container. Inspect
remaining processes/containers and incomplete outputs; stop only task-owned
processes within the authorization already given. Do not reuse partially written
artifacts as proof. For operations whose interruption could damage shared state,
use existing logs until a disposable reproduction is available.

If the failure does not reproduce, compare revision, dirty state, input hashes,
image, SDK, cache, display/devices, and environment. Report the mismatch
instead of manufacturing a change.

## Layer matrix

| Layer | Check first |
| --- | --- |
| CLI environment | `holoscan` on `PATH`, CLI version versus `requirements.txt` or module pins, Python version, root discovery |
| Create/setup/lint | Template selection, unattended inputs, privileges, pre-commit environment, hook auto-fixes |
| Discovery/metadata | Project/language/mode, schema, registration, `list --json`, `modes --json` |
| Host | `env-check --json`, platform, driver, Docker, disk, display, devices/permissions |
| Image/container | Base image and its SDK, Dockerfile, image age and its CLI pin, mounts, workdir, user, display |
| Configure/build | Selected SDK root, first CMake/compiler error, targets, include/link paths, bindings, cache |
| Test | Selected CTest driver, `--cmake-options`, language, xvfb, registration, first failure |
| Module install/package | Selected Python, build dir/hook, metadata, generator, artifact/consumer |
| Data/model | Origin/license, exact path/hash, container path translation, permissions, format, engine compatibility |
| Application | Ports, tensors, conditions, resources, scheduler, termination, CUDA streams |
| Visual/recording | Mode, Holoviz layers, render path, display/headless, decoded pixels |
| Benchmark | Healthy normal app, instrumentation, runner command, logs, analyzer, restoration |

Test the boundary between the two most plausible layers before editing. For
example, compare the SDK named in the dry run with the application's SDK range
to separate SDK selection from a source bug, inspect mounts to separate
host-data absence from an application path bug, or decode a recording to
separate successful execution from blank output.

## Observe, hypothesize, and fix

- Prefer existing logs, finite verdicts, focused tests, and artifacts.
- Increase Holoscan log level only for the smallest reproduction and remove the
  temporary verbosity.
- Enter the project environment through a previewed `holoscan run-container`;
  do not run project tools or debuggers directly on the host by default.
- Obtain approval for host execution, debugger capabilities, core dumps, or
  device/permission changes.
- Decode representative frames or a short clip for visual failures.

State one falsifiable hypothesis tied to the selected layer. Choose the least
invasive observation that distinguishes it, change one variable, record the
result, and revert diagnostic-only edits. Read source only after ownership is
narrowed. Fix the cause without unrelated refactoring, blanket suppression,
broad dependency upgrades, or public-contract changes; add a focused
deterministic regression test when possible.

## Cleanup and proof

Treat cleanup as a separate hypothesis. If evidence proves stale build state,
preview only the required scope:

```bash
holoscan clear-cache --build --dryrun
```

Even this scope removes the build folders of every project under the root, and
plain `clear-cache` also removes data and install folders. Review resolved
paths and obtain explicit approval before the action.
Never manually delete build, data, or install trees, and never prefix the CLI
with `sudo`.

For a mutating reproduction, preview the post-fix identical command before
re-running it and the nearest focused test. Inspect corrected frames, consumed
packages, or benchmark restoration as appropriate. Remove diagnostic-only
changes, run `git diff --check`, compare final status with the baseline, and
follow the current checkout's lint workflow before a requested commit. In a
dirty checkout, keep auto-fixing lint scoped to task paths, for example
`holoscan lint <path>`, and run the required full lint for an exact commit
candidate in a clean disposable checkout.

In holoscan-applications, rerun `python .github/scripts/holoscan_ci.py` and the
hosted SDK-free tests after metadata, mode, registration, or test changes;
they do not replace the SDK build proof. In a standalone Module, `test` runs
the repository-wide suite; validate declared operators, demo apps, and
consumer behavior as well.
