---
name: holoscan-module-lifecycle
description: "Develop external Holoscan Modules with the Holoscan CLI and use them in Holoscan applications: scaffold, test, Python bindings, DEB/WHEEL packaging, and clean-consumer proof."
license: Apache-2.0
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  compatibility: "Holoscan CLI 5.x (verified with 5.0.0); Python 3.12–3.14; packaged template requires Holoscan SDK 5.0.0 or later"
  github-url: "https://github.com/nvidia-holoscan/holoscan-applications"
  tags:
    - holoscan
    - modules
---

# Holoscan Module lifecycle

## Purpose

Take a reusable Holoscan Module from layout and API contract through producer,
consumer, and clean artifact-install proof. Use `uv run holoscan` in a
standalone module, or `holoscan` in its activated pip environment.

A Holoscan Module extends `holoscan::core` with operators, resources, and
helpers for a domain capability, plus examples and tests. Core modules ship
with the SDK; external modules live in their own repositories, and
applications, including those in nvidia-holoscan/holoscan-applications,
consume their installed packages. This skill covers external Holoscan Module
development and consumption. For SDK core-module implementation, follow the
selected SDK checkout’s `modules/DEVELOPER.md` and module-specific guide; do
not apply the external CLI template or packaging workflow to SDK core modules.

## Inputs

Required: the API contract, workspace, identities, SDK floor, license,
maintainer, package formats, and acceptance checks, from the request or an
authoritative repository source. Optional: a consuming application. If
consequential release metadata remains unknown, stop before scaffolding or
packaging rather than guessing.

- a reusable producer/consumer API contract rather than an ordinary app;
- the user-provided standalone module repository, or the parent directory for
  a new one;
- language, module/operator/package identities, minimum SDK version, license,
  maintainer details, and requested package formats;
- the consuming application, if any, such as one in a holoscan-applications
  checkout;
- finite producer, consumer, and artifact acceptance checks.

First honor any request for planning only (for example, “do not run commands”);
preserve that restriction when routing. Check the session’s available skill
catalog for the destination skill. Route a concrete failing or wrong
`holoscan` command to `holoscan-app-debug-build-run`. If the matching skill is
unavailable, preserve the handoff context and name the skill to install
instead of broadening this workflow. An ordinary application without a
reusable module API is outside this skill; follow the target repository's own
guidance, such as `README.md` and `CONTRIBUTING.md` in holoscan-applications.

## Prerequisites

- For module concepts and producer and consumer walkthroughs, use the
  [Holoscan Modules tutorials](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules):
  [create a module](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules/create-a-module)
  and [use a module](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules/use-a-module).
  Recheck their version-sensitive commands against the selected checkout and
  CLI contract. If they are unavailable, use the summary above, the bundled
  references, and the selected CLI’s packaged template and help; identify the
  unavailable tutorial and do not claim to have checked it.
- Always read the [Holoscan CLI contract](references/holoscan-cli-contract.md)
  and [Holoscan CLI configuration](references/holoscan-cli-configuration.md).
- Read [module development](references/module-development.md) for layout,
  naming, scaffold, metadata, implementation, build, and test scope.
- Read [consumers and packaging](references/module-consumer-packaging.md) when
  a consuming app, Python binding consumer, or binary artifact is in scope.

The module's own `README.md`, `DEVELOPER.md`, and `AGENTS.md`, a consuming
checkout's guidance, local help, schemas, templates, tutorials, and nearby
examples are the live technical authority where they do not conflict with
user, system, or safety constraints.

## Instructions

For a planning-only request, return the proposed workflow and proof requirements
without executing commands or changing files. Otherwise continue below.

At any step, a failing effect-bearing CLI command ends this happy path;
follow Troubleshooting with its exact context. Parse read-only diagnostic
results and stop only when a failed capability is required by the module,
consumer, packaging environment, or requested proof. For example, a failed
Docker capability blocks a container build; an unavailable display need not
block an explicitly headless consumer.

1. **Preserve and orient.** Record full HEAD, concise status, selected CLI
   version/environment, and discovery output. Preserve the checkout revision
   and unrelated work.
2. **Choose the layout.** Use a standalone `holoscan-<name>` repository with
   its own demo applications. holoscan-applications keeps applications only
   and directs reusable operators and Modules to independent repositories.
3. **Fix identities before files.** Keep display name, `snake_case`
   module/namespace, `holoscan-<kebab-case>` repository/package, and
   `snake_case_op` operator slug consistent.
4. **Scaffold with the packaged template.** Follow the uv-first creation
   sequence in the development reference, from a parent directory outside any
   application checkout. Preview and run the same non-interactive
   `holoscan create` command with explicit language (`cpp`, or `cpp python` to
   add Python bindings) and agreed metadata. Always specify
   `--context holoscan_version=<agreed-sdk-floor>` to override configured or
   template defaults; the template requires 5.0.0 or later, and its preview
   does not check that floor. Examples explicitly select 5.0.0; use the agreed
   SDK floor for the actual module. No launcher or holoscan-applications
   checkout is required. Preserve existing Git state;
   reject populated destinations. Inspect generated files and Git status
   before continuing.
5. **Implement the smallest public contract.** Validate metadata, expose one
   useful operator/API, add deterministic tests, and add a finite demo that
   consumes the same public surface as another project. Keep demo policy out of
   the reusable operator.
6. **Build and test honestly.** Preview and run container-first module and demo
   commands, passing `--language cpp` or `--language python` when a project
   declares both. The module's `test` runs its repository-wide suite; combine
   it with focused pytest or CTest targets and report the scope.
7. **Prove a real consumer.** An application consumes the installed package
   with `find_package(holoscan-<name> CONFIG REQUIRED)` and
   `holoscan::<module_slug>`. In holoscan-applications, a metadata dependency
   alone neither fetches nor builds the module; follow the
   [application consumer procedure](references/module-consumer-packaging.md#consume-from-an-application-in-holoscan-applications)
   and do not change the shared root CMake without maintainer agreement. For
   Python bindings, import from the exact build tree or install the
   `holoscan-<name>-python` component into a task-owned prefix; follow the
   [Python consumer cleanup procedure](references/module-consumer-packaging.md#python-bindings-consumer):
   record the interpreter, build directory, prefix, and installed paths, always
   remove the task-owned prefix after the consumer attempt, and verify it is
   absent. The packaged template stages no editable-import hooks, so do not
   substitute `install --dev`. If cleanup fails, stop using that environment
   and report incomplete cleanup; a missing debugging skill does not permit
   continuing to package or publish. Discard a task-owned disposable
   environment only under existing cleanup authorization; a shared environment
   needs the owner’s repair decision.
8. **Package only after behavior proof.** Preview and build the requested
   artifacts, inspect identity/metadata/dependencies/payload, and install each
   format in a clean artifact-only consumer with an import/link and finite
   smoke check. The packaged template produces a C++-only DEB and no wheel;
   request WHEEL only for a module whose own `pyproject.toml` builds one. For
   DEB, verify the declared `holoscan` Debian dependency in the consumer’s
   package database: a successful Python import or C++ build is insufficient.
   If it is absent, report DEB installation unverified and follow the
   [DEB prerequisite guidance](references/module-consumer-packaging.md#package-and-inspect);
   do not force dependency checks or claim archive extraction proves installability.
9. **Validate and hand off.** Rerun producer and clean-consumer checks,
   `git diff --check`, and final status. In a dirty checkout, restrict
   auto-fixing lint to task paths; before a requested commit, validate the
   exact candidate change with the full lint required by that repository's
   guidance, in a clean disposable checkout. If the required commands cannot
   be determined, report lint unverified. Confirm no generated artifacts or
   install prefixes are staged. Do not commit, push, upload, publish, or
   release unless requested.

## Troubleshooting

If a CLI command fails, stop the happy path and hand off its exact command,
revision, dirty state, inputs, and observed result to
`holoscan-app-debug-build-run`.

## Examples

- Scaffold and package a reusable operator with clean-consumer proof: use this
  skill.
- Package an ordinary app or diagnose a failing build: route out of this skill.

## Limitations

- Preserve unrelated work. Do not reset, clean, remove unrelated files, change
  host configuration, broaden privileges, commit, push, upload, publish, or
  release without authorization.
- Never prefix `holoscan` with `sudo`; obtain approval for `holoscan setup`,
  host packages, host-local execution, root containers, devices/capabilities,
  or permission changes.
- Keep build, data, install prefixes, and package output out of commits.
- Protect credentials, proprietary source, patient data, private media, and
  identifying metadata.
- Claim support, reproducibility, installability, performance, or safety only
  for the producer, artifacts, and clean-consumer evidence actually collected.

## Output

Return layout and identities, public API/demo, metadata and registrations,
preview and real command results, honest test scope, consumer proof, artifact
identity/contents/install results, licensing and publication limits, and final
worktree state.

For a planning-only request, return the proposed order, authoritative metadata
sources, approval boundaries, and proof requirements without claiming results.
