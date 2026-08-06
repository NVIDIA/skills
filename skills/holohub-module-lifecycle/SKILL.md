---
name: holohub-module-lifecycle
description: "Use for reusable Holoscan Module work with ./holohub: scaffold, tests, editable install, DEB/WHEEL packaging, and clean-consumer proof."
license: Apache-2.0
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  compatibility: "holoscan-cli>=4.5.0"
  github-url: "https://github.com/nvidia-holoscan/holohub"
  tags:
    - holoscan
    - holohub
    - modules
---

# HoloHub Holoscan Module lifecycle

## Purpose

Take a reusable Holoscan Module from layout and API contract through producer,
consumer, and clean artifact-install proof.

## Inputs

Require the API contract, checkout, layout, identities, SDK floor, license,
maintainer, package formats, and acceptance checks from the request or an
authoritative repository source. If consequential release metadata remains
unknown, stop before scaffolding or packaging rather than guessing.

- a reusable producer/consumer API contract rather than an ordinary app;
- the user-provided HoloHub checkout;
- external self-contained repository or in-tree descriptor layout;
- language, module/operator/package identities, minimum SDK version, license,
  maintainer details, and requested package formats;
- finite producer, consumer, and artifact acceptance checks.

Route an ordinary app to `holohub-app-lifecycle` and a concrete failing or
wrong `./holohub` command to `holohub-debug-build-run`. If the matching skill
is unavailable, preserve the handoff context and name the skill to install
instead of broadening this workflow.

## Prerequisites

- Always read the [CLI contract](references/holohub-cli-contract.md).
- Read [module development](references/module-development.md) for layout,
  naming, scaffold, metadata, implementation, build, and test scope.
- Read [consumer and packaging](references/module-consumer-packaging.md) when a
  consuming app, editable install, or binary artifact is in scope.
- Use HoloHub's current [create-a-module](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/holoscan-modules/create-a-module)
  and [use-a-module](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/holoscan-modules/use-a-module)
  tutorials as producer and consumer walkthroughs, then recheck
  version-sensitive commands against the selected checkout and CLI contract.

The selected checkout's `AGENTS.md`, local help, module guidance, schemas,
templates, tutorials, and nearby examples are the live technical authority
where they do not conflict with user, system, or safety constraints.

## Instructions

At any step, a failing effect-bearing wrapper command ends this happy path;
follow Troubleshooting with its exact context. Parse read-only diagnostic
results and stop only when a failed capability is required by the module,
consumer, packaging environment, or requested proof.

1. **Preserve and orient.** Record full HEAD, concise status, local wrapper
   version/environment, and discovery output. Preserve the checkout revision
   and unrelated work.
2. **Choose the layout.** Use an external `holoscan-<name>` repository for
   independent release or an in-tree descriptor under `modules/` for code
   maintained with HoloHub.
3. **Fix identities before files.** Keep display name, `snake_case`
   module/namespace, `holoscan-<kebab-case>` repository/package, and
   `snake_case_op` operator slug consistent.
4. **Scaffold external modules safely.** Preview template setup, inspect its
   host dependency installation, and obtain explicit user authorization before
   the real setup. Only after setup succeeds, preview and run the fully
   specified non-interactive create command, treating preview as potentially
   mutating. Require an existing parent, inspect the generated staged Git
   repository, and never push it without a request.
5. **Implement the smallest public contract.** Validate metadata, expose one
   useful operator/API, add deterministic tests, and add a finite demo that
   consumes the same public surface as another project. Keep demo policy out of
   the reusable operator.
6. **Build and test honestly.** Preview and run container-first module and demo
   commands. In HoloHub, test every declared operator and demo because
   `test <module>` is not module-scoped. In a generated repository, combine its
   repository-wide test with focused pytest or CTest targets.
7. **Prove a real consumer.** Declare the exact metadata dependency and an
   immutable full SHA for external source. Use a mounted source override for
   fast iteration or an editable install only in the environment reported by
   `env-info --json`. Record the hook and remove it even if a later step fails;
   if removal fails, report the exact environment and hook state.
8. **Package only after behavior proof.** Preview and build requested DEB/WHEEL
   artifacts, inspect identity/metadata/dependencies/payload, and install each
   format in a clean artifact-only consumer with an import/link and finite
   smoke check.
9. **Validate and hand off.** Rerun producer and clean-consumer checks,
   `git diff --check`, and final status. In a dirty checkout, restrict
   auto-fixing lint to task paths; before a requested commit, validate the
   exact candidate change with the repository-required full lint in a clean
   disposable checkout. Confirm no generated artifacts or editable hooks are
   staged. Do not commit, push, upload, publish, or release unless requested.

## Troubleshooting

If a wrapper command fails, stop the happy path and hand off its exact command,
revision, dirty state, inputs, and observed result to
`holohub-debug-build-run`.

## Examples

- Scaffold and package a reusable operator with clean-consumer proof: use this
  skill.
- Package an ordinary app or diagnose a failing build: route out of this skill.

## Limitations

- Preserve unrelated work. Do not reset, clean, remove unrelated files, change
  host configuration, broaden privileges, commit, push, upload, publish, or
  release without authorization.
- Never run `sudo ./holohub`; obtain approval for host packages, host-local
  execution, root containers, devices/capabilities, or permission changes.
- Keep build, data, install, package output, and editable hooks out of commits.
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
