# Module consumers and packaging

Use `uv run holoscan` in a standalone module (or `holoscan` in an activated
pip environment). A consuming application uses the module's installed
package; in holoscan-applications, run `holoscan` from that checkout's root.

## Declare the dependency

A module declares dependencies under `module.dependencies[]`, and an
application can declare modules under `application.dependencies.modules[]`.
Each entry names the dependency exactly as the producer's
`metadata.json:module.name`, declares `provides_operators`, and, for external
source, supplies `source.git_url` plus an immutable 40-character commit SHA in
`source.ref`; tags and branches are rejected. The CLI parses these entries on
every build and run, including dry runs, and fails on one without that
source, while metadata schema validation does not check them. Validate
against current schemas and real targets. A CMake consumer of an installed
module uses `find_package(holoscan-<name> CONFIG REQUIRED)` and links the
aggregate `holoscan::<module_slug>` target or an individual operator target.

During `build` and `run`, the CLI turns declared source dependencies into
`external_operators_manifest.cmake` in the build directory and honors a
`HOLOSCAN_CLI_LOCAL_<SANITIZED_MODULE_NAME>` source override there. Derive the
key from the dependency's `name`: replace non-alphanumeric runs with `_`, trim
leading/trailing underscores, and uppercase, so `holoscan-my-module` becomes
`HOLOSCAN_CLI_LOCAL_HOLOSCAN_MY_MODULE`. Only a consuming root
`CMakeLists.txt` that includes that manifest fetches and builds the module.

## Consume from an application in holoscan-applications

holoscan-applications' root `CMakeLists.txt` does not include the manifest, so
neither a declared dependency nor a local override fetches or builds the
module there. Do not add FetchContent or change the shared root CMake without
maintainer agreement. Prove consumption from an installed package built
against the same SDK and image family as the application:

1. Create an empty task-owned prefix directory and install the module's
   default (C++) component into it from the exact producer build directory,
   in the environment that produced the build. A `--local` build installs on
   the host. A default container build records container paths, so install it
   inside the module's development container with the prefix mounted, after
   previewing the launch:

   ```bash
   uv run holoscan run-container --add-volume <task-owned-prefix> --dryrun --verbose -- \
     'cmake --install <container-build-dir> --prefix /workspace/volumes/<prefix-name>'
   ```

   Alternatively, install the module's DEB in a disposable consumer image with
   approval.
2. In the application's own CMake, call
   `find_package(holoscan-<name> CONFIG REQUIRED)` and link
   `holoscan::<module_slug>` or an operator target. Record the dependency in
   the application's README, and in its metadata only as a complete source
   entry.
3. Point configure at the package without replacing the CLI's SDK prefix. In
   container mode, mount the prefix and pass its container path; keep preview
   and action identical except for `--dryrun`:

   ```bash
   holoscan run <consumer-app> <mode> --language cpp \
     --add-volume <task-owned-prefix> \
     --configure-args='-Dholoscan-<name>_DIR=/workspace/volumes/<prefix-name>/lib/cmake/holoscan-<name>' \
     --dryrun --verbose
   ```

   Dry runs hide CMake option values. In the real run, confirm from configure
   output that the package resolved from the mounted prefix, then require a
   finite consumer result.
4. Remove the task-owned prefix afterwards with the guaranteed cleanup below.

## Python bindings consumer

A module created with `--language cpp python` exposes its C++ operators to
Python as `holoscan_<snake_name>`. The packaged template stages no
editable-import hooks: `install --dev` installs only hooks that a build staged
as `holoscan_<snake_name>_dev.py`, so for this template its preview can
announce an install that the real action rejects. Do not use it here.

Build first, then prove Python consumption in one of two ways:

- from the build tree, with only that build's `python/lib` directory and the
  SDK's Python paths on `PYTHONPATH`; or
- from the `holoscan-<name>-python` CMake component installed into a fresh
  task-owned prefix:

```bash
cmake --install <exact-build-dir> --component holoscan-<name>-python --prefix <task-owned-prefix>
```

The component is excluded from the default install and from the DEB. Run the
install in the environment that produced the build, as for the C++ component
above. Before installing, record the interpreter executable, the exact build
directory, and the prefix; confirm the prefix is absent or empty and is not
shared with other work. Record the installed paths that `cmake --install`
reports.

Use a `finally` block or equivalent guaranteed cleanup path around the consumer
attempt. Remove only the task-owned prefix, even if the consumer failed.
Cleanup succeeds only when removal returns zero and the prefix no longer
exists; verify this with `Path.exists()` or an equivalent filesystem check. A
successful removal of a different path is not proof.

On a removal failure or a remaining prefix, stop consumer/package work and
report the interpreter, prefix and its state, and the error in a redacted local
handoff. Do not add privileges or delete unrelated files. Route the
failure to `holoscan-app-debug-build-run` if available. If it is unavailable,
end with cleanup incomplete and the affected environment unsuitable for
further validation; do not claim restoration. A task-owned disposable
environment can be discarded under the existing cleanup authorization, but a
shared environment requires the owner's repair decision.

## Package and inspect

Package only after producer and consumer behavior is proved:

```bash
uv run holoscan package <module> --pkg-generator DEB --dryrun --verbose
uv run holoscan package <module> --pkg-generator DEB --verbose
```

The packaged template produces only a DEB. It contains the C++ operators,
installs under `/usr` when built against SDK 5, declares
`holoscan (>= <minimum-sdk-version>)`, and excludes the Python bindings; the
template builds no wheel. A module whose own `pyproject.toml` builds a wheel
can request `--pkg-generator DEB,WHEEL`; `WHEEL` also needs the Python `build`
package.

With holoscan-cli 5.0.0, the package dry run does not create package
directories. It also honors the explicit `<module>` argument; a real package
action fails when that module produces no CPack configuration. Record actual
artifact paths from the action. Inspect wheel file lists and
`*.dist-info/METADATA`; compare name, version, dependencies, and namespace
with module metadata. Inspect `dpkg-deb --info` and `--contents`; compare
package identity, dependencies, and payload paths.

## Clean consumer proof

Install every requested format in a separate disposable consumer inside a
SDK-equipped test environment. Import/link the declared namespace and run a finite
demo or minimal smoke. If platform, policy, dependency, or privilege prevents
this, mark that format unverified; package exit zero is insufficient.
Installing a DEB changes the package database; obtain approval before doing so
outside a disposable consumer.

For DEB, first inspect `dpkg-deb --info` and check each dependency with
`dpkg-query` in the clean consumer. The 5.0.0 template declares
`holoscan (>= <minimum-sdk-version>)`, the Debian alias of `holoscan-cuda-13`
on SDK 5.0 platforms. An SDK container or Python wheel can provide working
Holoscan libraries without registering that Debian package; SDK 4.6.0's CUDA 13
image exhibits this distinction. `dpkg -i` then leaves the module
unconfigured. Use a consumer with the actual declared SDK Debian dependency
installed, or report DEB installation unverified. Do not force dependencies,
fabricate a provider package, or equate archive extraction with a successful
package installation.

Prefer an artifact-only environment. For a subprocess-based consumer, use a
neutral working directory such as `/tmp` and a process-local environment with
user-site packages disabled and only documented SDK paths on `PYTHONPATH`.
Producer checkout/build paths and local-module overrides must be absent from
that subprocess environment. Keep the parent environment, skill files, and
repository configuration unchanged. Record the imported module's resolved
file: it must belong to the fresh artifact, not the producer checkout,
`/workspace`, or a build tree.

Reconcile `binary_packages.install_commands` with what worked. Do not upload to
PyPI, Debian repositories, releases, or registries without authorization and a
redistribution review.

## Review validation

Rerun focused producer tests and clean consumer smoke. Run `git diff --check`,
inspect the complete diff/status, and ensure packages, install prefixes, and
build trees are not staged.

Use each repository's own lint workflow: the generated module's guidance, and
`holoscan lint` plus DCO sign-off for a consumer in holoscan-applications. In a
dirty checkout, scope auto-fixing lint to task paths and validate an exact
requested commit with the required full lint in a clean disposable checkout.
Preview supported lint commands, inspect auto-fixes, and do not commit, push,
publish, or release unless requested.
