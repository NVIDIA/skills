# Module consumer iteration and packaging

## Declare the dependency

An application declares modules under `application.dependencies.modules[]`; a
module uses `module.dependencies[]`. Each entry names the dependency exactly as
the producer's `metadata.json:module.name`, declares `provides_operators`, and,
for external source, supplies `source.git_url` plus an immutable 40-character
commit SHA in `source.ref`. Validate against current schemas and real targets.

For an immutable declared dependency, prefer the ordinary top-level wrapper
workflow and keep its preview and action identical except for `--dryrun`:

```bash
./holohub run <consumer-app> <mode> --language <cpp-or-python> --dryrun --verbose
./holohub run <consumer-app> <mode> --language <cpp-or-python> --verbose
```

## Mounted-source iteration

Use this advanced path only when the consumer must exercise a local module
working tree instead of its immutable declared source.

The current `./holohub` source override is
`HOLOSCAN_CLI_LOCAL_<SANITIZED_MODULE_NAME>`. Derive it from the consuming
metadata dependency entry's `name`: replace non-alphanumeric runs with `_`,
trim leading/trailing underscores, and uppercase. For example,
`holoscan-my-module` becomes `HOLOSCAN_CLI_LOCAL_HOLOSCAN_MY_MODULE`.
The public consolidated CLI defines this mapping in
[`external_resolver.py`](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/src/holoscan_cli/utils/external_resolver.py).
Confirm the selected CLI version with `./holohub version --json` before
relying on version-sensitive override behavior.

A host export is not automatically forwarded into the normal project
container, and an outer dry run does not execute the inner resolver. The
nesting below is therefore a two-layer preview requirement, not a preference
for nested commands. Mount the source and set the override to its container
path inside one workflow. First preview only the outer container launch:

```bash
./holohub run-container <consumer-app> <mode> --language <cpp-or-python> \
  --add-volume /absolute/path/holoscan-my-module \
  --dryrun --verbose -- \
  'set -eu; module_root=/workspace/volumes/holoscan-my-module; export HOLOSCAN_CLI_LOCAL_HOLOSCAN_MY_MODULE="$module_root"; test -f "$module_root/metadata.json"; ./holohub run <consumer-app> <mode> --local --language <cpp-or-python> --dryrun --verbose'
```

The outer dry run does not execute its quoted shell. After reviewing it, launch
the previewed container with the same mounts and options so the single nested
run dry run executes and validates the mounted override:

```bash
./holohub run-container <consumer-app> <mode> --language <cpp-or-python> \
  --add-volume /absolute/path/holoscan-my-module --verbose -- \
  'set -eu; module_root=/workspace/volumes/holoscan-my-module; export HOLOSCAN_CLI_LOCAL_HOLOSCAN_MY_MODULE="$module_root"; test -f "$module_root/metadata.json"; ./holohub run <consumer-app> <mode> --local --language <cpp-or-python> --dryrun --verbose'
```

Require the inner preview to accept the override without a fallback. If its
output does not identify the selected source, do not claim that it did. After a
fresh review, repeat the same outer invocation with only the inner `--dryrun`
removed. Require configure/build evidence that uses the mounted path and a
finite consumer result. If using `--docker-opts`, preserve required mode
options because the override flag replaces rather than extends them.

## Editable install

`install --dev` writes an editable hook immediately into the environment
selected by `./holohub`; it is not the normal project-container install path.
Use it only when that exact environment is the consumer:

```bash
./holohub env-info --json
./holohub install <module> --dev --build-dir <exact-build-dir> --dryrun --verbose
./holohub install <module> --dev --build-dir <exact-build-dir> --verbose
# verify import with the exact environment reported by env-info --json
./holohub install <module> --dev --uninstall --dryrun --verbose
./holohub install <module> --dev --uninstall --verbose
```

Name the module and exact build directory in multi-project trees. Omitting the
project can affect every discovered hook. A host hook does not automatically
exist inside an ephemeral project container.

## Package and inspect

Package only after producer and consumer behavior is proved:

```bash
./holohub package <module> --language <cpp-or-python> \
  --pkg-generator DEB,WHEEL --dryrun --verbose
./holohub package <module> --language <cpp-or-python> \
  --pkg-generator DEB,WHEEL --verbose
```

With holoscan-cli 4.5.0 or newer, the package dry run does not create package
directories. It also honors the explicit `<module>` argument; a real package
action fails when that module produces no CPack configuration. Record actual
artifact paths from the action. Inspect wheel file lists and
`*.dist-info/METADATA`; compare name, version, dependencies, and namespace
with module metadata. Inspect `dpkg-deb --info` and `--contents`; compare
package identity, dependencies, and payload paths.

## Clean consumer proof

Install every requested format in a separate disposable consumer inside a
wrapper-managed container. Import/link the declared namespace and run a finite
demo or minimal smoke. If platform, policy, dependency, or privilege prevents
this, mark that format unverified; package exit zero is insufficient.

Prefer an artifact-only environment. Otherwise use a neutral directory such as
`/tmp`, unset local-module overrides, disable user-site packages, remove
checkout/build paths from `PYTHONPATH` while preserving documented SDK paths,
and record the imported module's resolved file. It must come from the fresh
artifact, not the producer checkout, `/workspace`, or a build tree.

Reconcile `binary_packages.install_commands` with what worked. Do not upload to
PyPI, Debian repositories, releases, or registries without authorization and a
redistribution review.

## Review validation

Rerun focused producer tests and clean consumer smoke. Run `git diff --check`,
inspect the complete diff/status, and ensure packages, staged installs, and
editable hooks are not staged.

Use the current `AGENTS.md` lint workflow in HoloHub and the generated
repository's own guidance externally. In a dirty checkout, scope auto-fixing
lint to task paths and validate an exact requested commit with the required
full lint in a clean disposable checkout. Preview supported lint commands,
inspect auto-fixes, and do not commit, push, publish, or release unless
requested.
