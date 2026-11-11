# External Holoscan Module development

## Layout and naming

Use a standalone `holoscan-<name>` repository for each external module, with
its demo applications and tests. holoscan-applications keeps end-to-end
applications only and directs reusable operators and Modules to independent
repositories; its applications consume a module's installed package. Keep the
display name, `snake_case` namespace, `holoscan-<kebab-case>`
package/repository, and `snake_case_op` operator consistent. The packaged
template rejects module slugs owned by SDK core modules: `core`, `vision`,
`visualization`, and `inference`.

## Standalone scaffold: uv first

The following commands pin `holoscan-cli` 5.0.0 from PyPI and work with
Python 3.12–3.14; see "Python and CLI version" in the Holoscan CLI
configuration reference.
`uvx` supplies the isolated creation environment; `uv run` uses the generated
project's pinned development group. There is no need to pin a local Python
patch version or activate an environment. Start in a parent directory outside
any other project checkout and use the CLI's packaged Module template:

```bash
uvx --from 'holoscan-cli[create]==5.0.0' holoscan create my-module --language cpp --interactive false --context holoscan_version=5.0.0 --dryrun
uvx --from 'holoscan-cli[create]==5.0.0' holoscan create my-module --language cpp --interactive false --context holoscan_version=5.0.0
cd holoscan-my-module
uv run holoscan list --json
```

The template implements operators in C++ for SDK 5.0.0 or later with CUDA 13,
CMake 4.0 or later, and C++20. Use `--language cpp python` in both commands to
add Python bindings that register the C++ operators, a Python version of the
example, and a pytest suite. `--language python` alone is rejected; the
template does not generate Python-implemented operators, editable-import
hooks, or wheels.

Replace the illustrative name, language, and SDK minimum with the requested
values in both commands. Supply repeated `--context key=value` options for
release metadata: `project_name`, `module_slug`, `module_repo_name`,
`operator_slug`, `full_name`, `affiliation`, `version`, `description`,
and `contact_email`. `_license` must remain `Apache-2.0` to match the
generated license. Inspect derived defaults rather than publishing
placeholder metadata. Read the preview before the real creation; it does not
run the template's validation, and uv may still download tooling during a CLI
dry run.

The default output is a new `holoscan-<name>` directory under the current
parent. Use `--directory` to choose another parent. Missing, empty, or `.git`-only
destinations are supported, including fresh worktrees; populated destinations
are rejected. Outside an existing worktree, creation initializes Git on `main`
and stages the scaffold without committing. Preserve Git history and inspect
generated status. Do not create or copy a launcher script.

The generated `requirements-cli.txt`, development dependency group, and
pre-commit configuration pin the creating CLI version for host and container
use. The generated `pyproject.toml` routes `holoscan-cli` to NVIDIA's package
index for `uv run`, and `requirements-cli.txt` documents the matching
`--extra-index-url`; the 5.0.0 wheel is identical on PyPI and that index. Keep
these pins consistent and change them together only with authorization.

### Fallback: .venv with pip

If uv is unavailable, create a Python 3.12 virtual environment in the parent,
activate it, and install the creation extra:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install 'holoscan-cli[create]==5.0.0'
```

Use the same preview/action `holoscan create` arguments above without the
`uvx` prefix. Inside the generated module, use its own environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --extra-index-url https://pypi.nvidia.com -r requirements-cli.txt
```

Then omit `uv run` from development commands. Do not install into system Python.

## Iterate on the module

```bash
uv run holoscan build holoscan-my-module --dryrun --verbose
uv run holoscan build holoscan-my-module --verbose
uv run holoscan run my_module_pipeline --language cpp --dryrun --verbose
uv run holoscan run my_module_pipeline --language cpp --verbose
uv run holoscan test --dryrun --verbose
uv run holoscan test --verbose
```

Use actual names from `list` and `modes`. With Python bindings, the example
declares both languages: an omitted `--language` selects Python, `--language
cpp` builds without the bindings, and `--language python` runs the Python
example through them. Run from nested source directories or put
`--project-root /path/to/module` before the verb. Keep shared configuration
in `[tool.holoscan]` as described in the Holoscan CLI configuration reference.
Lifecycle commands use the development container unless `--local` is passed;
follow the generated `AGENTS.md` for hosts without a GPU. Generated operator/demo
bodies are TODO stubs: scaffold/import success is not functional proof.
Python bindings need the SDK's `holoscan` Python package in the build
environment, which SDK 5 Debian packages do not provide. Verify the SDK
import in the selected environment first.

## Producer and demo contract

Validate against the selected CLI's packaged Module schema
(`holoscan_cli/metadata/module.schema.json` in the installed package) and the
generated examples. Keep identity, languages, namespaces, SDK range,
platforms, operators, source, license, tests, `support_level`, and
`binary_packages` consistent. The module's languages must cover its
operators' languages. Declare `binary_packages.wheel` only after the module
publishes a wheel; `binary_packages.pypi` is a deprecated alias. Implement the
smallest reusable operator/API, preserve native and binding/import contracts,
and add deterministic unit tests. Register each new bound operator in the
generated Python extension and cover it in `tests/python`. Add a finite demo
that consumes the same public surface as another project and exits nonzero on
mismatch. Keep demo policy out of the operator.

Register only necessary targets and tests. Exclude data, models, engines,
build/install trees, wheels, debs, and recordings from source control.

## Test scope

The bundled CTest driver does not scope `test <module>` to one module, so the
module repository's `test` runs its whole suite either way and empties and
reuses `<root>/build-` on every run, separate from
`HOLOSCAN_CLI_BUILD_PARENT_DIR`; inspect the driver before running tests. Combine it with the
documented focused pytest or CTest targets and report scope honestly. A
passing generated compile-plan, import, or placeholder test is not a finite
pipeline proof; replace TODO bodies and assert actual processing results.

Require build success, focused tests, finite demo evidence, and visual
inspection when rendering is affected. After one full proof, reuse an
unchanged image with `--no-docker-build`; rebuild local artifacts after source
changes.
