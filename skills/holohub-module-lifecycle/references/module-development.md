# Holoscan Module development through `./holohub`

## Layout and naming

- **External module:** generate a self-contained `holoscan-<name>` repository
  from `modules/template`, including its wrapper, operators, demos, tests,
  Dockerfile, packaging, and CI. Prefer this for independent releases.
- **In-tree module:** create a descriptor under `modules/holoscan-<name>/`
  while sources remain under HoloHub operators/applications. Prefer this when
  maintained and tested with the monorepo.

Keep these identities consistent:

- display name: free text;
- module slug and namespace: `snake_case`;
- repository/package: `holoscan-<kebab-case>`;
- operator slug: `snake_case`, normally ending `_op`.

Do not copy an in-tree descriptor into a standalone repository without adding
self-contained build/test/package structure.

## External scaffold

From HoloHub, preview template setup in the selected wrapper environment before
`create`:

```bash
./holohub setup --scripts template --dryrun
```

Inspect the previewed host dependency installation and obtain explicit user
authorization before running `./holohub setup --scripts template` in the same
environment. If setup fails, stop before creation. Then preview the complete
create command.
`--directory` is required even for unattended dry-run in the tested revision.

```bash
./holohub create <unprefixed-name> \
  --template modules/template \
  --directory <existing-parent> \
  --language <cpp-or-python> \
  -i False \
  --context project_name="<display-name>" \
  --context module_slug=<snake_case> \
  --context module_repo_name=holoscan-<kebab-case> \
  --context operator_slug=<snake_case_op> \
  --context full_name="<author>" \
  --context affiliation="<organization>" \
  --context version=<initial-semver> \
  --context holoscan_version=<minimum-sdk-version> \
  --context description="<short-description>" \
  --context _license=<spdx-license> \
  --context contact_email=<maintainer-email> \
  --dryrun
```

Review target and context and obtain any required authorization before
repeating the identical create command without `--dryrun`. The parent must
exist. The template initializes and stages a new Git repository; inspect it and
never push without an explicit request. Record the source HoloHub checkout's
full SHA as generator provenance, not as the generated repository's revision.

## In-tree descriptor

Inspect a nearby descriptor such as `modules/holoscan-gstreamer/`, then add only
the required metadata, `pyproject.toml`, Dockerfile, and operator/demo
registrations. Follow repository guidance before shared schema, Dockerfile, or
CMake macro changes. Preview every supported mutating wrapper command.

## Producer and demo contract

Validate against `utilities/metadata/module.schema.json` and nearby examples.
Keep identity, languages, namespaces, SDK range, platforms, operators, source,
license, tests, and `binary_packages` consistent. Implement the smallest
reusable operator/API, preserve native and binding/import contracts, and add
deterministic unit tests. Add a finite demo that consumes the same public
surface as another project and exits nonzero on mismatch. Keep demo policy out
of the operator.

Register only necessary targets and tests. Make in-tree descriptor declarations
match actual targets. Exclude data, models, engines, build/install trees,
wheels, debs, and recordings from source control.

## Build and test scope

Preview and run matching explicit-language commands:

```bash
./holohub build <module> --language <cpp-or-python> --dryrun --verbose
./holohub build <module> --language <cpp-or-python> --verbose
./holohub run <demo-app> <mode> --language <cpp-or-python> --dryrun --verbose
./holohub run <demo-app> <mode> --language <cpp-or-python> --verbose
```

Current HoloHub test drivers do not scope `test <module>` to one module. Test
declared operators and demo apps individually:

```bash
./holohub test <operator> --language <cpp-or-python> --dryrun --verbose
./holohub test <operator> --language <cpp-or-python> --verbose
./holohub test <demo-app> --language <cpp-or-python> --dryrun --verbose
./holohub test <demo-app> --language <cpp-or-python> --verbose
```

In a small generated repository, intentionally use repository-wide
`./holohub test` plus documented focused pytest/CTest targets. Report scope
honestly. Require build success, focused tests, finite demo evidence, and visual
inspection when rendering is affected. After one full proof, reuse an unchanged
image with `--no-docker-build`; rebuild local artifacts after source changes.
