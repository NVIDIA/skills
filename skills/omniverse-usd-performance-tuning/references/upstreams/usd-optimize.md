# usd-optimize / Usd Optimize Package Handoff

Usd Optimize operation mechanics are owned by upstream `usd-optimize` and
ship with the prebuilt Usd Optimize package. This package owns digital twin
workflow routing, runtime setup context, validation scope, output workspace
policy, batch orchestration, and reporting.

- Public repository: [https://github.com/NVIDIA-Omniverse/usd-optimize/](https://github.com/NVIDIA-Omniverse/usd-optimize/)
- Prebuilt packages: **GitHub Releases** on the repository above
  (`https://github.com/NVIDIA-Omniverse/usd-optimize/releases`). Each release
  carries Linux x86_64, Linux aarch64, and Windows x86_64 zips.
- **Verified release: 1.3.1.** Published 2026-10-08; its CHANGELOG entry is
  dated 2026-10-07. Check the release page before selecting another version,
  then verify its package and live API.
- Package pattern: `usd_optimize_usd_<usd>_py_<python>@<version>.<platform>.release.zip`.
  The operation catalog and validator concepts describe 1.3.1: 48 operations,
  including `moveMaterials`, the `editStageMetrics.stopAtSkeletonRoot` argument
  and the `shrinkwrap` temporal arguments. 1.1.0 and earlier lack those and
  register validator rules under the bare names the concept registry no longer
  uses, so they are not supported.
- **Pick the USD flavor explicitly.** 1.3.1 ships three USD flavors per platform
  (`usd_25.05`, `usd_25.11` and `usd_26.08`), and a bare
  `-p '*manylinux*x86_64*'` matches all three. Use `usd_25.11` unless a caller
  needs another:
  ```bash
  gh release download v1.3.1 -R NVIDIA-Omniverse/usd-optimize \
    -p 'usd_optimize_usd_25.11_py_3.12@*manylinux_2_35_x86_64.release.zip'
  ```
  (or pick the asset from the releases page in a browser).
- Asset sizes vary by flavor, so quote the flavor when you quote a size. For
  1.3.1 Linux x86_64: `usd_25.11` is ~121 MiB (~540 MB extracted), `usd_25.05`
  is ~913 MiB, and `usd_26.08` is ~105 MiB.
- Package operation guides: `docs/operations/<operation>.rst`
- Package operation runner skill: `.agents/skills/run-operations/SKILL.md`
- Package validator runner skill: `.agents/skills/run-validators/SKILL.md`
- Package validator interpretation skill: `.agents/skills/interpret-validators/SKILL.md`
- Package proxy skill: `.agents/skills/create-proxy/SKILL.md`
- Package install skill: `.agents/skills/prebuilt-package/SKILL.md`

## Operation Guide Resolution

For any operation key listed in `references/operations/operations.json`, derive
the upstream mechanics path instead of storing per-operation package details in
this repo. Resolve it under the selected package root (`$USD_OPTIMIZE_ROOT`),
without cloning the source repo. This is the single place this rule is stated;
other skill files point here.

- Package path template: `$USD_OPTIMIZE_ROOT/docs/operations/<operation-key>.rst`.
- Sidecars: operation index `docs/operations.rst`, pipeline guidance
  `docs/choosing-operations.rst`, invocation `docs/cli.rst`. The six presets in
  `config_presets/` ship in the package and are what
  `usdOptimize -c config_presets/<name>.json` reads.
- Upstream web URL template, pinned to the verified tag:
  `https://github.com/NVIDIA-Omniverse/usd-optimize/blob/v1.3.1/docs/operations/<operation-key>.rst`.

The package root must contain `docs/operations/` and `docs/operations.rst`, plus
the runtime sentinels `python/`, `usdpy/`, `lib/`, and `extraLibs/` when it is
also used for standalone execution.

The package ships `.agents/skills/`, including the five skills listed above. It
ships no `.claude` or `.codex` compatibility aliases; use `.agents` paths in
handoffs.

The package also ships a native `bin/usdOptimize` CLI and a
`getOperationDocumentation()` method on the core instance. The native CLI does
not start a Python interpreter, so the four Python-hosted operations
(`pythonScript`, `deleteHiddenPrims`, `removeUntypedPrims`, `moveMaterials`) are
unavailable through it and report `invalid operation specified`; it lists the
other 44. This skill drives Usd Optimize through the Python bindings, where all
48 operations are available.

If no package root exists, download and extract the published
`usd_optimize_...release.zip` package for the target platform from GitHub
Releases, or use the package archive path, release-asset URL, or extracted
package root supplied by the user. Extract with a tool that restores symbolic
links, such as `unzip`: the shared libraries are symlink chains. Package-internal
paths (`.agents/...`, `docs/operations/...`, `config_presets/`, `python/`,
`usdpy/`, `lib/`, `extraLibs/`) were last verified on 2026-10-12 against the
extracted 1.3.1 Linux x86_64 `usd_25.11` package. If web or raw GitHub fetch is
available, the public repository URL can be used for docs-only reads.
Do not clone the source repo just to read operation parameters, defaults, or
implementation gotchas.

The package ships no Python distribution metadata (no `*.dist-info`,
`*.egg-info`, or `entry_points.txt` anywhere in the tree), so
`importlib.metadata.entry_points()` cannot discover the bundled validator
plugin. Consequently `usd_optimize.validators` requires an explicit
`register_all()` call — importing the module alone registers **no** Usd
Optimize rules; `register_all()` registers 25 (19 under `Usd:Performance`, 6
under `Omni:Geometry`). They register as `UsdOptimize<Name>`, for example
`UsdOptimizeWindingsChecker`: the module rewrites each rule's `__name__` at
import to keep it clear of identically named usd-validation-nvidia rules.
`validator-concepts.json` names those classes.

## Version Reporting

The verified prebuilt package does not stamp a version into its bindings.
Record the changelog as version evidence through [setup](../setup.md):

| Candidate source | 1.3.1 |
|---|---|
| `usd_optimize.core.__version__` | absent |
| `CHANGELOG.md` first heading | `## [1.3.1] - 2026-10-07` |

Record the package changelog heading and its source path as version evidence.
Distinguish changelog evidence from a version reported by the bindings.

Use `references/operations/operations.json` — the single catalog carrying both
routing metadata and the nested `curation` block (generated `status` +
authored `wired_into`; `rationale` only on overrides) — for digitaltwin
routing, risk, confirmation, and recommendation
posture. Before invoking any operation, consume
`<output_path>/setup-preflight.json` and confirm the op appears in
`usdOptimize.operationsAvailable`.
