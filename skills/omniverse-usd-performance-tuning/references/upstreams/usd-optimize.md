# usd-optimize / Usd Optimize Package Handoff

Usd Optimize operation mechanics are owned by upstream `usd-optimize` and
ship with the prebuilt Usd Optimize package. This package owns digital twin
workflow routing, runtime setup context, validation scope, output workspace
policy, batch orchestration, and reporting.

- Public repository: [https://github.com/NVIDIA-Omniverse/usd-optimize/](https://github.com/NVIDIA-Omniverse/usd-optimize/)
- Prebuilt packages: **GitHub Releases** on the repository above
  (`https://github.com/NVIDIA-Omniverse/usd-optimize/releases`). Each release
  carries Linux x86_64, Linux aarch64, and Windows x86_64 zips.
- **Current released version: 1.1.0** (published 2026-07-20; CHANGELOG entry
  dated 2026-07-13). Anything numbered above that — 1.1.1 and up — is unreleased
  upstream HEAD, not a downloadable package.
- Package pattern: `usd_optimize_usd_<usd>_py_<python>@<version>.<platform>.release.zip`.
  usd-optimize 1.0.4 is the minimum supported runtime for this skill; 1.1.0 is
  verified. The two layouts differ only in where per-operation docs live:
  1.1.x packages ship them at `docs/operations/<key>.rst`, 1.0.x packages at
  `.agents/operations/<key>.md`. Operation inventory (47 keys), arguments, and
  defaults are the same across both — verified by diffing the live
  `getOperations()` / `getOperationArguments()` output of the extracted 1.0.4 and
  1.1.0 packages. The one delta is `decimateMeshes.reductionFactor`, which gained
  a `rejectOutOfRange=true` metadata flag in 1.1.0; its name, type, and `50.0`
  default are unchanged.
- **Pick the USD flavor explicitly.** 1.1.0 ships two USD flavors per platform
  (`usd_25.05` and `usd_25.11`); 1.0.4 shipped only `usd_25.11`. A bare
  `-p '*manylinux*x86_64*'` matches **both** 1.1.0 x86_64 assets and pulls
  ~1.1 GB. Use `usd_25.11` unless a caller needs 25.05:
  ```bash
  gh release download v1.1.0 -R NVIDIA-Omniverse/usd-optimize \
    -p 'usd_optimize_usd_25.11_py_3.12@*manylinux_2_35_x86_64.release.zip'
  ```
  (or pick the asset from the releases page in a browser).
- Asset sizes vary by flavor, so quote the flavor when you quote a size:
  1.1.0 `usd_25.11` Linux x86_64 is ~174 MiB (~650 MB extracted), 1.1.0
  `usd_25.05` Linux x86_64 is ~911 MiB, and 1.0.4 Linux x86_64 is ~331 MiB
  (~920 MB extracted).
- Package operation guides: `docs/operations/<operation>.rst` (1.1.x) or `.agents/operations/<operation>.md` (1.0.x)
- Package operation runner skill: `.agents/skills/run-operations/SKILL.md`
- Package validator runner skill: `.agents/skills/run-validators/SKILL.md`
- Package validator interpretation skill: `.agents/skills/interpret-validators/SKILL.md`
- Package proxy skill: `.agents/skills/create-proxy/SKILL.md`
- Package install skill: `.agents/skills/prebuilt-package/SKILL.md`

## Operation Guide Resolution

For any operation key listed in `references/operations/operations.json`, derive
the upstream mechanics path instead of storing per-operation package details in
this repo. Resolve it with a version-tolerant lookup under the selected package
root (`$USD_OPTIMIZE_ROOT`), without cloning the source repo. This is the single
place this rule is stated; other skill files point here.

- Package path template (prefer): `$USD_OPTIMIZE_ROOT/docs/operations/<operation-key>.rst`
  (1.1.x packages).
- Fallback: `$USD_OPTIMIZE_ROOT/.agents/operations/<operation-key>.md` (1.0.x
  packages, which predate the auto-generated docs tree).
- Sidecars follow the same rule. Operation index: `docs/operations.rst` (1.1.x)
  or `.agents/operations/INDEX.md` (1.0.x). Pipeline/preset guidance:
  `docs/choosing-operations.rst` (1.1.x) or `.agents/operations/PIPELINES.md`
  (1.0.x). Invocation: `docs/cli.rst` (1.1.x) or
  `.agents/operations/INVOCATION.md` (1.0.x). **`config_presets/` is not shipped
  in either package** — it exists only in the upstream source tree, so do not
  resolve preset guidance to it.
- Upstream web URL template (1.1.x `main`):
  `https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/docs/operations/<operation-key>.rst`.
  To document the 1.0.x layout, pin the tag:
  `https://github.com/NVIDIA-Omniverse/usd-optimize/blob/1.0.4/.agents/operations/<operation-key>.md`.

Each root above must contain the per-operation doc set — `docs/operations/` with
`docs/operations.rst` (1.1.x) or `.agents/operations/INDEX.md` (1.0.x) — plus the
runtime sentinels `python/`, `usdpy/`, `lib/`, and `extraLibs/` when it is also
used for standalone execution.

`.agents/` itself ships in both 1.0.4 and 1.1.0; only the `operations/` subtree
moved. 1.1.0 keeps `.agents/skills/` (including `run-operations`,
`run-validators`, `interpret-validators`, `create-proxy`, and
`prebuilt-package`), so the package skill paths listed above resolve on both
versions. Neither package ships `.claude` or `.codex` compatibility aliases;
use `.agents` paths in handoffs.

Two things 1.1.0 adds that 1.0.4 lacks: a native `bin/usdOptimize` CLI, and a
`getOperationDocumentation()` method on the core instance. The native CLI does
not start a Python interpreter, so the three Python-plugin operations
(`pythonScript`, `deleteHiddenPrims`, `removeUntypedPrims`) are unavailable
through it and report `invalid operation specified`. This skill drives Usd
Optimize through the Python bindings, where all 47 operations are available.

If no package root exists, download and extract the published
`usd_optimize_...release.zip` package for the target platform from GitHub
Releases, or use the package archive path, release-asset URL, or extracted
package root supplied by the user. Package-internal paths (`.agents/...`,
`docs/operations/...`, `python/`, `usdpy/`, `lib/`, `extraLibs/`) were last
verified on 2026-08-04 against the extracted 1.0.4 and 1.1.0 Linux x86_64
`usd_25.11` packages. If web or raw GitHub fetch is available, the public
repository URL can be used for docs-only reads. Do not clone the source repo
just to read operation parameters, defaults, or implementation gotchas.

Neither package ships Python distribution metadata (no `*.dist-info`,
`*.egg-info`, or `entry_points.txt` anywhere in the tree), so
`importlib.metadata.entry_points()` cannot discover the bundled validator
plugin. 1.1.0 did not change this. Consequently `usd_optimize.validators`
requires an explicit `register_all()` call — importing the module alone
registers **zero** rules; `register_all()` registers 25 (19 under
`Usd:Performance`, 6 under `Omni:Geometry`), identically on 1.0.4 and 1.1.0.

## Version Reporting

The prebuilt package does not stamp a version into the bindings, so the probe
fallback chain in `setup-usd-performance-tuning/references/runtime-probe.md`
lands on its changelog step for both releases. Measured behavior:

| Candidate source | 1.0.4 | 1.1.0 |
|---|---|---|
| `usd_optimize.core.__version__` | absent | absent |
| `CHANGELOG.md` first heading | `## [1.0.4] - 2026-06-09` | `## [1.1.0] - 2026-07-13` |

So the reported value is `0.0.0+changelog:## [<version>] - <date>` on both. No
binding-level source reports a package version on either release, so the
changelog step is the one that identifies the package — treat it as
load-bearing, not as a last-ditch fallback.

Use `references/operations/operations.json` — the single catalog carrying both
routing metadata and the nested `curation` block (generated `status` +
authored `wired_into`; `rationale` only on overrides) — for digitaltwin
routing, risk, confirmation, and recommendation
posture. Before invoking any operation, consume
`<output_path>/setup-preflight.json` and confirm the op appears in
`usdOptimize.operationsAvailable`.
