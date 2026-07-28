# usd-optimize / Usd Optimize Package Handoff

Usd Optimize operation mechanics are owned by upstream `usd-optimize` and
ship with the prebuilt Usd Optimize package. This package owns digital twin
workflow routing, runtime setup context, validation scope, output workspace
policy, batch orchestration, and reporting.

- Public repository: [https://github.com/NVIDIA-Omniverse/usd-optimize/](https://github.com/NVIDIA-Omniverse/usd-optimize/)
- Prebuilt packages: **GitHub Releases** on the repository above
  (`https://github.com/NVIDIA-Omniverse/usd-optimize/releases`). Each release
  carries Linux x86_64, Linux aarch64, and Windows x86_64 zips (~330-360 MB).
- Package pattern: `usd_optimize_usd_<usd>_py_<python>@<version>.<platform>.release.zip`.
  usd-optimize 1.0.4 is the minimum supported runtime for this skill; 1.1.x is
  also supported. The two layouts differ only in where per-operation docs live:
  1.1.x packages ship them at `docs/operations/<key>.rst`, 1.0.x packages at
  `.agents/operations/<key>.md`. Operation inventory, arguments, and defaults are
  the same across both.
- Download example:
  `gh release download v1.0.4 -R NVIDIA-Omniverse/usd-optimize -p '*manylinux*x86_64*'`
  (or pick the asset from the releases page in a browser).
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
  `docs/choosing-operations.rst` plus `config_presets/*.json` (1.1.x) or
  `.agents/operations/PIPELINES.md` (1.0.x). Invocation: `docs/cli.rst` (1.1.x)
  or `.agents/operations/INVOCATION.md` (1.0.x).
- Upstream web URL template (1.1.x `main`):
  `https://github.com/NVIDIA-Omniverse/usd-optimize/blob/main/docs/operations/<operation-key>.rst`.
  To document the 1.0.x layout, pin the tag:
  `https://github.com/NVIDIA-Omniverse/usd-optimize/blob/1.0.4/.agents/operations/<operation-key>.md`.

Each root above must contain the per-operation doc set — `docs/operations/` with
`docs/operations.rst` (1.1.x) or `.agents/operations/INDEX.md` (1.0.x) — plus the
runtime sentinels `python/`, `usdpy/`, `lib/`, and `extraLibs/` when it is also
used for standalone execution. The package may include `.claude` and `.codex`
compatibility aliases, but handoffs should use `.agents` paths.

If no package root exists, download and extract the published
`usd_optimize_...release.zip` package for the target platform from GitHub
Releases, or use the package archive path, release-asset URL, or extracted
package root supplied by the user. Package-internal paths (`.agents/...`,
`python/`, `usdpy/`, `lib/`, `extraLibs/`) were last verified against the
110.x packages; re-verify against the extracted 1.0.x package on first use. If web or raw GitHub fetch is available, the public
repository URL can be used for docs-only reads. Do not clone the source repo
just to read operation parameters, defaults, or implementation gotchas.

Use `references/operations/operations.json` — the single catalog carrying both
routing metadata and the nested `curation` block (generated `status` +
authored `wired_into`; `rationale` only on overrides) — for digitaltwin
routing, risk, confirmation, and recommendation
posture. Before invoking any operation, consume
`<output_path>/setup-preflight.json` and confirm the op appears in
`usdOptimize.operationsAvailable`.
