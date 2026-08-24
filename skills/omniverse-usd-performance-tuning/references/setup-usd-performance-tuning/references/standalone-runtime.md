# Standalone Runtime Setup

Use this reference when the user chooses standalone libraries instead of Kit or
when no Kit candidate is available.

## Statuses

- `ready-standalone`: standalone Usd Optimize and usd-validation-nvidia paths are
  selected and verified.
- `needs-runtime-choice`: setup cannot continue without the user choosing Kit,
  standalone, or installation.
- `blocked_missing_usd_optimize`: the user requested Usd Optimize but no
  supported SO runtime can be selected or installed.

## Usd Optimize Prompt

When standalone Usd Optimize is missing, ask before invoking
`install-usd-optimize-standalone`. The prompt must include:

- Python 3.12 hard requirement.
- Approximate download size — quote the flavor, since it varies: the current
  1.1.0 `usd_25.11` Linux x86_64 asset is ~174 MiB (~650 MB extracted); the
  1.0.4 Linux x86_64 asset is ~331 MiB (~920 MB extracted).
- Intended install location.
- Requirement for a published prebuilt Usd Optimize release package
  (asset name + download: `references/upstreams/usd-optimize.md`)
  archive path, direct archive URL, or extracted package root when no package
  root is already available.
- Usd Optimize validators need an explicit `usd_optimize.validators.register_all()`
  after both packages are importable in the same Python environment. Import alone
  registers nothing, and the package ships no entry-point metadata.
  `register_all()` adds 25 rules (19 `Usd:Performance`, 6 `Omni:Geometry`) on
  both 1.0.4 and 1.1.0.
- Limitation that render-time profiling needs Kit.

Offer:

1. Proceed with standalone Usd Optimize install.
2. Install Kit instead.
3. Stop and produce diagnosis-only output from available evidence.

If the user proceeds and Python 3.12 is missing, install or select Python 3.12
first, then invoke `install-usd-optimize-standalone`.

## Expected Standalone Layout

Usd Optimize standalone uses:

```text
<USD_OPTIMIZE_ROOT>/docs/operations.rst          # op index, 1.1.x packages
<USD_OPTIMIZE_ROOT>/docs/operations/<key>.rst    # per-op docs, 1.1.x packages
<USD_OPTIMIZE_ROOT>/.agents/operations/INDEX.md  # op index, 1.0.x packages
<USD_OPTIMIZE_ROOT>/.agents/operations/<key>.md  # per-op docs, 1.0.x packages
<USD_OPTIMIZE_ROOT>/python
<USD_OPTIMIZE_ROOT>/usdpy
<USD_OPTIMIZE_ROOT>/lib
<USD_OPTIMIZE_ROOT>/extraLibs
```

The per-operation doc sentinel is version-tolerant: a root is valid when either
`docs/operations.rst` (1.1.x packages) or `.agents/operations/INDEX.md` (1.0.x
packages) exists. The runtime dirs (`python`, `usdpy`, `lib`, `extraLibs`) must
be present regardless.

Verified 2026-08-04 against the extracted 1.0.4 and 1.1.0 Linux x86_64
`usd_25.11` packages: the sentinel passes on both, and per-operation doc lookup
resolves all 47 operation keys — to `docs/operations/<key>.rst` on 1.1.0 and to
`.agents/operations/<key>.md` on 1.0.4. `.agents/` ships on both; only its
`operations/` subtree moved in 1.1.x.

Invoke `install-usd-optimize-standalone` when `USD_OPTIMIZE_ROOT`, `USD_OPTIMIZE_ROOT`,
or `WU_SO_PACKAGE_DIR` is missing or does not point at an extracted package with
the sentinel paths above. Do not clone the Usd Optimize source repository to
satisfy standalone setup.

For standalone Omni usd-validation-nvidia, invoke `install-usd-validation-nvidia-standalone`
when the runtime is absent. Test that with the command the distribution actually
installs:

```bash
nvidia_usd_validate --version
```

A non-zero exit means the runtime is missing and the installer should run. Exit 0
means it is present, so skip the installer. Check once per setup pass and act on
the result. Re-running the installer against an environment that already answers
exit 0 is a defect; it never terminates.

Install into the same venv that Usd Optimize uses, then call
`usd_optimize.validators.register_all()` to pull the 25 Usd Optimize rules into
the validator registry.

Do not use the Usd Optimize package's bundled `validator-venv` as the
default usd-validation-nvidia runtime — it may lack `numpy` and is slower on large
stages.

## Handoff

After standalone setup, return to:

- `omniverse-usd-performance-tuning` for broad performance requests.
- `usd-validation-runner` for validation.
- `usd-optimize-run-operations` only after Usd Optimize operation availability is
  verified and recorded in `<output_path>/setup-preflight.json`.
