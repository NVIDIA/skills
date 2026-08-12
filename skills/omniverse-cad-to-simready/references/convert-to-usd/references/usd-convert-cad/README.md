# Convert CAD to USD

## When to Use

Use this reference for NVIDIA-backed source conversion. Conversion delegates to the standalone `usd-convert-cad` Python wheel, a self-contained converter that bundles its own OpenUSD (`pxr`) runtime. There is no Omniverse Kit app, no extension-registry pull, no `config.env`, and no EULA prompt.

Guardrail: `usd-convert-cad` is the only allowed converter backend for this reference's NVIDIA-backed source conversion on all supported architectures, including Linux arm64. Do not fall back to `usd-convert-asset`, hand-authored USD, mesh converters, or other substitute CAD converters.

## Upstream Reference

- NVIDIA Omniverse `usd-convert-cad` repository: `https://github.com/NVIDIA-Omniverse/usd-convert-cad`
- Upstream CAD conversion skill (authoritative Supported Formats and converter options for the tested runtime): `https://github.com/NVIDIA-Omniverse/usd-convert-cad/blob/v0.2.0/skills/omniverse-cad-to-usd/SKILL.md`

Preflight clones the ref pinned in `upstream-versions.lock.json` so this SKILL.md is available locally; `run.py --probe` parses that tested Supported Formats table instead of silently adopting changes from a newer upstream release. Browser, raw-file fetches, or unauthenticated GitHub access can fail depending on access level. If that happens, use an authenticated local clone of `https://github.com/NVIDIA-Omniverse/usd-convert-cad` at the pinned ref and read the referenced paths from that checkout.

The wheel bundles its own OpenUSD (`pxr`) runtime, so install it into a dedicated Python 3.12 interpreter to avoid clashing with any other `pxr` distribution in a shared environment. Point this reference at that interpreter with `--usd-convert-cad-python` or the `USD_CONVERT_CAD_PYTHON` environment variable. When neither is set, the reference reads the interpreter from the preflight manifest (`usd_convert_cad` runtime), then falls back to the interpreter running the script.

## Inputs

Collect a source file, an output directory, and optionally the converter interpreter via `--usd-convert-cad-python` (or `USD_CONVERT_CAD_PYTHON`).
Supported source suffixes come from the upstream Supported Formats table in the
`usd-convert-cad` SKILL.md (`skills/omniverse-cad-to-usd/SKILL.md`). Because the
wheel does not expose a machine-readable format registry, `run.py` reads that
table directly from a checkout of the repo and resolves the SKILL.md in this
order:

1. `--usd-convert-cad-skill <path-to-SKILL.md>`
2. `USD_CONVERT_CAD_SKILL` (SKILL.md file or a checkout directory)
3. `--usd-convert-cad-root <checkout>`
4. the preflight manifest's `usd_convert_cad` upstream checkout
5. `USD_CONVERT_CAD_ROOT`
6. the shared upstream checkout (`OMNIVERSE_CAD_TO_SIMREADY_UPSTREAM_ROOT/usd-convert-cad`)

When none of these is reachable (for example, a fully offline run), `run.py`
falls back to a built-in snapshot of the table (`SUPPORTED_CAD_SUFFIXES`), so
conversion is never blocked purely by a missing checkout. Examples such as
`.stp`, `.step`, `.igs`, `.iges`, `.dgn`, `.ifc`, `.ifczip`, `.jt`, and
proprietary CAD files route to `usd-convert-cad`, never to a substitute
converter. Mesh/scene formats also route here when the table lists them as
supported; otherwise they are reported unsupported rather than sent to
`usd-convert-asset`.
The legacy backend-selection argument (`--backend`) is still accepted for
compatibility but ignored; the wheel selects its converter internally.
`--usd-convert-cad-root` now selects the SKILL.md checkout (see above) instead of
a conversion runtime.

## Dependency Check

Require:

- The manifest-pinned `usd-convert-cad==0.2.0` wheel installed into a Python 3.12 environment
  (`python -m pip install usd-convert-cad==0.2.0`). If the wheel is hosted on an NVIDIA
  package index, add it, for example `--extra-index-url https://pypi.nvidia.com`.
- An isolated interpreter for that wheel (its bundled `pxr` can conflict with
  another OpenUSD in a shared interpreter), exposed via `--usd-convert-cad-python`,
  `USD_CONVERT_CAD_PYTHON`, or the preflight manifest.

Do not silently install missing dependencies. If the wheel is not importable in
the resolved interpreter, run the wrapper and preserve its blocked conversion
report with the install hint. `check_dependencies.py` verifies the wheel imports
(`import usd_convert_cad; print(usd_convert_cad.get_version())`) and is the CAD
readiness gate before batching per-asset conversions.

## Conversion Workflow

1. Confirm the source asset exists.
2. Confirm the source suffix appears in the `usd-convert-cad` SKILL.md Supported Formats table (parsed live from the resolved checkout, with the built-in snapshot as fallback).
   On Linux aarch64, reject `.dwg`, `.dxf`, `.rvt`, and `.rfa` before invoking
   the wheel because the upstream DWG/Revit readers are unavailable there.
3. Resolve the converter interpreter from `--usd-convert-cad-python`, `USD_CONVERT_CAD_PYTHON`, or the preflight manifest.
4. If the wheel is not installed or does not match `upstream-versions.lock.json`, follow the install hint (`python -m pip install usd-convert-cad==0.2.0` into an isolated Python 3.12 environment).
5. Run this reference's portable script. It invokes `python -m usd_convert_cad -i <source> -o <output_dir>/<stem>.usd` with the resolved interpreter.
6. Treat the converter exit code as authoritative: exit `0` with the expected USD present is success; non-zero or a missing output is a failure with the converter's stderr recorded in the report.
7. If USD is generated, hand it to `validate-usd-minimum`.
8. If blocked, report the exact failure, such as an uninstalled or unimportable wheel, a wrong Python version, an unsupported source format, a conversion error, or a CAD license dependency.

## CLI Pattern

Default STEP conversion (uses `USD_CONVERT_CAD_PYTHON` or the current interpreter):

```bash
python3 scripts/run.py asset.step output_dir \
  --report output_dir/conversion.json
```

Explicit converter interpreter:

```bash
python3 scripts/run.py asset.jt output_dir \
  --usd-convert-cad-python /path/to/.venv/bin/python \
  --report output_dir/conversion.json
```

Forward documented converter options straight through to the wheel:

```bash
python3 scripts/run.py asset.jt output_dir \
  --report output_dir/conversion.json \
  --tess-lod 4 --no-materials
```

When running from outside the reference directory, use the installed reference path:

```bash
python3 /path/to/skills/omniverse-cad-to-simready/references/convert-to-usd/references/usd-convert-cad/scripts/run.py asset.step output_dir --report output_dir/conversion.json
```

Check dependencies with:

```bash
python3 scripts/check_dependencies.py --report dependency-check.json
```

## Output Format

This repo normalizes the conversion into the shared conversion report contract and includes:

- `source_asset_path`
- `source_format: cad`
- `converter_skill: usd-convert-cad`
- `converter_tool: usd-convert-cad`
- `converter_command`, the `python -m usd_convert_cad -i <input> -o <output>` invocation (plus any forwarded converter options)
- `output_directory`
- `output_usd_path`
- `generated_files`
- `sidecar_inputs`, including the resolved converter interpreter
- `warnings`, including the resolved interpreter and installed wheel version
- `errors`
- `next_step: validate-usd-minimum`

The wheel writes only the requested USD output (and any sidecars the chosen USD format implies); it does not emit a separate JSON status report.

## Known Caveats

- The `usd-convert-cad` wheel bundles its own OpenUSD and CAD conversion runtime; install it into an isolated Python 3.12 environment so its `pxr` cannot conflict with another OpenUSD distribution.
- Python 3.12 is required by the wheel.
- On Linux aarch64, AutoCAD `.dwg`/`.dxf` and Revit `.rvt`/`.rfa` inputs are
  unsupported by the upstream wheel and are reported as a platform limitation.
- USDZ (`.usdz`) export is not supported yet; use `.usd`, `.usda`, or `.usdc`.
- Proprietary CAD formats can require CAD Converter licensing.
- Detailed converter option names must come from the upstream skill or `usd-convert-cad --help` for the installed version.
- A successful CAD conversion does not imply simulation readiness.
