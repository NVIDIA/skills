# Convert to USD

## When to Use

Use this reference as the conversion router. Ask the converter references which source they support, choose the highest-priority supported reference, preserve conversion metadata, and hand off the result to minimum USD validation before deeper `omni-asset-validate`, `omni-asset-validate-geometry`, `omni-asset-validate-physics`, and `simready-validate` checks.

Do not perform detailed format-specific conversion here when a specialized converter reference exists.

For NVIDIA-backed conversion references, preserve the upstream tool names. The router does not maintain a mesh/CAD/Gaussian-splat extension table; it calls each reference's probe mode and lets that reference read its upstream capability source.

## Prerequisites

- Python 3.12 and `uv` (per repo `README.md`).
- Prefer a ready `PHYSICAL_AI_PREFLIGHT_MANIFEST` from the `preflight`
  reference. NVIDIA-backed converter references consume prepared upstream roots
  and executables from that manifest before falling back to direct legacy
  discovery. When `PHYSICAL_AI_REQUIRE_PREFLIGHT=1` is set, missing converter
  readiness blocks at the preflight guardrail.
- The selected converter backend must be installed and reachable before
  conversion runs.
- CAD routes require the NVIDIA Omniverse `usd-convert-cad` checkout,
  `omniverse-kit`, supported CAD core extensions, and CAD Converter licensing
  on supported architectures, including Linux arm64.
- First CAD runs need network access to download Kit extensions from the NVIDIA
  registry.

## Routing

Do not classify NVIDIA-backed inputs from a router-owned extension table. Existing USD files are detected locally, then the router queries converter references in priority order:

| Probe source | Route |
|---|---|
| URDF reference probe | `urdf-usd-converter` |
| MuJoCo reference probe, including XML `<mujoco>` root inspection | `mujoco-usd-converter` |
| Upstream `usd-convert-gsplat` CLI source inspected by the `usd-convert-gsplat` reference | `usd-convert-gsplat` |
| Upstream `usd-convert-cad` SKILL.md Supported Formats table inspected by the `usd-convert-cad` reference probe | `usd-convert-cad`; NVIDIA-backed source conversion delegates to upstream `usd-convert-cad` |
| Existing OpenUSD layer or package signature | Skip conversion and route to `validate-usd-minimum` |

If more than one converter reference reports support, the router selects by converter-reference priority and records a warning. If no reference reports support, return an unsupported report rather than guessing.

For ambiguous mesh-like suffixes such as STL, rely on upstream converter capability from each converter reference probe instead of hard-coding a local route in the router.

## USD Exchange Backing

Most conversion routes eventually hand off to a usdex consumer:

| Conversion path | Underlying consumer |
|---|---|
| `urdf-usd-converter` | `urdf_usd_converter`, direct `usdex.core` |
| `mujoco-usd-converter` | `mujoco_usd_converter`, direct `usdex.core` |
| `usd-convert-cad` routes | Self-contained `usd-convert-cad==0.2.0` wheel pinned by `upstream-versions.lock.json`, with its bundled OpenUSD and CAD conversion runtime |

`usd-convert-asset` is intentionally not an active route until a public PyPI
package is available. Downstream validation still follows the OpenUSD Exchange
SDK 2.3 stage and metadata contract where applicable; use the OpenUSD Exchange
SDK / usdex reference for repo authoring rules and `omni-asset-validate` for
the workflow validation wrapper.

## Instructions

1. Locate the source asset and relevant sidecar files.
2. Check whether the input is already OpenUSD.
3. Identify required asset roots such as mesh folders, texture folders, ROS package roots, or MJCF asset directories.
4. Run each converter reference probe and select the first supported reference in router priority order.
5. Confirm the installed reference script and selected converter dependency are available before running conversion.
6. For CAD inputs, let the `usd-convert-cad` reference verify the
   `usd-convert-cad` wheel is importable in the resolved interpreter before it
   invokes conversion. Do not import the wheel or start a converter runtime
   directly from this router.
7. Convert to a dedicated output directory with `convert-to-usd`, not next to source files unless the user requested that location.
8. Record a conversion report with source, converter, output, warnings, and next validation step.
9. Hand off the generated USD artifact to `validate-usd-minimum`.

## CLI Pattern

Prefer the installed reference-local script instead of assembling ad hoc commands:

```bash
python3 scripts/run.py /path/to/source_asset /path/to/output_dir --report /path/to/conversion_report.json
```

When running from outside the skill directory, use the installed skill path:

```bash
python3 /path/to/skills/omniverse-cad-to-simready/references/convert-to-usd/scripts/run.py /path/to/source_asset /path/to/output_dir --report /path/to/conversion_report.json
```

Check router dependencies with:

```bash
python3 scripts/check_dependencies.py --report dependency-check.json
```

## Output Format

Every conversion path should produce or report:

| Field | Meaning |
|---|---|
| `source_asset_path` | Original input file or directory. |
| `source_format` | Detected format, such as `urdf`, `mjcf`, `gsplat`, `cad`, or `usd`. |
| `converter_reference` | Reference selected for conversion. |
| `converter_tool` | Library, CLI, or application used by the converter reference. |
| `output_directory` | Directory containing the generated USD artifact and sidecar output. |
| `output_usd_path` | Primary USD layer or package generated by conversion. |
| `sidecar_inputs` | Meshes, textures, package roots, or extra files needed to convert. |
| `warnings` | Non-fatal issues, assumptions, and missing optional information. |
| `errors` | Blocking failures or unsupported features. |
| `next_step` | Usually `validate-usd-minimum`. |

Markdown is acceptable for the first report. JSON can be added later when schemas exist.

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/run.py` | Query converter reference probes, route to the selected reference, and produce a normalized conversion report. | Execute: `python3 scripts/run.py <source_asset> <output_dir> --report <conversion_report.json>`. Calls external converters via subprocess (network on first CAD run). |
| `scripts/check_dependencies.py` | Verify the converter dependencies referenced by `run.py` are reachable. | Execute: `python3 scripts/check_dependencies.py --report <dependency_report.json>`. Read-only; no network. |
| `scripts/report_schema.json` | JSON Schema for the conversion report shape. | Reference: read for expected report structure. |

## Limitations

- This router does not perform detailed format-specific conversion when a
  specialized converter reference exists.
- This router does not own NVIDIA-backed source-extension tables. Update the
  upstream converter reference or upstream repo when format capability changes.
- NVIDIA-backed source conversion must delegate to the `usd-convert-cad`
  reference. Do not switch to any other converter or substitute tooling when the
  selected runtime is unavailable.
- Ambiguous source types are only routed when a converter reference probe
  reports support.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| NVIDIA-backed conversion is blocked | The `usd-convert-cad` wheel is not installed/importable, the resolved interpreter is not Python 3.12, `USD_CONVERT_CAD_PYTHON` points at the wrong interpreter, or converter licensing is unavailable | Return a `blocked` conversion report with the specific readiness or conversion dependency and the install hint. Do not switch to another converter or substitute tooling. NVIDIA-backed source conversion must delegate to the `usd-convert-cad` reference. |
| CAD conversion is blocked on import or version | The pinned `usd-convert-cad==0.2.0` wheel is not installed in the resolved interpreter, a different version is installed, or its bundled `pxr` conflicts with another OpenUSD in a shared interpreter | Install the pinned wheel into an isolated Python 3.12 environment (`python -m pip install usd-convert-cad==0.2.0`) and point `USD_CONVERT_CAD_PYTHON` (or `--usd-convert-cad-python`) at that interpreter, then rerun. |
| A source routes to an unexpected converter | More than one reference reported support, or a capability snapshot changed | Inspect the report warnings. The router records the priority-selected reference and warns when multiple probes report support. |

## Runtime Availability

Do not imply that `uv sync` installs every source converter runtime. URDF,
MuJoCo/MJCF, and the repo Python dependencies are handled by the project
environment, but NVIDIA-backed source conversion requires an installed and
validated `NVIDIA-Omniverse/usd-convert-cad` checkout. If that runtime is
missing or does not support the source, preserve the blocked conversion report
and its `install_hint` instead of attempting an unrequested local build or
substituting another converter.

## Unsupported Cases

Do not pretend a conversion succeeded if a converter is unavailable or cannot parse the source. Produce a clear blocked report that includes:

- detected format
- missing converter or dependency
- source files inspected
- expected converter reference
- recommended next action

NVIDIA-backed source conversion must delegate to the `usd-convert-cad`
reference, including the source types listed in the upstream `usd-convert-cad`
SKILL.md Supported Formats table. The `usd-convert-cad` reference probe parses
that table live from a checkout of the repo (falling back to a built-in
snapshot when offline); conversion itself runs the self-contained
`usd-convert-cad` pip wheel, which bundles its own OpenUSD and CAD conversion
runtime (no Omniverse Kit app or converter extensions).
If the wheel is not importable in the resolved Python 3.12 interpreter, or
platform support, licensing, or conversion support is unavailable, return a
blocked conversion report rather than switching to any other converter or
substitute tooling. The higher-level router must not override converter
capability with its own source-extension list.
