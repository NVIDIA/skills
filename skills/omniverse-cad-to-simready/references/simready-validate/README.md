# SimReady Validate Profile

## When to Use

Use this reference after Content Agents assignment, post-assignment
`simready-conform-profile`, `validate-usd-minimum`, and `omni-asset-validate`
have run, and the user has selected, or needs help selecting, a SimReady
Foundation profile. This is a validation-only skill: it reports profile
conformance and blockers, but does not repair or stamp assets unless explicitly
requested. For end-to-end CAD-to-SimReady workflows where material/physics
assignment will run, do not run this reference before Content Agents. The only
pre-assignment validation gate should be `validate-usd-minimum`.

SimReady Foundation organizes validation in four layers:

- Requirements: atomic checks such as `UN.006` or `VG.MESH.001`
- Capabilities: grouped requirements such as `units` or `geometry`
- Features: use-case bundles such as `FET001_BASE_NEUTRAL`
- Profiles: named bundles of features such as robotics prop or robot-body profiles

## Dependency Check

Require:

- Prefer a ready `PHYSICAL_AI_PREFLIGHT_MANIFEST` from the `preflight`
  reference. This wrapper consumes the prepared SimReady Foundation root and
  `simready-validate` executable from that manifest before falling back to
  direct legacy discovery. When `PHYSICAL_AI_REQUIRE_PREFLIGHT=1` is set,
  missing profile-validation readiness blocks at the preflight guardrail.
- `simready.validate` / `simready-validate` from NVIDIA SimReady Foundation, or a source checkout with `requirements.txt` or `nv_core/validator_sample/requirements.txt`
- Upstream source: `https://github.com/NVIDIA/simready-foundation` pinned by `upstream-versions.lock.json` to `v2026.04.1`
- Deterministic OpenUSD runtime: NVIDIA OpenUSD Exchange SDK package `usd-exchange==2.3.0` from `https://github.com/NVIDIA-Omniverse/usd-exchange`, with the complete SimReady venv package set pinned in the same manifest
- SimReady Foundation spec files: `capabilities/`, `features/`, and `profiles/profiles.toml`

Check installed reference dependencies with:

```bash
python3 scripts/check_dependencies.py --report dependency-check.json
```

If `--foundation-root`, `--foundation-spec-root`, `SIMREADY_FOUNDATION_ROOT`, and `SIMREADY_FOUNDATION_SPEC_ROOT` are not configured and no installed `simready.validate` specs are available, provide a checkout under `$HOME/.omniverse-cad-to-simready/upstreams/simready-foundation` or `$OMNIVERSE_CAD_TO_SIMREADY_UPSTREAM_ROOT/simready-foundation`, checked out to the manifest pin (`v2026.04.1`), and load `nv_core/sr_specs/docs` plus `nv_core/validator_sample` from that checkout.

If `simready-validate` is not on `PATH`, do not stop there. `scripts/run.py` installs the exact runtime set from `upstream-versions.lock.json` into a dedicated venv and verifies the installed distribution versions before using that executable. Override the venv with `PHYSICAL_AI_SIMREADY_VALIDATE_VENV`; otherwise the default is `$XDG_CACHE_HOME/omniverse-cad-to-simready/simready-validate-venv` or `$HOME/.cache/omniverse-cad-to-simready/simready-validate-venv`.

This locked runtime is also the Linux aarch64 solution: `usd-exchange==2.3.0`
provides OpenUSD Python modules and shared libraries, and
`simready-validate==2026.4.8` is installed with `--no-deps`. The manifest pins
`omniverse-asset-validator==1.18.0`,
`omniverse-usd-profiles==1.10.22`, and every transitive package used by that
venv, preventing a future resolver run from silently selecting an incompatible
combination.

Do not fall back to local profile presets or direct `omni_asset_validate` feature/capability flags for validation. Report `BLOCKED` when the executable is unavailable, no usable Foundation checkout/spec root exists, the locked runtime cannot be installed, or the installed versions do not match the manifest.

## Target Selection

Supported formal profiles are loaded from SimReady Foundation `profiles.toml`. The default profile is:

```text
Prop-Robotics-Neutral@1.0.0
```

Use `--list-profiles` to expose selectable profile options before running validation:

```bash
simready-validate --list-profiles --foundation-root /path/to/simready-foundation
```

Recognize these common profile names:

| Profile | Use |
|---|---|
| `Prop-Robotics-Neutral` | Neutral robotics prop profile. |
| `Prop-Robotics-Physx` | Robotics prop with PhysX rigid-body simulation requirements. |
| `Prop-Robotics-Isaac` | Isaac Sim-oriented robotics prop profile. |
| `Robot-Body-Neutral` | Neutral robot body profile. |
| `Robot-Body-Runnable` | Runnable robot body profile with PhysX/articulation/drive requirements. |
| `Robot-Body-Isaac` | Isaac Sim robot body profile. |

For URDF or MuJoCo robot assets, prefer `Robot-Body-Runnable` unless the user names another profile. For generic CAD/mesh props, prefer the default `Prop-Robotics-Neutral`. Use `Prop-Robotics-Physx` when the user asks for PhysX-specific prop validation.

## Instructions

1. Confirm the asset is an existing USD asset path.
2. Confirm Content Agents and post-assignment conformance have already run when
   property assignment is in scope. If the request is explicitly
   validation-only or property assignment was skipped, record that exception.
3. Confirm earlier validation has passed, or state that minimum USD and generic Asset Validator checks should run first.
4. Select a formal SimReady Foundation profile from user intent and asset type.
5. Resolve the SimReady Foundation source checkout from `--foundation-root` or `SIMREADY_FOUNDATION_ROOT`; alternatively resolve specs from `--foundation-spec-root` or `SIMREADY_FOUNDATION_SPEC_ROOT`. If no path is configured, use `$OMNIVERSE_CAD_TO_SIMREADY_UPSTREAM_ROOT/simready-foundation` or `$HOME/.omniverse-cad-to-simready/upstreams/simready-foundation`, checked out to the ref pinned in `upstream-versions.lock.json`.
6. Run this reference's portable `scripts/run.py`, which installs and verifies the locked SimReady runtime when the CLI is missing on `PATH`, then uses Foundation `simready-validate` behavior to load Foundation `capabilities`, `features`, and `profiles/profiles.toml`.
7. Parse profile, feature, requirement, issue, warning, and error results from the Foundation validation runtime.
8. Inspect the asset topology with OpenUSD. Treat `RB.MB.001` as non-blocking when the asset has only one mesh component or one `GeomSubset` component, because there is no reusable multi-body component structure to promote. Preserve the ignored issue under `ignored_issues`, add a warning, and pass the profile if no other failures remain.
9. Fail when any selected profile feature fails or any issue has `ERROR` or `FAILURE` severity after applying the single-component `RB.MB.001` policy.
10. Report a structured SimReady profile validation result.

## CLI Pattern

Prefer the installed reference-local script for runtime checks:

```bash
python3 scripts/run.py asset.usda \
  --profile Prop-Robotics-Neutral \
  --report report.json

SIMREADY_FOUNDATION_ROOT=/path/to/simready-foundation \
  python3 scripts/run.py asset.usda --profile Prop-Robotics-Neutral --report report.json

python3 scripts/run.py asset.usda \
  --profile Robot-Body-Runnable \
  --foundation-root /path/to/simready-foundation \
  --report report.json
```

Do not use `--fix`, `--stamp`, or profile adaptation unless the user explicitly asks for those operations.

When running from outside the reference directory, use the installed reference path:

```bash
python3 /path/to/skills/omniverse-cad-to-simready/references/simready-validate/scripts/run.py asset.usda --profile Prop-Robotics-Neutral --report report.json
```

## Output Format

Reports should follow:

```text
scripts/report_schema.json
```

Include:

- `asset_path`
- `validator_skill`
- `validator_tool`
- `passed`
- `status`
- `profile_name`
- `profile_target`
- `command`
- `available_profiles`
- `profile_results`
- `feature_results`
- `requirement_counts`
- `issue_counts`
- `issues`
- `ignored_issues`
- `asset_topology`
- `validation_policy`
- `warnings`
- `errors`
- `next_step`

## Pass/Fail Policy

Fail when:

- required validator dependencies are missing
- the selected SimReady Foundation profile is unknown or not present in `profiles.toml`
- the Foundation validation runtime returns `FAIL` or `ERROR`
- any issue has severity `ERROR` or `FAILURE` after the single-component `RB.MB.001` policy is applied
- any selected feature reports failed requirements after the single-component `RB.MB.001` policy is applied

Warn when:

- the target is narrower than the user's stated use case
- profile stamping or adaptation is requested but not available in the runtime
- `RB.MB.001` is ignored as non-blocking because the USD has only one mesh component or one `GeomSubset` component

## Next Steps

Use this handoff:

| Result | Next step |
|---|---|
| Passes selected profile | Report validation result and preserve the JSON report. |
| Fails selected profile feature | Send issues to a post-assignment repair loop through `simready-conform-profile`, then rerun this reference on the newest authored USD. |
| SimReady Foundation runtime blocked | Provide the manifest-pinned `simready-foundation` checkout with `--foundation-root` or `SIMREADY_FOUNDATION_ROOT`, then retry so `scripts/run.py` can install and verify the exact runtime from `upstream-versions.lock.json`. |
