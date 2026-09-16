# Changelog — omniverse-usd-performance-tuning

## 0.4.1

### Fixed

- **The validation aggregate no longer reports `PASS` when nothing was
  validated.** The completion gate treats `blocked_validation_runtime`,
  `timeout_recorded` and `user_declined` as resolved dispositions, which they
  are — but all three mean no validator ran, and a unit that never ran cannot
  contribute a finding, so `errorCount` stayed 0 and the summary read `PASS`.
  Reported on Windows with all 15 requested Tier 1 concepts blocked; the
  aggregate has no platform branch and reproduces anywhere. `summary.status` is
  now `BLOCKED` unless every planned unit actually ran, so a single unchecked
  safety gate withholds the pass. Findings and timeouts are unchanged otherwise.
- **The validator subprocess starts on Windows.** The child environment is built
  from an explicit allowlist rather than a wholesale `os.environ` copy, and that
  allowlist carried only POSIX names. Without `SYSTEMROOT`, CPython cannot
  initialise Winsock while importing `asyncio` and the worker dies in
  `asyncio/_overlapped` with WinError 10106 before reading its job — which is
  what produced the all-blocked set above. `SYSTEMROOT`, `SYSTEMDRIVE`,
  `WINDIR`, `COMSPEC`, `PATHEXT`, `PROCESSOR_ARCHITECTURE` and
  `NUMBER_OF_PROCESSORS` are now allowed through. It is still an allowlist.

## 0.4.0

Behaviour changes to the decimation contract, the Phase 2e gate, and merge
scoping, plus a new briefing reference. All five came out of six end-to-end runs
against a converted customer CAD station and a comparison against a hand-optimized
reference for the same asset.

### Changed

- **A stated triangle target is a ceiling, not a mode switch.** Naming a polygon
  count previously satisfied the condition that switches `decimateMeshes` out of
  error-budget mode, zeroing `maxMeanError` and leaving the quota as the only stop
  condition. Two runs of the same brief differing only in that clause lost 3.84%
  and 0.00% of the asset's surface. A stated target now bounds the result while the
  quality budget stays active.
- **Welding is a validated precondition of decimation**, not an ordering
  preference. When `vertex_weld` fires, weld before reducing: on unwelded CAD
  nearly every edge is a boundary edge, so `pinBoundaries` pins almost the whole
  mesh and decimation barely moves. It also restores `reductionFactor`'s meaning —
  it is keep-percent of the *welded* count, so on ~2.4x redundant input `90.0`
  retained 35.5%.
- **Phase 2e resolves unattended instead of halting.** The gate previously told a
  non-interactive agent to stop and forbade substituting a default, so a brief that
  never mentioned restructuring produced no optimized asset at all. It now resolves
  from measured reuse at the candidate frontier (>=30% recoverable ->
  `deduplicate-internally`, <15% -> `optimize-as-is`, between -> halt as before) and
  records the measurement in `decision_basis`. Not a fixed default: measured reuse
  at the same depth is 10.3% on a production line and 83% on a rack, so one fixed
  answer is wrong for one of them whichever way it points. Interactive behaviour is
  unchanged.
- **Merge scoping now leads with scope, not material.** Phase 4c states the rule
  from the merge spec that decides whether output is usable: one scoped call per
  boundary, never a single global call. A global call buckets by material
  stage-wide and `mergePoint` resolves to the stage root — measured, that produced
  298 meshes of which 39.6% each spanned more than half the asset, with 0 of 617
  named units surviving. The `GeomSubset` ceiling is also quantified: 6,674 subsets
  measured 12 FPS against 50 for the same geometry at 298 draw calls, while RTX was
  66 either way.

### Added

- **`references/briefing-the-skill.md`** — what a request should state and why,
  with three worked briefs for the three asset shapes (production line, rack,
  building/MEP) that need three different dominant operations. Names the two
  phrasings that silently cost quality: a stated triangle count and a stated mesh
  count.
- **A required composition check after every structural op.** Counts and bounding
  boxes are not evidence the output survived. Four observed runtime failures each
  reported success while leaving the layer composing to nothing or to mangled
  prims. Three lines — reference the written layer from a throwaway stage and
  assert the mesh count — catch all four.

### Known issues carried into this release

The five runtime behaviours behind the composition check belong to `usd-optimize`, not
the skill, and are unfixed upstream:

- `merge` on a masked stage with `rootPath` unset deletes every mesh, authors
  nothing, and returns success. Both conditions are required; passing an absolute
  `rootPath` under the target makes merging on a masked stage safe.
  `meshPrimPaths` is not involved.
- `merge` with `rootPath` unset parks output outside the `defaultPrim`; with
  `rootPath` set the path has been seen mis-parsed as relative.
- `flattenHierarchy` at its upstream default, run whole-stage, has deleted the
  `defaultPrim`.
- `merge` leaves the source Xforms standing after fusing their meshes, so a merge
  pass needs a stage-level `pruneLeaves` to finish. The composition check does not
  catch this one: it asserts the mesh count, which is correct on a stage still
  carrying the empty scaffolding.

All are classed `apply_authority: auto` / `loss_class: lossless`. Until they are
fixed, `auto` means "no user decision needed", never "cannot break the layer".

## 0.3.0

First release under this version scheme. See the release notes issued with the
hub sync.
