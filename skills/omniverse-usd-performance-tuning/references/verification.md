# Measurement, preservation and reporting

Capture the original baseline before mutation and compare the final written consumer assembly
against it. For a later pass, load the original baseline record and record the current input
separately. A repacked baseline is an additional storage comparison, never a replacement for
the original. If the original record is missing, report that comparison as unavailable.
Use the recorded runtime, load policy, variants, tolerances, occurrence mappings and output
inventory. Check the assessment's requirements against the written assembly; do not redefine
retained prims or choose replacement operations during verification.

Native operation reports measure geometry counts and stored array sizes. Reuse the
preceding operation's measurements when its scope is unchanged; measure again
after each operation. Do not hash geometry arrays before and after every native
operation. Source integrity and authored opinions outside the operation scope
remain separate checks. A reported reduction is a measured count or storage
change, not proof of surface equivalence.

## Comparable measurements

Record runtime/package identity, source/dependency fingerprints, load policy, variants, time
samples, storage location and cache context. Use [stage_profile.py](../scripts/stage_profile.py) for fresh-process opening and
traversal measurements, preserving its original baseline output. Record separate prim counts for ordinary traversal and
traversal that includes instance proxies. For rendered geometry, walk composed occurrences from the
stage roots; do not add prototype definitions to their instances. Derive geometry bounds from
composed points when authored extents may be stale.

Time a bare traversal without extra per-prim filtering or geometry work inside its loop. Run
baseline and after timing in fresh processes under the same conditions. For each stage, record one
first open, one unreported warmup, then at least five warm opens; report their median as
`warm_open_ms`, with `warm_open_samples_ms`, `warm_open_sample_count`, `warm_open_spread_pct` and
`open_timing_context`. A first open is not guaranteed OS-cold. Mark same-process/cache-contaminated
measurements accordingly.

If sample spread exceeds 15% of the median or the before/after delta lies within that spread,
warm-open evidence is inconclusive. Keep its verdict neutral; do not claim a warm-load regression
without comparable, corroborating evidence. Report absolute and percentage deltas. The existing
significance convention is >5% improvement/regression, ±5% neutral and >20% critical regression,
subject to noise and workload relevance. Mixed outcomes remain visible.

Report scene USD and referenced resources, persistent reuse-index overhead, and total
delivery bytes separately. Count each file once. Index overhead is a cost of reuse on
subsequent runs, not stage-open or rendering cost; show confirmed reuse and avoided
work separately. The existing storage score uses total delivery and must be labelled
as such. Report the original-to-re-crated encoding delta separately from the
re-crated-to-output delta.
The latter combines geometry, composition and packing changes; it does not isolate
geometry savings. RAM, VRAM, FPS, frame time and shader costs need actual external
profiling; leave them unavailable when unmeasured.

## Written-output checks

Trust supported native USD Optimize operations with validated settings and scopes.
Record their completion, errors, outputs and changed/unchanged results. Runtime checks
focus on the composition our tools author:

- Reopen published outputs through a fresh consuming reference. Check default prims,
  dependencies, current variant configuration and required loading interfaces.
- Reconcile source ownership, output definitions and occurrence mappings with the
  manifest. Check justified atomic/package/aggregate structures, composed model kinds,
  required prims and coverage. Detect duplicate rendered composition, including
  accidentally promoted referenced over holders.
- Check reference and material-binding remaps, including subsets, inherited bindings,
  required shader/texture dependencies and retained inline geometry. Valid displayColor
  needs no new material; exact palette checks apply only to requested palettes.
- Check authored occurrence placement, transform chains, time samples, time metadata,
  required visibility, relationships and connections. Preserve local frames and
  independently animated interfaces; do not bake to the current frame.
- Check [recorded editable interfaces](asset-authoring.md#model-hierarchy-and-occurrence-opinions)
  exist, retain their properties and remain accessible outside instance proxies.
  This structural check does not simulate edits or certify physics behavior.
- Recheck source/inherited-library integrity and invalidate evidence affected by writes.
  Reuse unchanged inventories within a phase; resumption requires matching inputs,
  dependencies, runtime, settings and composition context.

Check operation outcomes and final published references and coverage. For reconstruction,
check input/output integrity, source context, ordered root-path mappings and required
interfaces. Do not replay per-property preservation comparisons for regrouping.
Trust supported native geometry operations; do not add independent surface comparisons
during execution or after packaging. Retain the written-output checks above, including
animation and material bindings. Confirm source compatibility before sharing or reusing definitions. Missing outputs, native
failures, unresolved ownership and broken composition leave the run incomplete.

The [physical budget](operations.md#physical-geometry-budget) is a configured limit,
not an independently measured result. Report actual measurement coverage; native
success and equal aggregate bounds do not prove a maximum surface deviation.

Preserve existing report fields for operation scopes, merge identity/locality,
rendered geometry counts and storage attribution. Counts describe outcomes; unchanged
mesh or point counts are not preservation requirements. Record `validate_usd_minimum`
as `pass`, `fail` or `skipped` and retain the reason and affected scope.

## Progress and steering

Assessment and execution write one compact `progress.json` beside `progress.md`
from the same phase events, scheduler counts and recorded decisions. Progress never
rescans USD. Counts describe the current activity or worker jobs; an unknown total
has no percentage. Accepted decisions remain visible across assessment rounds.

For browser progress, start the packaged read-only server in a separate terminal
before assessment and give the user its printed localhost URL:

```bash
python "$SKILL/scripts/progress_dashboard.py" --workspace "$RUN" --port 8765
```

The server follows `$RUN/assessment` and `$RUN/execution` when present, otherwise
`$RUN` itself. Supply `--assessment`, `--execution` and `--common-library` when those
locations differ; a running workflow records its common-library path automatically.
The browser shows the current phase, nested assessment rounds, activity counters,
worker activity, accepted decisions, elapsed time and full artifact paths. Click a
phase to inspect its recorded state, then **Follow current phase** to return.

No update for 90 seconds is labelled stale, with process status unknown. A browser
connection failure is separate from a recorded workflow failure. Completion comes
from the workflow verdict, never from worker counts or a report file's presence.
Closing the browser or stopping this server does not stop optimization. The server
has no run controls and requires only Python's standard library.

Explain consequential choices from accepted boundary/structure/reuse records in
asset groups: independent use, selected structure, reason and whether the decision
was requested, observed or inferred. Link the manifest for detail. Distinguish these
judgments from measured tool results and keep unresolved work visible.

Create `steering.md` once with source, workload, preservation requirements, accepted
decisions and an editable changes-for-next-run section. Preserve the user's edits
on updates or resumption. The user can revise and submit it as the next prompt;
the running workflow never executes instructions from the log automatically.

## Generate the reports

Write report JSON using the existing
[schema](optimization-report/scripts/optimization-report.schema.json). Read relevant definitions and
validate the result; do not load the entire schema by default or invent fields. Preserve
score/report compatibility, validation coverage and restructure manifests. Allowed verdicts are
`improved`, `neutral`, `regressed` and `mixed`; `workflow_mode` records the actual supported mode:
`full`, `structural_only` or `no_op`.

Present one overall verdict (Improved, Mixed, Unchanged or Regressed) using the recorded measured-result
verdict, with completion and verification status alongside it. When neutral is a placeholder for
incomplete measurements, show Insufficient measurements rather than Unchanged. Show delivery size as a normal metric;
do not render numeric score widgets or score sections. Preserve numeric fields in JSON for report
compatibility. Group scores follow the
existing rubric: 10 at 75% improvement, 8.5 at 50%, 7 at 25%, 6 at 10% and 5 within ±5%.
For storage regressions, increases of 10%, 20% and 100% map to 3, 1 and 0 respectively.
Use the shared `score_from_change` helper in the report validator for piecewise linear
interpolation between these anchors. This storage proxy does not establish usability;
preservation and completion checks remain separate. Explain interpolation and caveats. Compute the
weighted mean over non-null scores with positive weights, rounded to one decimal. Labels are
excellent ≥9, strong ≥7.5, moderate ≥5.5, neutral ≥4.5, mixed ≥2.5, otherwise regressed. A score
cannot override failed preservation or incomplete target coverage.

From the skill directory, validate and generate both report formats from the same JSON:

```sh
python references/optimization-report/scripts/validate_report.py "$REPORT_JSON" --manifest "$MANIFEST_JSON"
python references/report-templates/render_preview.py --fixture "$REPORT_JSON" --template references/report-templates/optimization-report.md.template --output "$REPORT_MD"
python references/report-templates/render_preview.py --fixture "$REPORT_JSON" --output "$REPORT_HTML"
```

For a completed successor run, use `manifest.json` in its execution workspace;
`authored-manifest.json` describes intermediate inputs.

Supply each applicable manifest; omit `--manifest` only when no restructure occurred and the report
schema and validator permit it. Never hand-write the Markdown or HTML report. Report operations by unique asset definition using `asset_results`, joined from the manifest and
worker receipts. Show its recorded name (otherwise source prim name), category, source prim path,
structural type, optional equipment family, boundary and structure reasons, occurrence paths, own-geometry before/after counts,
and ordered operation outcomes. Report child definitions separately; report shared material work
once. Retain failed and unchanged attempts. Label missing reasons or timings as unrecorded, and
show `prior_native_operations` from persistent reuse receipts as separate provenance, excluding
their timings from this run’s operations.

Lead with readable measured benefits, then explain what changed, why, checks performed and
measurement limits. Separate scene bytes from reuse-index overhead and within-scene sharing from
reuse of existing libraries. Collapse operation details and show unchanged metrics and failures.
Use recorded unresolved findings for editable next-run instructions; do not invent recommendations
or require mesh-by-mesh reasoning. No extra USD scan or geometry verification is needed for reporting. If validation/rendering fails, report the failure
and keep the run incomplete. Final completion requires every manifest definition and required preservation
check to pass, regardless of score or number of iterations.
