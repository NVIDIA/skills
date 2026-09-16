# Cosmos3 DEFT Report Rendering

Use `"$PYTHON" scripts/render_report.py` as the only renderer for the source template at
`references/DEFT_Loop_Report.html`. It writes one self-contained
`${RESULTS_DIR}/DEFT_Loop_Report.html` from canonical disk evidence through a
temporary sibling and atomic replacement. `init_deft_state.py` invokes it at
loop start and `commit_stage.py` invokes it after every valid commit, so report
freshness never depends on an agent remembering a final task.

The Cosmos3 report intentionally shares the ChangeNet report's visual contract:
the same 1280 px single-column hero and cards, NVIDIA dark palette, typography,
32 px vertical rhythm, section headers, table treatment, KPI banner, and status
panels. Model-specific content and evidence differ, but the report must not
introduce a separate grid, card geometry, or visual hierarchy. The renderer
test compares these shared CSS properties across both source templates.

Render a hero KPI banner only when the KPI is met or the run ends at a committed
hard stop. For in-progress runs and terminal runs that remain short of target,
render no banner. Never show an informational `i`, `BEST RESULT RECORDED`, or
`from target after the approved iteration budget` callout; the outcome panel
and metric tables already carry that evidence.

Required sections:

1. **Run Configuration & Outcome** — platform, model, bare mode, KPI contract,
   GPU/node shape and exact GPU model, timestamps, current/terminal status; a
   two-column DEFT experiment summary with baseline/end KPI, routing, total and
   average measured runtime, and SDG throughput; then a Training Set Growth
   table with `Iteration`, `KNN Raw Mined`, `SDG Generated`, `New Unique Images
   (After Dedup)`, `Training Set Total`, and `Δ`. `SDG Generated` comes from
   the committed CSV row count. `assemble_summary.json.output_records` owns the
   cumulative total; its difference from the preceding total owns both New
   Unique and `Δ`. The committed JSON length is a compatibility fallback, while
   `unique_target_images.new_after_dedup` remains a batch diagnostic in
   Augmentation Volume.
2. **Benchmark KPI Trend** — immediately follows Run Configuration & Outcome;
   only frozen Benchmark results carry a pass/fail verdict.
3. **Dataset Isolation** — Proxy/Benchmark/Mining input paths, per-iteration
   AnomalyGen output, generated Train artifacts by iteration, OK/NG counts,
   Benchmark SHA-256, and role ownership. Attribute every Train record to its
   producer: mined real pairs or synthetic AnomalyGen pairs.
4. **Prompt Examples** — up to three distinct first-user prompts read from the
   recorded Proxy, Benchmark, and Mining annotations. Group identical prompts
   across roles, show their record count and exact `OK`/`NG` response contract,
   escape file-derived text, and truncate only the displayed preview at 600
   characters. Do not embed the rest of an annotation record.
5. **Iteration Metrics** — for baseline and every completed iteration:
   accuracy, NG recall, NG precision, NG F1, false accepts, false rejects,
   unknowns, KPI pass/fail. Label every figure with its source split; only
   Benchmark figures carry the KPI verdict.
6. **Pipeline Execution** — ordered committed stage events and positive,
   measured durations from `deft_state.json.events`. Summary totals exclude the
   administrative `loop_stop` event and label partial historical logs instead
   of interpreting zero as elapsed time.
7. **Augmentation Volume** — per iteration and per producer. Mining: raw
   candidates, cosine-kept paths. AnomalyGen: requested `num_SDG`,
   AMP-allocated (the sum of the committed `allocation.json`), generated, and
   the per-defect-type breakdown, or the documented skip and its reason. Then
   mining candidates, already-mined rejects, newly selected novel target
   images, and cumulative training records by iteration. Read the
   per-iteration history summary and run-level ledger; surface any
   recommendation to increase top-K or expand the pool.
8. **Artifacts** — baseline base-model reference, iteration checkpoints,
   Proxy/Benchmark results, RCCA, mining, AnomalyGen `SDG_result.csv`,
   `allocation.json`, and generated ShareGPT, assembled JSON, and
   validation-report links/paths.
9. **Hard Stops / Warnings** — committed error events, if present.

Cosmos3 bare mode is a discrete OK/NG classifier.

## Terminal iterations have no Proxy artifacts

The loop evaluates the frozen Benchmark gate first and runs Proxy evaluate and
RCCA only when it continues to another iteration. The iteration that ends a run
— gate passed, or `N = max_iterations` — therefore has Benchmark results and
KPI evidence but no `proxy_results_json`, `proxy_gaps_summary`,
`false_accepts_json`, or `false_rejects_json`. Its `stage_completed` is
`benchmark_metrics` rather than `proxy_rcca`.

This is the designed shape, not a gap. Render those cells as
`not run (terminal iteration)`, not as `not available`, and never present the
absence as a missing artifact, an incomplete iteration, or a failure. Do not
carry a previous iteration's Proxy numbers forward to fill them.

Escape all file-derived text before inserting it into HTML. Do not embed
credentials, environment values, full annotation records, or full model logs.
Mark missing optional artifacts as `not available`; never fabricate a metric or
stage.
