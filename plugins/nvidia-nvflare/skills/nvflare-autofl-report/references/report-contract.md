# Auto-FL Final Report Contract

## Inputs

The report generator consumes product Auto-FL artifacts from the job directory:

- `results.tsv`: append-only result evidence;
- `autofl.yaml`: requested and imported campaign contract;
- `.nvflare/autofl/campaign_state.json`: stop/finalization state;
- `.nvflare/autofl/candidates/*/candidate_manifest.json`: optional detailed
  candidate provenance;
- `progress.png`: existing plot to refresh safely.

The current ledger contract includes status, name, score, runtime, changed
files, hypothesis/diff summary, exact run command, artifacts, failure reason,
candidate manifest, base candidate, patch hash, metric name/source/artifact,
candidate kind, algorithm family, and literature event ID. Older ledgers may
omit newer fields; the report discloses missing provenance rather than failing
when the essential status and score columns remain readable.

## Termination

`final_response_allowed=true` is the preferred deterministic proof that a
campaign may be finalized. For abrupt process termination, the user and agent
may explicitly confirm interruption. This assertion is report provenance only:
it must not mutate campaign state. `--confirm-interrupted` bypasses only stale
stop state. Finalization is refused when campaign state reports pending work,
the ledger has a `candidate` row, or an available manifest is not in a
recognized terminal status: `keep`, `discard`, `crash`, or `abandoned`.
Missing, unknown, or unreadable manifest status also blocks finalization
because the helper cannot prove that its candidate is complete. Finalize or
abandon such candidates before generating an authoritative report.

The reporter acquires the same nonblocking
`.nvflare/autofl/campaign.lock` used by runner lifecycle actions before reading
campaign evidence and holds it through plotting and report writes. If another
action owns the lock, reporting fails cleanly rather than publishing a mixed
snapshot. On POSIX, the persisted lock file is a stable lock target rather than
proof of a live owner. The reporter can therefore read a read-only campaign
archive when the lock file already exists and the three outputs point to a
writable location. NVFlare supports POSIX platforms; reporting fails clearly
when `fcntl` locking is unavailable.

## Candidate Lineage

Candidates are based on the current best source snapshot. A retained candidate
can therefore record `changed_files=none` while inheriting algorithm code from
its `base_candidate`. Follow `base_candidate` links back to baseline and union
changed files across the chain. Mark lineage partial when an ancestor row or
manifest is unavailable.

Ledger paths may refer to the container or host where a candidate ran. When a
recorded manifest path is unavailable, the reporter checks the campaign-local
`.nvflare/autofl/candidates/<candidate_id>/candidate_manifest.json` path and
accepts it only when the manifest `candidate_id` exactly matches the ledger
candidate name. The JSON summary preserves `recorded_path`, `resolved_path`,
and whether resolution used the recorded or campaign-local path.

## Literature Evidence

A `literature` row opens a checkpoint and records a `literature_event_id`.
Associate candidate rows carrying that ID with the checkpoint, even when a
later review was recorded before the batch completed. Candidates without an
event ID are not attributed to a literature checkpoint; the report does not
guess from row order. Compare the best finalized `keep` or `discard` score in
that evidence with the retained incumbent immediately before the review.
Pending candidates and crashes are attempts but never scored improvements,
even if a malformed crash row contains a numeric score:

- `helped`: a following candidate improved the incumbent;
- `matched`: the best following candidate tied the incumbent;
- `not_confirmed`: candidates ran but none matched or improved the incumbent;
- `failed`: attempts produced no score;
- `not_evaluated`: no candidate attempt followed the checkpoint;
- `incomplete`: authoritative campaign state shows that fewer than the required
  exploration candidates completed before the campaign stopped. The report
  preserves measured evidence but does not present the partial batch as a
  confirmed negative result.

Preserve `[src: ...]` markers from the checkpoint. These are campaign-recorded
source identifiers, not independently verified citations.

## Result Selection

`best` is strictly the best scored retained result: a baseline or `keep` row.
`best_observed` may identify a better scored `discard` as unretained evidence.
`candidate` and `crash` rows are excluded from retained best selection,
running-best milestones, and literature improvements. Literature tables render
status first, so scored crashes remain `crash` and unscored discards remain
`n/a`.

Baseline identity is strict: only `status=baseline` is a baseline. Names and
run-command text do not change result status. The report carries the runner's
per-row metric extraction provenance, candidate kind, algorithm family, and
literature linkage into the JSON summary and best-candidate Markdown section.

## Outcome Synthesis

The human-readable report leads with three deterministic views built from the
same ledger evidence:

- **Selected Candidate And Why** identifies the best scored retained result,
  its baseline delta, declared family and kind, hypothesis, and retained
  lineage. It does not claim robustness from a single run.
- **What Helped** contains only `keep` rows that strictly improved the retained
  incumbent at that point in the campaign. Major entries are selected by the
  first, final, and largest measured improvements.
- **What Did Not Help** selects a representative scored `discard` for each
  explicit algorithm-family/literature-event group and groups crashes or
  unscored discards by their recorded failure reason.

`outcome_summary.families` reports `helped`, `mixed`, `not_confirmed`, or
`failed` from explicit `algorithm_family` values; missing values remain
`unclassified`. The helper never infers a family or mechanism from candidate
names. `outcome_summary.literature` provides a compact count and strongest
helped checkpoint while `literature_reviews` retains every detailed event.

Trajectory compression preserves the first and final running-best rows and
fills the remaining limit with the largest objective improvements, restoring
chronological order for presentation. A scored discarded row may remain
visible as observed trajectory evidence, but it never enters **What Helped** or
retained selection.

The objective contract records two distinct provenance fields. `metric_source`
describes where measurements came from and defaults to `NVFlare metric
artifacts`. `metric_contract_source` records how the importer selected the
metric, for example `user_request`, `arg:key_metric`, `core_default` when the
job uses NVFlare's resolved default key metric, or `default` when the importer
uses an unresolved metric fallback.
Product Auto-FL campaigns preserve the imported `objective.mode` (`min` or
`max`) and its provenance. Reported improvement is direction-adjusted and
therefore positive when the objective improves, while ledger scores remain raw.
A report refuses legacy minimization evidence without direction provenance or
a malformed/conflicting direction contract.

## Comparability

`autofl.yaml` describes the imported/declarative contract; exact commands in
`results.tsv` describe what executed. Report both. Compare baseline and best
command options and flag changes to clients, rounds, local epochs/steps, batch
sizes, data partitioning, seed, model architecture, model-size limits,
aggregation, or final-evaluation clients.

The report derives candidate attempts, baseline score, and improvement from
the ledger and cross-checks those values against authoritative campaign state.
It also verifies that the state's `results` pointer names the ledger being
reported. It exposes state/ledger disagreements as warnings rather than
silently choosing one account. The state-derived `abandoned_candidates` count
remains visible because abandoned manifests do not necessarily have ledger
rows.

When a test-like metric guided multiple candidate decisions, state that the
selected candidate needs one final evaluation on an untouched holdout. Do not
silently present repeated test-set selection as an unbiased final estimate.

## Outputs

- **autofl_final_report.md**: human-readable review artifact;
- `autofl_report_summary.json`: machine-readable summary using schema
  `nvflare.autofl.report.v1`;
- `progress.png`: refreshed using the `nvflare-autofl` product plotter when
  plotting is available.

The Markdown report embeds an available progress plot using a path relative to
the report file. JSON and the report artifact appendix retain absolute paths
for provenance.

The JSON summary remains `nvflare.autofl.report.v1` and includes
`artifacts.progress_plot_available` and `objective.metric_contract_source`,
plus `selection`, `outcome_summary`, `abandoned_candidates`, and
`state_accounting` for the concise evidence synthesis and consistency check.
Each `literature_reviews` entry carries `completion`, either an object with
`completed`, `required`, `complete`, and `reason`, or `null` when campaign state
does not record a matching exploration batch.
Missing plotting dependencies or an invalid existing PNG produce warnings and
`progress_plot_available=false`, but do not suppress the Markdown or JSON
report. The invalid or failed plot artifact is preserved and is not embedded
in Markdown.

All relative helper path arguments, including `--plotter`, resolve from the
campaign directory rather than the invoking shell's current working directory.
Before plotting or writing, the helper canonicalizes the three writable output
paths and rejects aliases with the ledger, state, config, agent context,
plotter, campaign lock, candidate manifests, `job.py`, or files listed in
`trust_contract.allowed_edit_paths`. Outputs within the campaign directory
must not match `trust_contract.allowed_create_patterns`. The progress,
Markdown, and JSON outputs must also be mutually distinct. Case-folded paths
and existing symlink or hard-link aliases are treated as collisions so a
configuration is portable to case-insensitive filesystems.

Report generation must be independent of Git and must not edit campaign source,
the ledger, manifests, or state.
