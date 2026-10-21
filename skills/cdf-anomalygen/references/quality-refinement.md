# Quality-refinement reference — Step 6 (quality_refine)

Progressive-disclosure detail for **Step 6 — quality refinement** in the parent `SKILL.md`.
Read this for the draw strategy (uniform vs. Bayesian optimization), the ranges/seeding, and `select`.
If this file conflicts with `SKILL.md`, `SKILL.md` wins.

## What the step does

For each generation sample, search a better `(guidance, crop_ratio)` than the Step 4 default. A round =
**draw** a candidate per sample → **generate** → **evaluate**; after `num_search_run` rounds, **select**
keeps each sample's best-scoring render across the original bucket + requested rounds into `searched/`.

`quality_refine` has three subcommands. **`run`** is the one `SKILL.md` Step 6 calls — it drives the
whole loop and gates each round. `draw` (write one round's `testcase.jsonl`) and `select` (assemble
`searched/`) are the pieces it composes, and stay separately callable for re-running a single round by
hand. **`select` always runs** — with 0 rounds it clones the original bucket, so `searched/` always
exists for Step 7.

> **"select ran" ≠ "refinement ran."** The default is `num_search_run=3`: run the
> draw→generate→evaluate loop **3 times before** `select`. Running only `select` (clone-only) is correct
> **only** if you deliberately chose `num_search_run=0` — otherwise you skipped refinement and gain no
> per-sample quality lift. Tell-tale of a skipped Step 6: no `${OUT}/rounds/round_*/`, and `searched/`'s
> `source` column is entirely `original`.

## `draw` — how each round is proposed

Per sample:

- **Round 1 (or no history): uniform.** With only the original observation so far, `draw` samples
  `(guidance, crop_ratio)` uniformly from the ranges.
- **Rounds 2+: per-sample Bayesian optimization.** When `--original`/`--original_kpi` (+ `--rounds_dir`)
  supply history, `draw` fits an **independent GP** on *that sample's own* `(guidance, crop_ratio) →
  score` observations (inputs scaled to `[0,1]²` by the ranges; Matérn + white-noise kernel) and
  **Thompson-samples** one point from `--bo_candidates` (default 1024) candidates.
- A sample falls back to **uniform** if it has fewer than `--bo_min_obs` (default 2) observations, or
  fewer than 2 *distinct* `(guidance, crop_ratio)` points (a GP needs spread to fit).

Key flags:

| Flag                            | Default     | Note                                                                                      |
| ------------------------------- | ----------- | ----------------------------------------------------------------------------------------- |
| `--base_testcase`               | —           | the generation `testcase.jsonl` to redraw                                                 |
| `--output`                      | —           | this round's `testcase.jsonl`                                                             |
| `--seed`                        | `42`        | vary per round (pass the round number) for reproducible draws                             |
| `--recipe`                      | —           | the run-dir recipe; supplies the trained `crop_ratio_range`                               |
| `--guidance_range`              | `1.0 10.0`  | draw range for guidance                                                                   |
| `--crop_ratio_range`            | recipe band | crop_ratio draw range; the recipe's trained band unless given, narrowed to it (see below) |
| `--original` / `--original_kpi` | off         | original bucket + its `kpi.json` — **enables BO**                                         |
| `--rounds_dir`                  | off         | prior `round_<r>/` buckets (each with a `kpi.json`) pooled into each sample's history     |
| `--current_round`               | inferred    | current round; excludes this/later stale rounds (`run` always supplies it)                |
| `--score`                       | `nn`        | per-sample metric to optimize (see the choices below)                                     |
| `--bo_min_obs`                  | `2`         | min observations before a sample uses BO instead of uniform                               |

`--crop_ratio_range` defaults to the band `--recipe` trained under (fallback `1.5 2.5`). Generation
clamps every `crop_ratio` into that band, so a wider draw only repeats its edge: an explicit range is
narrowed to the band (with a warning), one entirely outside it is refused, and `LOW` must be at least
`1.5`, the floor a recipe's band has.

Without `--original`/`--rounds_dir`, every sample draws uniformly (a plain random round).
When calling `draw` by hand against existing history, pass `--current_round <r>` unless `--output`
uses the canonical `round_<r>/testcase.jsonl` path. An ambiguous path is rejected rather than pooling
current or future rounds left by an earlier run. This boundary applies to every score mode.

`--score` accepts `nn`, `mnn`, `completeness`, `precision`, `boundary_iou`, `aq_nn`, or `aq_rank`.
`nn` is the default. Every score except `aq_rank` is an absolute per-sample value read directly from
each round's `kpi.json`.

`aq_rank` is deliberately cumulative. Immediately before Round `r`, refinement pools Original plus
the completed rounds `< r`, groups candidates by anomaly type, and calls the shared
AQ-Rank scorer with compass diagnostics over each group. Adding a round therefore re-ranks earlier
candidates too; the bucket-local `aq_rank_score` written by `evaluate.py` is not reused whenever
candidates compete. The re-ranked values become the per-sample BO observations for that draw. Final
selection repeats the same calculation over Original plus the rounds generated by the current `run`
and chooses each sample's winner from that bounded pool. In the zero-round clone path, a stored value
may appear in the audit CSV but cannot affect selection.

AQ-Rank ranks only candidates that carry finite `nn_score`, `completeness_score`,
`precision_score`, and `boundary_iou_score` values. A candidate with a missing/non-finite input is
logged and excluded from the rank pool, so it cannot be rewarded when an axis is oriented
negatively; other candidates in the bucket continue normally. A bucket with no eligible candidate,
a missing SAM2 checkpoint affecting every candidate, or a CSV/KPI join mismatch is a hard failure.
If one sample has no eligible render across any otherwise-valid bucket, final selection logs that
sample and keeps its Original render as an explicit fallback, not as an AQ-Rank win.
When a per-type pool has fewer than five eligible candidates, the compass direction is unreliable;
the code logs this and ranks that type by NN only.

The compass directions are fitted **between candidates across the pooled samples**, while BO and
final selection use those directions to compare **variants of the same sample**. Those relationships
can disagree (for example, an axis may correlate with defect size between samples but improve within
every sample), so cumulative AQ-Rank does not guarantee that each enabled direction is correct for
within-sample refinement. Every re-ranking logs each anomaly type's pool size and every axis's AUROC
and `+1`, `-1`, or `disabled` decision. Retain those logs when auditing a run. A within-sample
orientation rule would be a different algorithm and requires separate experiment and human validation.
The current `|AUROC - 0.5| >= 0.05` gate is deliberately permissive: it filters constant or nearly
signalless axes, but it is neither a significance test nor independent evidence that an enabled sign
will improve a within-sample choice.

### Choosing `nn` or `aq_rank`

- Use `nn` for the default, correspondence-focused search and the fewest metric dependencies.
- Use `aq_rank` when selection should jointly consider correspondence, defect coverage, spill, and
  boundary agreement. It is an alternative objective, not a cross-pass validation KPI or a
  human-alignment guarantee; inspect the logged compass directions.

On the same 253 outputs across seven datasets, NN and AQ-Rank selected candidates from Rounds 2–3 at
similar rates (53.8% and 51.4%). A separate blinded 2AFC analysis of those outputs with one reviewer
judged 63% of the final-output pairs similar and detected no statistically significant perceptual
difference. This supports both as usable search objectives while keeping NN as the conservative
default; it does not establish perceptual equivalence or AQ-Rank superiority.

## `run` — the round loop

For `r` in `1..num_search_run` (default 3): `draw` → `generate.py` (same `--checkpoint`/`--recipe` as
Step 4, plus any `--base_checkpoint`/`--base_seed`/`--guardrail` given to `run`) → `evaluate.py`
writing `round_<r>/kpi.json`, then `select` and a final `evaluate.py` on
`searched/`. Each round's `kpi.json` feeds the next round's BO history via `--rounds_dir`, and each
round launches its own `torchrun --standalone` so back-to-back rounds never contend for a fixed
rendezvous port.

> **A round counts only with one image per testcase row, a matching generation CSV, AND its
> `kpi.json`.** Both consumers key off
> `kpi.json` — `select` scores buckets from it, `draw` pools it into the next round's BO history — so a
> round that stopped early would be **silently dropped from both**: GPU time wasted, and later rounds
> searching with less history. `run` checks this after every round and **exits 1** rather than
> assembling a stale `searched/`; it names what was short. Repair that round in place — re-run its
> `generate.py` (it overwrites) and `evaluate.py` — then finish with `select` plus the final
> `evaluate.py`. Do **not** re-run `run`: it has no resume, so it would redraw and regenerate every
> round from 1, discarding the ones that already succeeded.

`run` takes `draw`'s tuning flags (`--guidance_range`, `--crop_ratio_range`, `--score`, `--bo_min_obs`,
`--bo_candidates`) and forwards them to every round; the per-round `--seed` and internal
`--current_round` are the round number. Use `--dry_run` to print each stage's command without running
it.

## `select` — assembling `searched/`

Inside `run`, `select` reads the original bucket plus `round_1` through the requested
`num_search_run`; stale later rounds cannot compete. A standalone `select` reads all positive numeric
`round_<r>/` buckets by default and ignores other `round_*` names. Pass `--max_round <N>` to reproduce
a bounded run manually (`--max_round 0` clones Original). Per sample index, it keeps the render with
the best `--score` (`nn` by default) into `searched/`. For `aq_rank`, it first recomputes one per-type
ranking over the bounded candidate pool. `searched/` is what Step 7 pseudo-labels.

`${TASK}_generation_result.csv` gains four columns: `nn_score` and `mnn_score` (the winner's *actual*
metrics, always under their own names regardless of `--score`), `selected_by` (the metric that made
the decision), and `source` (which bucket the winner came from). `search_summary.csv` is
the per-sample audit trail: `source`, `selected_by`, `score`, `original_score`, `improved`.

The `source` distribution is the readout on the search — how many samples came from `original` (no
improvement found) vs. each round. All-`original` means refinement bought nothing.

## Scoring the result

`select` does not evaluate what it assembled, so `run` closes with `evaluate.py` on `searched/`.
Under `--score nn`, compare `Average.nn_score` with the Step 5 `${OUT}/${NAME}_kpi.json`; per-sample
selection can improve the assembled batch even when no whole round beats Original on average.

Do not compare AQ-Rank means across Original, rounds, and `searched/`: percentile ranks are defined
by the pool being ranked, so those means are not a cross-pass quality scale. Under `--score aq_rank`,
use `search_summary.csv` for Original/round winner sources, compare the selected images' paired raw
metrics, and confirm the trade-off with human review or downstream detector performance.

## Notes

- BO is **per sample and independent** — there is no shared/global surrogate; each sample explores its
  own response surface, so a hard sample keeps searching while an easy one settles.
- Round 1's log line reads `mode=uniform (no sample had enough history)` even with `--original` passed:
  one observation per sample isn't enough to fit a GP, so round 1 explores blind by design. From round 2
  the line reports the real split (`bayesopt <n>/<total> samples`).
- `select` only ever *keeps* the best render across the original + rounds under the chosen numerical
  objective. That is not a guarantee of better human-perceived quality, especially for the AQ-Rank
  orientation limitation above. If a sample is unscored in the original bucket (a `NaN` score in
  `kpi.json`), any scored round wins it by default because NaN sorts worst. Check for empty
  `original_score` cells in `search_summary.csv` if a `source` distribution looks surprising.
