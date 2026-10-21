# Verification reference — per-step gates

Progressive-disclosure detail for the **Verification** section in the parent `SKILL.md`: one gate per
step, named after the step (`dataset`, `amp`, `train`, `generate`, `eval`, `refine`, `label`) — run it
before treating that step as done. If this file conflicts with `SKILL.md`, `SKILL.md` wins.

Uses the shared vars set at the top of the SKILL.md step list (`${DATASET_DIR}`, `${VAL_DIR}`,
`${GEN_DIR}`, `${OUT}`, `${NAME}`, `${TASK}`, `${NUM_SDG}`, `${NUM_SEARCH_RUN}`, `${AMP_DIR}`,
`${RUN_DIR}`), plus `${MASK_BACKEND}` — the run's `mask_backend`, which the `dataset` gate branches on
(unset ⇒ `amg`, the skill's default) and which selects gate **amg** or **amp** at Step 2. The `amg`
gate holds each set to `${NUM_SDG}` — the same row-count rule as **amp**; both backends split that
total by the same allocation.

## `dataset` — contract layout on disk (Step 1)

`${DATASET_DIR}` is whatever the caller passed to `set_pipeline_vars.sh --dataset_dir`; the script does
no I/O, so **this gate is the first thing that touches the path at all**. A typo, a stale path or a
dataset that was never staged surfaces here and nowhere earlier.

Before anything else, `${DATASET_DIR}` must hold, per `[texture, defect]` pair,
`{texture}/clean_image/`, `{texture}/anomaly_image/{defect}/`, and `{texture}/mask/{defect}/` (each mask
is the anomaly's stem + `_mask`). `defect_spec.jsonl` is required **only when `mask_backend=amp`** —
it is what allocate → pair → place routes on; the default `amg` never reads it (nor does any step
after 2), so a dataset without one is complete under `amg`, not a failure. `cad` and `cad_tht`
defects (an `amp` routing) also need `{texture}/cad_mask/<clean_stem>.png` — one per clean image, no
`{defect}` level — plus `semantic_segmentation_labels.json`. For `cad_tht` each `cad_mask` must
additionally be the **same size** as its clean image, and the label file must declare `pad`,
`hole`, and a lead class — `pin_tip` or `pin`.

Set `MASK_BACKEND` to the run's `mask_backend` first (it defaults to `amg`, matching the skill):

```shell
if [ "${MASK_BACKEND:-amg}" = amp ] && [ ! -f "${DATASET_DIR}/defect_spec.jsonl" ]; then
  echo "dataset gate FAIL: mask_backend=amp requires ${DATASET_DIR}/defect_spec.jsonl"
elif ls -d "${DATASET_DIR}"/*/clean_image "${DATASET_DIR}"/*/anomaly_image/*/ "${DATASET_DIR}"/*/mask/*/ >/dev/null 2>&1; then
  echo OK
else
  echo "dataset not in contract layout"
fi
```

**The directories existing is not enough — check the pairing too.** The loader pairs each anomaly image
with `{stem}_mask` *in the image's own extension, falling back to `.png`*, and an image whose mask is
missing is **silently skipped**, not reported. So a typo'd or absent mask quietly shrinks the training
and reference sets, and you only get a hard error when *every* mask for a defect is missing:

```shell
miss=0; seen=0
for img in "${DATASET_DIR}"/*/anomaly_image/*/*; do
  # Also drops the pattern itself when nothing matched: it ends in `*`, so it fails this test.
  case "${img,,}" in *.png|*.jpg|*.jpeg) ;; *) continue ;; esac
  seen=$((seen + 1))
  tex="${img%/anomaly_image/*}"; defect="${img#*/anomaly_image/}"; defect="${defect%%/*}"
  base="$(basename "${img}")"; ext="${base##*.}"; stem="${base%.*}"
  [ -f "${tex}/mask/${defect}/${stem}_mask.${ext}" ] || [ -f "${tex}/mask/${defect}/${stem}_mask.png" ] \
    || { echo "unpaired: ${img}"; miss=$((miss + 1)); }
done
# `seen` is load-bearing: without it a wrong DATASET_DIR sweeps nothing and still reports OK.
if [ "${seen}" = 0 ]; then echo "no anomaly images under ${DATASET_DIR} — wrong path or empty dataset"
elif [ "${miss}" = 0 ]; then echo "pairing OK (${seen} images)"
else echo "${miss}/${seen} anomaly image(s) with no mask — fix before training"; fi
```

A missing dir or an unpaired `_mask` will otherwise surface as empty AMP placements or a
training/scoring error later — fix the layout now.

## `amg` — both mask sets delivered (Step 2, the default backend)

Under `--num-sdg` (what `SKILL.md` Step 2 runs) AMG delivers **exactly `num_sdg` rows**, split across
the types by the same rule AMP allocates by — so this is the same row-count check as the `amp` gate
below, applied to every set the mode produced: `full` builds both, `finetune_only` only the validation
set, `generation_only` only the generation set.

```shell
EXPECT="${NUM_SDG}"   # with --per-defect-counts instead: the SUM of that dict
for d in "${VAL_DIR}" "${GEN_DIR}"; do    # finetune_only: VAL_DIR only · generation_only: GEN_DIR only
  [ -f "$d/testcase.jsonl" ] || { echo "FAIL $d: no testcase.jsonl"; continue; }
  rows=$(wc -l < "$d/testcase.jsonl")
  [ "$rows" = "${EXPECT}" ] && echo "OK   $d: ${rows} rows" || echo "FAIL $d: ${rows} != ${EXPECT}"
done
# Both logs, because re-running only one set would leave the other's failure in place.
grep -hoE 'degraded\[(type_failed|model_smoke_failed)[^]]*\]' "${LOG_DIR}"/amg_*_"${NAME}".log \
  | sort | uniq -c | sort -rn
```

- **Rows = `num_sdg` exactly**, however the total divides: 25 over 3 types is 9/8/8, not 9 each.
  Only a run sized with `--per-defect-counts` expects something else — that override's sum wins, and
  is authoritative, exactly as AMP's `--per_defect_counts` is.
- **More mask files on disk than rows is normal.** The fill loop delivers in beam-sized batches and
  then trims to the target, logging `trimmed from N`; the surplus goes to
  `{texture}/_overflow/{defect}/` + `testcase_overflow.jsonl` and is deliberately excluded from
  `testcase.jsonl`. `N` well above a type's share is expected at small counts — count rows, not files.
- **Short row count → re-run the same command, do NOT pass `--overwrite`.** Matching settings are the
  resume switch: a re-run counts what is already delivered and produces only the shortfall.
  `--overwrite` destroys the out-dir's masks and starts over, throwing away everything the run earned.
- **`type_failed`** → a type delivered nothing; the run says why and already exited non-zero. Fix that
  cause and re-run rather than proceeding with a short set — every downstream step is blind to which
  types are missing.
- **`model_smoke_failed`** → the capability probe rejected a model before the batch spent anything.
  Fix the credential/model name; do not reach for `--skip-smoke`, which just moves the same failure
  into the middle of the batch.

## `amp` — testcase built (Step 2)

`roi_allocate`+`roi_pair` alone (just `allocation.json`+`amp_samples.json`) are **not** a finished AMP —
`roi_place` must run and write `testcase.jsonl`; `roi_place_pipeline` stops with the offending stage
named if one does not. The **generation** set must then match the row count the allocation asked for:

```shell
EXPECT="${NUM_SDG}"   # with --per_defect_counts: the SUM of that dict instead, which is authoritative
[ "$(wc -l < "${GEN_DIR}/testcase.jsonl")" = "${EXPECT}" ] && echo OK || echo "AMP count != ${EXPECT}"
```

- **Plain `--num_sdg`** → uniform allocation sums to *exactly* `num_sdg`, so `EXPECT=${NUM_SDG}`.
- **With `--per_defect_counts`** → the override sum wins by design and a differing total is **correct,
  not a failure** (see `mask-placement.md`); set `EXPECT` to that sum before
  judging the gate.
- **Fewer rows than `EXPECT`** → a defect lost placements (see *AMP short of allocation* in Common
  Pitfalls); check the `roi_place` logs and fix before generating.
- **When every clean image must receive an output, the row count is not enough.** The diversity
  pairing hits the right row count while reusing some clean images and skipping others, so count the
  *distinct* ones too:

  ```shell
  DISTINCT=$(python3 -c \
    'import json,sys; print(len({json.loads(l)["image_filename"] for l in sys.stdin if l.strip()}))' \
    < "${GEN_DIR}/testcase.jsonl")
  [ "${DISTINCT}" = "${EXPECT}" ] && echo OK || echo "AMP covered ${DISTINCT}/${EXPECT} joints"
  ```

  A shortfall means the run used the diversity pairing, which is the default. Report it and say that
  `per_clean_pairing=on` is what covers every clean (`mask-placement.md`) — the option is the user's
  to set, so do not re-run with it unasked. Only run this check when the batch has that requirement;
  a validation set, and any batch whose cleans are a reusable pool, covers fewer by design.
- **More rows than `EXPECT`** → most often **stale placed masks from an earlier AMP run** in a reused
  `--output_dir`: `roi_place` globs `*__seed*.png` back up, so leftovers from a previous call are
  counted as this one's. Clear `${AMP_DIR}` and re-run rather than adjusting `EXPECT`. Note rows =
  the sum of each record's `n_seeds` (one row per placed mask), which is exactly what
  `allocation.json` sums to — so a mismatch is a stale-file or wrong-call problem, not an arithmetic
  one (see `mask-placement.md`).

## `train` — training finished (Step 3)

Reached `trainer.max_iter` (not just an early `save_iter` checkpoint), and the best-checkpoint pointer
written at train end exists — it names the iteration Steps 4–7 generate from. Gate Step 4 on
`train_control.sh wait` returning **0**, not on the first checkpoint appearing — see
`fine-tuning.md`.

`Done with training` is logged only at `max_iter`, so it already implies the final checkpoint:

```shell
grep -q 'Done with training' "results/train_${NAME}.log" \
  && [ -s "${RUN_DIR}/checkpoints/best_checkpoint.txt" ] \
  && echo "train gate PASS: best = $(cat "${RUN_DIR}/checkpoints/best_checkpoint.txt")" \
  || echo "train gate FAIL"
```

A missing pointer after a completed run means validation never scored (check `testcase_jsonl`) or the
callback warned — see the log. Fix the run; do **not** fall back to `latest_checkpoint.txt`, which names
the last iteration rather than the best.

## `generate` — generation complete and well-formed (Step 4)

`generate.py` emits **one image per testcase row** and never drops by quality (`PSNR` is recorded in the
CSV, never a filter). The **one** legitimate shortfall is the content-safety guardrail, on by default:
a blocked caption or composite writes no image and is logged to `guardrail_blocked.csv` instead. So
images + blocked must equal the rows, and the CSV must carry the canonical header:

```shell
BLOCKED=$([ -f "${OUT}/guardrail_blocked.csv" ] && echo $(( $(wc -l < "${OUT}/guardrail_blocked.csv") - 1 )) || echo 0)
head -1 "${OUT}/${TASK}_generation_result.csv" | grep -Eq '^output_filename,image_filename,mask_filename,anomaly_type,.*,index\r?$' \
  && [ "$(( $(ls "${OUT}"/reconstructed_image/*.png | wc -l) + BLOCKED ))" = "$(wc -l < "${GEN_DIR}/testcase.jsonl")" ] \
  && echo "OK (${BLOCKED} guardrail-blocked)" || echo "generation partial/malformed — re-run Step 4"
```

`guardrail_blocked.csv` is written whenever the guardrail ran, header-only when it blocked nothing, and
not at all under `--no-guardrail` — so an absent file is zero either way. A **nonzero** count is not a
failure: it is the guardrail doing its job, and those rows are simply absent from the batch. Read the
`message` column before re-running, since a re-run reproduces the same block.

A shortfall the blocked count does *not* explain, or a non-canonical header, means Step 4 didn't finish
— **re-run it; never hand-write or patch the CSV**. Step 6 reports a malformed
`${TASK}_generation_result.csv` as an input error and stops before using incomplete history.

## `eval` — KPI computed (Step 5)

`${OUT}/${NAME}_kpi.json` exists with a per-type block and an `Average` block carrying
`nn_score`/`mnn_score`(/`fid`). `nn_score` is the primary KPI.

## `refine` — every round actually finished (Step 6)

Unless `num_search_run=0`, Step 6 runs the draw→generate→evaluate loop before `select`. Catch two
failures: no rounds at all (clone-only), and — subtler — a round that started but didn't finish.

**Existing round directories are not enough.** A round dir appears with `generate.py`'s first image, so
an interrupted round looks fine; `select` then **silently ignores** it for want of a `kpi.json`. A round
can also be complete on files *and* counts yet still be unrankable — see the `${SCORE}` check below.
Run this after each round, and again before `select`:

```shell
: "${SCORE:?set SCORE to the exact value passed to Step 6 --score before running the refine gate}"
fail=0
rankable() {    # rankable <kpi.json> <score> -> True when the score's required inputs are usable
  python3 -c '
import json, math, sys
def ok(v):
    try: return math.isfinite(float(v))
    except (TypeError, ValueError): return False
d = json.load(open(sys.argv[1]))
score = sys.argv[2]
rows = [row for k, v in d.items() if k != "Average" and isinstance(v, dict)
        for row in v.get("per_sample", [])]
if score == "aq_rank":
    required = ("nn_score", "completeness_score", "precision_score", "boundary_iou_score")
    print(any(all(ok(row.get(key)) for key in required) for row in rows))
else:
    print(any(ok(row.get(f"{score}_score")) for row in rows))
' "$1" "$2" 2>/dev/null || echo False
}

# The ORIGINAL bucket competes against the rounds, so it must carry the selected score's inputs too.
# AQ-Rank needs at least one candidate with every raw axis; invalid candidates are excluded.
# With zero rounds no comparison occurs: select only clones Original, so this score gate is skipped.
if [ "${NUM_SEARCH_RUN}" -gt 0 ] &&
   [ "$(rankable "${OUT}/${NAME}_kpi.json" "${SCORE}")" != True ]; then
  echo "FAIL original: required ${SCORE} inputs are unusable in ${OUT}/${NAME}_kpi.json"
  fail=1
fi

n=$(seq 1 "${NUM_SEARCH_RUN}" | while read -r i; do [ -d "${OUT}/rounds/round_${i}" ] && echo "$i"; done | wc -l)
[ "$n" = "${NUM_SEARCH_RUN}" ] || { echo "only ${n}/${NUM_SEARCH_RUN} requested round dirs"; fail=1; }
for i in $(seq 1 "${NUM_SEARCH_RUN}"); do
  r="${OUT}/rounds/round_${i}/"
  img=$(ls "$r"reconstructed_image/*.png 2>/dev/null | wc -l)
  row=$(wc -l < "$r"testcase.jsonl 2>/dev/null || echo 0)
  # Rounds run the same guardrail-enabled generate.py as Step 4, so a blocked sample writes no image
  # by design. Counting images against rows alone fails a correct round, and the fix below (re-run
  # generate.py) reproduces the same block — the loop would never terminate.
  blk=$([ -f "$r"guardrail_blocked.csv ] && echo $(( $(wc -l < "$r"guardrail_blocked.csv) - 1 )) || echo 0)
  # Can this round actually be ranked? evaluate.py degrades to NN/MNN(+FID) whenever anomaly_quality
  # raises, so an AQ/geometry mode can have complete artifacts but unusable score inputs.
  scored=$(rankable "$r"kpi.json "${SCORE}")
  if [ -f "$r"kpi.json ] && [ "$(( img + blk ))" = "$row" ] && [ "$img" != 0 ] && [ "$scored" = True ]; then
    echo "OK   $(basename $r): ${img}+${blk}/${row} images, kpi.json present, ${SCORE} inputs usable"
  else
    echo "FAIL $(basename $r): ${img}+${blk}/${row} images, kpi.json $([ -f "$r"kpi.json ] && echo present || echo MISSING), ${SCORE} inputs usable=${scored}"
    fail=1
  fi
done
[ "$fail" = 0 ] && echo "refine gate PASS" || echo "refine gate FAIL — fix every FAIL line above before select"
```

- **Short image count (beyond what `guardrail_blocked.csv` explains) or missing `kpi.json`** → re-run that round's `generate.py` (it overwrites) and its
  `evaluate.py`, then re-run `select` and Step 7. Do **not** accept the existing `searched/`: it was
  assembled without that round.
- **Fewer round dirs than `${NUM_SEARCH_RUN}`** → rounds were skipped entirely.
- **`${SCORE} inputs usable=False`** → the round rendered fine but cannot be ranked, so re-running
  `generate.py` fixes nothing. AQ/geometry metrics can hit this because `evaluate.py` degrades to
  NN/MNN(/FID) whenever `anomaly_quality` raises, and a **missing SAM2 checkpoint** at
  `checkpoints/facebook/sam2.1-hiera-large/` is the usual cause. Grep that round's evaluate log for
  `anomaly_quality computation failed`, fix the checkpoint, and re-run **`evaluate.py` only**. Left
  unchecked, every sample scores NaN, the original wins each pick, and the search burns its rounds
  to reproduce Step 4 exactly. `Step 6` itself now stops on this rather than exiting 0.
- **`FAIL original: required ${SCORE} inputs are unusable`** → the Original bucket cannot be compared
  fairly with the rounds. Same causes as above, plus a `${NAME}_kpi.json` scored before the
  anomaly-quality axes were available. Re-run **Step 5's `evaluate.py`** so the KPI carries the
  score's required inputs, then re-run Step 6. `quality_refine.py` refuses this before the first
  round, so no GPU time is spent on a search that was already decided.

For `SCORE=aq_rank`, this shell gate checks that at least one KPI row has all four finite raw inputs.
The built-in `quality_refine.py run` gate additionally logs and excludes individual invalid
candidates, verifies that the generation CSV names exactly match the rendered images, and requires
every generated candidate to join to a KPI row; do not bypass it.

Also confirm `${OUT}/searched/${NAME}_kpi.json` exists (the closing evaluation).

An all-`original` `source` column in `${OUT}/searched/${TASK}_generation_result.csv` is **not** a
failure on its own — it is the honest result when no round beat the Step 4 render. Read it together
with the round check above: round dirs present *and* all-`original` means the search ran and found
nothing (accept it, and consider wider ranges or more rounds); **no round dirs** at a nonzero
`${NUM_SEARCH_RUN}` is the actual failure, and that is what the loop above catches.

## `label` — pseudo-labels written (Step 7)

`${OUT}/searched/pseudo_labels/coco_annotations.json` + `classification/classes.txt` exist.
