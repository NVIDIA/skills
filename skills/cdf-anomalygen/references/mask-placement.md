# Mask-placement reference — Step 2 (AMP)

Progressive-disclosure detail for **Step 2 — mask placement**, `mask_backend=amp`, in the parent `SKILL.md`. Read this
for allocation modes, the validation KPI floor, custom counts, ROI caching, `spatial_dependency`
routing, and pairing coverage. If this file conflicts with `SKILL.md`, `SKILL.md` wins.

## Pipeline

`roi_place_pipeline` (what `SKILL.md` Step 2 invokes) runs all three stages below as one command,
deriving every intermediate path from `--output_dir` and threading `n_seeds` through for you. Reach
for the individual CLIs only to inspect or hand-edit an intermediate. Either way the stages are:

| #   | CLI            | In → out                                                                                                                      |
| --- | -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 1   | `roi_allocate` | `defect_spec` + on-disk mask counts → `allocation.json` (`{defect_type: n}`)                                                  |
| 2   | `roi_pair`     | `allocation.json` → `amp_samples.json` (+ `amp_samples.json.n_seeds`); pairs each count with clean images + submask templates |
| 3   | `roi_place`    | `amp_samples.json` → placed masks + **`testcase.jsonl`** (row = clean image + placed mask + gen params)                       |

Build one AMP set per bucket with a **distinct `--seed`**: **full** builds both (validation +
generation), **finetune_only** builds validation, **generation_only** builds generation.

## Allocation modes (`roi_allocate --mode`)

| Mode                      | Allocation                                                   | KPI floor                                         | Use                          |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------- | ---------------------------- |
| `inference` (the default) | **uniform** across defect types                              | none — a type may get 0 via `--per_defect_counts` | the generation batch         |
| `validation`              | **proportional** to training mask counts (largest-remainder) | **≥ 1 per defect, enforced**                      | the fine-tune validation set |

There are only these two modes. `inference` is the one the **generation** batch uses — the name is the
CLI's, not the pipeline stage's, so don't reach for a `generation` mode; it does not exist and
`--mode generation` is rejected. It is also the default, so omitting `--mode` gives the same thing.
`num_sdg` is split across the defect types listed in `defect_spec`.

**Exact-count invariant (generation):** uniform allocation sums to **exactly `num_sdg`** — `base =
num_sdg // N` per type, then the first `num_sdg % N` types get **+1** (so a type is at most `base+1`,
never `base+2`). Passing `--num_sdg N` alone yields a `testcase.jsonl` with exactly `N` rows. A different
total is **not** a rounding effect: it means the call carried a `--per_defect_counts` override (whose sum
wins) or a different `num_sdg`. (`validation` mode refuses a run in which a type's share would round to 0 —
`validation coverage broken` — instead of delivering it.) After AMP, assert the generation set's `wc -l testcase.jsonl` against **whatever the
allocation asked for** — `num_sdg` for a plain call, the override sum when `--per_defect_counts` was
passed. Checking against `num_sdg` unconditionally fails correct custom-count runs.

### Validation KPI floor (raises)

In `--mode validation` every defect type must receive ≥ 1 sample so validation can score it. If a type
rounds to 0 (e.g. `num_sdg` too small relative to a rare type's mask share), `roi_allocate` **raises**
with a "coverage broken" error naming the starved defect — raise `num_sdg` (or rebalance the dataset)
and re-run. `nn_score` needs ≥ ~3 samples/type to be stable, so size the validation set with headroom.

## Custom counts (`--per_defect_counts`, generation only)

Pass a JSON dict to set exact counts, e.g. `--per_defect_counts '{"Phone+oil": 5, "Phone+scratch": 10}'`:

- Only valid with `--mode inference`; **rejected in `validation`**.
- A type absent from the dict gets **0**; an unknown key is **ignored with a warning**.
- If the dict sum ≠ `--num_sdg`, a warning is logged and the **override sum** is used.

## `spatial_dependency` routing

`roi_place` routes each defect by its `defect_spec` `spatial_dependency`:

| Value     | Placement                                                                                            | Extra inputs                                                                                                                         |
| --------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `free`    | random ROI in the clean image (area-ratio bounded)                                                   | —                                                                                                                                    |
| `text`    | ROI grounded by a VLM (`--model_id`, default `nvidia/Cosmos3-Nano`) from a prompt                    | `roi_prompt_defect_location` per defect in `defect_spec`                                                                             |
| `cad`     | ROI from CAD / segmentation candidates                                                               | `semantic_segmentation_labels.json` + `{texture}/cad_mask/<clean_stem>.png`                                                          |
| `cad_tht` | ROI from through-hole joint geometry CAD / segmentation candiates (pad ring / hole ring / lead face) | `semantic_segmentation_labels.json` + `{texture}/cad_mask/<clean_stem>.png`, whose classes are `pad`, `hole`, and `pin_tip` or `pin` |

`cad` and `cad_tht` read the same two artefacts and differ only in what the classes mean. Routing is by
this field.

## Pairing coverage (`--per_clean_pairing`)

The default pairing maximises **(submask, clean) diversity**, not coverage: it reuses some clean images
and skips others, while still emitting the right number of records. Coverage is
worst near `S = sqrt(num_sdg)`, so a larger submask bank makes it worse. That is the right trade when
clean images are a reusable pool of backdrops, and the wrong one when each is a work item that must
receive an output.

`--per_clean_pairing on` switches to one record per clean; `off` (the default) is the round-robin.
It is a user input (`inputs.md`), not something to decide here: which of the two a batch wants
follows from what the clean images *are* — a pool of backdrops or a list of work items — and nothing
in the dataset says which. Absent, the default applies.

One record per clean means one defect per clean, so `on` needs **exactly one defect type with a
non-zero allocation** — select it with `--per_defect_counts '{"<texture>+<defect>": N}'`. The
allocation must fit the pool: larger than the clean count **raises**, smaller **warns** with the
shortfall.

## ROI caching

The text/cad ROIs cost a VLM + SAM2 pass, so `roi_place` **caches** each placed ROI mask and reuses it
on re-run; pass `--refresh_roi` to regenerate. `--roi_only` generates ROIs without writing the AMP masks
/ JSONL (useful to warm the cache).

## `n_seeds`

`roi_pair` writes the recommended `n_seeds` next to `amp_samples.json` (as `…json.n_seeds`).
`roi_place_pipeline` passes it through automatically; only when running the stages by hand do you
thread it yourself — `roi_place --n_seeds "$(cat …/amp_samples.json.n_seeds)"`. It is the per-sample
placement multiplicity that makes the placed-mask count reach the allocation; leave it to the pipeline.

Under `--per_clean_pairing on` it is **1**: each clean is used exactly once, so there is nothing for AMP
multi-placement to make up.

## Preflight (`_validate_amp_inputs`)

`roi_allocate` cross-checks the `(dataset_dir, clean_dir, defect_spec)` triple before allocating and
fails with a list if any of these break:

- a `defect_type` not in `TEXTURE+ANOMALY` form,
- an unknown `spatial_dependency`,
- a `text` defect missing a non-blank `roi_prompt_defect_location`,
- a missing submask source dir (`{texture}/mask/{defect}/`),
- no clean images,
- for `cad` and `cad_tht`: missing `semantic_segmentation_labels.json`, or a clean without a matching
  `cad_mask`,
- for `cad_tht`: a label file without `pad`, `hole` and a lead class, a `defect_type` outside
  the published enum, or a `cad_mask` sized differently from its clean image.

## Outputs

`${AMP_DIR}/` holds `allocation.json`, `amp_samples.json` (+ `.n_seeds`), the placed masks, and
`testcase.jsonl` (the row set Steps 3–6 consume). A defect that yields zero placements is dropped
**warn-only** — check the `roi_place` log if a bucket comes out short.
