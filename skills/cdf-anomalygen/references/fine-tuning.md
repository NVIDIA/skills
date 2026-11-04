# Fine-tuning reference — Step 3

Progressive-disclosure detail for **Step 3 — fine-tuning** in the parent `SKILL.md`. Read this when you
need the recipe knobs, launch mechanics, early stopping, or checkpoint selection. If this file conflicts
with `SKILL.md`, `SKILL.md` wins.

## Recipe — `ag_config/exp_${TASK}_<name>.yaml`

`ag_config/exp_texture_ft_phone_screen.yaml` is both the worked example and the template. Copy it and
edit the **required dataset block**; every other key ships at a tested default (deleting a key falls back
to the default in the task's builder under `anomalygen/configs/`).

```shell
TEMPLATE=ag_config/exp_texture_ft_phone_screen.yaml   # the worked example doubles as the template
[ -f "${USER_RECIPE:-}" ] && TEMPLATE="${USER_RECIPE}"   # a recipe the user passed replaces it
[ "${RECIPE}" = "${TEMPLATE}" ] || cp "${TEMPLATE}" "${RECIPE}"   # same file (name=phone_screen or a passed recipe): edit it in place
```

> **Replace every key in the dataset block — all four.** They carry real phone_screen values, not `<name>`
> placeholders, so one left behind aims training at the phone_screen paths instead of failing loudly.

Required — training fails or silently misaligns without these:

| Key              | Example                                           | Note                                                                                         |
| ---------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `dataset_name`   | `phone_screen`                                    | run-dir name segment                                                                         |
| `dataset_path`   | `datasets/phone_screen`                           | Step 1 contract-layout root                                                                  |
| `anomaly_types`  | `[[Phone, oil], [Phone, scratch]]`                | **every** `[texture, defect]` pair on disk; sorted at parse, the sorted list fixes class ids |
| `testcase_jsonl` | `datasets/validation_phone_screen/testcase.jsonl` | Step 2 **validation** set (AMG or AMP) — what each validation pass scores                    |

Key optional knobs. Every value here **is** the builder default in
`anomalygen/configs/texture/exp_config.py` — the example recipe is the reference that file is kept in
sync with — so deleting a key preserves behaviour rather than changing it:

| Key                                            | Value                 | Note                                                                   |
| ---------------------------------------------- | --------------------- | ---------------------------------------------------------------------- |
| `model_size`                                   | `nano`                | frozen backbone + base checkpoint — see below                          |
| `per_class_lora_rank` / `per_class_lora_alpha` | `8` / `8`             | per-defect LoRA capacity on the frozen base                            |
| `max_iter` / `cycle_lengths`                   | `15000`               | total training iterations / learning-rate schedule length; keep equal  |
| `save_iter`                                    | `1000`                | checkpoint every N iterations → `iter_<N>.pt`                          |
| `validation_iter`                              | `1000`                | validate (write `valid/<N>/`) every N iterations                       |
| `run_validation_on_start`                      | `true`                | one validation at iteration 0 (baseline)                               |
| `lr` / `max_sequence_length`                   | `0.001` / `5120`      | AdamW learning rate / token budget per step                            |
| `activation_checkpointing_mode`                | `none`                | `none` / `selective` / `full`; trades recompute for memory only        |
| `input_size_ladder`                            | `[256,512,1024,1536]` | allowed input sizes; one entry pins a fixed size                       |
| `crop_mode`                                    | `scale`               | framing source: `scale` (context + resample) or `ratio` (bbox × ratio) |
| `crop_ratio_range`                             | `[1.5,2.5]`           | context ratio around the defect; `1.5 <= low < high`, never pinned     |
| `scale_range`                                  | `[0.333,3.0]`         | resample-scale draw; teaches defect-size invariance                    |

> **`model_size` has to be right before training starts.** It picks the frozen backbone (`nano` =
> Qwen3-VL-8B, `edge` = Nemotron-3 Dense VL 2B — experimental) *and* the base checkpoint generation
> loads (`checkpoints/Cosmos3-Nano` / `checkpoints/Cosmos3-Edge`). Adapters trained at one size
> **cannot** be generated with the other, and the mismatch is not caught for you. Omitting the key
> falls back to `nano`, so a recipe that never mentions it is a `nano` run — say so rather than
> leaving it implicit.
>
> **Do not set the augmentation probabilities.** `inst_aug_prob` defaults to `0.2`, so instance drop
> is on; `background_dropout_prob` and `ring_jitter_prob` default to `0.0`, i.e. off. These are the
> regime the model was validated under, not per-dataset dials. They are deliberately absent from the
> example recipe: omit them and you get the validated behaviour. Set them only for a deliberate
> ablation the user asked for. The one exception is the `legacy_fixed_size` block in `SKILL.md`
> Step 3, which sets all three as part of the legacy mechanism — append it whole.

`anomaly_types` is sorted and deduplicated at recipe parse, so class ids never depend on the order a
recipe lists them; the checkpoint stamps the sorted list and refuses a recipe whose *set* differs.
Use the **run-dir copy** of the recipe downstream so the set never drifts.

## Launch — detached, then wait for it to exit

A fine-tune at the default `max_iter=15000` runs for hours (dataset-dependent; the measured times are
in the tutorial's timing table, `docs/tutorial/texture/7-agentic-end-to-end-flow.md`) — too long for one foreground
call, so `scripts/skill_utility/train_control.sh` launches it detached and polls it in slices.
`SKILL.md` Step 3 holds the invocation; this section covers what the script is doing and why.

`start` builds `torchrun --nproc_per_node=${NUM_GPUS} ${SCRIPTS}/train.py --config=… --recipe=… --
experiment=anomalygen_${TASK}`, with `IMAGINAIRE_OUTPUT_ROOT` pointing at `./results`:

- `--config` is a **`.py` path**, not a dotted module — `get_config_module` rejects anything not ending
  in `.py` ("Config file cannot be specified as module") and does the `/`→`.` conversion itself. It
  resolves against the installed `cosmos_framework` package, not the repo, so `ls` on it from the repo
  root fails (the checkout is `cosmos-framework/`, with a hyphen) while the path is still correct.
- The bare `--` separates the launcher args from the experiment override.
- **Why `setsid`:** a plain `… &` already survives the launching command returning — it reparents to
  init either way. The one case it does *not* survive is a harness that cleans up by **session**
  (`pkill -s <sid>`), because a plain background job stays in the launching shell's session; that is
  what has reaped long runs here before. `setsid` gives training its own session, so a session-scoped
  kill misses it. `nohup` does not substitute (it only ignores `SIGHUP`), and `disown` is unnecessary
  once the process has left the shell's session.
- `results/train_${NAME}.pid` holds the **real** `torchrun` PID, because the detached shell records
  `$$` and then `exec`s. `$!` would name the `setsid` wrapper, which exits immediately — `stop` would
  then kill nothing. The poll checks that PID rather than `pgrep -f "${SCRIPTS}/train.py"`, which also
  matched any process whose command line merely contained that path, including the polling shell.
- `start` refuses to launch over a live run of the same `--name`: both would append to one log and the
  second would orphan the first. Use a different `--name`, or `stop` first.
- **Quick dry run:** append `-- trainer.max_iter=20 trainer.validation_iter=10 checkpoint.save_iter=10`
  to `start` to smoke-test the full loop in a couple of minutes. Anything after `--` is passed through
  as a hydra override. A dry run can leave `cycle_lengths` as it is.

`wait` blocks for `--timeout` seconds (default 570) and reports the run as its **exit code**: `0`
finished, `1` failed, `2` still running. Repeat it while it returns 2 — one call is ~9.5 min, so
the `15000` default takes dozens of them. Exit 0 requires the `Done with training`
marker *and* waits for the process to actually exit, so the GPUs are released before Step 4 generates
on them. A dead process without the marker is exit 1, never a quiet success.

**Gate Step 4 on exit 0, never on a checkpoint appearing** — the trainer writes one every `save_iter`
and all but the last are intermediate. **Waiting is an action, not a stopping point:** the common
failure is to launch training, arm a background watcher, and end the turn, after which training
finishes and *nothing runs Steps 4–7*. Report progress only after `wait` returns 0 or 1.

If the run *is* killed, `start` again — the trainer resumes from `checkpoints/latest_checkpoint.txt`.

> **`job_name` lives in two places and they must agree.** `${RUN_DIR}`'s last segment is the recipe's
> `job_name`, so renaming it in the recipe — the documented way to get a clean run dir instead of
> resuming an old checkpoint — also needs `--job_name <new>` passed to `set_pipeline_vars.sh`. Change
> only the recipe and Steps 4-7 read the *old* run dir: a stale checkpoint and a stale recipe copy,
> both of which load without complaint. `legacy_fixed_size` renames it to `anomalygen_texture_ft_legacy`
> on both sides.

Run dir `${RUN_DIR}` (`results/anomalygen/${NAME}/${JOB_NAME}`), containing
`checkpoints/model/iter_<N>.pt`, `checkpoints/best_checkpoint.txt` (peak-scoring iteration, written at
train end), `valid/<N>/` (`reconstructed_image/`, `valid_kpi.csv`), `training_loss.png`,
`training_curves.png`, and a copy of the recipe.

## Checkpoint selection — latest vs. best

> **The two pointers have different jobs. `best_checkpoint.txt` selects the checkpoint to *generate
> from*; `latest_checkpoint.txt` is what training *resumes from*, and must stay that way.** They are read
> by separate code paths — inference resolves `--checkpoint` through
> `anomalygen/inference/inpaint.py`, while resume goes through `SelectiveCheckpointer.load`, which only
> ever reads `latest_checkpoint.txt`. Never repoint resume at the best pointer: the `iter_<N>.pt` files
> under `checkpoints/model/` are model-only, and resume needs the `optim`/`scheduler`/`trainer`
> components saved at the *same* iteration — rewinding to an earlier best would both discard progress
> and desync the optimizer state.

`--checkpoint ${RUN_DIR}` resolves via `best_checkpoint.txt`, then `latest_checkpoint.txt`, else the
highest `sorted(model/iter_*.pt)`. That last fallback is the **latest** iteration, and small datasets
routinely peak early then drift, so the latest is often *not* the best — the resolver logs a warning
whenever it lands there.

**The trainer selects for you.** At train end the `TrainingReport` callback writes
`checkpoints/best_checkpoint.txt` — the bare filename of the peak-scoring iteration, same format as
`latest_checkpoint.txt` (left untouched for resume) — and logs the pick. Passing `${RUN_DIR}` therefore
already picks the best; pass the explicit path anyway so the run record names the exact checkpoint:

```shell
CKPT="${RUN_DIR}/checkpoints/model/$(cat "${RUN_DIR}/checkpoints/best_checkpoint.txt")"   # iter_<best>.pt
```

```text
Best checkpoint by nn (max): iteration 2000 = 0.6622 -> .../checkpoints/model/iter_<best>.pt
```

Selection rules:

- Scores the **`Average`** column of each `valid/<N>/valid_kpi.csv` (macro-mean over defect types), not
  any single type.
- Metric is `TrainingReport.best_metric`, which the recipe builder sets from the run's own
  `early_stop_metric` (default `nn`), so the kept checkpoint is scored on whatever training was
  monitored on; `METRIC_SPECS` supplies the direction, so `fid` correctly selects the **minimum**.
- Only iterations with a checkpoint on disk are eligible, so the pointer can't dangle —
  `run_validation_on_start` scores iteration 0, which is never checkpointed.
- **Iterations below `CKPT_WARMUP_ITER` (7500) are excluded when `max_iter` is above it.** `nn` swings
  widely before the adapters settle, and the pick is otherwise a plain best-of, so an early spike
  would beat a genuinely better late checkpoint. Short runs (dry runs, `max_iter <= 7500`) keep every
  iteration eligible so they still get a pointer.
- **A run that ends before the warm-up falls back to the full set and warns — and early stopping
  slips past the gate.** A *crash* is already caught: no `Done with training`, so the **train** gate
  fails and you stop. Early stopping is different — it shrinks `max_iter` so the trainer exits
  normally, the gate **passes**, and Step 4 would proceed on a sub-warm-up checkpoint with nothing
  but a log line. (The guard still fires, because it compares against the `max_iter` captured at
  train start, before the shrink.) So whenever `early_stop_enabled: true`, read the iteration out of
  `best_checkpoint.txt` before Step 4. If it is below 7500, **report it and let the user choose**:
  accept the plateau early stopping found, or re-run with `early_stop_enabled: false` (or a larger
  `early_stop_patience`) so training reaches the warm-up.

Nothing is written when validation never scored or no checkpoint was saved; the callback is non-fatal
(warns, training unaffected). Missing after a completed run = **train** gate failure — fix the run
rather than falling back to the latest. If the run *did* validate but no validated iteration was
checkpointed, the warning names the cause: set `validation_iter` to a multiple of `save_iter`.

## Early stopping

Off by default. Enable it to stop once the monitored metric plateaus instead of always running to
`max_iter`:

```yaml
early_stop_enabled: true
early_stop_metric: nn            # nn | mnn | fid | aq_nn | completeness | precision | boundary_iou
                                 # fid is lower-better; every other metric is higher-better
early_stop_patience: 5           # validations without improvement before stopping
early_stop_min_delta: 0.0
early_stop_min_delta_mode: rel   # rel | abs
```

The callback reads the metric from each `valid/<N>/valid_kpi.csv` and, after `patience` validations
without improvement, shrinks `max_iter` so the trainer exits (rank 0 decides; broadcast to all ranks).
Because it stops `~patience × validation_iter` iterations **after** the peak, the latest checkpoint still
overshoots the best — select the peak-`nn_score` iteration (see **Checkpoint selection**) for generation.

## Validation KPI

Each validation pass scores the `testcase_jsonl` set with the same correspondence KPI as final evaluation
and writes `valid/<N>/` (`reconstructed_image/` renders + `valid_kpi.csv`). **`nn_score` is the
primary KPI** (higher is better); `mnn_score` and `fid` are secondary. With `compute_anomaly_quality`
(on by default) each pass also scores the anomaly-quality axes — `completeness`, `precision`,
`boundary_iou` — and the composite `aq_nn` (= `completeness` + `nn_score`). `training_curves.png`
plots **six** metrics (`nn`, `mnn`, `fid`, `completeness`, `precision`, `boundary_iou`); `aq_nn` goes
to `valid_kpi.csv` and the training report (and is selectable for early stopping) but is not drawn on
the curve. Watch these to judge quality vs. iteration and to choose the best checkpoint.

> **Cost of `compute_anomaly_quality`.** The axes add one SAM2 forward per validation sample. Measured
> (SAM2-hiera-large, GPU): a one-time model load of ~2.3 s (cached for the whole run) plus **~0.4 s per
> sample**. Per validation pass that is `≈ 0.4 s × |testcase_jsonl|` (e.g. ~10 s for 25 testcases, ~40 s
> for 100). Over a default 4 h / `max_iter=15000` run — ~16 passes at `validation_iter=1000`, ~31 at
> `500` — it totals roughly **3–5 min for a 25-case set (~1–2 %)** and **~11–21 min for a 100-case set
> (~5–9 %)**. Small enough to keep on by default; if you validate a large set (~100) frequently and want
> the time back, set `compute_anomaly_quality: false` in the recipe.
>
> **`nn`/`mnn` scoring config changed.** `nn_score` / `mnn_score` are now measured with `region_policy=zoom`,
> `layer=12`, `readout=worst25`, `inst_agg=min` (was `full` / final-layer / `mean`). This lowers the raw
> numbers (a checkpoint's `nn` is a different value than before), so **don't compare a baseline recorded
> under the old config**, and re-tune any `early_stop_min_delta_mode: abs` threshold. `evaluate.py` and
> `filter.py` expose `--nn_region_policy` / `--nn_layer` / `--nn_readout` / `--nn_inst_agg` to reproduce
> the old scoring if you need to compare against pre-existing numbers.
>
> **GPU footprint of `zoom` scoring.** This is a property of the correspondence KPI itself, so it applies
> equally to the three callers that share it: the in-training `ValidationKPI`, the standalone
> `evaluate.py`, and `filter.py`. `zoom` runs DINOv2-large on one 518×518 crop per masked instance,
> batched together, so peak GPU memory grows with the instance count. Measured (fp32, layer-12): N=1 →
> 1.4 GB, **N=8 → 2.7 GB**, N=32 → 7.1 GB (weights 1.2 GB + activations). The per-mask instance count is
> capped at `_MAX_ZOOM_INSTANCES=8` (largest by area), so this stays ~2.7 GB regardless of how fragmented
> the mask is — negligible even in the tightest case (validation, where it co-resides with the Nano-8B
> being trained), and a speckled mask can't OOM the GPU.

## Error handling

| Symptom                                             | Action                                                                                                                                                                                                            |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Validation scores nothing / `testcase_jsonl` errors | Step 2 (validation set) not run or not wired into the recipe — build it first                                                                                                                                     |
| `anomaly_types` mismatch downstream                 | Use the **run-dir recipe copy** for Steps 4–7; a different set is refused by the checkpoint stamp                                                                                                                 |
| Trainer resumes from an old checkpoint              | It resumes from any checkpoint in the run dir — use a new `job_name` (or `IMAGINAIRE_OUTPUT_ROOT`) for a clean start; see the note below                                                                          |
| OOM during training                                 | Set `activation_checkpointing_mode: selective` (or `full`) first — memory only, training unchanged; then lower `max_sequence_length` or drop the largest `input_size_ladder` size; reduce `validation_batch_size` |
| Loss diverges / NaN                                 | Lower `lr` (e.g. `0.0005`); check the dataset masks                                                                                                                                                               |
| Multi-GPU hang / NCCL error                         | Verify `CUDA_VISIBLE_DEVICES` and driver; retry with `--nproc_per_node=1` to isolate                                                                                                                              |
| Dies at step `cycle_lengths + 1` with `TypeError`   | `list indices must be integers or slices, not NoneType` means `max_iter` is larger than `cycle_lengths`; set them equal and relaunch                                                                              |
