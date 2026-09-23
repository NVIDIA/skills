# Cosmos AnomalyGen — Cosmos3 DEFT AOI Reference

Read this when the loop runs the `anomalygen` stage. The underlying skill
`tao-skill-bank:tao-generate-anomalies` owns the standalone pipeline, the parameter
reference, the invariants that gate a run, and the AMP no-ROI failure mode —
**read it before invoking**. This file is the Cosmos3 overlay: how synthetic
defects become bare OK/NG records, when the stage is skipped, and what is
committed.

The generation mechanics are identical to the VCN loop; only the consumer
differs. `skills/applications/tao-run-deft-aoi/references/tao-generate-anomalies.md`
holds the shared shell setup, the container-owned one-time post-gate bootstrap,
and the per-iteration `docker run` pair. Its download command explicitly
requests Text2Image 2B for both loops. Do not duplicate them here — read that
file for the commands and apply the Cosmos3 values below.

## The published checkpoint is flat — normalize it first

In network-enabled mode, the shared bootstrap auto-downloads the fine-tuned
checkpoint from `nvidia/Cosmos-AnomalyGen-PCB-2B` (~5 GB, iter 14000) by
default; a pre-staged BYO checkpoint overrides it. The published checkpoint is
compatible with the pinned `paidf-anomalygen:1.0.1` container, and any override
must match the container's PAIDF major.minor.

The official HF repo ships `iter_000014000.pt` and `ag_config.yaml` side by
side, with no
`checkpoints/latest_checkpoint.txt` and no `checkpoints/model/`, while the
container loads the nested layout. Every run hits it.

Build a symlink view under the run directory rather than touching the source
cache, then take `step` from the view:

```bash
CKPT=$(find -L "$AG_CHECKPOINT_DIR" -path '*/ag_config.yaml' -print -quit | xargs -r dirname)
: "${CKPT:?FATAL: checkpoint auto-fetch or BYO staging did not produce ag_config.yaml under $AG_CHECKPOINT_DIR}"
if [ ! -f "$CKPT/checkpoints/latest_checkpoint.txt" ]; then
  mapfile -t staged < <(
    find -L "$CKPT" -maxdepth 1 -type f -name 'iter_[0-9]*.pt' ! -name '*_reg_model.pt' | sort
  )
  [ "${#staged[@]}" -eq 1 ] || { echo "FATAL: expected one flat iter checkpoint under $CKPT" >&2; exit 2; }
  name=$(basename "${staged[0]}")
  CKPT_VIEW="$RUN_DIR/checkpoint_view"
  mkdir -p "$CKPT_VIEW/checkpoints/model"
  ln -sfn "$(realpath "$CKPT/ag_config.yaml")" "$CKPT_VIEW/ag_config.yaml"
  ln -sfn "$(realpath "${staged[0]}")" "$CKPT_VIEW/checkpoints/model/$name"
  printf '%s\n' "$name" > "$CKPT_VIEW/checkpoints/latest_checkpoint.txt"
  CKPT=$CKPT_VIEW
fi
STEP=$(sed 's/^iter_0*\([0-9]*\)\.pt$/\1/' "$CKPT/checkpoints/latest_checkpoint.txt")
```

This is a path adapter, not a download. Requiring exactly one root checkpoint
is deliberate — several means the intended one is ambiguous, and silently
picking is worse than stopping. The published checkpoint produces `STEP=14000`.

## Why the stage exists

Mining finds *real* boards similar to the ones the model already gets wrong. It
cannot manufacture a defect the pool never contained. AnomalyGen paints new
defects onto clean boards, so it addresses the failure mining cannot: the model
under-detecting defects it has too few examples of.

`NG` is the positive class, so the gap this stage closes is **Proxy false
accepts** (NG → OK). That is also its skip condition.

## Position in the loop

```text
routing -> anomalygen -> data_mining -> assemble_data -> validate_data -> train
```

`anomalygen` and `data_mining` are independent producers writing into the same
`assemble_data` stage: synthetic NG pairs from this stage, real mined pairs from
the next. Neither reads the other's output.

## Parameters

Run `mode=inference_only`. The loop needs only Phase 2 (`prep_testcase.sh`, AMP
routing) and Phase 3 (`run_sdg.sh`, SDG diffusion); Phases 4–7 are SDG-quality
optimization and contribute no training pairs. Skip them with
`num_search_run=0` and `nn_threshold=0`.

Read every value from `deft_state.json::config.anomalygen`; do not re-infer
conventions. `init_deft_state.py` records these already symlink-resolved, and
accepts `--anomalygen-checkpoint-dir`, `--anomalygen-dataset-dir`, and
`--cosmos-models-dir` when the assets live outside the workspace convention.
Mount every recorded path that falls outside `$WS` in addition to `$WS` itself
— mounting only `-v $WS:$WS` leaves a symlinked subtree dangling inside the
container, which surfaces as a "not found" error for a path that plainly exists
on the host.

| Param | Source |
|---|---|
| `mode` | `inference_only` |
| `project` | `config.anomalygen.project` — the directory label for this project's checkpoint + dataset |
| `checkpoint_dir` | `config.anomalygen.checkpoint_dir` (holds `ag_config.yaml` + `checkpoints/`) |
| `step` | parsed from `${checkpoint_dir}/checkpoints/latest_checkpoint.txt` |
| `dataset_dir` / `clean_dir` | `config.anomalygen.dataset_dir`, passed verbatim |
| `defect_spec` | `config.anomalygen.defect_spec` |
| `num_SDG` | `config.anomalygen.num_SDG` |
| `cosmos_models_dir` | `config.anomalygen.cosmos_models_dir` |
| `output_dir` | `${RESULTS_DIR}/iter${N}/anomalygen/sdg/` — the stage-bound results directory |
| `num_search_run` / `nn_threshold` | `0` / `0` |

DEFT AOI is always a PCB workflow and uses the pre-staged, version-compatible
fine-tuned checkpoint. AnomalyGen runs in its own container
(`config.anomalygen.container`, `images.metropolis_sdg.paidf_anomalygen`), not
the Cosmos-RL image.

After Phase 2, parse `<output_dir>/allocation.json` before launching Phase 3. A
large requested-vs-allocated gap means the cad_mask or anomaly-mask geometry is
wrong, and Phase 3's GPU cost is fixed regardless of yield — abort and diagnose
with the underlying skill's failure-mode section instead of generating four
samples for the price of twenty.

## Output to bare OK/NG records

AnomalyGen writes one synthetic defect per row of `SDG_result.csv`:

```text
<output_dir>/
├── allocation.json                       # Phase 2 defect -> AMP count proof
├── SDG_result.csv
├── reconstructed_image/<T>+<A>_<idx>.png   # generated defect  -> images[0], the AOI board
└── original_image/<T>+<A>_<idx>.png        # clean source      -> images[1], the golden reference
```

That is already the Cosmos3 pair shape, so each generated sample becomes one
record whose assistant response is exactly `NG`:

```bash
"$PYTHON" "$SKILL_ROOT/scripts/emit_sdg_sharegpt.py" \
  --sdg-csv "$RESULTS_DIR/$LABEL/anomalygen/sdg/SDG_result.csv" \
  --media-root "$MEDIA_ROOT" \
  --prompt-from "$MINING_ANNOTATIONS" \
  --emit-relative \
  --output "$RESULTS_DIR/$LABEL/anomalygen/sdg_sharegpt.json" \
  --summary "$RESULTS_DIR/$LABEL/anomalygen/emit_sdg_summary.json"
```

The emitter supports both producer contracts without editing the CSV:

- documented paths relative to the CSV/output directory, such as
  `reconstructed_image/X.png` and `original_image/X.png`;
- PAIDF 1.0.1 `output_filename` values that echo the repo-root-relative
  `--output_dir`, such as `results/run/sdg/reconstructed_image/X.png`, plus its
  `image_filename` clean-source column.

For a relative path it tries the CSV parent, the CSV parent with an echoed
output prefix removed, then the optional `--sdg-root <paidf-repo-root>`. Use
`--sdg-root` when the CSV was moved away from the producer tree. If the
recorded clean source is unavailable, it derives
`original_image/<generated-name>` beside the resolved reconstructed directory.
Missing/empty failures list every attempted path. A future schema with no
recognized generated-image column fails with its available columns instead of
silently guessing.

`--prompt-from` inherits the single inspection prompt recorded in the Mining
pool so synthetic and mined records ask the model the same question; it
hard-stops when the pool carries more than one distinct prompt. Use `--prompt`
to pass the exact string instead. Never write a different prompt for synthetic
records, and never emit a label other than `NG` — the defect was painted on
deliberately.

The emitter hard-stops on a missing or empty image on either side of a pair,
and de-duplicates by resolved generated image.

Wrap Phase 3 with these two one-line hard gates, using the shared VCN
reference's `$COSMOS` and `$SDG_LOG` values:

```bash
test -s "$COSMOS/nvidia/Cosmos-Guardrail1/video_content_safety_filter/safety_filter.pt" || { echo "FATAL: AnomalyGen Guardrail checkpoint missing; guardrail=not_run" >&2; exit 2; }
test -s "$SDG_LOG" && ! grep -Fqi "post-generation image checks are DISABLED" "$SDG_LOG" || { echo "FATAL: AnomalyGen Guardrail log missing or screening disabled; guardrail=not_run" >&2; exit 2; }
```

The first runs before SDG and the second after it. On either failure, end the
stage as `anomalygen --status error` and do not emit generated data downstream.
Record `passed` only when screening ran and accepted the rows, `failed` when it
rejected them, and `not_run` when it was disabled or failed to initialize;
`not_run` is unscreened and must never be reported as passed.

## Skip condition

When the driving Proxy RCCA recorded **zero false accepts**, skipping is
*permitted* — there is no under-detection gap. It is not recommended by
default. Generating anyway is always legal, and on the one dataset measured so
far it was the better call: see *Measured counter-evidence* below. Prefer the
skip when generator cost matters or the clean-image pool is unrepresentative.

To skip, commit a documented branch skip instead of launching the generator:

```bash
PYTHON=$(bash "$SKILL_ROOT/scripts/deft_python.sh")
"$PYTHON" "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label "iter${N}" --stage anomalygen \
  --skip --duration-sec "$STAGE_DURATION_SEC" \
  --summary "no Proxy false accepts; synthetic defects not indicated"
```

The driving RCCA is `baseline` for `iter1` and `iter${N-1}` for later
iterations. Read that phase's recorded `false_accepts_json` and use `--skip`
only when it contains no entries. This is the only legal way to omit the
stage: a failed generator with false accepts outstanding is a hard stop.

Note the asymmetry: zero false accepts *permits* the skip, it never forces it.
Nothing blocks generating when the skip would also have been legal.

### Measured counter-evidence

The reasoning behind the skip condition — "NG is the positive class, so
synthetic NG only closes false accepts" — is too narrow. A 2026-07-29 NVPCB run
had zero false accepts on both splits at baseline, making the skip legal, and
generated 20 synthetic pairs anyway. On the frozen Benchmark:

| | baseline | iter1 |
|---|---:|---:|
| accuracy | 0.7727 | 0.9273 |
| false rejects (OK->NG) | 25 | 8 |
| recall_ng | 1.000 | 1.000 |

Accuracy rose and over-rejection fell sharply. Paired (defect, clean) examples
appear to teach the OK/NG boundary itself, not just add NG mass — which helps a
model whose actual failure is over-rejection (baseline precision_ng 0.545).

Treat this as one data point, not a rule: n=1 dataset, a single iteration, and
the same training set also gained 13 mined records, so the synthetic
contribution is not cleanly separable. It is enough to say the skip should not
be automatic.

## Commit

```bash
PYTHON=$(bash "$SKILL_ROOT/scripts/deft_python.sh")
"$PYTHON" "$SKILL_ROOT/scripts/commit_stage.py" \
  --results-dir "$RESULTS_DIR" --iter-label "iter${N}" --stage anomalygen \
  --anomalygen-sdg "$RESULTS_DIR/iter${N}/anomalygen/sdg/SDG_result.csv" \
  --anomalygen-allocation "$RESULTS_DIR/iter${N}/anomalygen/sdg/allocation.json" \
  --anomalygen-sharegpt "$RESULTS_DIR/iter${N}/anomalygen/sdg_sharegpt.json" \
  --duration-sec "$STAGE_DURATION_SEC" \
  --summary "SDG: requested=N, AMP-allocated=M, generated=K by defect type; guardrail=passed"
```

`commit_stage.py` derives `M` by summing the committed defect-to-count
`allocation.json`. When `M < N`, report
both requested and allocated counts — that gap is the
signal a reviewer needs to spot an allocation bottleneck rather than a
generation one.

All three artifacts must land under the stage's bound results directory.
`--skip` and the artifact flags are mutually exclusive; do not record both.

## Commercial-training eligibility

Synthetic records enter Train, so they inherit the Mining pool's obligation:
only synthetic data approved for commercial training may be used. Track that
approval outside the annotation payload — see `references/data-layout.md`.
Generated boards are also evaluation-isolated: `validate_split_contract.py
--synthetic` proves no generated target appears in Proxy or Benchmark.
