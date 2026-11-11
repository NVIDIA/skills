# Inputs and packaged defaults

Read this during intake. Preserve explicit user values. Apply defaults only to
omitted values, show their source in the launch review, and freeze the approved
policy once.

## Required inputs

- An installed supported execution platform; never choose among peers silently.
- `max_iterations`.
- Four normalized roles described in `data-contract.md`.
- One trainable RT-DETR base checkpoint. Every iteration starts from this same
  checkpoint; a prior iteration checkpoint is never the next initializer.
- A baseline mode: `cold_start` (default) or `checkpoint`. Cold start avoids
  binary inference with a multiclass training initializer; checkpoint mode
  requires a binary-compatible checkpoint.
- Separate KPI and test roles.
- An explicit synthesis decision.

When synthesis is enabled, require a normalized reference pool,
`defect_spec.jsonl`, Cosmos3-Nano assets, and either an existing
checkpoint/recipe pair or a complete one-time fine-tuning block for every KPI
`dataset_id`. Read `anomalygen-pool.md`.

## Packaged algorithm values

The authoritative values live in `assets/default_policy.yaml`. Important
defaults are:

- inference confidence `0.001`;
- baseline mode `cold_start`;
- loose/strict gap confidence `0.3` / `0.8`;
- match IoU `0.5`;
- background-like FP boundary below `0.05`;
- SigLIP `google/siglip-base-patch16-224`;
- square-context preprocessing, with `tight_context` available for the
  aspect-ratio-preserving crop behavior;
- defect context scale `1.5`;
- square-context output size `224`;
- clean grids `[1, 2]`;
- initial minimum similarity `-1.0`;
- retrieval selection strategy `round_robin_similarity`, with
  `max_similarity` available for global maximum-similarity ranking;
- max-similarity candidate overfetch `15` and independent round-robin refill
  depths `[5, 15, 50, 200, 100000]`;
- round-robin audit shortlist floor `20` candidates per query;
- real mining factor range `1..6`;
- clean dose `2` per routed FP and cumulative cap `1.0` per admitted real;
- RT-DETR train shape of four GPUs, batch size eight, base LR `1e-4`, and
  backbone LR `1e-5`;
- 36 epochs for iterations 1–2;
- three ten-epoch probes from `probes_start_iteration` onward, defaulting to
  iteration 3;
- adaptive main budget of 24–48 epochs;
- one 12-epoch late-best extension;
- synthesis disabled;
- synthesis top-3 distinct clean candidates with one successful neighbor
  retained per FN;
- synthetic cumulative cap `0.25` relative to admitted real defects.

For compatibility with pre-split inputs, an explicit
`retrieval.candidate_overfetch` is copied to `synthesis.candidate_topn` when the
latter is omitted. Already-frozen policies without `candidate_topn` also use
the retrieval value during synthesis preparation. Set both values explicitly
to opt into independent non-default values.

A value frozen in the policy is no longer a default. Changing it starts a new
contract rather than silently mutating an existing run.
