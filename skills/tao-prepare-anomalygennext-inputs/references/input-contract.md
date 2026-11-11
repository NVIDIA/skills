# Normalized AnomalyGenNext preparation contract

The preparation leaf accepts normalized identity instead of legacy path rules.
The producer must freeze `dataset_id`, `texture_id`, `defect_class`,
`anomaly_type`, and `fn_mask_source` on every gap row before launch.

`bbox` is `[x1, y1, x2, y2]` in source-image pixels. `fn_mask_source` must be a
same-size pixel mask; a detector box is not a replacement. `split` is an opaque
selection bucket such as `kpi` or `test`.

Preparation evaluates each selected FN independently. A source FN mask must be
readable, nonempty, non-full, have a tight extent smaller than the full frame,
match the source-image dimensions, and contain pixels inside its FN box. Donor
masks must satisfy the same mask-content gates; deterministic selection falls
through to later same-type donors when an earlier donor is invalid. The source
pool is never changed.

The clean-reference directory for each normalized texture must contain at least
one supported image. Missing and empty directories produce the stable per-FN
reason `no_clean_reference_images`. Those FNs are absent from query, embedding,
mask, and AMP inputs; eligible FNs for other types continue independently.

The YAML `datasets` mapping assigns each `dataset_id` an existing checkpoint
and recipe. This action verifies that both files exist and that the recipe's
`anomaly_types` contains the normalized `TEXTURE+TYPE`. A later synthesis
integration may resolve those two paths from a completed fine-tuning handoff,
but they must be concrete before invoking this leaf.

`pool_dataset_root` is one user-level typed folder input, not an incidental path
hidden only inside YAML. The platform stages the source once, records its
source-to-compute binding, and passes the resolved compute path to preparation.
Preparation freezes that path in `filtering_config.yaml` and in every pool
filepath carried by the clean embedding parquet. It also records the pool under
`input_contract.json.downstream_inputs` as a required read-only folder for the
clean-image embedding and AMP actions. Each action has a separate container
mount namespace, so staging may be reused but the source must be remounted at
the same frozen compute path for each consumer.

The frozen contract shape is:

```json
{
  "downstream_inputs": {
    "clean_embeddings": {
      "pool_dataset_root": {
        "type": "folder",
        "compute_path": "/inputs/anomalygen_pool",
        "read_only": true
      }
    },
    "run_amp": {
      "pool_dataset_root": {
        "type": "folder",
        "compute_path": "/inputs/anomalygen_pool",
        "read_only": true
      }
    }
  }
}
```

`compute_path` is the required mount destination, not a second source input.
The platform retains the original user-supplied source binding and uses it to
construct each later container mount.

The emitted embedding YAML files remain native Data Services configs; do not
add mount-only fields to them. The platform action request carries the typed
folder mount separately. The clean embedding consumes the frozen pool binding.
The FN embedding does not consume the pool, but its container must separately
mount every source-image root referenced by `fn_embedding_inputs.parquet`.

Selection supports:

- `all_eligible`: every eligible FN in the listed datasets.
- `per_dataset`: the first deterministic `per_dataset` rows from the named
  `split` for each listed dataset.

The same frozen encoder identity is written to both embedding specs. The next
action must use those specs unchanged so clean and FN vectors remain comparable.
`input_contract.json` records eligible IDs, skipped IDs, and skip reasons. An
all-ineligible selection is a typed `SKIPPED` result and emits no embedding or
AMP inputs with top-level reason `no_eligible_false_negatives`. Detailed
`skip_counts`, per-FN reason `no_clean_reference_images`, and warning evidence
record clean-reference failures without creating a separate stage-level reason.

`run_amp` joins the unique source-image embedding back to every box-level FN,
ranks distinct clean image paths within the normalized texture and anomaly-type
pool, and creates two AMP requests for every eligible pair. When the embedding
action emits several crop vectors for one clean image, its maximum crop
similarity is the image-level score; crop rows never occupy separate neighbor
ranks. It rejects missing, non-finite, zero-norm, or width-mismatched
embeddings. It always loads the frozen config from the
prepared root; no caller-supplied config is compared or accepted. The typed pool
argument is generated from the retained binding and must resolve to the pool
path frozen during preparation. AMP itself remains owned by the
container-native `anomalygen.scripts.auto_mask_placement.roi_place` entry point.

The `run_amp` action also requires the complete AnomalyGenNext checkpoint root
mounted at `/workspace/paidf-anomalygen/checkpoints`. Before native AMP starts,
it verifies the pinned Qwen cache, the configured `amp.model_id` as a complete
direct-local model or Hugging Face snapshot, and
`facebook/sam2.1-hiera-large/sam2.1_hiera_large.pt`; a standalone
SAM2 file or a checkpoint tree mounted at another path is rejected.

`finalize_inputs` retains the configured number of successful neighbors per FN.
Both aligned mask branches must match the clean image and cover neither zero nor
all pixels. It copies accepted masks under the prepared root and hashes every
generation contract artifact, without changing the source training pool.
