# Customer Adapter Contracts

The controller invokes customer logic only through command templates. Adapter
parameters are ordinary YAML values under `actions.<stage>.parameters` and can
be referenced as `{parameter_name}`. Existing local parameter files are hashed
into `data.lock.json`; secrets must be supplied through the execution platform,
not the workflow configuration.

## Prepare fixed mining embeddings

Run preparation inside the allocated DS container, using its installed TAO
packages. Mining uses a fixed encoder separate from the changing DINOv3 scoring
checkpoint. The shipped image producer supports CLIP and SigLIP; C-RADIO is not
required and no C-RADIO producer is shipped. Use the same immutable model and
processor snapshot for source and target and keep their embeddings fixed across rounds.

Prepare separate source and target Parquets with unique absolute `filepath`,
globally unique `sample_id`, matching `path`, and `storage_type: file` columns.
Targets also need `task` and `role` (`query` or `reference`), with reference rows
for each query task. For `grit-score`, each task needs more than `settling_k`
(default 50) references and more than the largest `view_ks` value (default 32)
queries; `validate` rejects smaller cohorts. Preserve the intended
source/query/reference splits.
The producer preserves extra columns; duplicate filepaths must be removed before
its metadata join. Archive members require an explicitly prepared file view.

1. Run `python -m nvidia_tao_ds.mining.embedding.scripts.image_embeddings`
   separately for source and target, supplying `input_parquet=...`,
   `output_parquet=...`, `model=CLIP` (or `SigLIP`), and
   `model_path=/models/fixed-snapshot`. Pre-stage both model and processor there.
2. Run `python -m nvidia_tao_ds.mining.dinov3.internal.refinement register-store`
   with `--store-root <source-shard-directory>`, `--output-dir <new-store-directory>`,
   and `--source-payload-contract <immutable-payload-contract.json>`.
3. Run the same refinement module's `write-target-contract` with
   `--targets <target-embeddings.parquet>`,
   `--source-store-manifest <new-store-directory>/embedding_store.json`,
   and `--output-dir <new-target-contract-directory>`.

Both registration commands require `--encoder-name`,
`--encoder-checkpoint-digest`, `--input-resolution`, and `--normalization`.
Declare the actual matching encoder/processor identity, not just matching vector
dimensions; include processor/configuration identity in the normalization
descriptor when needed. Use fresh output directories for publication.

Set `data.target_manifest` to the target Parquet, `data.source_store_manifest`
to the registered store, and `data.target_embedding_contract` to the generated
`target_embedding_contract.json`, then validate the configured workflow.
The writer validates target vectors and records input digests, but the recorded
target digest is lineage only: workflow consumers enforce encoder, dimension and
source-inventory compatibility, not equality to that recorded target digest.
Regenerate the contract when targets change and approve a fresh run. Neither the
writer nor matching dimensions certify encoder identity or task/split semantics.

## Source Data

`data.source_payload_contract` is a small JSON document binding source locators
to versioned, immutable payload roots. It has this shape:

```json
{
  "schema_version": "1.0",
  "immutability": "immutable",
  "datasets": [
    {
      "dataset_id": "wfm-aoi",
      "version": "dss-snapshot-12345",
      "root_uri": "file:///datasets/wfm-aoi-v12345"
    }
  ]
}
```

Use a DSS snapshot, object version, archive-set digest, or an equivalent stable
version identifier. The workflow hashes this contract on approval and resume.
Registered embedding stores are committed artifacts whose `shards` entries must
each include `relative_path`, `bytes`, and `sha256`; the controller verifies the
actual bytes and binds the resulting inventory digest into `data.lock.json`.
Create them with Data Services `register-store --source-payload-contract ...`.
Registration checks every locator against a contracted root and binds the exact
contract file plus a deterministic locator-audit digest into the store artifact.
The legacy `data.source_parts` path recomputes the same locator audit during
approval and resume; use a registered store for large corpora.
This gives image/archive provenance without materializing or hashing every
individual image in a large pool.

## Task Scoring

DS ships a scorer only for GRIT. `multi-task-round-robin` needs the user's own
score adapter, because weakness comes from the user's task heads and labels.
The packaged recipe names the placeholder `/path/to/customer_score_adapter`;
every command that loads the config (`validate`, `plan`, `run`, `resume`, ...)
rejects it. If the user has no adapter that meets
the contract below, offer `grit-score` instead. The DS document
[docs/dinov3_refinement.md "Multi-task score adapter"](https://github.com/NVIDIA-TAO/tao-data-services/blob/main/docs/dinov3_refinement.md#multi-task-score-adapter)
gives the full adapter contract: the command-template keys (`{checkpoint}`,
`{target_manifest}`, `{output_dir}`, `{round}` and the `parameters` keys) and the
`score_request.json` and `score_audit.json` digest sources.

Custom score adapters must declare `actions.score.implementation_files`; custom
evaluators must declare `actions.evaluate.implementation_files`. These lists are
the complete executable dependency closures approved for the run, not merely the
entrypoint. The packaged GRIT adapter closure is discovered and sealed
automatically.

Large CUDA GRIT cohorts require `execution.capabilities.gpu_faiss: true`.
Local execution probes `StandardGpuResources` and `index_cpu_to_gpu` before the
score command is launched. External runners receive the capability in the stage
request and must reject an image that cannot provide it. The default TAO pip
dependency is CPU-only; build the scoring image with the supported GPU FAISS
package instead of allowing an implicit CPU fallback.

Runner `status` and `cancel` responses must echo the exact requested
`client_job_id`; an absent or different identity is rejected. Status is one of
`PENDING`, `RUNNING`, `COMPLETE`, `ERROR`, `CANCELED`, or `UNKNOWN`; any other
value fails immediately instead of polling forever. `cancel` is idempotent and
returns `COMPLETE`, `ERROR`, or `CANCELED` only after the backend workload is
terminal. Unknown or nonterminal acknowledgements leave the workflow in
`canceling`; a later `cancel` or `run` reconciles it. Every external runner verb
is bounded by `execution.call_timeout_seconds`.

The scoring command receives the current DINO checkpoint, the fixed adaptive
target manifest, customer head configuration, and an output directory. It must
write `task_scores.parquet`, `score_commit.json`, and `_SUCCESS`.

Required Parquet columns:

| Column | Contract |
| --- | --- |
| `sample_id` | Identity from the target manifest |
| `task` | One configured `multi_task.tasks` value |
| `weakness_score` | Finite scalar; larger always means weaker |
| `embedding` | Fixed mining-encoder vector used only for source relevance |

Each `(sample_id, task)` pair must be unique. The adapter may use a frozen head
or refit a head on a fixed training split, but that policy and its inputs must
remain unchanged within a run.

The output identity set must exactly equal the target manifest identity set.
For GRIT this means every row whose `role` is `query`; for multi-task scoring it
means every declared `(sample_id, task)` pair. Partial score coverage is an
error, not an implicit sampling policy. Every output embedding must also match
the corresponding target-manifest mining vector exactly; scoring changes only
the weakness value, never the mining geometry.

`score_commit.json` must bind `input_sha256` for the target manifest,
`checkpoint_sha256` for the checkpoint used to produce weakness, and
`output_sha256` for `task_scores.parquet`. It must also record a nonempty
`implementation_sha256` covering the adapter and head-scoring implementation.
List imported implementation dependencies under
`actions.score.implementation_files`; the built-in GRIT action locks its CLI,
formula, and extraction pipeline automatically.
The controller also supplies `TAO_REFINEMENT_REQUEST_SHA256` and
`TAO_REFINEMENT_ENTRYPOINT_SHA256`, plus the approved implementation digest.
The commit must echo them as `request_sha256`, `entrypoint_sha256`, and
`implementation_sha256`; the trusted entrypoint must calculate its own digests
rather than copying environment values.
`_SUCCESS` contains the SHA-256 of `score_commit.json`. The controller verifies
this chain before target selection and again when adopting a cached score stage;
a score from a prior checkpoint is never accepted by path alone.

The controller converts `targets_per_round` and `task_weights` into a fixed
deterministic task allocation. Data Services ranks weakness independently in
each task, alternates among active tasks, and deduplicates `sample_id` globally.
Task preference therefore does not inflate the total per-round target budget.
For `multi_task.policy: balanced`, Data Services computes the allocation over
all configured tasks even when scoring has no actionable rows for one of them.
The unfilled quota is preserved. Materialization joins every selected neighbor
to its query task, keeps the cumulative manifest unique, and publishes a
separate task-balanced training view.

## Training

Data Services composes a complete immutable native DINOv3 experiment spec from
the reviewed base spec, cumulative manifest, immutable original
`model.base_checkpoint`, pass count, and resolved allocation. The leaf command is only
`dinov3 train -e {training_spec}`. After it exits successfully, Data Services
selects the exact terminal EMA-teacher export, publishes `checkpoint.pth`, and
seals `training_contract.json`, `training_commit.json`, the runtime
`experiment.yaml`, and `_SUCCESS`. The controller recomputes the full chain
before evaluation and whenever a completed run is reopened.

With `training.node_scaling`, the controller derives the allocation from the
cumulative manifest row count, the base spec's `dataset.batch_size`, and an
explicit optimizer-update floor. It writes `training_allocation.json` before
the leaf runs, writes the same values into the native spec, and includes them in
the stage request. A platform runner must honor that allocation exactly.

Changing worker count does not silently rewrite optimizer or scheduler values.
The native spec preserves the base learning rate, warmup, decay, checkpoint
cadence, and freeze policy. The allocation and expected optimizer-step count are
recorded per round so an intentional schedule change remains an explicit base
spec revision.

### Short-round schedule sizing

Size the schedule for each candidate, not the sum of all rounds. Every candidate
restarts from the original checkpoint and its optimizer step starts at zero.
For `R` cumulative training-manifest rows, per-GPU batch size `B`, world size
`W = num_nodes * gpus_per_node`, and `P = training.passes_per_round`, the expected budget is
`N = ceil(ceil(R / W) / B) * P`, recorded as `total_optimizer_steps` in
`training_contract.json`.

Compare `N` with both `train.schedulers.learning_rate.warm_up_steps` and
`train.schedulers.last_layer_learning_rate.warm_up_steps`, plus
`train.schedulers.last_layer_learning_rate.freeze_steps`.
Training uses steps `0` through `N - 1`: `N` must be strictly
greater than a threshold to leave that phase. Data Services warns during native
training-spec preparation when an LR warm-up or freeze covers the entire round.
The warning identifies the round, budget and each phase's covered fraction; it
does not reject intentional freezing or rewrite the schedule. The trigger is
full coverage, not a quality threshold: a phase covering 96% alone does not
trigger it.

The **pre-launch check is manual**: apply the formula to the anticipated
training-manifest size and allocation, then compare the resolved schedules.
`validate` and `plan` do not emit this warning. During `run` (or resumed training
preparation), the controller knows the actual materialized row count and writes
a `stage: train`, `status: schedule_warning` event to `<run_dir>/events.jsonl`
before submitting the training leaf. It also emits a Python warning on controller
stderr, visible on Docker with `docker logs "$JOB_ID"`. This is not leaf output:
do not expect it in `jobs/*.log` or `logs <run_dir> <client_job_id>`.
Inspect the durable event and prepared `refinement_input.yaml` for each round.

These diagnostics require an image containing the Data Services
[schedule-warning fix](https://github.com/NVIDIA-TAO/tao-data-services/pull/56).
Older images do not emit that schedule warning; the manual sizing check is the only check there.
Retain the packaged release-readiness checks before offering a launch.

For example, 768 rows, batch size 16, one GPU and two passes give 96 updates;
1536 rows give 192. The shipped ViT-B spec's 10000-step LR warm-ups and
1250-step last-layer freeze cover both rounds. The last-layer learning rate
stays zero throughout; successfully sealed artifacts do not demonstrate useful
adaptation.

For a **96-step smoke test only**, explicitly edit a copy of the base training
spec as follows, retaining its other settings, and point `training.base_spec`
at that copy:

```yaml
train:
  schedulers:
    learning_rate:
      warm_up_steps: 20
    last_layer_learning_rate:
      warm_up_steps: 20
      freeze_steps: 0
```

These are not quality-tuned defaults. Choose production schedules explicitly,
inspect other schedules such as teacher temperature, and validate held-out
metrics. Longer passes are another explicit choice; review the resulting
compute budget. Absence of this warning is not a quality guarantee. Schedule
changes require a new approved base spec/run, not edits to sealed round outputs.

### Native training execution

Manifest-backed training uses a deterministic shard-aware distributed sampler.
Each backing file or archive is assigned to ranks in bounded windows, every
rank receives the same number of rows, and ranks are padded to full batches so
all manifest rows are retained.
Archive handles use an explicit bounded cache. Random access is limited to
ordinary files, uncompressed tar, and zip archives.

There is no DEFT-specific training adapter. The Data Services round contract
requires `base_checkpoint_each_round`, and automatic checkpoint discovery is
disabled for prepared specs so every candidate starts from
`model.base_checkpoint` and cannot adopt an earlier candidate accidentally.

For distributed training, run the native TAO command once for the declared
static allocation and let Lightning create and coordinate DDP/FSDP workers. Do
not prepend `torchrun` or launch one independent training action per GPU. The
runner must submit all declared nodes as one gang, cancel every peer on failure,
and retry the full gang; partial rank retry is invalid. Every node sees the same
immutable spec, manifest, checkpoint, and shared output. Platform-provided GPU
visibility must match the allocation. Data Services alone finalizes the output
after the runner reports successful completion.

Runtime placement, code transport, cache placement, and cleanup belong to the
platform runner, not this workflow. A runner may use an installed package,
container image, content-addressed archive, shared filesystem, or another
platform-native mechanism as long as it preserves the immutable request and
committed-output contracts. Deployments may impose stricter storage policies
without changing the controller or action interface.

GRIT feature arrays are disk-backed and require `actions.score.settings.work_dir`
or `TAO_LOCAL_SCRATCH`. The image reader projects locator columns only and loads
fixed mining embeddings only for query rows. View neighborhoods are computed
once at the largest configured `k`; cohorts above 50,000 rows require exact
FAISS so the reference index remains resident instead of being recopied for
every query block.

## Runtime and provenance contracts

### Held-out benchmark isolation

This protection requires a DS image that contains the
[benchmark-isolation change](https://github.com/NVIDIA-TAO/tao-data-services/pull/57).
Older images accept the identity sidecar but never screen the source pool, so
benchmark rows can be mined into training without an error. Check this before
`validate`: `preflight` must report `"contracts": {"benchmark_isolation": 1}`
or a later version. If the field is missing or lower, stop before launch and
do not claim isolation.

Whenever a held-out benchmark is declared, provide `data.benchmark_acquisition_units`
as a nonempty Parquet identity table, even when evaluation is disabled. An empty
path is rejected. Without a held-out benchmark, omit both
`data.benchmark_manifest` and `data.benchmark_acquisition_units`.
The table must contain `sample_id` and `data.acquisition_unit_column` (default
`acquisition_unit_id`). Use `sample_id` as the acquisition-unit column only
when each sample really is an independent acquisition unit. A source store
whose `id_column` is not `sample_id` cannot use that column as the unit, because
mined rows carry it as `sample_id`. To exclude byte-identical copies with
different IDs/units, also supply `content_sha256`. No DS producer writes it per
row (`register_embedding_store` hashes shard files), so declare it only when the
embedding job writes it next to `path`; otherwise tell the user that renamed
copies are not excluded.
The evaluator's `data.benchmark_manifest` remains opaque; it is not the
identity table used for mining exclusions.

Targets, every source shard and inherited training manifests must contain each
declared identity column with non-null, nonempty values; declaring
`content_sha256` makes it mandatory on all of them. IDs and units are strings
or integers, compared exactly and case-sensitively after surrounding whitespace
is removed. SHA-256 values use 64 hexadecimal digits, optionally prefixed with
`sha256:`, and compare case-insensitively. Produce these identities before
registering the immutable source store. `validate` fails closed on missing
metadata before any stage runs, rather than silently reducing protection to ID
equality. Content hashes are trusted metadata, not recomputed from image bytes
at run time; acquisition-unit grouping must correctly represent the held-out split.

Any matching sample ID, acquisition unit, or declared content hash excludes
the source row before exact or indexed selection. Cumulative materialization
rejects overlap before publishing its artifact seal. Parent history, cached
training inputs and adopted balanced views are checked as well: contaminated
history is rejected, not silently edited after its checkpoint was trained.
This protects the declared identities; it cannot identify undeclared semantic
near-duplicates. The original base checkpoint's training history also remains
the user's responsibility.
Explicitly enabled `scope: diagnostic_replay` evaluation is not held-out
evaluation and permits overlap when no held-out identity sidecar is declared.
An explicitly supplied `benchmark_acquisition_units` always activates the
protection, regardless of evaluation scope. Disabling evaluation does not
disable protection for a declared benchmark.

When evaluation is enabled, round 0 evaluates the immutable base checkpoint.
Every later metrics payload must have the exact same unique `(task, name)`,
direction, and sample count contract, with finite numeric values. The evaluator
commit binds the checkpoint, metrics, benchmark identity, request digest, and
locked implementation. Declare imported evaluator dependencies through
`actions.evaluate.implementation_files`. The report presents raw values and
direction-normalized gain versus round 0. Evaluation remains optional; customer
heads and evaluators use this same leaf contract.

Registered source stores have two validation modes. `full_sha256` hashes every
shard at approval and resume and works without a prior local seal.
`sealed_inventory` uses the content seal emitted by `register-store` with
`--source-payload-contract` after one complete shard hash pass; no separate
`bind-store-payload` step is needed. Metadata-only `--no-hash-content`
registration cannot supply this seal. Subsequent checks compare the committed
digest inventory and POSIX device, inode, size, mtime, and ctime identities.
The seal is mount-bound: copying shards or changing mounts may invalidate it
despite identical bytes. Re-register into a fresh output directory on the target
mount, or use `full_sha256` for portable content verification.

GRIT actions declare node-local scratch through
`resources.local_scratch.path_environment`. The runner maps that environment
to platform scratch; the leaf computes required feature bytes from the approved
manifest and model width and fails before inference when capacity is insufficient.

Parent and cumulative training identities are always sent to retrieval as
exclusions so already-trained rows cannot consume the finite neighbor budget.
`parent_history_policy: exclude` also makes materialization fail closed if an
adapter nevertheless replays a row. A zero-row novel delta stops the loop only
after retrieval had the opportunity to search unseen neighbors.
Use `full_sha256` when a platform can replace files without changing those
identities or when the storage contract does not guarantee POSIX semantics.

`execution.backend: local` supports one controller process. With the default
`actions.train.execution_mode: runner`, multi-node fixed resources or node scaling
require `execution.backend: external` and these declared capabilities:
`gang_scheduling`, `gang_retry`, `attempt_scoped_launch_id`, and
`shared_filesystem`. An adapter that durably submits and adopts its own training
allocation instead uses `actions.train.execution_mode: adapter_managed`; its
`wrapper_resources` must remain one node, while the resolved allocation is
recorded separately in `execution_contract.adapter_managed_resources`.

The same execution choice applies to score, data, search, and evaluation
actions. This keeps the loop independent of the compute platform: a command may
be launched directly by a local, Kubernetes, Slurm, or custom runner, or it may
be a durable adapter that owns a platform-native child job. Adapter-managed
actions must make submission idempotent, adopt an existing matching child job,
and return only after committed outputs or a terminal failure are visible.

Each training output directory is bound to one preparation request. Repeating a
completed identical request is an idempotent success; changing its base spec,
manifest, parent checkpoint, passes, resources, or checkpoint policy requires a
new output directory. The adapter fails rather than replacing existing lineage.

## Evaluation

Evaluation is optional. A configured command receives the current checkpoint,
sealed benchmark manifest, customer parameters, and output directory. It must
write `metrics.json` matching `metrics.schema.json`,
`evaluation_commit.json`, and `_SUCCESS`. The commit binds the exact checkpoint
fingerprint, evaluation scope, and metrics file identity; `_SUCCESS` contains
the commit file's SHA-256. Additional per-sample outputs should also be listed
in the commit. A scheduler job ID alone is not a success seal.

Evaluation output is recorded in the report only. It cannot change selection,
mining radius, persistence, or stopping. The controller validates that the
adaptive target cohort and benchmark-unit manifest are disjoint.

With `scope: diagnostic_replay`, overlap is allowed for operational feedback,
but the output is not held-out evidence and must be labeled accordingly.
Omit the held-out identity sidecar for replay-only evaluation; changing scope
does not override an explicitly declared benchmark isolation constraint.

## Continuation

Never rewrite a prior run's state or lock files to adopt work produced under a
different workflow release. A continuation uses a new `run_dir`, sets
`workflow.start_round`, keeps `model.base_checkpoint` as the immutable training
initializer, sets `model.initial_scoring_checkpoint` to the accepted candidate,
and supplies `data.previous_training_manifest` plus a `continuation` contract.
The contract names the prior run, adopted round, sealed training contract,
commit, success marker, and prior data/release locks.

Run `adopt-training` before `run`. Adoption validates manifest bytes and rows,
base checkpoint, pass count, native topology, optimizer steps, final EMA teacher
name, runtime spec, checkpoint, and every seal. It records old and new lock
identities additively, marks only the adopted training stage complete, clears
stale jobs, and leaves evaluation and the round incomplete. The next `run`
performs exactly that evaluation before starting a new mining round.
The adoption record also seals the previous `state.json` and carries forward
`persistent_targets`, so recovery cannot reset noisy-sample suppression.

## Large-Scale Search

Use `actions.search.backend: audited_ann` for a persistent large-pool index.
Configure a committed dense-store manifest, index manifest, recall-audit
manifest, audited probe count, audited candidate depth, and separate candidate
and rerank command prefixes. The controller appends the concrete Data Services
subcommand and arguments. Candidate retrieval commits `ann_candidates.npz` in
`search_candidates`; reranking commits the normal search outputs below. These
are independent resumable jobs and may use different Python/CUDA runtimes.

The recall audit must belong to the configured index and approve the exact
probe count and candidate depth. cuVS output is never a final mining decision:
the rerank runtime reads immutable float32 mining vectors and applies exact
cosine relevance, parent/round exclusion, adaptive radius, and global duplicate
rejection. Relevance-radius expansion and duplicate rejection retain the same
meaning as exact search.

An optional `actions.search.backend: custom` plus `command` remains available
for customer indexes. It receives queries, the registered embedding-store
manifest, cumulative exclusion manifest, mining thresholds, and output
directory. The exclusion manifest includes held-out benchmark rows; an adapter
that returns them causes materialization to reject the round. Both indexed implementations publish:

- `neighbors.parquet`
- `search_summary.json`
- `artifact.json`
- `_SUCCESS`

The committed artifact records the exact locked search input: every
`source_shard` for direct Parquets, the `source_store_manifest` identity and
artifact ID for a registered exact/custom store, or the `dense_vector_store`
identity and artifact ID for audited ANN. Registered searches also record the
`query_embedding_contract`. The controller compares these inputs with
`data.lock.json` after every search, so a same-path store replacement fails the
stage instead of entering the cumulative training manifest.

`search_summary.json.search_proof` is either `exact_all_declared_shards` or an
audited ANN identifier beginning with `ann_audited_`. An empty exact result can
prove `radius_exhausted`; an empty ANN result produces
`search_budget_exhausted` unless the eligible source count is zero. Exact search
remains the terminal proof path when corpus exhaustion must be distinguished
from insufficient ANN depth.
