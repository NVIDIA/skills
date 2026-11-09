# Loader acceptance and stream contracts

Read when constructing shards, diagnosing missing samples or claiming resumable input
order. The [dataset README](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/README.md) owns layouts, keys, presets and
preparation commands; the module docstring of [wds_utils.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/wds_utils.py)
owns sample order, shard counts and multi-rank behavior.

## A readable shard is not a usable batch

Trace `WDSLoader.filter_items` and `_preprocess` in
[wds_dataloaders.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/wds_dataloaders.py) and the selected decoder.

- A sample missing a `key_map` member is dropped unless a `files_map`/`presets_map`
  constant provides that key, in which case the constant silently stands in. Test
  required fields with and without their member and count accepted samples after
  preprocessing.
- Collation requires identical nested keys across a batch before shapes are compared;
  optional members (an absent audio track) fail it, and so do variable-length members such
  as audio at mixed native sample rates unless the loader resamples them to one target rate
  ([`target_sample_rate`](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/README.md#videoaudioloaderconfig)).
  Homogenise, resample or filter the data, then check that the model accepts the remaining
  mode.
- Every sample carries `idx`; methods take the batch size from it even without a
  conditioning tensor. Augmentation and raw-versus-cached equivalence belong to the
  [network input contract](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-add-network/references/network-contract.md#raw-batches-become-model-inputs-once),
  not to a second preprocessing path in a data script.

## Check the selected stream, not its directory

Read `BaseWDSLoader`, `DeterministicWDS` and the batching helpers in `wds_utils.py`.

- `shards_per_worker` truncates **or repeats** shard URLs to reach its target; equal shard
  counts do not mean equal unique samples. Record the selected URLs, rank and worker
  counts and the accepted samples.
- `partial` controls incomplete batches; deterministic finite `drop_last` controls
  complete groups of data-parallel samples. The trainer's synchronised iterator detects
  unequal exhaustion but does not enforce equal batch sizes, so validate availability on
  every participating rank and worker.
- Shuffled or resampled loaders ignore the resume offset; only deterministic loading keeps
  order, and changing the worker count changes the shard selection when
  `shards_per_worker` is set. For exact accounting precompute exclusions, use strict
  decoding and rebuild shard-count files after any data edit, since resume-skipped
  shards are not re-counted.

For a tiny fixture prove at least one complete batch under the intended settings; for a
long run prove that repetition or finite coverage matches the iteration budget. Report
unique usable examples and consumed samples separately.
