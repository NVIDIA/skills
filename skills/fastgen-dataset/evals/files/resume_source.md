# FastGen source fixture

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Excerpts below are input evidence, not an installed checkout.

## fastgen/datasets/wds_utils.py:14

```text
``sampler_start_idx`` counts samples before decoding, after removing the union of all ``ignore_index_paths``.
The trainer's resume offset matches the batches it received only if each remaining sample produces exactly one
output. A sample that fails to decode or preprocess, lacks a required field, or is dropped by the pipeline is
skipped with a warning naming it. Each skip pulls that worker's next sample forward, so the stream runs ahead of the
trainer's offset and a resume repeats one sample per skip: with one worker (the deterministic default) the last
samples before the checkpoint, with more the samples just after it that the skipping worker had already delivered.
``MAX_CONSECUTIVE_SKIPS`` skips in a row on one worker raise, since they point to a systematic problem.
``strict=True`` raises on the first such sample instead. ``presets_filter`` is rejected either way, because it drops
samples systematically; precompute its exclusions into ``ignore_index_paths``.

``DeterministicWDS`` documents the sample order. It is independent of the worker count only for a fixed shard
selection (setting ``shards_per_worker`` makes the selection depend on the worker count) and while no sample is
skipped, since a skip shifts only its worker's later batches. It requires the DataLoader's default in-order delivery.
Only ``train=True`` repeats the stream; otherwise a start index at or beyond the end yields no batches.

A ``shard_count_file`` must hold accurate raw counts (before ignore indices), and every ignored key must exist in
its shard, because the number of ignored keys is subtracted from these counts. Counts are checked for shards that
are read but not for shards skipped on resume; rebuild stale counts before resuming, or omit the file to read
through the offset.
```

## fastgen/datasets/wds_utils.py:393

```text
def get_shard_count(
    shard_count_file: str,
    ignore_index: Dict[str, Set[str]] | None = None,
) -> Dict[str, int]:
    """Load shard count from a JSON file.

    Args:
        shard_count_file: Path to JSON file with shard counts
        ignore_index: Optional ignore index to adjust counts

    Returns:
        Dictionary mapping shard names to sample counts
    """
    if s3.is_uri(shard_count_file):
        try:
            with os.popen(f"s5cmd cat {shlex.quote(shard_count_file)}", "r") as f:
                json_content = f.read()
                if not json_content.strip():
                    raise ValueError(f"Empty or failed to read S3 file: {shard_count_file}")
                shard_count = json.loads(json_content)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load shard count from S3 path {shard_count_file}: {e}")
            raise
    else:
        with open(shard_count_file, "r") as f:
            shard_count = json.load(f)

    if ignore_index is not None:
        for shard_name in shard_count.keys():
            if shard_name in ignore_index:
                shard_count[shard_name] -= len(ignore_index[shard_name])
    validate_shard_counts(shard_count)

    logger.info(
        f"Shard count file loaded from {shard_count_file}. Using {sum(shard_count.values())} samples from the dataset."
    )

    return shard_count
```

## fastgen/datasets/wds_utils.py:710

```text
def _resume_shards(urls: List[str], counts: Dict[str, int] | None, start: int, repeat: bool):
    """Return the remaining shard order and the offset within its first shard."""
    if counts is None:
        return urls, start
    total = sum(counts[shard_to_name(url)] for url in urls)
    if total == 0 or (not repeat and start >= total):
        return [], 0
    if repeat:
        start %= total
    for index, url in enumerate(urls):
        count = counts[shard_to_name(url)]
        if start < count:
            remaining = urls[index:] + urls[:index] if repeat else urls[index:]
            return remaining, start
        start -= count
    raise AssertionError("Validated shard counts did not contain the resume offset")
```
