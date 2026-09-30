# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bundled and adapted from the Apache-2.0 NVIDIA TAO Tutorials PAS DEFT utilities so the
# customer workflow does not depend on an external source checkout.

"""Data-mining and training-split utilities for DEFT pipelines.

- :func:`materialize_pas_eval_split` — materialize PAS eval image list and pairs file.
- :func:`materialize_pas_training_split` — write real/non-excluded seed training rows.
- :func:`materialize_pas_pool_split` — write mining pool rows.
- :func:`convert_vcn_csv_to_parquet` — VCN CSV → per-image parquet with absolute filepaths.
- :func:`convert_mined_parquet_to_csv` — reverse mined filepaths back to VCN training CSV rows.
- :func:`merge_train_csvs` — merge augmentation CSVs with the base training set.
- :func:`convert_clip_image_list_to_parquet` — CLIP image list → embedding parquet.
- :func:`convert_mined_parquet_to_clip_image_list` — mined parquet → CLIP image-list + pairs.
- :func:`summarize_knn_mining` — post-k-NN mining summary (txt + json).
- :func:`track_cumulative_mined_unique_names` — cumulative mined names across iterations.
- :func:`write_iteration_summary` — per-iteration artifact/checkpoint summary.
"""


def materialize_pas_eval_split(
    eval_pairs_source_file: str,
    eval_image_list_file: str,
    eval_pairs_file: str,
    query_types: str = "",
    val_image_list_file: str = "",
    val_sample_size: int = 512,
):
    """Materialize the PAS eval image list and pairs file from test_pairs.json.

    Args:
        eval_pairs_source_file:  Path to the source test_pairs.json.
        eval_image_list_file:    Output image list for eval rows.
        eval_pairs_file:         Output pairs JSON for eval rows.
        query_types:             Optional comma-separated query type filter.
        val_image_list_file:     Optional output path for a small sampled
                                 validation image list used during TAO training.
        val_sample_size:         Number of images to sample for the validation
                                 list (default 512).
    """
    import json
    import os
    import random

    def _split_csv(value):
        return {item.strip() for item in str(value or "").split(",") if item.strip()}

    def _infer_dataset(image_path):
        normalized = str(image_path or "").replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
        for marker in ("images", "data"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return parts[idx + 1].strip()
        return parts[0].strip() if len(parts) > 1 else ""

    def _normalize_row(row):
        unique_name = str(row.get("unique_name") or "").strip()
        caption = str(row.get("caption") or "").strip()
        image_path = str(row.get("image_path") or "").strip()
        dataset = str(row.get("dataset") or "").strip() or _infer_dataset(image_path)
        if not unique_name or not caption or not dataset:
            return None
        out_row = dict(row)
        out_row["dataset"] = dataset
        out_row["query_type"] = str(row.get("query_type") or "").strip()
        out_row["caption"] = caption
        out_row["unique_name"] = unique_name
        return out_row

    def _iter_json_records(path):
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
            second = f.readline()
        compact = (
            second.lstrip().startswith("{")
            and second.rstrip().rstrip(",").endswith("}")
        )
        if compact:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s in ("[", "]"):
                        continue
                    if s.endswith(","):
                        s = s[:-1]
                    if s:
                        yield json.loads(s)
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                yield row

    required = [p for p in (eval_image_list_file, eval_pairs_file, val_image_list_file) if p]
    if required and all(os.path.isfile(p) for p in required):
        print("PAS eval split already exists; skipping")
        return

    if not eval_pairs_source_file:
        raise ValueError("eval_pairs_source_file must be set")
    if not os.path.isfile(eval_pairs_source_file):
        raise FileNotFoundError(f"PAS eval pairs file not found: {eval_pairs_source_file}")

    qtypes = _split_csv(query_types)
    for path in required:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    eval_rows = 0
    eval_images = []
    eval_seen = set()
    eval_skipped = 0
    pairs_handle = None
    first = True
    try:
        if eval_pairs_file:
            pairs_handle = open(eval_pairs_file, "w", encoding="utf-8")
            pairs_handle.write("[\n")
        for row in _iter_json_records(eval_pairs_source_file):
            query_type = str(row.get("query_type") or "").strip()
            if qtypes and query_type not in qtypes:
                eval_skipped += 1
                continue
            out_row = _normalize_row(row)
            if out_row is None:
                eval_skipped += 1
                continue
            name = out_row["unique_name"]
            if name not in eval_seen:
                eval_seen.add(name)
                eval_images.append(name)
            if pairs_handle:
                if not first:
                    pairs_handle.write(",\n")
                pairs_handle.write(json.dumps(out_row, ensure_ascii=False))
                first = False
            eval_rows += 1
    finally:
        if pairs_handle:
            pairs_handle.write("\n]\n")
            pairs_handle.close()

    if eval_image_list_file:
        with open(eval_image_list_file, "w", encoding="utf-8") as f:
            for name in eval_images:
                f.write(f"{name}\n")

    if val_image_list_file and val_sample_size > 0 and eval_images:
        rng = random.Random(42)
        sample = rng.sample(eval_images, min(val_sample_size, len(eval_images)))
        val_parent = os.path.dirname(val_image_list_file)
        if val_parent:
            os.makedirs(val_parent, exist_ok=True)
        with open(val_image_list_file, "w", encoding="utf-8") as f:
            for name in sample:
                f.write(f"{name}\n")

    print(
        f"PAS eval split: {eval_rows} rows / {len(eval_images)} images "
        f"({eval_skipped} skipped) -> {eval_pairs_file}"
    )


def materialize_pas_training_split(
    train_pairs_source_file: str,
    seed_image_list_file: str,
    seed_pairs_file: str,
    seed_exclude_datasets: str = "CUHK_PEDES,ICFG_PEDES",
    augmented_suffix: str = "_Aug",
    query_types: str = "",
    max_seed_rows: int = 0,
):
    """Materialize the PAS seed/training image list and pairs file.

    Writes real, non-excluded rows from train_pairs.json to the seed
    training files. Skips silently if both output files already exist.

    Args:
        train_pairs_source_file:  Path to the source train_pairs.json.
        seed_image_list_file:     Output image list for real seed rows.
        seed_pairs_file:          Output pairs JSON for real seed rows.
        seed_exclude_datasets:    Comma-separated real datasets to exclude.
        augmented_suffix:         Dataset suffix identifying augmented data.
        query_types:              Optional comma-separated query type filter.
        max_seed_rows:            Optional cap; 0 means no cap.
    """
    import json
    import os

    def _split_csv(value):
        return {item.strip() for item in str(value or "").split(",") if item.strip()}

    def _infer_dataset(image_path):
        normalized = str(image_path or "").replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
        for marker in ("images", "data"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return parts[idx + 1].strip()
        return parts[0].strip() if len(parts) > 1 else ""

    def _normalize_row(row):
        unique_name = str(row.get("unique_name") or "").strip()
        caption = str(row.get("caption") or "").strip()
        image_path = str(row.get("image_path") or "").strip()
        dataset = str(row.get("dataset") or "").strip() or _infer_dataset(image_path)
        if not unique_name or not caption or not dataset:
            return None
        out_row = dict(row)
        out_row["dataset"] = dataset
        out_row["query_type"] = str(row.get("query_type") or "").strip()
        out_row["caption"] = caption
        out_row["unique_name"] = unique_name
        return out_row

    def _iter_json_records(path):
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
            second = f.readline()
        compact = (
            second.lstrip().startswith("{")
            and second.rstrip().rstrip(",").endswith("}")
        )
        if compact:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s in ("[", "]"):
                        continue
                    if s.endswith(","):
                        s = s[:-1]
                    if s:
                        yield json.loads(s)
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                yield row

    required = [p for p in (seed_image_list_file, seed_pairs_file) if p]
    if required and all(os.path.isfile(p) for p in required):
        print("PAS training split already exists; skipping")
        return

    if not train_pairs_source_file:
        raise ValueError("train_pairs_source_file must be set")
    if not os.path.isfile(train_pairs_source_file):
        raise FileNotFoundError(
            f"PAS train_pairs.json not found: {train_pairs_source_file}"
        )

    for path in required:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    excluded = _split_csv(seed_exclude_datasets)
    qtypes = _split_csv(query_types)
    max_seed_rows = int(max_seed_rows or 0)

    seed_count = 0
    skipped_query_type = 0
    skipped_unknown_dataset = 0
    first_seed = True

    seed_list_handle = None
    seed_pairs_handle = None
    try:
        if seed_image_list_file:
            seed_list_handle = open(seed_image_list_file, "w", encoding="utf-8")
        if seed_pairs_file:
            seed_pairs_handle = open(seed_pairs_file, "w", encoding="utf-8")
            seed_pairs_handle.write("[\n")

        for row in _iter_json_records(train_pairs_source_file):
            query_type = str(row.get("query_type") or "").strip()
            if qtypes and query_type not in qtypes:
                skipped_query_type += 1
                continue
            out_row = _normalize_row(row)
            if out_row is None:
                skipped_unknown_dataset += 1
                continue
            dataset = out_row["dataset"]
            unique_name = out_row["unique_name"]

            if "is_augmented" in out_row:
                value = out_row.get("is_augmented")
                if isinstance(value, bool):
                    is_aug = value
                else:
                    normalized_val = str(value).strip().lower()
                    if normalized_val in {"true", "1", "yes", "y", "on"}:
                        is_aug = True
                    elif normalized_val in {"false", "0", "no", "n", "off"}:
                        is_aug = False
                    else:
                        is_aug = bool(augmented_suffix) and dataset.endswith(augmented_suffix)
            else:
                is_aug = bool(augmented_suffix) and dataset.endswith(augmented_suffix)
            is_real = not is_aug

            if is_real and dataset not in excluded:
                if not max_seed_rows or seed_count < max_seed_rows:
                    if seed_pairs_handle:
                        if not first_seed:
                            seed_pairs_handle.write(",\n")
                        seed_pairs_handle.write(json.dumps(out_row, ensure_ascii=False))
                        first_seed = False
                    if seed_list_handle:
                        seed_list_handle.write(f"{unique_name}\n")
                    seed_count += 1

    finally:
        if seed_pairs_handle:
            seed_pairs_handle.write("\n]\n")
            seed_pairs_handle.close()
        if seed_list_handle:
            seed_list_handle.close()

    print(
        f"PAS training split: {seed_count} rows "
        f"({skipped_unknown_dataset} malformed, {skipped_query_type} query-type-filtered) "
        f"-> {seed_pairs_file}"
    )


def materialize_pas_pool_split(
    pool_pairs_source_file: str,
    aug_pool_image_list_file: str,
    aug_pool_pairs_file: str,
    augmented_suffix: str = "_Aug",
    query_types: str = "",
    max_aug_pool_rows: int = 0,
    mining_pool_mode: str = "augmented",
):
    """Materialize the PAS mining pool image list and pairs file.

    Writes pool rows from pool_pairs_source_file based on mining_pool_mode.
    Skips silently if both output files already exist.

    Args:
        pool_pairs_source_file:    Path to the source pairs JSON for the mining pool.
        aug_pool_image_list_file:  Output image list for pool rows.
        aug_pool_pairs_file:       Output pairs JSON for pool rows.
        augmented_suffix:          Dataset suffix identifying augmented data.
        query_types:               Optional comma-separated query type filter.
        max_aug_pool_rows:         Optional cap; 0 means no cap.
        mining_pool_mode:          Which rows go into pool:
                                   ``real``, ``augmented``, or
                                   ``real_and_augmented``.
    """
    import json
    import os

    def _split_csv(value):
        return {item.strip() for item in str(value or "").split(",") if item.strip()}

    def _infer_dataset(image_path):
        normalized = str(image_path or "").replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
        for marker in ("images", "data"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return parts[idx + 1].strip()
        return parts[0].strip() if len(parts) > 1 else ""

    def _normalize_row(row):
        unique_name = str(row.get("unique_name") or "").strip()
        caption = str(row.get("caption") or "").strip()
        image_path = str(row.get("image_path") or "").strip()
        dataset = str(row.get("dataset") or "").strip() or _infer_dataset(image_path)
        if not unique_name or not caption or not dataset:
            return None
        out_row = dict(row)
        out_row["dataset"] = dataset
        out_row["query_type"] = str(row.get("query_type") or "").strip()
        out_row["caption"] = caption
        out_row["unique_name"] = unique_name
        return out_row

    def _iter_json_records(path):
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
            second = f.readline()
        compact = (
            second.lstrip().startswith("{")
            and second.rstrip().rstrip(",").endswith("}")
        )
        if compact:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s in ("[", "]"):
                        continue
                    if s.endswith(","):
                        s = s[:-1]
                    if s:
                        yield json.loads(s)
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                yield row

    def _is_augmented_row(row, dataset):
        if "is_augmented" in row:
            value = row.get("is_augmented")
            if isinstance(value, bool):
                return value
            normalized = str(value).strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off"}:
                return False
        return bool(augmented_suffix) and dataset.endswith(augmented_suffix)

    def _mode_tokens(value):
        normalized = str(value or "").lower().replace("+", ",").replace(";", ",")
        tokens = _split_csv(normalized)
        if "real_and_augmented" in tokens or "all" in tokens:
            tokens.update({"real", "augmented"})
        if "aug" in tokens:
            tokens.add("augmented")
        return tokens or {"augmented"}

    required = [p for p in (aug_pool_image_list_file, aug_pool_pairs_file) if p]
    if required and all(os.path.isfile(p) for p in required):
        print("PAS pool split already exists; skipping")
        return

    if not pool_pairs_source_file:
        raise ValueError("pool_pairs_source_file must be set")
    if not os.path.isfile(pool_pairs_source_file):
        raise FileNotFoundError(
            f"Pool pairs file not found: {pool_pairs_source_file}"
        )

    for path in required:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    qtypes = _split_csv(query_types)
    max_aug_pool_rows = int(max_aug_pool_rows or 0)
    pool_modes = _mode_tokens(mining_pool_mode)
    include_real_pool = "real" in pool_modes
    include_aug_pool = "augmented" in pool_modes

    pool_count = 0
    skipped_query_type = 0
    skipped_unknown_dataset = 0
    first_pool = True

    pool_list_handle = None
    pool_pairs_handle = None
    try:
        if aug_pool_image_list_file:
            pool_list_handle = open(aug_pool_image_list_file, "w", encoding="utf-8")
        if aug_pool_pairs_file:
            pool_pairs_handle = open(aug_pool_pairs_file, "w", encoding="utf-8")
            pool_pairs_handle.write("[\n")

        for row in _iter_json_records(pool_pairs_source_file):
            query_type = str(row.get("query_type") or "").strip()
            if qtypes and query_type not in qtypes:
                skipped_query_type += 1
                continue
            out_row = _normalize_row(row)
            if out_row is None:
                skipped_unknown_dataset += 1
                continue
            dataset = out_row["dataset"]
            unique_name = out_row["unique_name"]
            is_aug = _is_augmented_row(out_row, dataset)
            is_real = not is_aug

            if (is_aug and include_aug_pool) or (is_real and include_real_pool):
                if not max_aug_pool_rows or pool_count < max_aug_pool_rows:
                    if pool_pairs_handle:
                        if not first_pool:
                            pool_pairs_handle.write(",\n")
                        pool_pairs_handle.write(json.dumps(out_row, ensure_ascii=False))
                        first_pool = False
                    if pool_list_handle:
                        pool_list_handle.write(f"{unique_name}\n")
                    pool_count += 1

    finally:
        if pool_pairs_handle:
            pool_pairs_handle.write("\n]\n")
            pool_pairs_handle.close()
        if pool_list_handle:
            pool_list_handle.close()

    print(
        f"PAS pool split: {pool_count} rows (mode={sorted(pool_modes)}) "
        f"({skipped_unknown_dataset} malformed, {skipped_query_type} query-type-filtered) "
        f"-> {aug_pool_pairs_file}"
    )


def convert_clip_image_list_to_parquet(
    image_list_file: str,
    image_dir: str,
    output_parquet: str,
    caption_dir: str = "",
    caption_file_suffix: str = "",
    pairs_file: str = "",
) -> str:
    """Convert a CLIP-format ``image_list_file`` to an embedding parquet.

    Args:
        image_list_file:     Path to a text file with one image basename per line.
        image_dir:           Flat directory of images.
        output_parquet:      Path where the output parquet will be written.
        caption_dir:         Optional caption directory.
        caption_file_suffix: Caption file suffix (e.g. ``.txt``).
        pairs_file:          Optional TAO-FT ``*_pairs.json``.

    Returns:
        Path to ``output_parquet``.
    """
    import json
    import os

    import pandas as pd

    def _infer_dataset(image_path):
        normalized = str(image_path or "").replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
        for marker in ("images", "data"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return parts[idx + 1].strip()
        if len(parts) > 1:
            return parts[0].strip()
        return ""

    def _iter_json_records(path):
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
            second = f.readline()
        compact = (
            second.lstrip().startswith("{")
            and second.rstrip().rstrip(",").endswith("}")
        )
        if compact:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s in ("[", "]"):
                        continue
                    if s.endswith(","):
                        s = s[:-1]
                    if s:
                        yield json.loads(s)
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                yield row

    out_dir = os.path.dirname(output_parquet)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(image_list_file, "r", encoding="utf-8") as f:
        basenames = [line.strip() for line in f if line.strip()]

    if pairs_file:
        wanted = set(basenames)
        records = []
        for row in _iter_json_records(pairs_file):
            name = str(row.get("unique_name") or "").strip()
            if name not in wanted:
                continue
            image_path = str(row.get("image_path") or "")
            dataset = str(row.get("dataset") or "").strip() or _infer_dataset(image_path)
            records.append({
                "filepath": os.path.abspath(os.path.join(image_dir, name)),
                "text": str(row.get("caption") or "").strip(),
                "unique_name": name,
                "image_path": image_path,
                "query_type": str(row.get("query_type") or "").strip(),
                "dataset": dataset,
            })
        out_df = pd.DataFrame(records)
        out_df.to_parquet(output_parquet, index=False)
        print(
            f"Converted CLIP pairs to parquet: "
            f"{len(records)} rows from {pairs_file} -> {output_parquet}"
        )
        return output_parquet

    include_text = bool(caption_dir) and bool(caption_file_suffix)
    records = []
    for name in basenames:
        rec = {"filepath": os.path.abspath(os.path.join(image_dir, name))}
        if include_text:
            stem = os.path.splitext(name)[0]
            caption_path = os.path.join(
                caption_dir, f"{stem}{caption_file_suffix}",
            )
            with open(caption_path, "r", encoding="utf-8") as f:
                rec["text"] = f.read().strip()
        records.append(rec)

    columns = ["filepath", "text"] if include_text else ["filepath"]
    out_df = pd.DataFrame(records, columns=columns)
    out_df.to_parquet(output_parquet, index=False)
    print(
        f"Converted CLIP image_list_file to parquet: "
        f"{len(basenames)} basenames -> {output_parquet}"
    )
    return output_parquet


def convert_mined_parquet_to_clip_image_list(
    mined_parquet: str,
    image_dir: str,
    caption_dir: str,
    caption_file_suffix: str,
    output_image_list_file: str,
    manifest_path: str,
    source_pairs_file: str = "",
    output_pairs_file: str = "",
    target_query_count: int = 0,
    caption_expansion_enabled: str = "false",
    caption_expansion_mode: str = "nearest",
    caption_expansion_max_pairs_per_image_path: int = 2,
    caption_expansion_max_expanded_pair_fraction: float = 0.25,
    caption_expansion_dedupe_normalized_caption: str = "true",
    caption_expansion_count_expanded_pairs_toward_target: str = "auto",
    source_embedding_shards_dir: str = "",
    write_detailed_csv: str = "true",
    source_embeddings_parquet: str = "",
    target_embeddings_parquet: str = "",
) -> str:
    """Convert mined filepaths into CLIP image-list, pairs, and stats files.

    When history-aware selection runs downstream, call this with
    ``target_query_count=0`` so the candidate set stays uncapped: the budget
    belongs to the selector, which spends it after partitioning novel from
    previously-committed names. ``write_detailed_csv`` should be off in that
    case, since the selector writes its own detail CSV over the committed
    selection and this one would only describe discarded candidates.

    The mine step emits only ``filepath``, so the candidate order it produces
    is its emission order — grouped by target query, not by similarity.
    Truncating that to ``target_query_count`` starves whichever weak queries
    happen to sit at the tail of the file. Pass ``source_embeddings_parquet``
    and ``target_embeddings_parquet`` (both are already written upstream) to
    recompute each candidate's best cosine similarity to any weak query and
    order by it first. Skipped when the mined parquet already carries a
    ``score`` column, so a future mine step that emits scores directly wins.
    """
    import json
    import math
    import os
    from collections import Counter

    import numpy as np
    import pandas as pd

    def _iter_json_records(path):
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
            second = f.readline()
        compact = (
            second.lstrip().startswith("{")
            and second.rstrip().rstrip(",").endswith("}")
        )
        if compact:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s in ("[", "]"):
                        continue
                    if s.endswith(","):
                        s = s[:-1]
                    if s:
                        yield json.loads(s)
            return
        with open(path, "r", encoding="utf-8") as f:
            for row in json.load(f):
                yield row

    def _counter_records(counter):
        total = sum(counter.values())
        rows = []
        for value, count in counter.most_common():
            rows.append({
                "value": value,
                "count": int(count),
                "pct": (100.0 * count / total) if total else 0.0,
            })
        return rows

    def _truthy(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _falsey(value):
        return str(value).strip().lower() in {"0", "false", "no", "n", "off"}

    def _pair_name(row):
        return str(row.get("unique_name") or "").strip().replace("\\", "/")

    def _pair_image_path(row):
        return str(row.get("image_path") or "").strip().replace("\\", "/")

    def _pair_caption(row):
        return str(row.get("caption") or row.get("text") or "").strip()

    def _normalize_caption(value):
        return " ".join(str(value or "").strip().lower().split())

    def _lexical_similarity(a, b):
        a_tokens = set(_normalize_caption(a).split())
        b_tokens = set(_normalize_caption(b).split())
        if not a_tokens and not b_tokens:
            return 1.0
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    def _dense_vector(value):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return None
        norm = float(np.linalg.norm(arr))
        if not math.isfinite(norm) or norm <= 0:
            return None
        return arr / norm

    def _load_embeddings_for_names(shards_dir, wanted_names):
        embeddings = {}
        if not shards_dir or not wanted_names:
            return embeddings, 0
        shard_dir = os.path.normpath(os.path.realpath(shards_dir))
        if not os.path.isdir(shard_dir):
            return embeddings, 0
        shard_paths = sorted(
            os.path.join(shard_dir, name)
            for name in os.listdir(shard_dir)
            if name.endswith("_embeddings.parquet")
        )
        wanted = set(wanted_names)
        rows_scanned = 0
        for shard_path in shard_paths:
            df = pd.read_parquet(
                shard_path,
                columns=["unique_name", "embedding"],
            )
            rows_scanned += len(df)
            if df.empty:
                continue
            names = df["unique_name"].fillna("").astype(str)
            mask = names.isin(wanted)
            if not mask.any():
                continue
            for name, embedding in zip(names[mask], df.loc[mask, "embedding"]):
                normed = _dense_vector(embedding)
                if normed is not None:
                    embeddings[str(name)] = normed
        return embeddings, rows_scanned

    image_root = os.path.normpath(os.path.realpath(image_dir))

    def _mined_unique_name(row):
        name = str(row.get("unique_name") or "").strip()
        if name:
            return name.replace("\\", "/")
        filepath = str(row.get("filepath") or "").strip()
        if not filepath:
            return ""
        abs_path = os.path.normpath(os.path.realpath(filepath))
        try:
            rel = os.path.relpath(abs_path, image_root)
        except ValueError:
            rel = ""
        if (
            rel
            and rel != os.curdir
            and rel != os.pardir
            and not rel.startswith(os.pardir + os.sep)
            and not os.path.isabs(rel)
        ):
            return rel.replace(os.sep, "/")
        return os.path.basename(filepath).replace("\\", "/")

    mined = pd.read_parquet(mined_parquet)
    if "filepath" not in mined.columns:
        raise ValueError(
            f"mined parquet at {mined_parquet} missing required "
            f"'filepath' column; found {list(mined.columns)}"
        )

    def _rescore_by_similarity(mined_df):
        """Order candidates by best cosine similarity to any weak query.

        Streams the source-pool embeddings in batches and keeps only the
        mined rows, so peak memory is one batch plus the mined subset rather
        than the whole pool (which runs to several GB).
        """
        import pyarrow.parquet as pq

        for label, path in (
            ("source", source_embeddings_parquet),
            ("target", target_embeddings_parquet),
        ):
            if not os.path.isfile(path):
                print(
                    f"{label} embeddings parquet not found at {path}; "
                    "keeping mined order"
                )
                return mined_df, False

        wanted = set(mined_df["filepath"].astype(str))
        target_df = pd.read_parquet(target_embeddings_parquet)
        if "embedding" not in target_df.columns or not len(target_df):
            print(
                "Target embeddings unusable for rescoring "
                f"({target_embeddings_parquet}); keeping mined order"
            )
            return mined_df, False

        paths, vectors = [], []
        parquet_file = pq.ParquetFile(source_embeddings_parquet)
        for batch in parquet_file.iter_batches(
            columns=["filepath", "embedding"], batch_size=65536,
        ):
            chunk = batch.to_pandas()
            chunk = chunk[chunk["filepath"].astype(str).isin(wanted)]
            if len(chunk):
                paths.extend(chunk["filepath"].astype(str).tolist())
                vectors.extend(chunk["embedding"].tolist())
        if not paths:
            print(
                "No mined filepath matched the source embeddings at "
                f"{source_embeddings_parquet}; keeping mined order"
            )
            return mined_df, False

        candidates = np.asarray(np.vstack(vectors), dtype=np.float32)
        targets = np.asarray(
            np.vstack(target_df["embedding"].to_numpy()), dtype=np.float32,
        )
        candidates /= np.maximum(
            np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12,
        )
        targets /= np.maximum(
            np.linalg.norm(targets, axis=1, keepdims=True), 1e-12,
        )
        similarity = candidates @ targets.T
        best_target = similarity.argmax(axis=1)
        best_score = similarity.max(axis=1)

        scored = pd.DataFrame({"filepath": paths, "score": best_score})
        # Recovering which weak query each candidate is nearest to also
        # restores the per-attribute mining-coverage breakdown.
        for column, out_name in (
            ("query_type", "target_query_type"),
            ("weak_attribute", "target_attribute"),
        ):
            if column in target_df.columns:
                scored[out_name] = target_df[column].to_numpy()[best_target]
        scored = (
            scored.sort_values("score", ascending=False)
            .drop_duplicates(subset=["filepath"])
        )

        merged = mined_df.merge(scored, on="filepath", how="left")
        unscored = int(merged["score"].isna().sum())
        merged = merged.sort_values(
            "score", ascending=False, na_position="last", kind="mergesort",
        ).reset_index(drop=True)
        print(
            f"Rescored {len(scored)} mined candidates against "
            f"{len(target_df)} weak queries "
            f"({unscored} had no source embedding and sort last); "
            f"score range {merged['score'].min():.4f}..{merged['score'].max():.4f}"
        )
        return merged, True

    rescored_by_similarity = False
    if "score" in mined.columns:
        # A mine step that emits scores directly wins; just honour it.
        mined = mined.sort_values(
            "score", ascending=False, na_position="last", kind="mergesort",
        ).reset_index(drop=True)
        rescored_by_similarity = True
        print("Mined parquet carries a score column; ordered by it directly.")
    elif source_embeddings_parquet and target_embeddings_parquet:
        try:
            mined, rescored_by_similarity = _rescore_by_similarity(mined)
        except Exception as exc:  # noqa: BLE001
            # Ordering is an optimisation; a failure here must not take the
            # whole conversion down. The warning below still fires, so a
            # degraded run is never silent.
            print(f"Similarity rescoring failed ({type(exc).__name__}: {exc})")
            rescored_by_similarity = False
    if not rescored_by_similarity:
        print(
            "WARNING: mined candidates are in the mine step's emission order, "
            "which is grouped by target query rather than by similarity. Any "
            "target_query_count truncation below will keep whichever queries "
            "come first in the file, not the nearest images. Pass "
            "source_embeddings_parquet and target_embeddings_parquet to fix."
        )

    raw_rows = len(mined)
    raw_unique_filepaths = int(mined["filepath"].drop_duplicates().shape[0])
    raw_unique_names = 0
    if "unique_name" in mined.columns:
        raw_unique_names = int(
            mined["unique_name"].fillna("").astype(str)
            .loc[lambda values: values.astype(bool)]
            .drop_duplicates()
            .shape[0]
        )
    target_query_count = int(target_query_count or 0)
    write_details = _truthy(write_detailed_csv)
    basenames = (
        mined.apply(_mined_unique_name, axis=1)
        .loc[lambda values: values.astype(bool)]
        .drop_duplicates()
        .tolist()
    )

    pairs_by_name = {}
    selected_pair_items = []
    missing_pairs = 0
    train_pairs_file = ""
    pairs_count = 0
    pre_budget_unique_basenames = len(basenames)
    pre_budget_recovered_pairs = 0
    budget_omitted_basenames = 0
    budget_reached = False
    source_pair_rows_scanned = 0
    target_pair_shortfall = 0
    expansion_enabled = _truthy(caption_expansion_enabled)
    expansion_mode = str(caption_expansion_mode or "nearest").strip().lower()
    if expansion_mode not in {"nearest", "all"}:
        raise ValueError(
            "caption_expansion_mode must be one of: nearest, all; "
            f"got {caption_expansion_mode!r}"
        )
    max_pairs_per_image_path = max(
        0,
        int(caption_expansion_max_pairs_per_image_path or 0),
    )
    max_expanded_pair_fraction = max(
        0.0,
        min(1.0, float(caption_expansion_max_expanded_pair_fraction or 0.0)),
    )
    dedupe_expansion_captions = _truthy(
        caption_expansion_dedupe_normalized_caption
    )
    count_expanded_raw = str(
        caption_expansion_count_expanded_pairs_toward_target or "auto"
    ).strip().lower()
    if count_expanded_raw in {"", "auto"}:
        count_expanded_pairs_toward_target = expansion_mode != "all"
    elif _truthy(count_expanded_raw):
        count_expanded_pairs_toward_target = True
    elif _falsey(count_expanded_raw):
        count_expanded_pairs_toward_target = False
    else:
        raise ValueError(
            "caption_expansion_count_expanded_pairs_toward_target must be "
            f"auto, true, or false; got {caption_expansion_count_expanded_pairs_toward_target!r}"
        )
    caption_expansion_stats = {
        "enabled": bool(expansion_enabled),
        "mode": expansion_mode,
        "uses_nearest_caps": bool(expansion_mode == "nearest"),
        "count_expanded_pairs_toward_target": bool(
            count_expanded_pairs_toward_target
        ),
        "target_budget_applies_to": (
            "total_pairs"
            if count_expanded_pairs_toward_target
            else "anchor_pairs"
        ),
        "max_pairs_per_image_path": int(max_pairs_per_image_path),
        "max_expanded_pair_fraction": float(max_expanded_pair_fraction),
        "dedupe_normalized_caption": bool(dedupe_expansion_captions),
        "source_embedding_shards_dir": source_embedding_shards_dir,
        "source_pair_rows_scanned_for_expansion": 0,
        "source_embedding_rows_scanned_for_expansion": 0,
        "candidate_image_paths": 0,
        "candidate_pair_rows": 0,
        "candidate_pair_names": 0,
        "candidate_pairs_after_dedup": 0,
        "planned_anchor_pairs": 0,
        "planned_expanded_pairs": 0,
        "final_anchor_pairs": 0,
        "final_expanded_pairs": 0,
        "final_unique_image_paths": 0,
        "missing_anchor_embeddings": 0,
        "missing_candidate_embeddings": 0,
        "deduped_caption_rows": 0,
        "similarity_backend": "none",
    }
    if source_pairs_file:
        if not output_pairs_file:
            stem, _ = os.path.splitext(output_image_list_file)
            output_pairs_file = f"{stem}_pairs.json"
        wanted = set(basenames)
        for row in _iter_json_records(source_pairs_file):
            source_pair_rows_scanned += 1
            name = _pair_name(row)
            if name in wanted:
                pairs_by_name.setdefault(name, []).append(row)
        missing_pairs = len(wanted) - len(pairs_by_name)
        if missing_pairs:
            print(
                f"Warning: {missing_pairs} mined basenames were not found "
                f"in {source_pairs_file}; they will be omitted from the "
                "mined image list to keep train_pairs_file aligned."
            )
        basenames = [name for name in basenames if name in pairs_by_name]
        anchor_basenames = list(basenames)
        pre_budget_unique_basenames = len(basenames)

        expansion_by_anchor = {}
        if expansion_enabled and basenames:
            anchor_image_paths = {}
            for name in basenames:
                pair_rows = pairs_by_name.get(name, [])
                image_path = _pair_image_path(pair_rows[0]) if pair_rows else ""
                if image_path:
                    anchor_image_paths[name] = image_path
            wanted_image_paths = set(anchor_image_paths.values())
            rows_by_image_path = {}
            seen_caption_keys = set()
            deduped_caption_rows = 0
            expansion_pair_rows_scanned = 0
            for row in _iter_json_records(source_pairs_file):
                expansion_pair_rows_scanned += 1
                image_path = _pair_image_path(row)
                if image_path not in wanted_image_paths:
                    continue
                if dedupe_expansion_captions:
                    caption_key = (image_path, _normalize_caption(_pair_caption(row)))
                    if caption_key[1] and caption_key in seen_caption_keys:
                        deduped_caption_rows += 1
                        continue
                    if caption_key[1]:
                        seen_caption_keys.add(caption_key)
                rows_by_image_path.setdefault(image_path, []).append(row)

            candidate_names = set()
            anchor_name_set = set(basenames)
            for rows in rows_by_image_path.values():
                for row in rows:
                    name = _pair_name(row)
                    if name and name not in anchor_name_set:
                        candidate_names.add(name)

            embeddings_by_name = {}
            embedding_rows_scanned = 0
            if expansion_mode == "nearest":
                embeddings_by_name, embedding_rows_scanned = _load_embeddings_for_names(
                    source_embedding_shards_dir,
                    candidate_names | anchor_name_set,
                )
                caption_expansion_stats["similarity_backend"] = (
                    "embedding" if embeddings_by_name else "lexical"
                )
            else:
                caption_expansion_stats["similarity_backend"] = "source_order"

            missing_anchor_embeddings = 0
            missing_candidate_embeddings = 0
            for anchor_name in basenames:
                anchor_rows = pairs_by_name.get(anchor_name, [])
                if not anchor_rows:
                    continue
                anchor_row = anchor_rows[0]
                anchor_image_path = anchor_image_paths.get(anchor_name, "")
                candidate_rows = rows_by_image_path.get(anchor_image_path, [])
                ranked = []
                anchor_embedding = embeddings_by_name.get(anchor_name)
                if expansion_mode == "nearest" and anchor_embedding is None:
                    missing_anchor_embeddings += 1
                for candidate_index, row in enumerate(candidate_rows):
                    candidate_name = _pair_name(row)
                    if (
                        not candidate_name
                        or candidate_name == anchor_name
                        or candidate_name in anchor_name_set
                    ):
                        continue
                    if expansion_mode == "nearest":
                        candidate_embedding = embeddings_by_name.get(candidate_name)
                        if anchor_embedding is not None and candidate_embedding is not None:
                            score = float(np.dot(anchor_embedding, candidate_embedding))
                        else:
                            if candidate_embedding is None:
                                missing_candidate_embeddings += 1
                            score = float(_lexical_similarity(
                                _pair_caption(anchor_row),
                                _pair_caption(row),
                            ))
                        ranked.append((score, candidate_index, row))
                    else:
                        ranked.append((0.0, candidate_index, row))
                if expansion_mode == "nearest":
                    ranked.sort(key=lambda item: (-item[0], item[1]))
                    if max_pairs_per_image_path > 0:
                        ranked = ranked[:max(0, max_pairs_per_image_path - 1)]
                else:
                    ranked.sort(key=lambda item: item[1])
                expansion_by_anchor[anchor_name] = [
                    (row, score) for score, _, row in ranked
                ]

            caption_expansion_stats.update({
                "source_pair_rows_scanned_for_expansion": int(
                    expansion_pair_rows_scanned
                ),
                "source_embedding_rows_scanned_for_expansion": int(
                    embedding_rows_scanned
                ),
                "candidate_image_paths": int(len(wanted_image_paths)),
                "candidate_pair_rows": int(
                    sum(len(rows) for rows in rows_by_image_path.values())
                ),
                "candidate_pair_names": int(len(candidate_names)),
                "candidate_pairs_after_dedup": int(
                    sum(len(rows) for rows in rows_by_image_path.values())
                ),
                "missing_anchor_embeddings": int(missing_anchor_embeddings),
                "missing_candidate_embeddings": int(missing_candidate_embeddings),
                "deduped_caption_rows": int(deduped_caption_rows),
            })

        expanded_pair_limit = None
        if (
            expansion_enabled
            and expansion_mode == "nearest"
            and count_expanded_pairs_toward_target
            and target_query_count > 0
        ):
            expanded_pair_limit = int(
                math.floor(target_query_count * max_expanded_pair_fraction)
            )

        planned_items = []
        planned_names = set()
        planned_expanded_pairs = 0

        def _append_planned(row, is_expansion, anchor_name, score=None):
            nonlocal planned_expanded_pairs
            name = _pair_name(row) or anchor_name
            if not name or name in planned_names:
                return False
            if is_expansion:
                if (
                    expanded_pair_limit is not None
                    and planned_expanded_pairs >= expanded_pair_limit
                ):
                    return False
                planned_expanded_pairs += 1
            planned_names.add(name)
            planned_items.append((row, bool(is_expansion), anchor_name, score))
            return True

        for name in basenames:
            for row in pairs_by_name.get(name, []):
                _append_planned(row, False, name, None)
            if expansion_enabled:
                for row, score in expansion_by_anchor.get(name, []):
                    _append_planned(row, True, name, score)

        pre_budget_recovered_pairs = len(planned_items)
        planned_anchor_pairs = sum(
            1 for _, is_expansion, _, _ in planned_items if not is_expansion
        )
        planned_expanded_pairs = sum(
            1 for _, is_expansion, _, _ in planned_items if is_expansion
        )
        caption_expansion_stats["planned_anchor_pairs"] = int(planned_anchor_pairs)
        caption_expansion_stats["planned_expanded_pairs"] = int(planned_expanded_pairs)
        if target_query_count > 0:
            budget_basis_count = (
                len(planned_items)
                if count_expanded_pairs_toward_target
                else planned_anchor_pairs
            )
            target_pair_shortfall = max(
                0, target_query_count - budget_basis_count
            )
            if target_pair_shortfall:
                budget_basis_label = (
                    "text-image pairs"
                    if count_expanded_pairs_toward_target
                    else "anchor text-image pairs"
                )
                print(
                    "Warning: mined candidates recover only "
                    f"{budget_basis_count} {budget_basis_label} before "
                    f"budgeting, short of target {target_query_count} by "
                    f"{target_pair_shortfall}. Increase mining.topn or the "
                    "weak-query selection breadth if this should be closer "
                    "to the requested budget."
                )
            if count_expanded_pairs_toward_target:
                selected_pair_items = planned_items[:target_query_count]
                budget_reached = len(planned_items) >= target_query_count
            else:
                selected_anchor_names = []
                seen_anchor_names = set()
                for _, is_expansion, anchor_name, _ in planned_items:
                    if is_expansion or anchor_name in seen_anchor_names:
                        continue
                    seen_anchor_names.add(anchor_name)
                    selected_anchor_names.append(anchor_name)
                    if len(selected_anchor_names) >= target_query_count:
                        break
                selected_anchor_name_set = set(selected_anchor_names)
                selected_pair_items = [
                    item for item in planned_items
                    if item[2] in selected_anchor_name_set
                ]
                budget_reached = planned_anchor_pairs >= target_query_count
        else:
            selected_pair_items = planned_items
        selected_anchor_names = {
            anchor_name
            for _, is_expansion, anchor_name, _ in selected_pair_items
            if not is_expansion
        }
        budget_omitted_basenames = max(
            0,
            len(anchor_basenames) - len(selected_anchor_names),
        )
        basenames = []
        seen_output_names = set()
        for row, _, anchor_name, _ in selected_pair_items:
            name = _pair_name(row) or anchor_name
            if name and name not in seen_output_names:
                seen_output_names.add(name)
                basenames.append(name)

        pairs_dir = os.path.dirname(output_pairs_file)
        if pairs_dir:
            os.makedirs(pairs_dir, exist_ok=True)
        with open(output_pairs_file, "w", encoding="utf-8") as f:
            f.write("[\n")
            first = True
            for row, _, _, _ in selected_pair_items:
                if not first:
                    f.write(",\n")
                f.write(json.dumps(row, ensure_ascii=False))
                first = False
                pairs_count += 1
            f.write("\n]\n")
        train_pairs_file = output_pairs_file
    elif target_query_count > 0:
        basenames = basenames[:target_query_count]
        budget_reached = len(basenames) >= target_query_count
        selected_pair_items = [
            ({"unique_name": name}, False, name, None)
            for name in basenames
        ]
    else:
        selected_pair_items = [
            ({"unique_name": name}, False, name, None)
            for name in basenames
        ]

    out_dir = os.path.dirname(output_image_list_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_image_list_file, "w", encoding="utf-8") as f:
        for name in basenames:
            f.write(f"{name}\n")

    detail_records = []
    dataset_counts = Counter()
    qtype_counts = Counter()
    dataset_qtype_counts = Counter()
    image_dataset_counts = Counter()
    final_image_paths = set()
    counted_image_paths_for_dataset = set()
    final_anchor_pairs = 0
    final_expanded_pairs = 0
    for pair_idx, (pair, is_expansion, anchor_name, expansion_score) in enumerate(
        selected_pair_items
    ):
        name = _pair_name(pair) or anchor_name
        dataset = str(pair.get("dataset") or "")
        qtype = str(pair.get("query_type") or "")
        image_path = _pair_image_path(pair)
        if image_path:
            final_image_paths.add(image_path)
        if dataset:
            dataset_counts[dataset] += 1
            image_dataset_key = image_path or name
            if image_dataset_key not in counted_image_paths_for_dataset:
                counted_image_paths_for_dataset.add(image_dataset_key)
                image_dataset_counts[dataset] += 1
        if qtype:
            qtype_counts[qtype] += 1
        if dataset or qtype:
            dataset_qtype_counts[f"{dataset}/{qtype}"] += 1
        if is_expansion:
            final_expanded_pairs += 1
        else:
            final_anchor_pairs += 1
        if write_details:
            detail_records.append({
                "unique_name": name,
                "pair_index": pair_idx,
                "filepath": os.path.abspath(os.path.join(image_dir, name)),
                "dataset": dataset,
                "query_type": qtype,
                "image_path": image_path,
                "caption": _pair_caption(pair),
                "is_caption_expansion": bool(is_expansion),
                "anchor_unique_name": anchor_name if is_expansion else "",
                "caption_expansion_score": (
                    "" if expansion_score is None else float(expansion_score)
                ),
            })
    caption_expansion_stats["final_anchor_pairs"] = int(final_anchor_pairs)
    caption_expansion_stats["final_expanded_pairs"] = int(final_expanded_pairs)
    caption_expansion_stats["final_unique_image_paths"] = int(len(final_image_paths))

    stats_json_path = os.path.join(out_dir or ".", "mined_stats.json")
    stats_txt_path = os.path.join(out_dir or ".", "mined_stats.txt")
    details_csv_path = ""
    if write_details:
        details_csv_path = os.path.join(
            out_dir or ".", "mined_samples_detailed.csv"
        )
        pd.DataFrame(detail_records).to_csv(details_csv_path, index=False)

    stats = {
        "mined_parquet": mined_parquet,
        "source_pairs_file": source_pairs_file,
        "raw_mined_rows": int(raw_rows),
        "raw_unique_filepaths": int(raw_unique_filepaths),
        "raw_unique_names": int(raw_unique_names),
        "source_pair_rows_scanned": int(source_pair_rows_scanned),
        "target_query_count": int(target_query_count),
        "pre_budget_unique_basenames": int(pre_budget_unique_basenames),
        "pre_budget_recovered_pairs": int(pre_budget_recovered_pairs),
        "pre_budget_anchor_pairs": int(
            caption_expansion_stats["planned_anchor_pairs"]
        ),
        "pre_budget_expanded_pairs": int(
            caption_expansion_stats["planned_expanded_pairs"]
        ),
        "target_budget_applies_to": caption_expansion_stats[
            "target_budget_applies_to"
        ],
        "avg_pairs_per_pre_budget_basename": (
            float(pre_budget_recovered_pairs) / pre_budget_unique_basenames
            if pre_budget_unique_basenames else 0.0
        ),
        "target_pair_shortfall": int(target_pair_shortfall),
        "final_unique_basenames": int(len(basenames)),
        "final_unique_image_paths": int(len(final_image_paths)),
        "recovered_pairs": int(pairs_count),
        "missing_pairs": int(missing_pairs),
        "budget_omitted_basenames": int(budget_omitted_basenames),
        "budget_reached": bool(budget_reached),
        "caption_expansion": caption_expansion_stats,
        "image_list_file": output_image_list_file,
        "train_pairs_file": train_pairs_file,
        "details_csv": details_csv_path,
        "dataset_counts": _counter_records(dataset_counts),
        "query_type_counts": _counter_records(qtype_counts),
        "dataset_query_type_counts": _counter_records(dataset_qtype_counts),
        "image_dataset_counts": _counter_records(image_dataset_counts),
    }
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    summary_lines = [
        "PAS mined sample stats",
        f"Mined parquet: {mined_parquet}",
        f"Raw mined rows: {raw_rows}",
        f"Raw unique filepaths: {raw_unique_filepaths}",
        f"Raw unique names: {raw_unique_names}",
        f"Source pair rows scanned: {source_pair_rows_scanned}",
        f"Requested mined query target: {target_query_count or 'unlimited'}",
        f"Pre-budget unique basenames: {pre_budget_unique_basenames}",
        f"Pre-budget recovered text-image pairs: {pre_budget_recovered_pairs}",
        "Pre-budget anchor text-image pairs: "
        f"{caption_expansion_stats['planned_anchor_pairs']}",
        "Pre-budget expanded text-image pairs: "
        f"{caption_expansion_stats['planned_expanded_pairs']}",
        "Target budget applies to: "
        f"{caption_expansion_stats['target_budget_applies_to']}",
        "Average source pairs per pre-budget basename: "
        f"{stats['avg_pairs_per_pre_budget_basename']:.3f}",
        f"Target pair shortfall before budget: {target_pair_shortfall}",
        f"Final unique basenames: {len(basenames)}",
        f"Final unique image paths: {len(final_image_paths)}",
        f"Recovered text-image pairs: {pairs_count}",
        f"Missing source-pair images: {missing_pairs}",
        f"Budget omitted basenames: {budget_omitted_basenames}",
        f"Budget reached: {budget_reached}",
        "Caption expansion enabled: "
        f"{caption_expansion_stats['enabled']}",
        f"Caption expansion mode: {caption_expansion_stats['mode']}",
        "Caption expansion final pairs: "
        f"{caption_expansion_stats['final_expanded_pairs']}",
        f"Image list: {output_image_list_file}",
        f"Pairs JSON: {train_pairs_file or '(not written)'}",
        f"Detailed CSV: {details_csv_path or '(disabled)'}",
        f"Stats JSON: {stats_json_path}",
        "",
        "By dataset (pair rows):",
    ]
    for row in stats["dataset_counts"]:
        summary_lines.append(
            f"  {row['value']}: {row['count']} ({row['pct']:.1f}%)"
        )
    summary_lines.append("")
    summary_lines.append("By query type (pair rows):")
    for row in stats["query_type_counts"]:
        summary_lines.append(
            f"  {row['value']}: {row['count']} ({row['pct']:.1f}%)"
        )
    summary_text = "\n".join(summary_lines) + "\n"
    with open(stats_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image_dir": image_dir,
                "caption_dir": caption_dir,
                "image_list_file": output_image_list_file,
                "caption_file_suffix": caption_file_suffix,
                "train_pairs_file": train_pairs_file,
                "target_query_count": target_query_count,
                "caption_expansion": caption_expansion_stats,
                "stats_json": stats_json_path,
                "stats_txt": stats_txt_path,
                "details_csv": details_csv_path,
            },
            f,
            indent=2,
        )

    print(summary_text.strip())
    print(
        f"Converted mined parquet to CLIP image list: "
        f"{raw_rows} rows -> {len(basenames)} unique basenames -> "
        f"{output_image_list_file} (+ manifest {manifest_path})"
    )
    if train_pairs_file:
        print(
            f"Recovered mined TAO-FT pairs: {pairs_count} rows -> "
            f"{train_pairs_file}"
        )
    return output_image_list_file


def summarize_knn_mining(
    mined_parquet: str,
    target_parquet: str,
    output_dir: str,
    topn: int = 0,
) -> str:
    """Write mining_summary.txt and mining_summary.json after k-NN mining.

    The ``tmm nearest_neighbors`` step already de-duplicates by filepath
    before writing, so raw-vs-deduped counts reflect that:
    raw_neighbor_rows is the pre-dedup estimate (target_queries * topn) and
    unique_mined_filepaths is the actual parquet row count.

    Args:
        mined_parquet:  Path to the k-NN output parquet (filepath column).
        target_parquet: Path to the target embeddings parquet used as k-NN queries.
        output_dir:     Directory where mining_summary.txt and .json are written.
        topn:           k-NN topn value; used to estimate raw_neighbor_rows.

    Returns:
        Path to the written mining_summary.txt.
    """
    import json
    import os

    import pandas as pd

    def _counts(df, column, limit=12):
        if df.empty or column not in df.columns:
            return ["  (none)"]
        counts = df[column].fillna("").astype(str).value_counts().head(limit)
        if counts.empty:
            return ["  (none)"]
        return [f"  {str(v) or '(blank)'}: {int(c)}" for v, c in counts.items()]

    os.makedirs(output_dir, exist_ok=True)

    target_df = pd.read_parquet(target_parquet)
    n_targets = len(target_df)

    mined_df = pd.read_parquet(mined_parquet)
    unique_mined_filepaths = len(mined_df)
    # Distinguish "the mine step did not emit this column" from a real
    # zero — the parquet carries only `filepath` unless the mine step is
    # configured to pass the target linkage through.
    _NA = "(not available: nearest_neighbors output has no such column)"
    unique_mined_names = (
        int(
            mined_df["unique_name"]
            .fillna("")
            .astype(str)
            .loc[lambda s: s.astype(bool)]
            .drop_duplicates()
            .shape[0]
        )
        if "unique_name" in mined_df.columns
        else None
    )

    # The nearest_neighbors step de-dupes before writing,
    # so estimate the raw pre-dedup row count from topn when available.
    raw_neighbor_rows = (n_targets * int(topn)) if topn else None

    summary_payload = {
        "target_parquet": target_parquet,
        "mined_parquet": mined_parquet,
        "target_queries": n_targets,
        "similar_items_per_query": int(topn) if topn else None,
        "raw_neighbor_rows_estimate": raw_neighbor_rows,
        "unique_mined_filepaths": unique_mined_filepaths,
        "unique_mined_names": unique_mined_names,
    }

    summary_lines = [
        "k-NN mining summary",
        f"Target parquet: {target_parquet}",
        f"Mined parquet: {mined_parquet}",
        f"Target queries: {n_targets}",
    ]
    if topn:
        summary_lines.append(f"Similar items per query: {topn}")
        summary_lines.append(
            f"Raw neighbor rows (estimate, pre-dedup): {raw_neighbor_rows}"
        )
    summary_lines += [
        f"Unique mined filepaths (post-dedup): {unique_mined_filepaths}",
        "Unique mined names (post-dedup): "
        + (_NA if unique_mined_names is None else str(unique_mined_names)),
        "",
        "Target query rows by query type:",
    ]
    summary_lines.extend(_counts(target_df, "query_type"))
    summary_lines.extend(["", "Target query rows by attribute:"])
    summary_lines.extend(_counts(target_df, "weak_attribute"))
    summary_lines.extend(["", "Mined rows by target query type:"])
    summary_lines.extend(
        _counts(mined_df, "target_query_type")
        if "target_query_type" in mined_df.columns else [f"  {_NA}"]
    )
    summary_lines.extend(["", "Mined rows by target attribute:"])
    summary_lines.extend(
        _counts(mined_df, "target_attribute")
        if "target_attribute" in mined_df.columns else [f"  {_NA}"]
    )
    summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    txt_path = os.path.join(output_dir, "mining_summary.txt")
    json_path = os.path.join(output_dir, "mining_summary.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(summary_text.strip())
    return txt_path


def track_cumulative_mined_unique_names(
    mined_pairs_file: str,
    base_experiment_path: str,
    iter_num: int,
    output_file: str,
) -> str:
    """Track the cumulative set of mined unique names across DEFT iterations.

    Reads the current iteration's mined pairs file, extracts unique names, and
    merges them with the previous iteration's cumulative set (when iter_num > 1)
    to produce a deduplicated cumulative list.

    Args:
        mined_pairs_file:       Path to the current iteration's mined_pairs.json.
        base_experiment_path:   Root directory for all experiment outputs.
        iter_num:               Current DEFT iteration number (1-based).
        output_file:            Path where the cumulative unique names JSON is written.

    Returns:
        Path to the written cumulative unique names JSON file.
    """
    import os

    import pandas as pd

    curr_pairs = pd.read_json(mined_pairs_file)
    curr_unique_names = (
        curr_pairs[["unique_name"]]
        if "unique_name" in curr_pairs.columns
        else pd.DataFrame(columns=["unique_name"])
    )

    if iter_num > 1:
        prev_cumulative_file = os.path.join(
            base_experiment_path,
            f"iter_{iter_num - 1}",
            "mining",
            "cumulative_mined_unique_names.json",
        )
        if os.path.isfile(prev_cumulative_file):
            prev_cumulative = pd.read_json(prev_cumulative_file)
            cumulative = (
                pd.concat([prev_cumulative, curr_unique_names])
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            cumulative = curr_unique_names
    else:
        cumulative = curr_unique_names

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cumulative.to_json(output_file, orient="records")
    print(
        f"Cumulative mined unique names: iter={iter_num}, "
        f"this_iter={len(curr_unique_names)}, "
        f"cumulative={len(cumulative)}, output={output_file}"
    )
    return output_file


def write_iteration_summary(
    experiment_dir: str,
    iter_num: int,
    gaps_parquet: str,
    mined_parquet: str,
    mined_pairs_file: str,
    training_checkpoint: str,
    next_checkpoint_path: str,
    experiment_id: str = "",
) -> str:
    """Write iteration_summary.json at the end of a DEFT iteration.

    Captures the key artifact paths and checkpoint state for each iteration.

    Args:
        experiment_dir:       Directory for this iteration's outputs.
        iter_num:             Current DEFT iteration number (1-based).
        gaps_parquet:         Path to the gap-analysis output parquet.
        mined_parquet:        Path to the raw k-NN mined parquet.
        mined_pairs_file:     Path to the final mined pairs JSON.
        training_checkpoint:  Checkpoint used as input for this iteration's training.
        next_checkpoint_path: Expected path of the best checkpoint produced by training.
        experiment_id:        Optional unique ID for this experiment run.

    Returns:
        Path to the written iteration_summary.json.
    """
    import json
    import os

    os.makedirs(experiment_dir, exist_ok=True)
    payload = {
        "iteration": int(iter_num),
        "experiment_id": experiment_id,
        "experiment_dir": experiment_dir,
        "training_checkpoint": training_checkpoint,
        "next_checkpoint_path": next_checkpoint_path,
        "gaps_parquet": gaps_parquet,
        "mined_parquet": mined_parquet,
        "mined_pairs_file": mined_pairs_file,
        "eval_results_dir": experiment_dir,
    }
    out_path = os.path.join(experiment_dir, "iteration_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Iteration {iter_num} summary written to {out_path}")
    return out_path
