# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bundled and adapted from the Apache-2.0 NVIDIA TAO Tutorials PAS DEFT utilities so the
# customer workflow does not depend on an external source checkout.

"""Post-KNN history-aware selection for PAS CLIP DEFT mining."""


def load_selection_history(history_file: str, iter_num: int) -> dict:
    """Load and validate the mining selection ledger.

    Returns the parsed state with every entry's ``selected_unique_names``
    normalized. A ledger that does not exist yet — legal only at iteration 1 —
    comes back with an empty ``iterations`` list and none of the
    experiment-level invariants (``mode``, ``continual_dataset``,
    ``replay_fraction``) set; :func:`select_history_aware_mined_pairs` stamps
    those when it commits iteration 1.
    """
    import hashlib
    import json
    import os

    iteration = int(iter_num)
    if iteration < 1:
        raise ValueError(f"iter_num must be >= 1, got {iteration}")

    history_path = os.path.abspath(history_file)
    if not os.path.isfile(history_path):
        if iteration != 1:
            raise FileNotFoundError(
                f"Selection history is required before iteration {iteration}: "
                f"{history_path}"
            )
        return {
            "version": 1,
            "identity": "unique_name",
            "source_pool_image_list_file": "",
            "source_pool_size": 0,
            "iterations": [],
        }

    with open(history_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict) or state.get("version") != 1:
        raise RuntimeError(f"Unsupported or malformed selection history: {history_path}")
    if state.get("identity") != "unique_name":
        raise RuntimeError("Selection history identity must be fixed to unique_name")

    entries = state.get("iterations")
    if not isinstance(entries, list):
        raise RuntimeError("Selection history iterations must be a list")
    iteration_numbers = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("Every selection history iteration must be an object")
        entry_iteration = int(entry.get("iteration", 0))
        iteration_numbers.append(entry_iteration)
        selected_names = entry.get("selected_unique_names")
        if not isinstance(selected_names, list):
            raise RuntimeError(
                f"History iteration {entry_iteration} has no selected_unique_names list"
            )
        normalized_names = [
            str(name or "").strip().replace("\\", "/") for name in selected_names
        ]
        if any(not name for name in normalized_names):
            raise RuntimeError(
                f"History iteration {entry_iteration} contains an empty unique_name"
            )
        if len(normalized_names) != len(set(normalized_names)):
            raise RuntimeError(
                f"History iteration {entry_iteration} contains duplicate unique_name values"
            )
        names_hash = hashlib.sha256(
            "".join(f"{name}\n" for name in normalized_names).encode("utf-8")
        ).hexdigest()
        if names_hash != str(entry.get("image_list_sha256") or ""):
            raise RuntimeError(
                f"History iteration {entry_iteration} names do not match its image list"
            )
        entry["selected_unique_names"] = normalized_names
    if iteration_numbers != list(range(1, len(entries) + 1)):
        raise RuntimeError(
            f"Selection history must contain contiguous iterations starting at 1; "
            f"found {iteration_numbers}"
        )
    return state


def select_history_aware_mined_pairs(
    candidate_pairs_file: str,
    candidate_manifest_file: str,
    output_image_list_file: str,
    output_pairs_file: str,
    manifest_path: str,
    history_file: str,
    source_pool_image_list_file: str,
    iter_num: int,
    target_query_count: int,
    continual_dataset: str,
    replay_fraction: float = 0.20,
    resume: str = "false",
) -> str:
    """Select final training pairs after KNN and source-pair recovery.

    ``continual_dataset=true`` selects only pairs never selected by an earlier
    iteration. The generated train config accumulates those disjoint per-iter
    datasets, so no replay rows are read or inserted here.

    ``continual_dataset=false`` keeps only the current iteration's dataset and
    therefore selects a controlled mixture of novel and replay pairs. Replay
    first follows the current KNN relevance order, then uses prior selected-pair
    files only when more replay rows are needed to reach the requested budget.

    The candidate pool is expected to be uncapped — the caller runs the
    conversion step with ``target_query_count=0`` into its own directory, so
    that this function owns the budget and spends it after the novel/replay
    partition. Given that, a non-zero ``selection_shortfall`` means KNN
    genuinely ran out of novel neighbours, rather than that the budget was
    spent upstream on rows discarded here.
    """
    import hashlib
    import json
    import math
    import os
    import tempfile

    def _truthy(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _pair_key(row):
        if not isinstance(row, dict):
            raise ValueError("Every mined pair must be a JSON object")
        key = str(row.get("unique_name") or "").strip().replace("\\", "/")
        if not key:
            raise ValueError(
                "Every history-aware candidate requires a non-empty unique_name"
            )
        return key

    def _iter_json_records(path):
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
            second = f.readline()
        line_delimited_array = (
            first.strip() == "["
            and second.lstrip().startswith("{")
            and second.rstrip().rstrip(",").endswith("}")
        )
        if line_delimited_array:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    value = line.strip()
                    if not value or value in {"[", "]"}:
                        continue
                    if value.endswith(","):
                        value = value[:-1]
                    yield json.loads(value)
            return
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array in {path}")
        for row in rows:
            yield row

    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _atomic_write_text(path, text):
        absolute = os.path.abspath(path)
        directory = os.path.dirname(absolute) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(absolute)}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, absolute)
            try:
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _json_text(value):
        return json.dumps(value, indent=2, ensure_ascii=False) + "\n"

    def _validate_committed_artifacts(entry):
        output_dir = os.path.dirname(os.path.abspath(output_image_list_file)) or "."
        expected_paths = {
            "pairs_file": os.path.abspath(output_pairs_file),
            "image_list_file": os.path.abspath(output_image_list_file),
            "manifest_file": os.path.abspath(manifest_path),
            "stats_file": os.path.join(output_dir, "mined_stats.json"),
        }
        if int(entry.get("target_query_count", 0)) != target:
            raise RuntimeError(
                f"Committed iteration target does not match current config: "
                f"{entry.get('target_query_count')} != {target}"
            )
        artifact_fields = (
            ("pairs_file", "pairs_sha256"),
            ("image_list_file", "image_list_sha256"),
            ("manifest_file", "manifest_sha256"),
            ("stats_file", "stats_sha256"),
        )
        for path_field, hash_field in artifact_fields:
            path = str(entry.get(path_field) or "")
            expected_hash = str(entry.get(hash_field) or "")
            if not path or not expected_hash:
                raise RuntimeError(
                    f"History entry {entry.get('iteration')} is missing {path_field} "
                    f"or {hash_field}"
                )
            if os.path.abspath(path) != expected_paths[path_field]:
                raise RuntimeError(
                    f"Committed {path_field} does not match current output path: "
                    f"{path} != {expected_paths[path_field]}"
                )
            if not os.path.isfile(path):
                raise RuntimeError(f"Committed history artifact is missing: {path}")
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Committed history artifact hash mismatch: {path} "
                    f"(expected {expected_hash}, got {actual_hash})"
                )

    iteration = int(iter_num)
    target = int(target_query_count)
    if iteration < 1:
        raise ValueError(f"iter_num must be >= 1, got {iteration}")
    if target < 1:
        raise ValueError(
            f"History-aware selection requires target_query_count >= 1, got {target}"
        )

    continual = _truthy(continual_dataset)
    configured_replay_fraction = float(replay_fraction)
    if not math.isfinite(configured_replay_fraction) or not (
        0.0 <= configured_replay_fraction <= 1.0
    ):
        raise ValueError(
            f"replay_fraction must be within [0, 1], got {replay_fraction}"
        )
    effective_replay_fraction = 0.0 if continual else configured_replay_fraction
    mode = "continual_novel_only" if continual else "controlled_replay"

    history_path = os.path.abspath(history_file)
    state = load_selection_history(history_file, iteration)
    if not state["iterations"]:
        # Fresh ledger — stamp the experiment-level invariants that every later
        # iteration is then checked against. Keyed on there being no committed
        # iterations rather than on "mode" being absent, so a persisted ledger
        # that lost its mode key fails the check below instead of being
        # silently re-stamped with whatever this run happens to request.
        state["mode"] = mode
        state["continual_dataset"] = continual
        state["replay_fraction"] = effective_replay_fraction

    if state.get("mode") != mode or state.get("continual_dataset") is not continual:
        raise RuntimeError(
            "Cannot change continual_dataset/history-aware mode inside one experiment"
        )
    recorded_replay_fraction = float(state.get("replay_fraction", 0.0))
    if abs(recorded_replay_fraction - effective_replay_fraction) > 1e-12:
        raise RuntimeError(
            "Cannot change replay_fraction inside one history-aware experiment"
        )

    pool_path = os.path.abspath(source_pool_image_list_file)
    if not os.path.isfile(pool_path):
        raise FileNotFoundError(f"Source pool image list is missing: {pool_path}")
    pool_stat = os.stat(pool_path)
    recorded_pool_path = str(state.get("source_pool_image_list_file") or "")
    if recorded_pool_path and recorded_pool_path != pool_path:
        raise RuntimeError(
            "Cannot change source_pool_image_list_file inside one experiment"
        )
    recorded_pool_bytes = int(state.get("source_pool_file_size_bytes", 0) or 0)
    recorded_pool_mtime = int(state.get("source_pool_mtime_ns", 0) or 0)
    if recorded_pool_bytes and recorded_pool_bytes != pool_stat.st_size:
        raise RuntimeError("Source pool image list size changed during the experiment")
    if recorded_pool_mtime and recorded_pool_mtime != pool_stat.st_mtime_ns:
        raise RuntimeError("Source pool image list mtime changed during the experiment")
    state["source_pool_image_list_file"] = pool_path
    state["source_pool_file_size_bytes"] = pool_stat.st_size
    state["source_pool_mtime_ns"] = pool_stat.st_mtime_ns

    # Structure, per-entry normalization, and contiguity are validated by
    # load_selection_history above.
    entries = state["iterations"]

    committed_entry = next(
        (entry for entry in entries if int(entry["iteration"]) == iteration),
        None,
    )
    if committed_entry is not None:
        _validate_committed_artifacts(committed_entry)
        if _truthy(resume):
            print(
                f"Skipping committed history-aware selection for iteration {iteration}: "
                f"{output_pairs_file}"
            )
            return output_image_list_file
        raise RuntimeError(
            f"Iteration {iteration} is already committed in {history_path}; "
            "use --resume or a new experiment results path"
        )
    if len(entries) != iteration - 1:
        raise RuntimeError(
            f"Cannot select iteration {iteration}; history currently ends at "
            f"iteration {len(entries)}"
        )

    historical_names = set()
    for entry in entries:
        historical_names.update(entry["selected_unique_names"])
    previous_iteration_names = (
        set(entries[-1]["selected_unique_names"]) if entries else set()
    )

    candidate_raw_count = 0
    candidate_duplicate_count = 0
    candidate_names = set()
    novel_candidates = []
    replay_candidates = []
    for row in _iter_json_records(candidate_pairs_file):
        candidate_raw_count += 1
        key = _pair_key(row)
        if key in candidate_names:
            candidate_duplicate_count += 1
            continue
        candidate_names.add(key)
        item = (candidate_raw_count - 1, key, row)
        bucket = replay_candidates if key in historical_names else novel_candidates
        if len(bucket) < target:
            bucket.append(item)

    with open(candidate_manifest_file, "r", encoding="utf-8") as f:
        candidate_manifest = json.load(f)
    if not isinstance(candidate_manifest, dict):
        raise ValueError(f"Malformed candidate manifest: {candidate_manifest_file}")

    has_history = bool(historical_names)
    requested_replay = (
        int(math.floor(target * effective_replay_fraction)) if has_history else 0
    )
    requested_novel = target - requested_replay

    selected = {}

    def _take(items, count):
        added = 0
        for source_index, key, row in items:
            if added >= count:
                break
            if key in selected:
                continue
            selected[key] = {
                "source_index": source_index,
                "key": key,
                "row": row,
                "from_current_candidates": True,
            }
            added += 1
        return added

    _take(novel_candidates, requested_novel)
    if not continual:
        _take(replay_candidates, requested_replay)
        history_files_loaded = 0

        def _history_rows():
            nonlocal history_files_loaded
            for entry in entries:
                prior_pairs_file = str(entry.get("pairs_file") or "")
                expected_hash = str(entry.get("pairs_sha256") or "")
                if not prior_pairs_file or not expected_hash:
                    raise RuntimeError(
                        f"History iteration {entry['iteration']} cannot provide replay rows"
                    )
                if not os.path.isfile(prior_pairs_file):
                    raise FileNotFoundError(
                        f"Replay pairs file is missing: {prior_pairs_file}"
                    )
                actual_hash = _sha256(prior_pairs_file)
                if actual_hash != expected_hash:
                    raise RuntimeError(
                        f"Replay pairs hash mismatch: {prior_pairs_file}"
                    )
                history_files_loaded += 1
                entry_names = set(entry["selected_unique_names"])
                for row in _iter_json_records(prior_pairs_file):
                    key = _pair_key(row)
                    if key not in entry_names:
                        raise RuntimeError(
                            f"Replay row {key} is not recorded in history iteration "
                            f"{entry['iteration']}"
                        )
                    yield key, row

        history_rows = _history_rows()

        def _take_history(count):
            if count <= 0:
                return 0
            added = 0
            for key, row in history_rows:
                if key in selected:
                    continue
                selected[key] = {
                    "source_index": None,
                    "key": key,
                    "row": row,
                    "from_current_candidates": False,
                }
                added += 1
                if added >= count:
                    break
            return added

        # Current-KNN historical rows are preferred, but the requested replay
        # quota is filled from prior outputs before novel fallback is allowed.
        selected_replay = sum(1 for key in selected if key in historical_names)
        _take_history(max(0, requested_replay - selected_replay))

        # If the historical pool is too small, fill the replay deficit with
        # extra novel rows. If novel supply is short, allow replay overflow.
        remaining = target - len(selected)
        if remaining > 0:
            _take(novel_candidates, remaining)
        remaining = target - len(selected)
        if remaining > 0:
            _take(replay_candidates, remaining)
        remaining = target - len(selected)
        if remaining > 0:
            _take_history(remaining)
    else:
        history_files_loaded = 0

    current_selected = sorted(
        (item for item in selected.values() if item["from_current_candidates"]),
        key=lambda item: item["source_index"],
    )
    historical_selected = [
        item for item in selected.values() if not item["from_current_candidates"]
    ]
    final_items = current_selected + historical_selected
    final_names = [item["key"] for item in final_items]
    final_rows = [item["row"] for item in final_items]
    selected_name_set = set(final_names)

    actual_replay = len(selected_name_set & historical_names)
    actual_novel = len(final_names) - actual_replay
    overlap_previous = len(selected_name_set & previous_iteration_names)
    overlap_all_prior = actual_replay
    cumulative_unique = len(historical_names | selected_name_set)
    replay_overflow = max(0, actual_replay - requested_replay)
    shortfall = max(0, target - len(final_names))
    if not final_names:
        raise RuntimeError(
            f"Iteration {iteration} has no selectable training pairs after "
            "history filtering; increase mining.topn or stop the experiment"
        )

    source_pool_size = int(state.get("source_pool_size", 0) or 0)
    if source_pool_size <= 0:
        with open(pool_path, "r", encoding="utf-8") as f:
            source_pool_size = sum(1 for line in f if line.strip())
    state["source_pool_image_list_file"] = pool_path
    state["source_pool_size"] = source_pool_size

    output_dir = os.path.dirname(os.path.abspath(output_image_list_file)) or "."
    stats_file = os.path.join(output_dir, "mined_stats.json")

    stats = {
        "iteration": iteration,
        "mode": mode,
        "continual_dataset": continual,
        "identity": "unique_name",
        "candidate_pairs_file": os.path.abspath(candidate_pairs_file),
        "candidate_raw_count": candidate_raw_count,
        "candidate_unique_count": len(candidate_names),
        "candidate_duplicate_count": candidate_duplicate_count,
        "candidate_novel_count": len(candidate_names - historical_names),
        "candidate_replay_count": len(candidate_names & historical_names),
        "target_query_count": target,
        "requested_novel_count": requested_novel,
        "requested_replay_count": requested_replay,
        "selected_count": len(final_names),
        "selected_novel_count": actual_novel,
        "selected_replay_count": actual_replay,
        "selected_replay_from_current_knn": sum(
            1
            for item in final_items
            if item["from_current_candidates"] and item["key"] in historical_names
        ),
        "selected_replay_from_history_files": sum(
            1 for item in final_items if not item["from_current_candidates"]
        ),
        "history_pair_files_loaded": history_files_loaded,
        "replay_overflow_count": replay_overflow,
        "actual_replay_fraction": (
            actual_replay / len(final_names) if final_names else 0.0
        ),
        "selection_shortfall": shortfall,
        "overlap_with_previous_iteration_count": overlap_previous,
        "overlap_with_previous_iteration_rate": (
            overlap_previous / len(final_names) if final_names else 0.0
        ),
        "overlap_with_all_prior_iterations_count": overlap_all_prior,
        "overlap_with_all_prior_iterations_rate": (
            overlap_all_prior / len(final_names) if final_names else 0.0
        ),
        "cumulative_unique_pair_count": cumulative_unique,
        "source_pool_size": source_pool_size,
        "selected_pair_coverage_rate": (
            len(final_names) / source_pool_size if source_pool_size else 0.0
        ),
        "cumulative_unique_pair_coverage_rate": (
            cumulative_unique / source_pool_size if source_pool_size else 0.0
        ),
        "image_list_file": os.path.abspath(output_image_list_file),
        "train_pairs_file": os.path.abspath(output_pairs_file),
        "history_file": history_path,
    }

    summary_text = "\n".join([
        "PAS history-aware mined sample stats",
        f"Iteration: {iteration}",
        f"Mode: {mode}",
        f"Unique post-KNN candidates: {len(candidate_names)}",
        f"Requested pairs: {target}",
        f"Selected pairs: {len(final_names)}",
        f"Novel pairs: {actual_novel}",
        f"Replay pairs: {actual_replay}",
        f"Replay overflow: {replay_overflow}",
        f"Selection shortfall: {shortfall}",
        f"Overlap with previous iteration: {overlap_previous}",
        f"Overlap with all prior iterations: {overlap_all_prior}",
        f"Cumulative unique pairs: {cumulative_unique}",
        "Cumulative source-pool coverage: "
        f"{100.0 * stats['cumulative_unique_pair_coverage_rate']:.4f}%",
        "",
    ])

    _atomic_write_text(output_image_list_file, "".join(f"{name}\n" for name in final_names))
    _atomic_write_text(output_pairs_file, _json_text(final_rows))
    _atomic_write_text(stats_file, _json_text(stats))

    manifest = {
        "image_dir": candidate_manifest.get("image_dir", ""),
        "caption_dir": candidate_manifest.get("caption_dir", ""),
        "image_list_file": os.path.abspath(output_image_list_file),
        "caption_file_suffix": candidate_manifest.get(
            "caption_file_suffix", ".txt"
        ),
        "train_pairs_file": os.path.abspath(output_pairs_file),
        "target_query_count": target,
        "history_aware": {
            "enabled": True,
            "mode": mode,
            "continual_dataset": continual,
            "replay_fraction": effective_replay_fraction,
            "history_file": history_path,
            "candidate_manifest": os.path.abspath(candidate_manifest_file),
        },
        "stats_json": stats_file,
    }
    _atomic_write_text(manifest_path, _json_text(manifest))

    entry = {
        "iteration": iteration,
        "mode": mode,
        "target_query_count": target,
        "selected_count": len(final_names),
        "novel_count": actual_novel,
        "replay_count": actual_replay,
        "selected_unique_names": final_names,
        "pairs_file": os.path.abspath(output_pairs_file),
        "pairs_sha256": _sha256(output_pairs_file),
        "image_list_file": os.path.abspath(output_image_list_file),
        "image_list_sha256": _sha256(output_image_list_file),
        "manifest_file": os.path.abspath(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "stats_file": stats_file,
        "stats_sha256": _sha256(stats_file),
    }
    entries.append(entry)
    state["iterations"] = entries
    _atomic_write_text(history_path, _json_text(state))

    print(summary_text.strip())
    print(
        f"Committed history-aware selection for iteration {iteration}: "
        f"{len(final_names)} pairs -> {output_pairs_file}"
    )
    return output_image_list_file
