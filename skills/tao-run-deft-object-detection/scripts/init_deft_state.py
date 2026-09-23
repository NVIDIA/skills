#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Freeze the approved DEFT OD run configuration into ``deft_state.json``.

Runs exactly once, immediately after the user approves the Pre-Flight Summary.
Everything the loop is allowed to decide was already decided at that gate, so
this writes it to disk and never re-derives it: a run resumed three days and
one context compaction later reads the same epochs, the same encoder, the same
thresholds. Without this freeze, a resumed run silently re-defaults a parameter
mid-flight and the iterations before and after the resume are no longer
comparable — the mAP trend the loop exists to produce becomes meaningless.

The encoder is the sharpest case. ``embedding_model_path`` must stay identical
to the one that produced the source-pool parquet for the whole run; a mismatch
does not raise, it just returns confidently wrong neighbours. Pre-Flight
resolves it once (check 9) and this file pins it.

Refuses to overwrite an existing ``deft_state.json`` without ``--force``, which
is the guard against clobbering a live run: state and log are the loop's only
memory, and re-initializing over them strands every artifact already on disk.

Stdlib only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deft_stages import (  # noqa: E402
    SCHEMA_VERSION,
    check_artifact,
    derive_rare_classes,
    log_path,
    state_path,
    write_log_atomic,
    write_state_atomic,
)

ALLOCATION_POLICIES = ("global", "class_stratified")
# The encoder families the embedding stage accepts. Kept in step with
# tao-generate-image-embeddings' verify_image_embeddings_spec.py, which enforces the
# same set against the spec.
EMBEDDING_MODELS = ("CLIP", "SigLIP")

# AP50 gates from the reference ITS pipeline, used when the caller does not supply
# their own. The asymmetry is deliberate: `car` is abundant and already well learned,
# so it is held to a strict gate and rarely drags an image into the weak set; the
# loop exists to lift the rare vulnerable-road-user classes, which get looser gates.
REFERENCE_AP50_THRESHOLDS = {"car": 0.99, "bicycle": 0.7, "person": 0.7}

# Gate applied to a target class the reference never covered. Set to the reference's
# rare-class value rather than something permissive: a gate that is too loose marks
# no image weak, and a run that mines nothing burns a full training cycle to arrive
# back where it started.
FALLBACK_AP50_THRESHOLD = 0.7

# commit_stage.py's crash-recovery record. init archives state and log; a journal
# left behind by a commit that died would then restore the archived run over the
# fresh one on the very next commit.
COMMIT_JOURNAL_NAME = ".deft_commit.journal"


def _abs(raw: str) -> Path:
    """Absolute host path. The loop mounts host==container paths, so every
    recorded path must be absolute and fully resolved."""
    return Path(raw).expanduser().resolve()


def _split_classes(values: list[str] | None) -> list[str]:
    """Flatten repeated and/or comma-separated class flags, preserving order."""
    classes: list[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item and item not in classes:
                classes.append(item)
    return classes


def _archive(path: Path, stamp: str) -> Path | None:
    """Move an existing file aside so --force leaves a coherent pair on disk."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    backup = path.with_name(f"{path.name}.bak.{stamp}")
    os.replace(path, backup)
    return backup


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--results-dir", required=True,
                        help="Run directory; holds deft_state.json and loop_log.jsonl.")
    parser.add_argument("--workspace", required=True,
                        help="Workspace root. Mounted into every container as itself.")
    parser.add_argument("--max-iterations", type=int, default=1,
                        help="Number of iterations after the baseline. Default 1: one mine, "
                             "train and score pass, which is the smallest run that produces a "
                             "comparison against the baseline.")

    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, required=True,
                        help="train.num_epochs for every iteration (resolved in Pre-Flight).")
    parser.add_argument("--learning-rate", type=float, required=True,
                        help="train.optim.lr for every iteration (resolved in Pre-Flight).")

    parser.add_argument("--zero-shot-checkpoint", required=True,
                        help="Checkpoint scored at baseline, and fine-tuned from on every "
                             "iteration. Pre-Flight resolves this from NGC when the user "
                             "supplies none (scripts/fetch_gdino_checkpoint.py).")
    parser.add_argument("--zero-shot-source", default="user",
                        help="Where the checkpoint came from — 'user' or an NGC version "
                             "string. Recorded so a resumed run can say which weights the "
                             "earlier iterations were measured against.")
    parser.add_argument("--train-spec-template", default=None,
                        # Defaults to assets/train_grounding_dino.yaml. Every value in
                        # it is one the reference already settled, and hand-authoring a
                        # template is how log_scale/class_embed_bias get reintroduced.
                        help="Grounding DINO train spec; dataset.train_data_sources must be a list.")

    parser.add_argument("--pool-dir", default=None,
                        help="Prepared-pool directory. Derives --source-pool-embeddings, "
                             "--source-pool-annotations, --source-detection-file, and "
                             "--pool-report from their standard names inside it.")
    parser.add_argument("--source-pool-embeddings", default=None,
                        help="Source-pool parquet (filepath + embedding) — the mining corpus.")
    parser.add_argument("--source-pool-annotations", default=None,
                        help="Source-pool ODVG directory (*.jsonl keyed by file_name).")
    parser.add_argument("--pool-report", default=None,
                        help="validate_pool_coco.py's report from the prep run. Cross-checks that "
                             "the pool was prepared for these target classes.")

    # Only consulted when the pool above does not exist yet, in which case `prep`
    # builds it before the baseline and all three are required.
    parser.add_argument("--pool-images", default=None,
                        help="Raw pool image directory. Required only when the pool artifacts "
                             "do not exist and `prep` must produce them.")
    parser.add_argument("--codetr-checkpoint", default=None,
                        help="Co-DETR detector checkpoint for pseudo-labeling. Required only "
                             "when `prep` must run.")
    parser.add_argument("--codetr-classmap", default=None,
                        help="Co-DETR's own vocabulary, one class per line (COCO-80 for a "
                             "COCO-trained checkpoint). Required only when `prep` must run.")

    parser.add_argument("--embedding-model", default="SigLIP",
                        help="Encoder family: SigLIP or CLIP. Not a model id — the weights "
                             "are --embedding-model-path. Must match the source-pool "
                             "parquet's encoder.")
    parser.add_argument("--embedding-model-path", required=True,
                        help="Resolved local snapshot directory, or a verified HuggingFace id.")

    parser.add_argument("--kpi-images-dir", required=True,
                        help="KPI image root. A trailing slash is stripped; kpi_analyze derives "
                             "Sequence Name from the second-to-last path component.")
    parser.add_argument("--ground-truth-labels-dir", required=True,
                        help="KPI ground-truth KITTI label directory.")
    parser.add_argument("--class-mapping", required=True,
                        help="Class-mapping YAML consumed by kpi_analyze.")

    parser.add_argument("--ap50-thresholds-json", default=None,
                        help='Per-class AP50 thresholds as a JSON object, e.g. \'{"car": 0.99}\'. '
                             "Omit to gate each target class at the reference value "
                             f"({json.dumps(REFERENCE_AP50_THRESHOLDS)}), falling back to "
                             f"{FALLBACK_AP50_THRESHOLD} for classes the reference did not cover.")
    parser.add_argument("--target-classes", action="append", default=None,
                        help="Target classes; repeatable and/or comma-separated. "
                             "Defaults to the keys of --ap50-thresholds-json, or to the "
                             "reference classes when neither flag is given.")

    parser.add_argument("--multiplier", type=int, default=3,
                        help="Mining budget = iteration 1 weak-image count x multiplier.")
    parser.add_argument("--allocation-policy", choices=ALLOCATION_POLICIES, default="global")
    parser.add_argument("--rare-class-list", default=None,
                        help='Comma-separated rare classes, e.g. "person,bicycle". '
                             "Required for class_stratified.")
    parser.add_argument("--source-detection-file", default=None,
                        help="Source-pool COCO JSON. Required for class_stratified.")
    parser.add_argument("--target-detection-file", default=None,
                        help="KPI COCO JSON. Required for class_stratified.")

    parser.add_argument("--distance-metric", default="euclidean")
    parser.add_argument("--candidate-expansion-factor", type=int, default=5,
                        help="Miner's internal candidate-pool growth seed. Not the loop iteration count.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--kpi-conf-threshold", type=float, default=0.0,
                        help="kpi.conf_threshold for every phase. Default 0.0: inference\n"
                             "writes its labels at 0.0, so scoring there covers the full\n"
                             "PR curve. A higher value truncates the low-confidence tail\n"
                             "and is not comparable to a run scored at 0.0.")

    parser.add_argument("--extra-mount", action="append", default=None, metavar="PATH",
                        help="Extra host path a container must read, repeatable. The mount "
                             "list is otherwise derived from the run's own inputs, so this is "
                             "the way to admit anything they do not name -- a spec, an "
                             "auxiliary dataset, a checkpoint a template points at. Files are "
                             "mounted by their parent directory.")

    parser.add_argument("--force", action="store_true",
                        help="Reinitialize over an existing run. The current state and a non-empty "
                             "log are archived as *.bak.<UTC stamp> first.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        errors: list[str] = []
        warnings: list[str] = []

        # ── numeric config ───────────────────────────────────────────────────
        if args.max_iterations < 1:
            errors.append(f"--max-iterations must be >= 1, got {args.max_iterations}")
        if args.num_gpus < 1:
            errors.append(f"--num-gpus must be >= 1, got {args.num_gpus}")
        if args.num_epochs < 1:
            errors.append(f"--num-epochs must be >= 1, got {args.num_epochs}")
        # `nan <= 0` and `inf <= 0` are both False, so a bare range check lets them
        # through — and json.dump then writes a NaN/Infinity literal that no other
        # JSON parser will read back. The frozen config has to stay valid JSON.
        if not math.isfinite(args.learning_rate):
            errors.append(f"--learning-rate must be a finite number, got {args.learning_rate}")
        elif args.learning_rate <= 0:
            errors.append(f"--learning-rate must be > 0, got {args.learning_rate}")
        if args.multiplier < 1:
            errors.append(f"--multiplier must be >= 1, got {args.multiplier}")
        if args.candidate_expansion_factor < 1:
            errors.append(
                f"--candidate-expansion-factor must be >= 1, got {args.candidate_expansion_factor}")
        if not 0 < args.iou_threshold <= 1:
            errors.append(f"--iou-threshold must be within (0, 1], got {args.iou_threshold}")
        if not 0 <= args.kpi_conf_threshold <= 1:
            errors.append(f"--kpi-conf-threshold must be within [0, 1], got {args.kpi_conf_threshold}")

        # ── AP50 thresholds ──────────────────────────────────────────────────
        requested_classes = _split_classes(args.target_classes)
        # The KPI mapping names the classes this run will be scored on, so it — not
        # the reference ITS set — is what an omitted --target-classes should follow.
        # Defaulting to the reference silently runs a bicycle/car/person loop against
        # an unrelated dataset, and every stage reports success while doing it.
        mapping_classes: list[str] = []
        mapping_path = Path(args.class_mapping).expanduser().resolve()
        if mapping_path.is_file():
            try:
                import yaml  # provisioned by deft_python.sh
                loaded = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    mapping_classes = [k for e in loaded if isinstance(e, dict) for k in e]
                elif isinstance(loaded, dict):
                    mapping_classes = list(loaded)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"--class-mapping could not be read for class derivation: {exc}")

        if not requested_classes and mapping_classes:
            warnings.append(
                f"--target-classes not given; derived {mapping_classes} from --class-mapping")
        elif not requested_classes and not mapping_classes:
            errors.append(
                "--target-classes not given and --class-mapping yielded no classes to derive "
                "from. Pass --target-classes explicitly rather than inheriting the reference "
                "ITS set, which would train and score a different dataset's classes.")

        if args.ap50_thresholds_json is None:
            # Gate the classes the caller actually asked for, not the reference's, so
            # the "every target class is gated" check below can never fail on a default.
            basis = requested_classes or mapping_classes or sorted(REFERENCE_AP50_THRESHOLDS)
            thresholds = {c: REFERENCE_AP50_THRESHOLDS.get(c, FALLBACK_AP50_THRESHOLD)
                          for c in basis}
            assumed = sorted(c for c in basis if c not in REFERENCE_AP50_THRESHOLDS)
            warnings.append(
                "--ap50-thresholds-json not given; defaulted to "
                + json.dumps(thresholds, sort_keys=True)
                + (f" — {assumed} are not reference classes and were gated at "
                   f"{FALLBACK_AP50_THRESHOLD} by assumption; confirm that is right for this "
                   "dataset, since too loose a gate mines nothing"
                   if assumed else " (the reference ITS gates)"))
        else:
            try:
                thresholds = json.loads(args.ap50_thresholds_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"--ap50-thresholds-json is not valid JSON: {exc}") from exc
            if not isinstance(thresholds, dict) or not thresholds:
                raise ValueError(
                    '--ap50-thresholds-json must be a non-empty JSON object, e.g. \'{"car": 0.99}\'')
        for name, value in thresholds.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"--ap50-thresholds-json[{name!r}] must be a number, got {value!r}")
            elif not 0 <= float(value) <= 1:
                errors.append(f"--ap50-thresholds-json[{name!r}] must be within [0, 1], got {value}")

        # `model` selects the loader inside the container and nothing between here
        # and there validates it, so a bad value is accepted at init and fails an
        # hour later at embed. The common mistake is putting the model id here.
        if args.embedding_model not in EMBEDDING_MODELS:
            hint = ""
            if "/" in args.embedding_model or args.embedding_model.count("-") >= 2:
                hint = (f"; {args.embedding_model!r} looks like a model id — did you mean "
                        "--embedding-model-path?")
            errors.append(
                f"--embedding-model takes the encoder family "
                f"({' or '.join(sorted(EMBEDDING_MODELS))}), got {args.embedding_model!r}{hint}")

        # ── class lists ──────────────────────────────────────────────────────
        target_classes = requested_classes or sorted(thresholds)
        # gap_analysis runs with default_ap50_threshold 0.0, so a target class with
        # no explicit threshold can never be reported weak and can never be mined for.
        untargeted = [c for c in target_classes if c not in thresholds]
        if untargeted:
            errors.append(
                f"target class(es) {untargeted} have no AP50 threshold; gap_analysis would "
                "score them against 0.0 and they could never be mined for")
        extra = [c for c in thresholds if c not in target_classes]
        if extra:
            warnings.append(f"AP50 threshold(s) for non-target class(es) {extra} will be ignored")

        rare_classes = _split_classes([args.rare_class_list] if args.rare_class_list else None)
        unknown_rare = [c for c in rare_classes if c not in target_classes]
        if unknown_rare:
            errors.append(f"--rare-class-list names non-target class(es) {unknown_rare}")

        # ── paths ────────────────────────────────────────────────────────────
        results_dir = _abs(args.results_dir)
        workspace = _abs(args.workspace)
        zero_shot_checkpoint = _abs(args.zero_shot_checkpoint)
        # No template supplied is the normal case: assets/train_grounding_dino.yaml
        # already carries every value the reference settled, and hand-authoring one
        # is how log_scale and class_embed_bias get reintroduced wrong.
        train_spec_template = _abs(
            args.train_spec_template
            or Path(__file__).resolve().parent.parent / "assets" / "train_grounding_dino.yaml"
        )
        # A prepared pool has a fixed internal layout, so one directory determines
        # all four paths. Explicit flags still win, for a pool assembled by hand.
        pool_dir = _abs(args.pool_dir) if args.pool_dir else None
        if pool_dir:
            args.source_pool_embeddings = (args.source_pool_embeddings
                                           or str(pool_dir / "source_embeddings.parquet"))
            args.source_pool_annotations = (args.source_pool_annotations
                                            or str(pool_dir / "odvg"))
            args.source_detection_file = (args.source_detection_file
                                          or str(pool_dir / "coco.json"))
            report_default = pool_dir / "pool_report.json"
            if not args.pool_report and report_default.is_file():
                args.pool_report = str(report_default)
        missing_flags = [f for f, v in (("--source-pool-embeddings", args.source_pool_embeddings),
                                        ("--source-pool-annotations", args.source_pool_annotations))
                         if not v]
        if missing_flags:
            raise ValueError(
                f"{missing_flags} required unless --pool-dir names a prepared pool")
        source_pool_embeddings = _abs(args.source_pool_embeddings)
        source_pool_annotations = _abs(args.source_pool_annotations)
        # ── the pool decides which classes are rare ──────────────────────────
        # pool_report.json records what the pool actually holds. Two things follow.
        #
        # First, a target class with no annotations in the pool can never be mined
        # for. Mining would not fail — it would return neighbours of something else —
        # and that class simply never improves while every stage reports success.
        #
        # Second, scarcity in the POOL is the right signal for class_stratified
        # allocation, not scarcity in the KPI set. Stratified allocation exists so
        # that classes the pool holds few of still get their share of the budget; a
        # class the pool is full of will be found by global allocation anyway.
        pool_counts: dict[str, int] = {}
        pool_provenance: dict = {}
        if args.pool_report:
            report_path = _abs(args.pool_report)
            if not report_path.is_file():
                errors.append(f"--pool-report: not a file: {report_path}")
            else:
                try:
                    report_data = json.loads(report_path.read_text(encoding="utf-8"))
                    pool_counts = {
                        str(c): int(n) for c, n in
                        report_data.get("annotations_by_class", {}).items()
                    }
                    pool_provenance = report_data.get("prep_inputs") or {}
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    errors.append(f"--pool-report: unreadable: {exc}")
                else:
                    unbacked = [c for c in target_classes if not pool_counts.get(c)]
                    if unbacked:
                        errors.append(
                            f"the source pool holds no annotations for target class(es) "
                            f"{unbacked} (prepared for "
                            f"{sorted(c for c, n in pool_counts.items() if n)}). Mining cannot "
                            "surface examples that are not in the pool, so those classes can "
                            "never improve — re-prep the pool for this class set")
                    # Prep is documented as idempotent: skip any step whose output
                    # exists. Existence cannot show a pool was built from the same
                    # checkpoint, fold, encoder or container as this run wants, and a
                    # stale pool is reused in silence. What the pool recorded about
                    # itself is the only evidence available.
                    if not pool_provenance:
                        warnings.append(
                            "--pool-report carries no prep_inputs, so nothing records which "
                            "checkpoint, fold or container built this pool. Re-run "
                            "validate_pool_coco.py with --record to make reuse checkable.")
                    else:
                        print("pool provenance:", file=sys.stderr)
                        for key, value in sorted(pool_provenance.items()):
                            print(f"  {key}: {value}", file=sys.stderr)
                        declared = pool_provenance.get("target_classes")
                        if declared:
                            declared_set = {c.strip() for c in str(declared).split(",") if c.strip()}
                            if declared_set != set(target_classes):
                                errors.append(
                                    f"the pool was prepared for {sorted(declared_set)} but this "
                                    f"run targets {sorted(target_classes)}. Its pseudo-labels "
                                    f"were folded to a different class set; re-prep rather than "
                                    f"reuse it")

        if args.allocation_policy == "class_stratified" and not rare_classes and pool_counts:
            rare_classes, detail = derive_rare_classes(pool_counts, target_classes)
            if rare_classes:
                warnings.append(
                    f"--rare-class-list not given; derived {rare_classes} from the pool's "
                    f"own class counts ({detail}). These are the classes the pool holds "
                    "fewest of, which is what stratified allocation exists to protect")

        kpi_images_dir = _abs(args.kpi_images_dir)
        ground_truth_labels_dir = _abs(args.ground_truth_labels_dir)
        class_mapping = _abs(args.class_mapping)

        required = [
            ("--workspace", workspace, "dir"),
            ("--zero-shot-checkpoint", zero_shot_checkpoint, "file"),
            ("--train-spec-template", train_spec_template, "file"),
            ("--kpi-images-dir", kpi_images_dir, "dir"),
            ("--ground-truth-labels-dir", ground_truth_labels_dir, "dir"),
            ("--class-mapping", class_mapping, "file"),
        ]
        pool = [
            ("--source-pool-embeddings", source_pool_embeddings, "file"),
            ("--source-pool-annotations", source_pool_annotations, "dir"),
        ]
        for flag, path, kind in required:
            problem = check_artifact(str(path), kind)
            if problem:
                errors.append(f"{flag}: {problem}")
        # A missing pool is not a reason to refuse the run — it is a reason to prep.
        # But prep needs inputs the loop otherwise never asks for, so the permission
        # is tied to having those rather than to a bare override flag. Without them
        # there is nothing to route to, and letting the run start anyway only defers
        # the failure to `mine`, six stages and a training run later.
        prep_inputs = {
            "--pool-images": args.pool_images,
            "--codetr-checkpoint": args.codetr_checkpoint,
            "--codetr-classmap": args.codetr_classmap,
        }
        missing_pool = [(flag, problem) for flag, path, kind in pool
                        if (problem := check_artifact(str(path), kind))]
        absent_inputs = sorted(f for f, v in prep_inputs.items() if not v)
        if missing_pool:
            if absent_inputs:
                for flag, problem in missing_pool:
                    errors.append(f"{flag}: {problem}")
                errors.append(
                    "the source pool does not exist yet, so `prep` has to build it — but "
                    f"{absent_inputs} were not supplied. Either point the pool flags at an "
                    "already-prepared pool, or give prep its inputs "
                    "(see references/prep-source-pool.md)")
            else:
                for flag, problem in missing_pool:
                    warnings.append(f"{flag}: {problem} — `prep` runs first and produces it")
                bad_inputs = [
                    f"{flag}: {problem}" for flag, value in prep_inputs.items()
                    if (problem := check_artifact(
                        str(_abs(value)), "dir" if flag == "--pool-images" else "file"))
                ]
                errors.extend(bad_inputs)

        # class_stratified mining needs both COCO detection files; TAO DS will not infer them.
        detection_files = {
            "--source-detection-file": args.source_detection_file,
            "--target-detection-file": args.target_detection_file,
        }
        if args.allocation_policy == "class_stratified":
            if not rare_classes:
                # Which classes are rare is a property of the pool's annotation
                # counts, and those counts are prep's output. Requiring the list
                # here would make init and prep each other's precondition, so on a
                # run that still has to prep it is left null and the prep commit
                # derives it from pool_report.json. `mine` is the first stage that
                # reads it, and prep is complete by then.
                message = ("--allocation-policy class_stratified requires "
                           "--rare-class-list")
                if missing_pool and not absent_inputs:
                    warnings.append(
                        f"{message} — left unset; `commit_stage.py --stage prep` derives it "
                        "from the pool's own class counts once prep has produced "
                        "pool_report.json")
                else:
                    errors.append(message)
            if not args.pool_report:
                # pool_report.json is the only artifact that cross-checks the prepared
                # pool against the requested classes. It is also prep's own output, so
                # demanding it on a run that still has to prep would make init and prep
                # each other's precondition — the same deadlock --source-detection-file
                # had. Required only when the pool already exists.
                message = (
                    "--allocation-policy class_stratified needs --pool-report "
                    "(validate_pool_coco.py writes it; --pool-dir picks it up "
                    "automatically when it sits beside the pool)")
                if missing_pool and not absent_inputs:
                    warnings.append(f"{message} — `prep` runs first and produces it")
                else:
                    errors.append(message)
            for flag, raw in detection_files.items():
                if not raw:
                    errors.append(f"--allocation-policy class_stratified requires {flag}")
        elif rare_classes:
            warnings.append(
                "--rare-class-list is set but --allocation-policy is global; rare classes are ignored")

        # A mount may be either a file or a directory, so this is existence only --
        # check_artifact would reject a directory as "not a file".
        for extra in (args.extra_mount or []):
            if not _abs(extra).exists():
                errors.append(f"--extra-mount: does not exist: {_abs(extra)}")

        resolved_detection: dict[str, str | None] = {}
        for flag, raw in detection_files.items():
            if not raw:
                resolved_detection[flag] = None
                continue
            path = _abs(raw)
            resolved_detection[flag] = str(path)
            problem = check_artifact(str(path), "file")
            if problem:
                # --source-detection-file is prep's own output (source_pool/coco.json),
                # so on a run that still has to prep, requiring it here would make init
                # and prep each other's precondition. Same downgrade as the pool
                # artifacts above; it is still required by the time `mine` reads it.
                if flag == "--source-detection-file" and missing_pool and not absent_inputs:
                    warnings.append(
                        f"{flag}: {problem} — `prep` runs first and produces it")
                else:
                    errors.append(f"{flag}: {problem}")
            elif path.suffix.lower() != ".json":
                warnings.append(f"{flag}: {path} is not a .json file; mining needs COCO JSON")

        # A local snapshot must be a directory; a bare HuggingFace id is left alone.
        model_path_raw = args.embedding_model_path
        looks_local = model_path_raw.startswith(("/", "~", ".")) or Path(model_path_raw).exists()
        if looks_local:
            model_path = str(_abs(model_path_raw))
            problem = check_artifact(model_path, "dir")
            if problem:
                errors.append(f"--embedding-model-path: {problem}")
            elif not (Path(model_path) / "config.json").is_file():
                warnings.append(
                    f"--embedding-model-path: no config.json in {model_path}; "
                    "the embed stage will fail to load this encoder")
        else:
            model_path = model_path_raw
            warnings.append(
                f"--embedding-model-path {model_path!r} is a HuggingFace id, not a local snapshot; "
                "HF_TOKEN and outbound access are required on every iteration")

        if errors:
            raise ValueError("invalid run configuration:\n  - " + "\n  - ".join(errors))

        # Containers see only "$WORKSPACE:$WORKSPACE"; anything outside is invisible inside them.
        # embedding_model_path is exempt: HF_HOME legitimately lives outside the workspace.
        # Every path a container has to read, not a subset. The encoder is included
        # despite living outside the workspace by design: prep's embed reads it from
        # inside the container, and an unmounted local snapshot is passed to
        # HuggingFace as a repo id, which fails as HFValidationError rather than as a
        # missing mount. Prep's own inputs are here for the same reason -- Co-DETR
        # reads the classmap and checkpoint from inside the container.
        mounted = [results_dir, zero_shot_checkpoint, train_spec_template, source_pool_embeddings,
                   source_pool_annotations, kpi_images_dir, ground_truth_labels_dir, class_mapping]
        if looks_local:
            # A HuggingFace hub snapshot is all symlinks into a sibling blobs/ dir, so
            # mounting the snapshot alone gives the container dangling links and the
            # loader reports "no file named model.safetensors found" -- a missing-model
            # error for a model that is present. Mount the repo root, which holds both.
            encoder = Path(model_path)
            if "snapshots" in encoder.parts:
                encoder = Path(*encoder.parts[:encoder.parts.index("snapshots")])
            mounted.append(encoder)
        for extra in (args.pool_images, args.codetr_checkpoint, args.codetr_classmap):
            if extra:
                mounted.append(_abs(extra))
        # `tmm unique_neighbor_matching` opens both COCO detection files from inside the
        # container under class_stratified allocation. The target file is the KPI
        # detections, which live with the read-only dataset rather than in the per-run
        # workspace, so without this the stage fails on a path the spec itself named.
        for detection in resolved_detection.values():
            if detection:
                mounted.append(Path(detection))
        # Whatever the derivation cannot know about. A run may legitimately reference a
        # path none of its own config keys name, and without this the only remedy is
        # editing the skill.
        for extra in (args.extra_mount or []):
            mounted.append(_abs(extra))
        # Anything outside the workspace needs its own -v or the container cannot read
        # it. Deriving the mounts here means the stages get a list to use rather than a
        # warning to act on, which is what left earlier runs to work it out mid-flight.
        outside = [p for p in mounted if not p.is_relative_to(workspace)]
        extra_mounts: list[str] = []
        for path in outside:
            anchor = path if path.is_dir() else path.parent
            if not any(anchor.is_relative_to(Path(m)) for m in extra_mounts):
                extra_mounts = [m for m in extra_mounts if not Path(m).is_relative_to(anchor)]
                extra_mounts.append(str(anchor))
        extra_mounts.sort()
        if extra_mounts:
            warnings.append(
                "path(s) outside --workspace need their own mount; every container launch "
                "must add: " + " ".join(f'-v "{m}:{m}"' for m in extra_mounts))
        if not any(p.suffix.lower() == ".txt" for p in ground_truth_labels_dir.iterdir()):
            warnings.append(
                f"--ground-truth-labels-dir has no .txt label files: {ground_truth_labels_dir}")

        # ── write ────────────────────────────────────────────────────────────
        # A non-empty log without state is a torn run, not a fresh one; overwriting only the
        # state would leave the audit's two sources of truth permanently disagreeing.
        live = [p for p in (state_path(results_dir), log_path(results_dir))
                if p.is_file() and p.stat().st_size > 0]
        if live and not args.force:
            raise FileExistsError(
                f"refusing to clobber an existing run: {', '.join(p.name for p in live)} already "
                f"present in {results_dir}. Resume it with audit_deft_run.py, or pass --force to "
                "reinitialize (the current state and log are archived as *.bak.<UTC stamp>).")

        now = datetime.datetime.now(datetime.timezone.utc)
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        results_dir.mkdir(parents=True, exist_ok=True)

        archived = [p for p in (_archive(f, stamp) for f in live) if p is not None]
        # A leftover commit journal describes the run being archived, not this one.
        (results_dir / COMMIT_JOURNAL_NAME).unlink(missing_ok=True)
        # Empty log first: a crash between the two writes leaves no state, so init is
        # still re-runnable without --force.
        if not log_path(results_dir).is_file():
            write_log_atomic(results_dir, [])

        state = {
            "schema_version": SCHEMA_VERSION,
            "workspace": str(workspace),
            "results_dir": str(results_dir),
            "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": {
                "max_iterations": args.max_iterations,
                "num_gpus": args.num_gpus,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "zero_shot_checkpoint": str(zero_shot_checkpoint),
                "zero_shot_source": args.zero_shot_source,
                "train_spec_template": str(train_spec_template),
                "source_pool_embeddings": str(source_pool_embeddings),
                "source_pool_annotations": str(source_pool_annotations),
                # Null on a run whose pool already existed. Frozen here so a `prep`
                # resumed after a compaction labels the same images with the same
                # detector — a pool built half from one checkpoint and half from
                # another is not a corpus anything can reason about.
                "pool_images": str(_abs(args.pool_images)) if args.pool_images else None,
                "codetr_checkpoint": (str(_abs(args.codetr_checkpoint))
                                      if args.codetr_checkpoint else None),
                "codetr_classmap": (str(_abs(args.codetr_classmap))
                                    if args.codetr_classmap else None),
                "embedding_model": args.embedding_model,
                "embedding_model_path": model_path,
                "kpi_images_dir": str(kpi_images_dir),
                "ground_truth_labels_dir": str(ground_truth_labels_dir),
                "class_mapping": str(class_mapping),
                "ap50_thresholds": thresholds,
                "multiplier": args.multiplier,
                "allocation_policy": args.allocation_policy,
                "target_classes": target_classes,
                "extra_container_mounts": extra_mounts,
                "rare_class_list": ",".join(rare_classes) if rare_classes else None,
                "source_detection_file": resolved_detection["--source-detection-file"],
                "target_detection_file": resolved_detection["--target-detection-file"],
                "distance_metric": args.distance_metric,
                "candidate_expansion_factor": args.candidate_expansion_factor,
                "iou_threshold": args.iou_threshold,
                "kpi_conf_threshold": args.kpi_conf_threshold,
            },
            "current_iteration": 0,
            "iterations": {},
            "status": "running",
        }
        write_state_atomic(results_dir, state)

        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        for backup in archived:
            print(f"archived {backup}", file=sys.stderr)

        print(f"wrote {state_path(results_dir)} (schema_version={SCHEMA_VERSION}, status=running)")
        print(f"  workspace       {workspace}")
        print(f"  iterations      max {args.max_iterations} · gpus {args.num_gpus} · "
              f"epochs {args.num_epochs} · lr {args.learning_rate}")
        print(f"  target classes  {', '.join(target_classes)}")
        print(f"  ap50            {json.dumps(thresholds, sort_keys=True)}")
        print(f"  mining          {args.allocation_policy} · multiplier {args.multiplier} · "
              f"rare {','.join(rare_classes) if rare_classes else 'none'}")
        print(f"  encoder         {args.embedding_model} @ {model_path}")
        print(f"  log             {log_path(results_dir)}")
        print(f"Next: audit_deft_run.py --results-dir {results_dir}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
