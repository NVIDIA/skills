#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mode-dispatched data preparation pipeline for the KERMT agent skills.

Composes the existing repo data-prep scripts (`scripts/clean_smiles.py`,
`scripts/save_features.py`, `scripts/build_vocab.py`, `scripts/split_data.py`)
into a single one-call entry point per workflow. Output lands in `--out` with
a `prepare_data.json` manifest that the downstream runners read.

Mode pipelines
--------------
pretrain  : clean -> (optional auto-split train into train+val by --val-frac)
            -> save_features (fgtasklabel) on each CSV
            -> vocab step: if --vocab-dir / --{atom,bond,smiles}-vocab given,
               copy those through (continue-pretrain case — the ckpt's vocab
               is authoritative); else if --skip-vocab, skip;
               else build_vocab on train (pretrain-from-scratch case)
            -> split_data (graph + feature shards + summary.txt) per CSV
finetune  : clean each provided CSV -> (optional random split when only one
            CSV is provided; emits a strong warning recommending scaffold-
            balanced pre-splits) -> save_features (rdkit_2d_normalized) per CSV
inference : clean -> save_features (rdkit_2d_normalized)
embed     : clean only (extract_embeddings.py featurizes on the fly)

Output convention
-----------------
The manifest under `<out>/prepare_data.json` captures every step's inputs,
outputs, duration, and skipped-due-to-existing flag, plus a top-level
`split_method` field (one of: "user_provided", "random", "n/a") that the
finetune runner uses to pass the correct `--split_type` to main.py.

Subprocess composition
----------------------
Each underlying script is invoked via `subprocess.run`. The PYTHONPATH=/workspace
env var (set by `scripts/kermt_container.sh`) makes the `kermt` package
importable inside the subprocesses; without it, build_vocab.py and split_data.py
fail with `ModuleNotFoundError: No module named 'kermt'`.

CLI
---
    prepare_data.py --mode {pretrain|finetune|inference|embed}
                    --csv <input.csv> --out <output-dir>
                    [--val-csv <path>] [--test-csv <path>]
                    [--val-frac 0.1] [--test-frac 0.1] [--seed 0]
                    [--sample-per-file 100000] [--vocab-format json]
                    [--dataset-name pretrain]
                    [--targets COL [COL ...]]
                    [--features-generator <name>]
                    [--smiles-column 0]
                    [--force] [--skip-clean] [--skip-features]
                    [--skip-vocab] [--skip-split]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

# sys.path tweak so `_utils` is importable regardless of how this script
# is invoked (kermt_run sets PYTHONPATH=/workspace; bare-Python launches
# from the host don't).
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import PRETRAIN_VOCAB_STEMS, resolve_kermt_repo, validate_vocab_file  # noqa: E402


REPO_ROOT = resolve_kermt_repo()
EXISTING_SCRIPTS = REPO_ROOT / "scripts"

DEFAULT_FEATURES_GENERATOR = {
    "pretrain": "fgtasklabel",
    "finetune": "rdkit_2d_normalized",
    "inference": "rdkit_2d_normalized",
    "embed": None,  # not used
}

VALID_MODES = ("pretrain", "finetune", "inference", "embed")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], step_name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Run a subprocess, append a step entry to manifest, raise on failure."""
    step: dict[str, Any] = {
        "name": step_name,
        "cmd": cmd,
        "duration_s": None,
        "ok": False,
        "stderr_tail": "",
        "skipped_due_to_existing": False,
    }
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    step["duration_s"] = round(time.time() - t0, 2)
    if proc.returncode != 0:
        step["stderr_tail"] = (proc.stderr or "").splitlines()[-20:]
        step["ok"] = False
        manifest["steps"].append(step)
        raise RuntimeError(
            f"step '{step_name}' failed (exit {proc.returncode}); "
            f"command: {' '.join(cmd)}\nstderr tail:\n" + "\n".join(step["stderr_tail"])
        )
    step["ok"] = True
    manifest["steps"].append(step)
    return step


def _skipped(step_name: str, output_path: str, manifest: dict[str, Any]) -> dict[str, Any]:
    step = {
        "name": step_name,
        "output": output_path,
        "ok": True,
        "duration_s": 0.0,
        "skipped_due_to_existing": True,
    }
    manifest["steps"].append(step)
    return step


def _exists_nonempty(path: Path) -> bool:
    """File exists with non-zero size, or directory exists with at least one entry."""
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        try:
            next(path.iterdir())
            return True
        except StopIteration:
            return False
    return False


# ---------------------------------------------------------------------------
# Per-script wrappers
# ---------------------------------------------------------------------------

def _resolve_smiles_column(csv_path: Path, explicit_value: int | None) -> int:
    """Return the 0-based index of the SMILES column in csv_path.

    Auto-detection rule when `explicit_value is None`:
      1. Read the CSV header (first non-empty row).
      2. Prefer an exact lowercase `smiles` column (kermt convention).
      3. Otherwise accept a single case-insensitive match
         (`SMILES`, `Smiles`, etc.).
      4. If no match (or multiple ambiguous matches), raise a ValueError
         that surfaces the header so the user can disambiguate via
         `--smiles-column N`.

    Real datasets routinely place SMILES at column index ≠ 0
    (e.g. openadmet's all.csv has "Molecule Name" at col 0 and "SMILES"
    at col 1). Auto-detection prevents the silent 0-row-clean failure
    mode where every row gets rejected because col 0 doesn't parse as
    a SMILES string.
    """
    if explicit_value is not None:
        return explicit_value

    if not csv_path.is_file():
        raise ValueError(f"input CSV not found: {csv_path}")

    import csv as _csv
    with csv_path.open("r", newline="") as f:
        reader = _csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"input CSV {csv_path} is empty")

    stripped = [c.strip() for c in header]
    # Prefer exact lowercase "smiles"
    exact = [i for i, c in enumerate(stripped) if c == "smiles"]
    if exact:
        return exact[0]
    # Then case-insensitive
    ci = [i for i, c in enumerate(stripped) if c.lower() == "smiles"]
    if len(ci) == 1:
        return ci[0]
    if len(ci) > 1:
        raise ValueError(
            f"input CSV {csv_path} has multiple SMILES-named columns: "
            f"{[header[i] for i in ci]} at indices {ci}. "
            "Pass --smiles-column N (0-based) to disambiguate."
        )
    raise ValueError(
        f"could not auto-detect a SMILES column in {csv_path}. "
        f"Header columns: {header}. "
        "Pass --smiles-column N (0-based) to specify which column holds SMILES."
    )


def _clean_smiles(
    input_csv: Path, output_csv: Path, smiles_column: int, manifest: dict[str, Any], force: bool
) -> Path:
    if not force and _exists_nonempty(output_csv):
        _skipped(f"clean_smiles({input_csv.name})", str(output_csv), manifest)
        return output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if force and output_csv.exists():
        # clean_smiles.py prompts interactively (input()) when the output file
        # already exists — that's an EOFError in a non-TTY subprocess. Pre-delete.
        output_csv.unlink()
    cmd = [
        sys.executable, str(EXISTING_SCRIPTS / "clean_smiles.py"),
        "--input", str(input_csv),
        "--output", str(output_csv),
        "--smiles_column", str(smiles_column),
    ]
    _run(cmd, f"clean_smiles({input_csv.name})", manifest)
    return output_csv


def _reduce_to_smiles_column(
    csv_path: Path, smiles_column: int, manifest: dict[str, Any]
) -> Path:
    """Rewrite an inference CSV to keep only the SMILES column (at index 0).

    Downstream `kermt.util.utils.get_data` -> `MoleculeDatapoint.__init__`
    floats every column after SMILES, which crashes on non-numeric passthrough
    columns (e.g. a 'split' label of 'train'/'val'/'test', or a 'Molecule Name'
    string). Inference does not need target columns, so drop them here.

    Note on skip semantics: this step is idempotent — running it on an
    already-single-column file is a no-op. We record that with
    `skipped_due_to_idempotent: True`, NOT `skipped_due_to_existing: True`.
    The two fields have different meanings: `_existing` means "I found a
    cached output file from a prior run and reused it" (overridden by
    `--force`); `_idempotent` means "the input is already in the desired
    state, so re-executing changes nothing" (safe to skip even under
    `--force`).
    """
    step_name = f"reduce_to_smiles_only({csv_path.name})"
    start = time.time()
    df = pd.read_csv(csv_path)
    if df.shape[1] == 1:
        manifest["steps"].append({
            "name": step_name,
            "output": str(csv_path),
            "ok": True,
            "duration_s": time.time() - start,
            "skipped_due_to_idempotent": True,
            "note": "already single-column",
        })
        return csv_path
    effective_col = smiles_column if 0 <= smiles_column < df.shape[1] else 0
    df.iloc[:, [effective_col]].to_csv(csv_path, index=False)
    manifest["steps"].append({
        "name": step_name,
        "output": str(csv_path),
        "ok": True,
        "duration_s": time.time() - start,
        "input_cols": int(df.shape[1]),
        "kept_col": effective_col,
        "kept_col_name": str(df.columns[effective_col]),
    })
    return csv_path


def _save_features(
    csv_path: Path, npz_path: Path, generator: str, manifest: dict[str, Any], force: bool
) -> Path:
    if not force and _exists_nonempty(npz_path):
        _skipped(f"save_features({csv_path.name}, {generator})", str(npz_path), manifest)
        return npz_path
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if force and npz_path.exists():
        npz_path.unlink()  # --restart still loads partial state if file exists; pre-delete to be safe
    cmd = [
        sys.executable, str(EXISTING_SCRIPTS / "save_features.py"),
        "--data_path", str(csv_path),
        "--save_path", str(npz_path),
        "--features_generator", generator,
        "--restart",
    ]
    _run(cmd, f"save_features({csv_path.name}, {generator})", manifest)
    return npz_path


def _resolve_vocab_inputs(args: argparse.Namespace) -> dict[str, Path | None] | None:
    """Returns {atom, bond, smiles}->Path|None when the user supplied vocab
    inputs (via --vocab-dir or --atom-vocab/--bond-vocab/--smiles-vocab),
    else None (signal to fall through to build_vocab).

    Conventional filenames inside --vocab-dir:
      pretrain_atom_vocab.{json,pkl}
      pretrain_bond_vocab.{json,pkl}
      pretrain_smiles_vocab.pkl
    """
    if args.vocab_dir:
        d = Path(args.vocab_dir).resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"--vocab-dir not found or not a directory: {d}")
        def _find(stem: str, exts: tuple[str, ...]) -> Path | None:
            for ext in exts:
                p = d / f"{stem}.{ext}"
                if p.is_file():
                    return p
            return None
        atom = _find(PRETRAIN_VOCAB_STEMS["atom"], ("json", "pkl"))
        bond = _find(PRETRAIN_VOCAB_STEMS["bond"], ("json", "pkl"))
        smiles = _find(PRETRAIN_VOCAB_STEMS["smiles"], ("pkl",))
        if atom is None and bond is None and smiles is None:
            stems = [PRETRAIN_VOCAB_STEMS[k] for k in ("atom", "bond", "smiles")]
            raise FileNotFoundError(
                f"--vocab-dir {d} contained no {{ {', '.join(stems) }}}.{{json,pkl}} "
                f"files. Expected at least {PRETRAIN_VOCAB_STEMS['atom']} + "
                f"{PRETRAIN_VOCAB_STEMS['bond']}."
            )
        return {"atom": atom, "bond": bond, "smiles": smiles}

    if args.atom_vocab or args.bond_vocab or args.smiles_vocab:
        return {
            "atom":   Path(args.atom_vocab).resolve()   if args.atom_vocab   else None,
            "bond":   Path(args.bond_vocab).resolve()   if args.bond_vocab   else None,
            "smiles": Path(args.smiles_vocab).resolve() if args.smiles_vocab else None,
        }

    return None


def _copy_provided_vocab(
    src: dict[str, Path | None], dst_dir: Path, dataset_name: str, manifest: dict[str, Any],
    force: bool,
) -> dict[str, Path]:
    """When the user supplies vocab files (use ckpt's vocab as-is),
    copy them into `<dst_dir>/<dataset_name>_<which>_vocab.<ext>` so the
    downstream pretrain command sees the conventional filenames.

    `src` is `{atom: Path|None, bond: Path|None, smiles: Path|None}`. The atom
    and bond entries must be both present or both absent (paired). smiles is
    optional (cmim/hybrid only).

    Returns the same dict of (resolved) destination paths.
    """
    import shutil
    if (src["atom"] is None) != (src["bond"] is None):
        raise ValueError(
            "vocab pass-through requires atom and bond vocab paths to be paired; "
            "got atom=" + str(src["atom"]) + ", bond=" + str(src["bond"])
        )
    out: dict[str, Path] = {}
    dst_dir.mkdir(parents=True, exist_ok=True)
    for which, path in src.items():
        if path is None:
            continue
        # Validate the source file IS a loadable KERMT vocab before copying.
        # Catches the "user pointed --smiles-vocab at a random pickle" case
        # early, with a clear error, instead of letting it surface as a cryptic
        # SMILESVocab.load_vocab failure at pretrain_ddp.py launch time.
        validate_vocab_file(path, kind=which)
        ext = path.suffix.lstrip(".")
        if which == "smiles":
            ext = "pkl"  # smiles vocab is always pickle
        dst = dst_dir / f"{dataset_name}_{which}_vocab.{ext}"
        if not force and _exists_nonempty(dst):
            _skipped(f"copy_vocab({which})", str(dst), manifest)
            out[which] = dst
            continue
        if force and dst.exists():
            dst.unlink()
        shutil.copy2(path, dst)
        manifest["steps"].append({
            "name": f"copy_vocab({which})",
            "src": str(path), "dst": str(dst), "ok": True,
            "duration_s": 0.0, "skipped_due_to_existing": False,
        })
        out[which] = dst
    return out


def _build_vocab(
    csv_path: Path, vocab_dir: Path, dataset_name: str, vocab_format: str,
    manifest: dict[str, Any], force: bool,
) -> dict[str, Path]:
    """Builds atom + bond (in --vocab-format) and smiles (always pickle) vocabs.
    Returns a dict of {atom, bond, smiles} -> Path."""
    suffix = "json" if vocab_format == "json" else "pkl"
    expected = {
        "atom": vocab_dir / f"{dataset_name}_atom_vocab.{suffix}",
        "bond": vocab_dir / f"{dataset_name}_bond_vocab.{suffix}",
        "smiles": vocab_dir / f"{dataset_name}_smiles_vocab.pkl",
    }
    if not force and all(_exists_nonempty(p) for p in expected.values()):
        _skipped(f"build_vocab({csv_path.name})", str(vocab_dir), manifest)
        return expected
    vocab_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for p in expected.values():
            if p.exists():
                p.unlink()
    cmd = [
        sys.executable, str(EXISTING_SCRIPTS / "build_vocab.py"),
        "--data_path", str(csv_path),
        "--vocab_save_folder", str(vocab_dir),
        "--dataset_name", dataset_name,
        "--vocab_format", vocab_format,
    ]
    _run(cmd, f"build_vocab({csv_path.name})", manifest)
    return expected


def _split_data(
    csv_path: Path, features_path: Path | None, sample_per_file: int, output_dir: Path,
    manifest: dict[str, Any], force: bool,
) -> Path:
    """Run split_data.py to produce shard dirs (graph/ + optionally feature/ + summary.txt)."""
    summary = output_dir / "summary.txt"
    if not force and _exists_nonempty(summary):
        _skipped(f"split_data({csv_path.name})", str(output_dir), manifest)
        return output_dir
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(EXISTING_SCRIPTS / "split_data.py"),
        "--data_path", str(csv_path),
        "--sample_per_file", str(sample_per_file),
        "--output_path", str(output_dir),
    ]
    if features_path is not None:
        cmd += ["--features_path", str(features_path)]
    _run(cmd, f"split_data({csv_path.name})", manifest)
    return output_dir


# ---------------------------------------------------------------------------
# Random splitter (used only when the user supplies a single CSV)
# ---------------------------------------------------------------------------

def _random_split_csv(
    src_csv: Path, dst_csvs: dict[str, Path], fractions: dict[str, float], seed: int,
    manifest: dict[str, Any], force: bool,
) -> None:
    """Shuffle src_csv and partition rows into dst_csvs by fractions.
    `dst_csvs` and `fractions` are dicts keyed by the split name (e.g. 'train', 'val').
    Sum of fractions must be 1.0 (within float tolerance). Writes each dst_csv with the
    same header as the input."""
    step = {
        "name": f"random_split({src_csv.name})",
        "seed": seed,
        "fractions": fractions,
        "ok": False,
        "duration_s": None,
        "skipped_due_to_existing": False,
        "row_counts": {},
    }
    if not force and all(_exists_nonempty(p) for p in dst_csvs.values()):
        step["skipped_due_to_existing"] = True
        step["ok"] = True
        manifest["steps"].append(step)
        return

    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0 (got {sum(fractions.values())})")

    t0 = time.time()
    df = pd.read_csv(src_csv).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    sizes: dict[str, int] = {}
    remaining = n
    split_names = list(fractions.keys())
    for name in split_names[:-1]:
        sizes[name] = int(round(fractions[name] * n))
        remaining -= sizes[name]
    sizes[split_names[-1]] = remaining

    start = 0
    for name in split_names:
        dst = dst_csvs[name]
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.iloc[start:start + sizes[name]].to_csv(dst, index=False)
        step["row_counts"][name] = sizes[name]
        start += sizes[name]

    step["duration_s"] = round(time.time() - t0, 2)
    step["ok"] = True
    manifest["steps"].append(step)


def _emit_random_split_warning(
    src_csv: Path, fractions: dict[str, float], seed: int, manifest: dict[str, Any]
) -> None:
    row_counts = manifest["steps"][-1].get("row_counts", {})
    n = sum(row_counts.values()) if row_counts else "?"
    lines = [
        f"WARNING: Auto-splitting {n} rows from {src_csv.name} into:",
    ]
    for name, frac in fractions.items():
        cnt = row_counts.get(name, "?")
        lines.append(f"  {name}: {cnt} rows ({frac * 100:.1f}%)")
    lines += [
        f"using random split with seed {seed}.",
        "",
        "This is a RANDOM split. For rigorous ADMET evaluation, scaffold-balanced",
        "(or other structure-aware) splits are strongly preferred — molecules with",
        "similar scaffolds can leak across splits and inflate apparent generalization.",
        "",
        "To use your own pre-computed splits instead, pass:",
        "    --train-csv <train.csv> --val-csv <val.csv> --test-csv <test.csv>",
        "",
        "To customize fractions:",
        "    --val-frac 0.15 --test-frac 0.15",
    ]
    warning = "\n".join(lines)
    print(warning, file=sys.stderr)
    manifest["warnings"].append(warning)


# ---------------------------------------------------------------------------
# Mode pipelines
# ---------------------------------------------------------------------------

def _prepare_embed(args, out: Path, manifest: dict[str, Any]) -> None:
    manifest["split_method"] = "n/a"
    if args.skip_clean:
        clean = Path(args.csv)
        manifest["steps"].append({"name": "clean_smiles", "skipped_by_flag": True, "ok": True})
    else:
        clean = _clean_smiles(Path(args.csv), out / "clean.csv", args.smiles_column, manifest, args.force)
    manifest["outputs"]["clean_csv"] = str(clean)


def _prepare_inference(args, out: Path, manifest: dict[str, Any]) -> None:
    manifest["split_method"] = "n/a"
    clean = _clean_smiles(Path(args.csv), out / "clean.csv", args.smiles_column, manifest, args.force)
    # Reduce to SMILES-only: downstream get_data/MoleculeDatapoint floats every
    # non-SMILES column, which crashes on non-numeric passthrough columns
    # (e.g. a 'split' label). Inference does not need target columns.
    _reduce_to_smiles_column(clean, args.smiles_column, manifest)
    manifest["outputs"]["clean_csv"] = str(clean)
    if args.skip_features:
        manifest["steps"].append({"name": "save_features", "skipped_by_flag": True, "ok": True})
        return
    generator = args.features_generator or DEFAULT_FEATURES_GENERATOR["inference"]
    npz = _save_features(clean, out / "clean.npz", generator, manifest, args.force)
    manifest["outputs"]["clean_npz"] = str(npz)


def _prepare_finetune(args, out: Path, manifest: dict[str, Any]) -> None:
    src_train = Path(args.csv)
    has_val = args.val_csv is not None
    has_test = args.test_csv is not None
    split_type = args.split_type

    if has_val and has_test:
        # User supplied explicit val + test CSVs: trust them, just clean + featurize.
        # split_type is irrelevant when val/test are given separately.
        manifest["split_method"] = "user_provided"
        clean_train = _clean_smiles(src_train, out / "clean_train.csv", args.smiles_column, manifest, args.force)
        clean_val = _clean_smiles(Path(args.val_csv), out / "clean_val.csv", args.smiles_column, manifest, args.force)
        clean_test = _clean_smiles(Path(args.test_csv), out / "clean_test.csv", args.smiles_column, manifest, args.force)
        manifest["outputs"]["clean_train_csv"] = str(clean_train)
        manifest["outputs"]["clean_val_csv"] = str(clean_val)
        manifest["outputs"]["clean_test_csv"] = str(clean_test)
        per_split = (("train", clean_train), ("val", clean_val), ("test", clean_test))
    elif has_val or has_test:
        raise ValueError(
            "for finetune mode, either provide BOTH --val-csv and --test-csv (user-provided splits) "
            "or NEITHER (run with --split-type {random|scaffold_balanced|index_predetermined}). "
            "Got one but not both."
        )
    elif split_type == "random":
        # Random auto-split — done here in prep so train.py gets ready-made CSVs.
        manifest["split_method"] = "random"
        manifest["split_seed"] = args.seed
        train_frac = max(0.0, 1.0 - args.val_frac - args.test_frac)
        manifest["split_fractions"] = {"train": train_frac, "val": args.val_frac, "test": args.test_frac}
        clean_full = _clean_smiles(src_train, out / "_clean_full.csv", args.smiles_column, manifest, args.force)
        dst = {
            "train": out / "clean_train.csv",
            "val": out / "clean_val.csv",
            "test": out / "clean_test.csv",
        }
        _random_split_csv(clean_full, dst, manifest["split_fractions"], args.seed, manifest, args.force)
        clean_train, clean_val, clean_test = dst["train"], dst["val"], dst["test"]
        manifest["outputs"]["clean_train_csv"] = str(clean_train)
        manifest["outputs"]["clean_val_csv"] = str(clean_val)
        manifest["outputs"]["clean_test_csv"] = str(clean_test)
        _emit_random_split_warning(src_train, manifest["split_fractions"], args.seed, manifest)
        per_split = (("train", clean_train), ("val", clean_val), ("test", clean_test))
    else:
        # Scaffold-balanced or index-predetermined: prep cleans + featurizes the full
        # CSV and defers actual splitting to task/train.py, which calls split_data
        # with the user-supplied seed and split_sizes.
        manifest["split_method"] = "deferred_to_runner"
        manifest["split_type"] = split_type
        manifest["split_seed"] = args.seed
        manifest["split_fractions"] = {
            "train": max(0.0, 1.0 - args.val_frac - args.test_frac),
            "val": args.val_frac,
            "test": args.test_frac,
        }
        clean_full = _clean_smiles(src_train, out / "clean_full.csv", args.smiles_column, manifest, args.force)
        manifest["outputs"]["clean_full_csv"] = str(clean_full)
        per_split = (("full", clean_full),)

    if args.skip_features:
        manifest["steps"].append({"name": "save_features", "skipped_by_flag": True, "ok": True})
        return
    generator = args.features_generator or DEFAULT_FEATURES_GENERATOR["finetune"]
    for split_name, csv in per_split:
        npz = _save_features(csv, csv.with_suffix(".npz"), generator, manifest, args.force)
        manifest["outputs"][f"clean_{split_name}_npz"] = str(npz)


def _prepare_pretrain(args, out: Path, manifest: dict[str, Any]) -> None:
    src_train = Path(args.csv)
    if args.val_csv is not None:
        manifest["split_method"] = "user_provided"
        clean_train = _clean_smiles(src_train, out / "clean_train.csv", args.smiles_column, manifest, args.force)
        clean_val = _clean_smiles(Path(args.val_csv), out / "clean_val.csv", args.smiles_column, manifest, args.force)
    else:
        manifest["split_method"] = "random"
        manifest["split_seed"] = args.seed
        train_frac = max(0.0, 1.0 - args.val_frac)
        manifest["split_fractions"] = {"train": train_frac, "val": args.val_frac}
        clean_full = _clean_smiles(src_train, out / "_clean_full.csv", args.smiles_column, manifest, args.force)
        dst = {"train": out / "clean_train.csv", "val": out / "clean_val.csv"}
        _random_split_csv(clean_full, dst, manifest["split_fractions"], args.seed, manifest, args.force)
        clean_train, clean_val = dst["train"], dst["val"]

    manifest["outputs"]["clean_train_csv"] = str(clean_train)
    manifest["outputs"]["clean_val_csv"] = str(clean_val)

    generator = args.features_generator or DEFAULT_FEATURES_GENERATOR["pretrain"]
    if args.skip_features:
        manifest["steps"].append({"name": "save_features", "skipped_by_flag": True, "ok": True})
        train_npz: Path | None = None
        val_npz: Path | None = None
    else:
        train_npz = _save_features(clean_train, out / "clean_train.npz", generator, manifest, args.force)
        val_npz = _save_features(clean_val, out / "clean_val.npz", generator, manifest, args.force)
        manifest["outputs"]["clean_train_npz"] = str(train_npz)
        manifest["outputs"]["clean_val_npz"] = str(val_npz)

    if args.skip_vocab:
        manifest["steps"].append({"name": "build_vocab", "skipped_by_flag": True, "ok": True})
        manifest["vocab_source"] = "skipped"
    else:
        # Resolve user-provided vocab paths from --vocab-dir or explicit flags.
        provided = _resolve_vocab_inputs(args)
        if provided:
            # Use the user-supplied (ckpt's) vocab as-is. Copy into the
            # conventional filenames the downstream pretrain command expects.
            vocabs = _copy_provided_vocab(provided, out, args.dataset_name, manifest, args.force)
            manifest["vocab_source"] = "user_provided"
        else:
            # Fall back to the existing build-from-corpus behavior. Used by
            # pretrain-from-scratch and by any continue case where the user
            # explicitly wants a fresh vocab (rare, usually wrong).
            vocabs = _build_vocab(clean_train, out, args.dataset_name, args.vocab_format, manifest, args.force)
            manifest["vocab_source"] = "built_fresh"
        if "atom" in vocabs:
            manifest["outputs"]["atom_vocab"] = str(vocabs["atom"])
        if "bond" in vocabs:
            manifest["outputs"]["bond_vocab"] = str(vocabs["bond"])
        if "smiles" in vocabs:
            manifest["outputs"]["smiles_vocab"] = str(vocabs["smiles"])

    if args.skip_split:
        manifest["steps"].append({"name": "split_data", "skipped_by_flag": True, "ok": True})
    else:
        train_dir = _split_data(clean_train, train_npz, args.sample_per_file, out / "train", manifest, args.force)
        val_dir = _split_data(clean_val, val_npz, args.sample_per_file, out / "val", manifest, args.force)
        manifest["outputs"]["train_dir"] = str(train_dir)
        manifest["outputs"]["val_dir"] = str(val_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "mode": args.mode,
        "input_csv": str(Path(args.csv).resolve()),
        "val_csv": str(Path(args.val_csv).resolve()) if args.val_csv else None,
        "test_csv": str(Path(args.test_csv).resolve()) if args.test_csv else None,
        "output_dir": str(out),
        "split_method": None,
        "steps": [],
        "outputs": {},
        "errors": [],
        "warnings": [],
    }
    try:
        if args.mode == "pretrain":
            _prepare_pretrain(args, out, manifest)
        elif args.mode == "finetune":
            _prepare_finetune(args, out, manifest)
        elif args.mode == "inference":
            _prepare_inference(args, out, manifest)
        elif args.mode == "embed":
            _prepare_embed(args, out, manifest)
        manifest["ok"] = True
    except Exception as exc:  # noqa: BLE001
        manifest["ok"] = False
        manifest["errors"].append(f"{type(exc).__name__}: {exc}")
    # Always write the manifest so partial-failure state is visible to the agent.
    (out / "prepare_data.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mode-dispatched data prep for the KERMT agent skills.")
    p.add_argument("--mode", required=True, choices=VALID_MODES)
    p.add_argument("--csv", required=True, help="Primary input CSV (train CSV for pretrain/finetune)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--val-csv", default=None, help="Optional separate val CSV (pretrain/finetune)")
    p.add_argument("--test-csv", default=None, help="Optional separate test CSV (finetune only)")
    p.add_argument("--val-frac", type=float, default=0.1, help="Auto-split val fraction (default 0.1)")
    p.add_argument("--test-frac", type=float, default=0.1, help="Auto-split test fraction (finetune only, default 0.1)")
    p.add_argument("--seed", type=int, default=0, help="Random split seed (default 0)")
    p.add_argument("--split-type", choices=["random", "scaffold_balanced", "index_predetermined"],
                   default="random",
                   help="(finetune only, when --val-csv/--test-csv are not given) how to split. "
                        "'random' splits in prep using --val-frac/--test-frac/--seed. "
                        "'scaffold_balanced' and 'index_predetermined' defer the actual split to the "
                        "runner (task/train.py invokes split_data with the appropriate algorithm "
                        "using the user-supplied seed); prep only cleans + featurizes the full CSV.")
    p.add_argument("--sample-per-file", type=int, default=100_000,
                   help="split_data shard size (pretrain only, default 100000)")
    p.add_argument("--vocab-format", choices=["json", "pkl"], default="json",
                   help="atom/bond vocab format (default json); smiles vocab is always pkl")
    # Vocab pass-through (pretrain mode): when continuing from a released ckpt,
    # pass its bundled vocab files in so we don't rebuild a mismatched vocab.
    p.add_argument("--vocab-dir", default=None,
                   help="(pretrain) directory containing pretrain_{atom,bond}_vocab.{json,pkl} "
                        "(+ pretrain_smiles_vocab.pkl for cmim/hybrid). When given, prepare_data "
                        "skips build_vocab and copies these files into the output dir under the "
                        "expected filenames. Used by kermt-continue-pretrain to bind the released "
                        "ckpt's vocab to the new corpus (the ckpt's vocab is authoritative).")
    p.add_argument("--atom-vocab", default=None,
                   help="(pretrain) explicit atom vocab path; pairs with --bond-vocab. Overrides "
                        "--vocab-dir's pretrain_atom_vocab.* discovery if both are given.")
    p.add_argument("--bond-vocab", default=None,
                   help="(pretrain) explicit bond vocab path; pairs with --atom-vocab.")
    p.add_argument("--smiles-vocab", default=None,
                   help="(pretrain, cmim/hybrid) explicit smiles vocab .pkl path. Optional for "
                        "vocab-only pretrain.")
    p.add_argument("--dataset-name", default="pretrain",
                   help="vocab filename prefix (default 'pretrain' so downstream pretrain commands "
                        "can reference pretrain_{atom,bond}_vocab.{json|pkl}, pretrain_smiles_vocab.pkl)")
    p.add_argument("--targets", nargs="+", default=None,
                   help="(finetune only) target column names; forwarded to the finetune runner via the manifest")
    p.add_argument("--features-generator", default=None,
                   help="Override the per-mode default (pretrain: fgtasklabel; finetune/inference: rdkit_2d_normalized)")
    p.add_argument("--smiles-column", type=int, default=None,
                   help="0-based column index of SMILES in the input CSV. "
                        "When omitted, auto-detected by header name "
                        "(prefers lowercase `smiles`; accepts case-insensitive "
                        "`SMILES`/`Smiles`). Pass explicitly to override.")
    p.add_argument("--force", action="store_true",
                   help="Re-run every step even if its outputs already exist")
    p.add_argument("--skip-clean", action="store_true", help="(embed mode) skip the cleaning step")
    p.add_argument("--skip-features", action="store_true", help="Skip feature generation")
    p.add_argument("--skip-vocab", action="store_true", help="(pretrain) skip vocab build")
    p.add_argument("--skip-split", action="store_true", help="(pretrain) skip shard split")
    args = p.parse_args(argv)

    # Forward --targets through the manifest so the finetune runner can see them.
    if args.mode == "finetune" and args.targets:
        pass  # captured in manifest below

    # Resolve the SMILES column index (auto-detect from header when the user
    # didn't pass --smiles-column). This is the only point where args.csv is
    # touched before downstream _clean_smiles calls fan it out.
    try:
        resolved_smiles_col = _resolve_smiles_column(Path(args.csv), args.smiles_column)
    except ValueError as exc:
        err_manifest = {
            "ok": False,
            "mode": args.mode,
            "errors": [f"smiles-column resolution failed: {exc}"],
        }
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "prepare_data.json").write_text(json.dumps(err_manifest, indent=2))
        print(json.dumps(err_manifest, indent=2))
        return 1
    if args.smiles_column is None:
        print(f"[prepare_data] auto-detected --smiles-column {resolved_smiles_col} "
              f"from {Path(args.csv).name} header", file=sys.stderr)
    args.smiles_column = resolved_smiles_col

    try:
        manifest = prepare(args)
    except Exception as exc:  # noqa: BLE001
        print(traceback.format_exc(), file=sys.stderr)
        print(json.dumps({"ok": False, "errors": [f"unhandled: {type(exc).__name__}: {exc}"]}, indent=2))
        return 1
    if args.targets:
        manifest["targets"] = list(args.targets)
    # Record the resolved SMILES column so the manifest is self-describing.
    manifest["smiles_column"] = args.smiles_column
    (Path(args.out) / "prepare_data.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0 if manifest.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
