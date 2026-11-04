# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bundled and adapted from the Apache-2.0 NVIDIA TAO Tutorials PAS DEFT utilities so the
# customer workflow does not depend on an external source checkout.

"""Visualization utilities for the CLIP DEFT pipeline.

Computes embeddings for weak-query samples, mined samples, and previous
training data, then produces a t-SNE scatter plot that overlays all three
categories with distinct colors.

Note: ``create_image_embeddings_task``, ``create_text_embeddings_task``, and
``create_video_embeddings_task`` from the Kratos pipeline are omitted — they
build Kubeflow tasks and are not part of this host utility. Prepare the
equivalent ``viz_*_embed`` action with ``run_deft_action.py`` and dispatch its
platform-neutral request through the selected TAO platform's four verbs. The
immutable action contract supplies the ``embedding image_embeddings`` argv,
mount aliases, image, and fresh-output checks; callers must not reconstruct a
platform-specific launch here.
"""


def prepare_clip_images_for_embedding(
    input_parquet: str,
    output_parquet_path: str,
    image_dir: str = "",
) -> str:
    """Build a unique image-filepath parquet ready for image embedding.

    Works on any CLIP-side parquet that carries a ``filepath`` (or
    ``unique_name``) column: the gap-analysis ``kpi_gaps.parquet``, the k-NN
    ``mined_samples.parquet``, or a previous-training-data pool parquet.  Those
    tables hold one row per (image, caption) query pair, so the same image
    appears many times; embedding them as-is re-encodes duplicates and chokes on
    rows whose ``filepath`` is empty.  This drops blanks, deduplicates on
    ``filepath``, and keeps the metadata columns t-SNE and the contact sheets
    use.

    Args:
        input_parquet:       Parquet with a ``filepath`` and/or ``unique_name``
                             column.
        output_parquet_path: Where to write the deduplicated parquet.
        image_dir:           Image root used to resolve ``unique_name`` when a
                             row has no ``filepath``.  Optional.

    Returns:
        The path to the written parquet file.
    """
    import os

    import pandas as pd

    keep_cols = (
        "filepath", "unique_name", "image_path", "text", "caption",
        "query_type", "dataset", "label", "weak_attribute",
    )

    df = pd.read_parquet(input_parquet)

    if "filepath" not in df.columns and "unique_name" not in df.columns:
        raise ValueError(
            f"{input_parquet} has neither a 'filepath' nor a 'unique_name' "
            f"column; found {list(df.columns)}"
        )

    out = df[[c for c in keep_cols if c in df.columns]].copy()

    if "filepath" in out.columns:
        out["filepath"] = out["filepath"].fillna("").astype(str).str.strip()
    else:
        out["filepath"] = ""

    # Fall back to image_dir/unique_name for rows the upstream step left blank.
    if image_dir and "unique_name" in out.columns:
        names = out["unique_name"].fillna("").astype(str).str.strip()
        missing = (out["filepath"] == "") & (names != "")
        if missing.any():
            out.loc[missing, "filepath"] = names[missing].map(
                lambda n: os.path.abspath(os.path.join(image_dir, n)),
            )

    total = len(out)
    out = out[out["filepath"] != ""]
    dropped = total - len(out)
    out = out.drop_duplicates(subset=["filepath"]).reset_index(drop=True)

    out_dir = os.path.dirname(output_parquet_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_parquet(output_parquet_path, index=False)

    print(
        f"Prepared {len(out)} unique images from {total} rows in "
        f"{input_parquet} ({dropped} rows had no filepath) -> "
        f"{output_parquet_path}"
    )
    return output_parquet_path


def export_clip_sample_contact_sheets(
    weak_parquet: str,
    mined_parquet: str,
    output_dir: str,
    source_pairs_file: str = "",
    max_samples_per_group: int = 12,
    max_total_samples: int = 96,
    tile_size: int = 192,
    host_path_map=None,
) -> str:
    """Export galleries and contact sheets with explicit mount-prefix aliases."""
    import base64
    import html
    import io
    import json
    import os
    import textwrap

    import pandas as pd

    def _iter_json_records(path):
        if not path or not os.path.isfile(path):
            return
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

    def _sample(df):
        if df.empty:
            return df
        group_cols = [c for c in ("dataset", "query_type") if c in df.columns]
        if not group_cols:
            return df.head(max_total_samples).copy()
        pieces = []
        for _, group in df.groupby(group_cols, dropna=False, sort=True):
            pieces.append(group.head(max_samples_per_group))
        out = pd.concat(pieces, ignore_index=True) if pieces else df.head(0)
        return out.head(max_total_samples).copy()

    path_prefixes = sorted(
        (
            (str(container).rstrip("/"), str(host).rstrip("/"))
            for container, host in (host_path_map or {}).items()
        ),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )

    def _mount_aliases(path):
        path = str(path or "")
        aliases = [path]
        for container, host in path_prefixes:
            if path == container or path.startswith(container + "/"):
                aliases.append(host + path[len(container):])
            if path == host or path.startswith(host + "/"):
                aliases.append(container + path[len(host):])
        return aliases

    def _path_candidates(path, image_path="", unique_name=""):
        path = str(path or "")
        image_path = str(image_path or "")
        unique_name = str(unique_name or "")
        raw_candidates = [path]
        if unique_name:
            raw_candidates.append(os.path.join(os.path.dirname(path), unique_name))
        if image_path:
            raw_candidates.append(image_path)
            base_dir = os.path.dirname(path)
            dataset_root = os.path.dirname(base_dir)
            raw_candidates.extend([
                os.path.join(base_dir, image_path),
                os.path.join(dataset_root, image_path),
                os.path.join(dataset_root, "images", image_path),
            ])
        candidates = []
        for raw in raw_candidates:
            candidates.extend(_mount_aliases(raw))
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                yield candidate

    def _resolve_path(path, image_path="", unique_name=""):
        for candidate in _path_candidates(path, image_path, unique_name):
            if os.path.isfile(candidate):
                return candidate
            if os.path.islink(candidate):
                target = os.readlink(candidate)
                if not os.path.isabs(target):
                    target = os.path.abspath(
                        os.path.join(os.path.dirname(candidate), target)
                    )
                for alias in _mount_aliases(target):
                    if os.path.isfile(alias):
                        return alias
        return ""

    def _add_resolution_columns(df):
        df = df.copy()
        if df.empty:
            df["resolved_filepath"] = []
            df["image_exists"] = []
            return df
        df["resolved_filepath"] = df.apply(
            lambda row: _resolve_path(
                row.get("filepath", ""),
                row.get("image_path", ""),
                row.get("unique_name", ""),
            ),
            axis=1,
        )
        df["image_exists"] = df["resolved_filepath"].astype(bool)
        return df

    def _thumbnail_data_uri(row, max_size=512):
        path = str(row.get("filepath") or "")
        resolved = str(row.get("resolved_filepath") or "") or _resolve_path(
            path,
            row.get("image_path", ""),
            row.get("unique_name", ""),
        )
        if not resolved:
            return ""
        try:
            from PIL import Image
            img = Image.open(resolved).convert("RGB")
            img.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85, optimize=True)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception as exc:
            print(f"Could not embed thumbnail for {path}: {exc}")
            return ""

    def _write_html(df, path, title):
        cards = []
        for _, row in df.iterrows():
            fp = str(row.get("filepath") or "")
            img_src = _thumbnail_data_uri(row)
            label = " / ".join(
                str(row.get(c) or "") for c in ("dataset", "query_type")
                if c in row and str(row.get(c) or "")
            )
            caption = str(row.get("caption") or row.get("text") or "")
            if img_src:
                media = f"<img src='{img_src}' alt='sample'>"
            else:
                media = (
                    "<div class='missing'>missing image<br>"
                    f"{html.escape(fp)}</div>"
                )
            cards.append(
                "<div class='card'>"
                f"{media}"
                f"<div class='meta'>{html.escape(label)}</div>"
                f"<div class='name'>{html.escape(os.path.basename(fp))}</div>"
                f"<div class='caption'>{html.escape(caption[:500])}</div>"
                "</div>"
            )
        body = "\n".join(cards) if cards else "<p>No samples.</p>"
        doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }}
.card {{ border: 1px solid #ddd; padding: 8px; }}
.card img {{ width: 100%; height: 220px; object-fit: contain; background: #f6f6f6; }}
.missing {{ width: 100%; height: 220px; display: flex; align-items: center; justify-content: center; text-align: center; background: #eee; color: #333; font-size: 12px; word-break: break-all; }}
.meta {{ font-weight: 700; margin-top: 6px; }}
.name {{ font-size: 12px; color: #555; word-break: break-all; }}
.caption {{ font-size: 12px; margin-top: 6px; white-space: pre-wrap; }}
</style></head><body><h1>{html.escape(title)}</h1><div class="grid">{body}</div></body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc)

    def _write_png(df, path, title):
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            print(f"Pillow unavailable; skipping PNG contact sheet: {exc}")
            return False
        if df.empty:
            return False
        cols = 4
        label_h = 96
        rows = (len(df) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile_size, rows * (tile_size + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for idx, (_, row) in enumerate(df.iterrows()):
            x = (idx % cols) * tile_size
            y = (idx // cols) * (tile_size + label_h)
            fp = str(row.get("filepath") or "")
            resolved = str(row.get("resolved_filepath") or "") or _resolve_path(fp)
            try:
                img = Image.open(resolved).convert("RGB")
                img.thumbnail((tile_size, tile_size))
                ix = x + (tile_size - img.width) // 2
                iy = y + (tile_size - img.height) // 2
                sheet.paste(img, (ix, iy))
            except Exception:
                draw.rectangle([x, y, x + tile_size - 1, y + tile_size - 1], fill="#eeeeee", outline="#bbbbbb")
                draw.text((x + 8, y + 8), "missing image", fill="black", font=font)
            label = " / ".join(
                str(row.get(c) or "") for c in ("dataset", "query_type")
                if c in row and str(row.get(c) or "")
            )
            caption = str(row.get("caption") or row.get("text") or os.path.basename(fp))
            wrapped = textwrap.wrap(f"{label} {caption}", width=34)[:5]
            draw.text((x + 6, y + tile_size + 6), "\n".join(wrapped), fill="black", font=font)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sheet.save(path)
        print(f"Saved {title} contact sheet to {path}")
        return True

    os.makedirs(output_dir, exist_ok=True)
    weak = pd.read_parquet(weak_parquet)
    mined = pd.read_parquet(mined_parquet)

    if "filepath" not in weak.columns:
        weak = pd.DataFrame(columns=["filepath"])
    if "filepath" not in mined.columns:
        mined = pd.DataFrame(columns=["filepath"])

    pair_by_name = {}
    for row in _iter_json_records(source_pairs_file) or []:
        name = str(row.get("unique_name") or "").strip()
        if name and name not in pair_by_name:
            pair_by_name[name] = row

    mined = mined.copy()
    mined["unique_name"] = mined["filepath"].map(os.path.basename)
    if pair_by_name:
        mined["dataset"] = mined["unique_name"].map(
            lambda n: str(pair_by_name.get(n, {}).get("dataset") or "")
        )
        mined["query_type"] = mined["unique_name"].map(
            lambda n: str(pair_by_name.get(n, {}).get("query_type") or "")
        )
        mined["caption"] = mined["unique_name"].map(
            lambda n: str(pair_by_name.get(n, {}).get("caption") or "")
        )
        mined["image_path"] = mined["unique_name"].map(
            lambda n: str(pair_by_name.get(n, {}).get("image_path") or "")
        )

    weak_sample = _add_resolution_columns(_sample(weak))
    mined_sample = _add_resolution_columns(
        _sample(mined.drop_duplicates(subset=["filepath"]))
    )

    weak_csv = os.path.join(output_dir, "weak_samples_visual.csv")
    mined_csv = os.path.join(output_dir, "mined_samples_visual.csv")
    weak_html = os.path.join(output_dir, "weak_samples_gallery.html")
    mined_html = os.path.join(output_dir, "mined_samples_gallery.html")
    weak_png = os.path.join(output_dir, "weak_samples_contact_sheet.png")
    mined_png = os.path.join(output_dir, "mined_samples_contact_sheet.png")

    weak_sample.to_csv(weak_csv, index=False)
    mined_sample.to_csv(mined_csv, index=False)
    _write_html(weak_sample, weak_html, "Weak PAS query samples")
    _write_html(mined_sample, mined_html, "Mined augmented PAS samples")
    weak_png_written = _write_png(weak_sample, weak_png, "weak samples")
    mined_png_written = _write_png(mined_sample, mined_png, "mined samples")

    summary_lines = [
        "PAS visual sample export",
        f"Weak sample rows: {len(weak_sample)} -> {weak_csv}",
        f"Weak image files resolved: {int(weak_sample['image_exists'].sum())}/{len(weak_sample)}",
        f"Mined sample rows: {len(mined_sample)} -> {mined_csv}",
        f"Mined image files resolved: {int(mined_sample['image_exists'].sum())}/{len(mined_sample)}",
        f"Weak gallery: {weak_html}",
        f"Mined gallery: {mined_html}",
        f"Weak PNG: {weak_png if weak_png_written else '(not written)'}",
        f"Mined PNG: {mined_png if mined_png_written else '(not written)'}",
    ]
    summary_path = os.path.join(output_dir, "sample_export_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    return output_dir


def prepare_prev_clip_data_for_embedding(
    prev_train_config_yaml: str,
    output_parquet_path: str,
    host_path_map=None,
) -> str:
    """Collect prior CLIP training data into a filepaths parquet.

    ``dataset.train.datasets`` in the prior train config is the authoritative
    record of what the last iteration trained on — under ``continual_dataset``
    it accumulates one entry per iteration, so walking it captures the whole
    history without reconstructing it here.

    That config is written for the TAO container, so its paths are
    container-absolute (``/results/...``).  When this runs on the host instead
    of inside the container, pass ``host_path_map`` to translate those prefixes
    for reading.  Emitted ``filepath`` values keep the config's own roots, since
    the parquet is consumed by the image-embedding container.

    Args:
        prev_train_config_yaml: Path to the CLIP train config YAML
                                from the previous training step.
        output_parquet_path:    Where to write the union parquet.
        host_path_map:          Container-prefix -> host-prefix mapping applied
                                when opening ``image_list_file``, e.g.
                                ``{"/results": "/abs/host/results"}``.  Omit
                                when running inside the container.

    Returns:
        Path to ``output_parquet_path``.
    """
    import os

    import pandas as pd
    import yaml

    path_map = sorted(
        (host_path_map or {}).items(), key=lambda kv: len(kv[0]), reverse=True,
    )

    def _to_host(path):
        """Rewrite a container path to its host equivalent, if mapped."""
        for container_root, host_root in path_map:
            if path == container_root or path.startswith(
                container_root.rstrip("/") + "/"
            ):
                return os.path.join(
                    host_root, os.path.relpath(path, container_root),
                )
        return path

    entries = []
    if prev_train_config_yaml and os.path.isfile(prev_train_config_yaml):
        with open(prev_train_config_yaml, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        dataset_cfg = config_data.get("dataset") or {}
        train_data_cfg = dataset_cfg.get("train") or {}
        entries = train_data_cfg.get("datasets") or []
        if not entries:
            entries = (config_data.get("train") or {}).get("datasets") or []
    else:
        print(
            f"No prior train config at {prev_train_config_yaml!r}; "
            "emitting empty previous-data parquet"
        )

    filepaths = []
    for idx, entry in enumerate(entries):
        entry_image_dir = entry.get("image_dir", "")
        entry_image_list_file = entry.get("image_list_file", "")
        if not entry_image_dir or not entry_image_list_file:
            print(
                f"train.datasets[{idx}] missing image_dir or "
                "image_list_file; skipping"
            )
            continue
        # Unfilled template entries carry "???" rather than a real path.
        if "???" in (entry_image_dir, entry_image_list_file):
            continue
        host_image_list_file = _to_host(entry_image_list_file)
        if not os.path.isfile(host_image_list_file):
            print(
                f"image_list_file {entry_image_list_file} from "
                f"train.datasets[{idx}] missing "
                f"(looked in {host_image_list_file}); skipping"
            )
            continue
        before = len(filepaths)
        with open(host_image_list_file, "r", encoding="utf-8") as lf:
            for line in lf:
                name = line.strip()
                if name:
                    filepaths.append(os.path.join(entry_image_dir, name))
        print(
            f"train.datasets[{idx}]: {len(filepaths) - before} images from "
            f"{entry_image_list_file}"
        )

    filepaths = sorted(set(filepaths))

    out_df = pd.DataFrame({"filepath": filepaths})
    out_dir = os.path.dirname(output_parquet_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_parquet(output_parquet_path, index=False)
    print(
        f"Prepared {len(filepaths)} previous CLIP training samples -> "
        f"{output_parquet_path}"
    )
    return output_parquet_path


def create_tsne_visualization(
    weak_embeddings_dir: str,
    augmented_embeddings_dir: str,
    previous_embeddings_dir: str,
    output_plot_path: str,
) -> str:
    """Create a t-SNE scatter plot of weak, augmented, and previous data.

    Args:
        weak_embeddings_dir:      Directory containing ``embeddings.parquet``
                                  for weak samples.
        augmented_embeddings_dir: Directory containing one or more
                                  ``*_embeddings.parquet`` files.
        previous_embeddings_dir:  Directory containing ``embeddings.parquet``
                                  for previous training data.
        output_plot_path:         File path for the output PNG.

    Returns:
        The path to the saved plot, or empty string if no data.
    """
    import glob
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.manifold import TSNE

    def _read_embedding_parquet(path):
        if not os.path.isfile(path):
            return None
        df = pd.read_parquet(path)
        if df.empty or "embedding" not in df.columns:
            return None
        return np.stack(df["embedding"].values)

    def _load_embeddings_from_dir(emb_dir, pattern="*.parquet"):
        if not os.path.isdir(emb_dir):
            print(f"No embeddings dir at {emb_dir}, skipping")
            return np.empty(0), []
        files = sorted(glob.glob(os.path.join(emb_dir, pattern)))
        arrays, sources = [], []
        for fp in files:
            embs = _read_embedding_parquet(fp)
            if embs is not None:
                fname = os.path.basename(fp)
                if fname.endswith("_embeddings.parquet"):
                    source = fname[: -len("_embeddings.parquet")]
                else:
                    source = fname[: -len(".parquet")]
                arrays.append(embs)
                sources.extend([source] * len(embs))
                print(f"  loaded {len(embs)} embeddings from {fname}")
        if not arrays:
            return np.empty(0), []
        return np.concatenate(arrays, axis=0), sources

    weak_embs, _ = _load_embeddings_from_dir(weak_embeddings_dir)
    aug_embs, aug_sources = _load_embeddings_from_dir(augmented_embeddings_dir)
    prev_embs, _ = _load_embeddings_from_dir(previous_embeddings_dir)

    categories = [
        ("Weak Samples", weak_embs, None),
        ("Augmented Samples", aug_embs, aug_sources),
        ("Previous Training Data", prev_embs, None),
    ]
    arrays, labels, sources = [], [], []
    for name, embs, srcs in categories:
        if embs.size > 0:
            arrays.append(embs)
            labels.extend([name] * len(embs))
            sources.extend(srcs if srcs else [""] * len(embs))

    if not arrays:
        print("No embeddings found — skipping plot")
        return ""

    all_embs = np.concatenate(arrays, axis=0)
    perplexity = min(30, max(1, len(all_embs) - 1))
    coords = TSNE(
        n_components=2, random_state=42, perplexity=perplexity,
    ).fit_transform(all_embs)

    labels_arr = np.array(labels)
    sources_arr = np.array(sources)
    color_map = {
        "Weak Samples": "#e74c3c",
        "Augmented Samples": "#2ecc71",
        "Previous Training Data": "#3498db",
    }
    aug_marker_map = {"mined": "^", "omniverse": "s"}
    fallback_markers = ["D", "P", "X", "*", "v", "<", ">"]

    draw_order = (
        "Previous Training Data",
        "Weak Samples",
        "Augmented Samples",
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    for category in draw_order:
        mask = labels_arr == category
        if not mask.any():
            continue
        if category == "Augmented Samples":
            unique_sources = sorted(set(sources_arr[mask].tolist()))
            for i, source in enumerate(unique_sources):
                sub_mask = mask & (sources_arr == source)
                marker = aug_marker_map.get(
                    source, fallback_markers[i % len(fallback_markers)],
                )
                label = (
                    f"{category} ({source}, n={sub_mask.sum()})"
                    if source else f"{category} (n={sub_mask.sum()})"
                )
                ax.scatter(
                    coords[sub_mask, 0], coords[sub_mask, 1],
                    c=color_map[category],
                    marker=marker,
                    label=label,
                    alpha=0.6, s=30,
                    edgecolors="white", linewidths=0.5,
                )
        else:
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color_map[category],
                label=f"{category} (n={mask.sum()})",
                alpha=0.6, s=30,
                edgecolors="white", linewidths=0.5,
            )

    ax.legend(fontsize=11, loc="best")
    ax.set_title("t-SNE: Weak vs Augmented vs Previous Data", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    fig.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"t-SNE plot saved to {output_plot_path}")
    return output_plot_path
