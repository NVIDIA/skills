# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for selective_download.

The parse path (extract_keys) is the silent-sample-drop risk, so it is tested
exhaustively across formats, ordering, dedup, nulls, list cells, and the
missing-column error. The download path uses an injected fake S3 client — no
network, no moto.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import selective_download as sd  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def records():
    # 4 rows, with a duplicate video and a null, to exercise dedup + null drop
    return [
        {"video": "clips/a.mp4", "caption": "x"},
        {"video": "clips/b.mp4", "caption": "y"},
        {"video": "clips/a.mp4", "caption": "z"},   # duplicate
        {"video": None, "caption": "w"},            # null -> dropped
    ]


def _write(tmp_path, records, fmt):
    df = pd.DataFrame(records)
    p = tmp_path / f"ann.{ 'jsonl' if fmt=='jsonl' else fmt }"
    if fmt == "parquet":
        df.to_parquet(p)
    elif fmt == "csv":
        df.to_csv(p, index=False)
    elif fmt == "jsonl":
        df.to_json(p, orient="records", lines=True)
    elif fmt == "json":
        df.to_json(p, orient="records")
    return p


class FakeS3:
    """Minimal stand-in for a boto3 S3 client (positional download_file)."""

    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.downloaded: list[str] = []

    def download_file(self, bucket, key, filename):
        if key not in self.objects:
            from botocore.exceptions import ClientError
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject"
            )
        Path(filename).write_bytes(self.objects[key])
        self.downloaded.append(key)


# --------------------------------------------------------------------------- #
# extract_keys — parse correctness
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fmt", ["parquet", "jsonl", "json", "csv"])
def test_extract_keys_roundtrip_all_formats(tmp_path, records, fmt):
    p = _write(tmp_path, records, fmt)
    got = sd.extract_keys(p, ["video"])
    assert got == ["clips/a.mp4", "clips/b.mp4"]  # dedup + order + null dropped


def test_extract_keys_format_inference(tmp_path, records):
    p = _write(tmp_path, records, "parquet")
    assert sd.infer_format(p) == "parquet"
    assert sd.extract_keys(p, ["video"], fmt=None) == ["clips/a.mp4", "clips/b.mp4"]


def test_extract_keys_dedup_preserves_first_seen_order(tmp_path):
    recs = [{"f": "z"}, {"f": "a"}, {"f": "z"}, {"f": "m"}, {"f": "a"}]
    p = _write(tmp_path, recs, "jsonl")
    assert sd.extract_keys(p, ["f"]) == ["z", "a", "m"]


def test_extract_keys_multiple_columns_in_key_order(tmp_path):
    recs = [{"image": "i1.png", "mask": "m1.png"}, {"image": "i2.png", "mask": "m2.png"}]
    p = _write(tmp_path, recs, "jsonl")
    assert sd.extract_keys(p, ["image", "mask"]) == ["i1.png", "i2.png", "m1.png", "m2.png"]


def test_extract_keys_list_valued_cells_flatten(tmp_path):
    recs = [{"frames": ["f1.jpg", "f2.jpg"]}, {"frames": ["f2.jpg", "f3.jpg"]}]
    p = _write(tmp_path, recs, "jsonl")
    assert sd.extract_keys(p, ["frames"]) == ["f1.jpg", "f2.jpg", "f3.jpg"]


def test_extract_keys_missing_column_raises_with_available(tmp_path, records):
    p = _write(tmp_path, records, "parquet")
    with pytest.raises(KeyError) as exc:
        sd.extract_keys(p, ["does_not_exist"])
    assert "video" in str(exc.value)  # lists what IS available


def test_extract_keys_empty_keys_raises(tmp_path, records):
    p = _write(tmp_path, records, "parquet")
    with pytest.raises(ValueError):
        sd.extract_keys(p, [])


def test_infer_format_unknown_extension_raises(tmp_path):
    p = tmp_path / "ann.weird"
    p.write_text("{}")
    with pytest.raises(ValueError):
        sd.infer_format(p)


# --------------------------------------------------------------------------- #
# download_selective — only-referenced, idempotency, fail-loud, nested dirs
# --------------------------------------------------------------------------- #

def test_download_only_referenced_files(tmp_path):
    objects = {
        "src/clips/a.mp4": b"a",
        "src/clips/b.mp4": b"b",
        "src/clips/UNREFERENCED.mp4": b"nope",  # present in bucket, not requested
    }
    s3 = FakeS3(objects)
    rel = ["clips/a.mp4", "clips/b.mp4"]
    out = sd.download_selective(s3, "bkt", "src", rel, tmp_path)
    assert len(out) == len(rel)                       # sample-count invariant
    assert sorted(s3.downloaded) == ["src/clips/a.mp4", "src/clips/b.mp4"]
    assert "src/clips/UNREFERENCED.mp4" not in s3.downloaded


def test_download_creates_nested_dirs_and_content(tmp_path):
    s3 = FakeS3({"p/deep/nested/x.bin": b"payload"})
    out = sd.download_selective(s3, "bkt", "p", ["deep/nested/x.bin"], tmp_path)
    assert (tmp_path / "deep/nested/x.bin").read_bytes() == b"payload"
    assert out == [str(tmp_path / "deep/nested/x.bin")]


def test_download_idempotent_skips_existing_nonempty(tmp_path):
    s3 = FakeS3({"src/a.mp4": b"a"})
    (tmp_path / "a.mp4").write_bytes(b"already")
    out = sd.download_selective(s3, "bkt", "src", ["a.mp4"], tmp_path)
    assert out == [str(tmp_path / "a.mp4")]
    assert s3.downloaded == []                         # not re-fetched
    assert (tmp_path / "a.mp4").read_bytes() == b"already"


def test_download_missing_object_fails_loud(tmp_path):
    s3 = FakeS3({"src/a.mp4": b"a"})
    with pytest.raises(RuntimeError) as exc:
        sd.download_selective(s3, "bkt", "src", ["a.mp4", "MISSING.mp4"], tmp_path)
    assert "1 of 2" in str(exc.value)
    assert "MISSING.mp4" in str(exc.value)


def test_download_empty_src_prefix_uses_bare_keys(tmp_path):
    s3 = FakeS3({"a.mp4": b"a"})
    out = sd.download_selective(s3, "bkt", "", ["a.mp4"], tmp_path)
    assert s3.downloaded == ["a.mp4"]
    assert out == [str(tmp_path / "a.mp4")]


# --------------------------------------------------------------------------- #
# end-to-end: parse then stage
# --------------------------------------------------------------------------- #

def test_extract_then_download_end_to_end(tmp_path, records):
    ann = _write(tmp_path, records, "jsonl")
    dest = tmp_path / "stage"
    s3 = FakeS3({"data/clips/a.mp4": b"a", "data/clips/b.mp4": b"b"})
    rel = sd.extract_keys(ann, ["video"])
    out = sd.download_selective(s3, "bkt", "data", rel, dest)
    # exactly the 2 unique, non-null referenced files staged
    assert len(out) == 2
    assert (dest / "clips/a.mp4").exists() and (dest / "clips/b.mp4").exists()
