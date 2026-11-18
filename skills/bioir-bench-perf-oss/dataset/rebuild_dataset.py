#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""rebuild_dataset.py — rebuild the benchmark dataset from public sources.

The published benchmark compares BioNeMo-IR against OSS baselines. Those numbers
only mean something if you can re-run them on the inputs we measured, so this
script reconstructs that input tree on your machine instead of us shipping it:

    ground_truth/*.cif   RCSB downloads          (wwPDB, CC0 1.0)
    templates/*.cif      RCSB downloads          (wwPDB, CC0 1.0)
    msa/*.a3m            NVIDIA MSA Search NIM   (UniRef, CC BY 4.0)

Everything it needs is in the three files shipped beside it: `spec_full.json`
and `spec_monomer.json` carry the sequences and the PDB entry each input is,
and `MANIFEST.json` carries the sha256 and size of every file the reference
build produced.

STRUCTURES ARE VERIFIED, ALIGNMENTS ARE RECORDED
------------------------------------------------
The two halves have different reproducibility guarantees and the script does not
pretend otherwise.

A structure is a fixed deposition, so a download either matches MANIFEST or it
does not, and a mismatch is checked at the coordinate level before it is called
a problem: RCSB re-releases entries for metadata reasons alone (4KL8 gained
revision 3.2 on 2026-08-12 — 11 KB of new annotation, all 14237 atoms
unchanged), and failing that build would be wrong. So a hash mismatch whose
ATOM/HETATM records still agree is reported as `revised` and accepted.

A mismatch that moves an atom is a new major version (7PV5 became 3.0 on
2026-09-02 when wwPDB remediated its gold-modified cysteine), and it would move
every lDDT for that target. RCSB serves only the newest version, so for an entry
file the script then asks the wwPDB versioned archive, which keeps the newest
minor version of every major version. A copy whose ATOM/HETATM records match is
reported as `archived` and used; none matching aborts. Assembly files have no
archived copies, so a coordinate change to one still aborts.

An alignment is a search result. It is reproducible for a fixed database version
and a fixed server, and the NIM pins its databases in the response
(`Uniref30_2302`), but we cannot promise the server behind the endpoint is the
same one that produced MANIFEST. So alignments are hashed, written to
BUILD.json, and diffed against MANIFEST for information. `--strict` promotes
that diff to an error, which is the right setting for CI that must reproduce a
specific published run.

    python3 rebuild_dataset.py --verify                 # check what is on disk
    python3 rebuild_dataset.py --structures-only        # no API key needed
    python3 rebuild_dataset.py                          # full rebuild
    python3 rebuild_dataset.py --only 7R1L,5SBJ --jobs 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import gzip
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from email.message import Message
from http.client import HTTPResponse
from pathlib import Path
from typing import BinaryIO, NoReturn

HERE = Path(__file__).resolve().parent
SPECS = ("spec_full.json", "spec_monomer.json")

RCSB_ENTRY = "https://files.rcsb.org/download/{pdb}.cif"
RCSB_ASSEMBLY = "https://files.rcsb.org/download/{pdb}-assembly{n}.cif"
WWPDB_VERSIONED = "https://files-versioned.wwpdb.org/pdb_versioned/data/entries/{mid}/pdb_0000{pdb}/"

MSA_BASE = "https://health.api.nvidia.com/v1/biology/colabfold/msa-search"
NVCF_STATUS = "https://health.api.nvidia.com/v1/status/{reqid}"

# The alignments we publish are UniRef-only, by design: it lets us name one
# upstream database and one licence for every distributed sequence. Asking the
# NIM for the environmental set would produce a deeper MSA and a different
# provenance story, so the database list is not a flag.
DATABASES = ["Uniref30_2302"]

# Depth ceilings of the HOSTED endpoint, measured, not chosen — and the two
# endpoints enforce them differently:
#
#   /predict         silently returns at most 101 rows (100 hits + the query) no
#                    matter what `max_msa_sequences`, `iterations` or `e_value`
#                    ask for. Nothing in the response says it truncated.
#   /paired/predict  rejects `max_msa_sequences > 500` with a 422, so at least
#                    it tells you.
#
# `max_msa_sequences` counts the query, and sending it can only LOWER the result
# — so neither is sent by default; passing one would silently cost a row.
#
# The reference build carries up to 10483 rows unpaired and 44638 paired, so a
# rebuild here is 50-100x shallower and will NOT reproduce the published
# accuracy (see `check_depth`). Note this is truncation, not a worse search:
# where the reference is below the ceiling the NIM finds MORE (7ROA 32 -> 77,
# 7ZCX 15 -> 80). Both ceilings look like gateway limits rather than model
# limits, so a self-hosted NIM is the way out.
RETRIES = 6

UNPAIRED_CEILING = 101
PAIRED_CEILING = 500

# Residues MMseqs2 will accept. The NIM rejects anything else with a 422, where
# the public ColabFold API simply returned a query-only alignment (5SBJ, whose
# 30-mer is flanked by X and whose reference alignment is exactly one sequence).
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# An input id is the RCSB entry it is. A suffix (`7WR3_A_C`) means the reference
# build took the biological assembly rather than the asymmetric unit — the two
# differ in bytes even when they hold the same chains, so the suffix, not the
# spec's `assembly` field, is what selects the URL. (`7PV5` carries
# `assembly: 1` and is still the plain entry file.)
_SUFFIXED = re.compile(r"^([0-9][A-Za-z0-9]{3})_.+$")


# --------------------------------------------------------------------------- io


def dataset_path(data: Path, rel: str | Path) -> Path:
    """Resolve a path strictly inside the dataset."""
    root = data.resolve()
    path = (root / rel).resolve()
    if root not in path.parents:
        raise ValueError(f"Dataset path escapes data root: {rel}")
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> NoReturn:
        """Never forward a service credential or download to a redirected host."""
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def _https_url(url: str) -> str:
    """Require TLS and reject embedded credentials and ambiguous request targets."""
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(c.isspace() or ord(c) < 32 for c in url)
    ):
        raise ValueError("Expected an HTTPS service URL without userinfo, fragments, or whitespace")
    return url


def _open_https(request: str | urllib.request.Request, timeout: int) -> HTTPResponse:
    """Open only validated HTTPS URLs, with certificate checks and no redirects."""
    _https_url(request if isinstance(request, str) else request.full_url)
    return urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout)


def http_get(url: str, timeout: int = 180, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            with _open_https(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:  # noqa: PERF203
            code = getattr(e, "code", None)
            if code and 400 <= code < 500 and code != 429:
                raise
            if attempt == tries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


# ------------------------------------------------------------------ structures

_ATOM_KEYS = (
    "label_asym_id",
    "label_seq_id",
    "auth_seq_id",
    "label_comp_id",
    "label_atom_id",
    "label_alt_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
)


def atom_records(text: str) -> list[tuple]:
    """Every ATOM/HETATM row, as the subset of _ATOM_KEYS the file declares.

    mmCIF loops declare their own column order and RCSB adds `atom_site` fields
    over time, so the header is read rather than assumed; positions the file does
    not carry come back as None, which makes the record unusable for a digest.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        j, cols = i + 1, []
        while j < len(lines) and lines[j].lstrip().startswith("_"):
            cols.append(lines[j].strip())
            j += 1
        if not cols or not cols[0].startswith("_atom_site."):
            i = j
            continue
        idx = {c.split(".", 1)[1]: k for k, c in enumerate(cols)}
        out: list[tuple] = []
        while j < len(lines) and lines[j].startswith(("ATOM", "HETATM")):
            row = lines[j]
            vals = row.split() if '"' not in row and "'" not in row else _quoted(row)
            out.append(tuple(vals[idx[f]] if f in idx and idx[f] < len(vals) else None for f in _ATOM_KEYS))
            j += 1
        return out
    return []


def _quoted(row: str) -> list[str]:
    import shlex

    return shlex.split(row)


def atoms_digest(text: str) -> str | None:
    """sha256 over the ATOM/HETATM records alone, annotation stripped.

    This is what makes a fresh build verifiable. A byte hash cannot distinguish
    "RCSB moved an atom" from "RCSB added an annotation field", and on a machine
    that has never held the reference file there is nothing to diff against — so
    the reference build records this alongside the byte hash, and a rebuild
    compares coordinates to coordinates. Returns None for a file that does not
    declare every field, which no RCSB entry does.
    """
    rows = atom_records(text)
    if not rows or any(v is None for v in rows[0]):
        return None
    h = hashlib.sha256()
    for row in rows:
        h.update(("\t".join(row) + "\n").encode())
    return h.hexdigest()


def structure_url(rel: Path, assembly: int | None) -> str:
    """RCSB URL for a `ground_truth/` or `templates/` path in the spec."""
    stem = rel.stem
    m = _SUFFIXED.match(stem)
    if m and rel.parent.name == "ground_truth":
        return RCSB_ASSEMBLY.format(pdb=m.group(1), n=assembly or 1)
    return RCSB_ENTRY.format(pdb=stem)


def archived_copy(pdb: str, want_atoms: str) -> tuple[str, str] | None:
    """(version, text) of the archived entry whose coordinates match, else None.

    The archive lists one file per major version, `..._xyz_v2-1.cif.gz`, and a
    later minor release replaces it under a new name, so the names are read
    from the directory index rather than recorded.
    """
    pdb = pdb.lower()
    base = WWPDB_VERSIONED.format(mid=pdb[1:3], pdb=pdb)
    index = http_get(base).decode("utf-8")
    found = re.findall(rf'href="(pdb_0000{pdb}_xyz_v(\d+)-(\d+)\.cif\.gz)"', index)
    for name, major, minor in sorted(found, key=lambda f: (-int(f[1]), -int(f[2]))):
        text = gzip.decompress(http_get(base + name)).decode("utf-8")
        if atoms_digest(text) == want_atoms:
            return f"{major}.{minor}", text
    return None


# ------------------------------------------------------------------------ msa


class MsaClient:
    """Minimal client for the MSA Search NIM.

    Handles NVCF's two response shapes: a 200 carrying the result, and a 202
    carrying `nvcf-reqid` to poll. Long queries take the second path, so a client
    that only understands 200 works until the first big sequence.
    """

    def __init__(self, key: str, base: str = MSA_BASE, timeout: int = 900, max_sequences: int | None = None):
        self.key, self.base, self.timeout = key, _https_url(base).rstrip("/"), timeout
        self.max_sequences = max_sequences

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        for attempt in range(RETRIES):
            try:
                with _open_https(req, timeout=self.timeout) as r:
                    if r.status == 202:
                        return self._poll(r.headers["nvcf-reqid"])
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode("utf-8", "replace")
                if e.code in (429, 500, 502, 503, 504) and attempt < RETRIES - 1:
                    # The gateway rate-limits a serial rebuild of 46 alignments,
                    # and a 35s ladder was not enough to ride it out. Honour
                    # retry-after when it is offered, otherwise back off in
                    # minutes — a rebuild is a one-off, waiting is free.
                    wait = e.headers.get("retry-after")
                    time.sleep(int(wait) if wait and wait.isdigit() else min(30 * 2**attempt, 300))
                    continue
                raise RuntimeError(f"{path} -> HTTP {e.code}: {detail}") from None
        raise AssertionError("unreachable")

    def _poll(self, reqid: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", reqid):
            raise ValueError("Invalid NVCF request ID")
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            req = urllib.request.Request(
                NVCF_STATUS.format(reqid=reqid), headers={"Authorization": f"Bearer {self.key}"}
            )
            with _open_https(req, timeout=120) as r:
                if r.status == 200:
                    return json.loads(r.read())
            time.sleep(5)
        raise RuntimeError(f"NVCF request {reqid} did not finish in {self.timeout}s")

    def unpaired(self, sequence: str, index: int) -> str:
        """UniRef-only a3m for one chain, headed `>seq{index}`.

        A sequence carrying a non-standard residue comes back 422; the reference
        build's answer for the one such chain is the query alone, so that is what
        we write rather than failing a whole input over it.
        """
        if set(sequence) - STANDARD_AA:
            return query_only(sequence, index)
        r = self._post(
            "/predict",
            {
                "sequence": sequence,
                "databases": list(DATABASES),
                "e_value": 0.0001,
                "iterations": 1,
                "output_alignment_formats": ["a3m"],
                **({"max_msa_sequences": self.max_sequences} if self.max_sequences else {}),
            },
        )
        return rename_query(_one_a3m(r.get("alignments", r)), index)

    def paired(self, sequences: list[str], strategy: str) -> list[str]:
        """One a3m per chain, already split.

        The paired endpoint returns `alignments_by_chain` keyed by a chain letter
        it assigns itself, and JSON gives no order guarantee (a two-chain reply
        arrives {"B": ..., "A": ...}). Blocks are therefore matched to inputs by
        their query sequence, not by key order or key name — getting this wrong
        writes chain A's alignment into chain B's file, which the models happily
        consume.

        It also takes a narrower payload than /predict: `iterations` and
        `output_alignment_formats` are rejected as extra inputs.
        """
        r = self._post(
            "/paired/predict",
            {
                "sequences": list(sequences),
                "databases": list(DATABASES),
                "e_value": 0.0001,
                "pairing_strategy": strategy,
                **({"max_msa_sequences": min(self.max_sequences, PAIRED_CEILING)} if self.max_sequences else {}),
            },
        )
        chains = r.get("alignments_by_chain")
        if not isinstance(chains, dict) or len(chains) != len(sequences):
            raise RuntimeError(
                f"paired response carries {len(chains) if hasattr(chains, '__len__') else '?'} "
                f"chain block(s), expected {len(sequences)}"
            )
        by_query: dict[str, list[str]] = {}
        for block in chains.values():
            a3m = _one_a3m(block)
            by_query.setdefault(query_of(a3m), []).append(a3m)
        out = []
        for i, seq in enumerate(sequences):
            got = by_query.get(seq)
            if not got:
                raise RuntimeError(f"paired response has no block whose query is chain {i} ({len(seq)} residues)")
            out.append(rename_query(got.pop(0) if len(got) > 1 else got[0], i))
        return out


def query_of(a3m: str) -> str:
    """The query sequence of an a3m: the line after the first header, with the
    lowercase insertion columns a3m allows dropped."""
    lines = a3m.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(">"):
            return "".join(c for c in lines[i + 1] if not c.islower()).replace("-", "")
    return ""


def rename_query(a3m: str, index: int) -> str:
    """Head the alignment `>seq{index}`, the convention every shipped file uses.

    The NIM labels the query `>Query|-|Query` unpaired and `>A|-|A` paired. The
    header is our label rather than data, and leaving it would make the rebuilt
    files gratuitously unlike the reference for readers diffing the two.
    """
    lines = a3m.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(">"):
            lines[i] = f">seq{index}"
            break
    return "\n".join(lines) + "\n"


def query_only(sequence: str, index: int) -> str:
    return f">seq{index}\n{sequence}\n"


def _one_a3m(node) -> str:
    """Pull the single a3m out of an alignments node.

    The response nests format and database inside `alignments`, and the exact
    nesting has moved between NIM versions, so this walks for the alignment text
    rather than hard-coding a path — and insists on finding exactly one, so a
    shape change surfaces as an error instead of a silently wrong database.
    """
    found: list[str] = []

    def walk(n):
        if isinstance(n, str):
            if n.lstrip().startswith(">"):
                found.append(n)
        elif isinstance(n, dict):
            for k, v in n.items():
                if k == "alignment" and isinstance(v, str):
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(node)
    uniq = list(dict.fromkeys(found))
    if len(uniq) != 1:
        raise RuntimeError(f"expected 1 alignment in the response, found {len(uniq)}")
    # MMseqs2 terminates a block with a NUL byte. OpenFold reads it as a residue
    # and dies in make_msa_features with KeyError: '\x00'.
    a3m = uniq[0].replace("\x00", "")
    return a3m if a3m.endswith("\n") else a3m + "\n"


# ------------------------------------------------------------------- planning


def load_specs(data: Path) -> list[dict]:
    items: dict[str, dict] = {}
    for name in SPECS:
        path = dataset_path(data, name)
        if not path.is_file():
            raise SystemExit(f"missing {path} — run this script from the bundle it ships in")
        for item in json.loads(path.read_text())["items"]:
            items.setdefault(item["id"], item)
    return [items[k] for k in sorted(items)]


def plan(items: list[dict]) -> tuple[dict[str, dict], list[dict]]:
    """(structures keyed by relative path, msa work grouped per item)."""
    structures: dict[str, dict] = {}
    msa_work: list[dict] = []
    for item in items:
        structures[item["gt"]] = {"assembly": item.get("assembly")}
        chains = []
        for poly in item["polymers"]:
            for tpl in poly["templates"]:
                structures[tpl["path"]] = {"assembly": None}
            if poly["polymer_type"] != "protein" or not poly["msas"]:
                continue
            unpaired = poly["msas"][0]["path"]
            chains.append(
                {
                    # `msa/7R1L_1.a3m` -> 1. The reference heads that file `>seq1`,
                    # so the index has to come from the name, not from the position
                    # in `chains` (which skips the non-protein polymers).
                    "index": int(Path(unpaired).stem.rsplit("_", 1)[-1]),
                    "sequence": poly["sequence"],
                    "unpaired": unpaired,
                    "paired": poly["paired_msas"][0]["path"] if poly["paired_msas"] else None,
                }
            )
        if chains:
            msa_work.append({"id": item["id"], "chains": chains})
    return structures, msa_work


# ------------------------------------------------------------------ execution


def fetch_structure(data: Path, rel: str, assembly, manifest: dict, force: bool) -> tuple[str, str]:
    """-> (status, note). status in {ok, cached, revised, archived, MISMATCH, ERROR}."""
    dest = dataset_path(data, rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    want = manifest.get(rel, {}).get("sha256")
    if dest.is_file() and not force:
        if want is None or sha256(dest) == want:
            return "cached", ""
        # An accepted revision never matches the byte hash again. Re-clear it
        # locally instead of re-downloading it on every run.
        want_atoms = manifest.get(rel, {}).get("atoms_sha256")
        if want_atoms and atoms_digest(dest.read_text()) == want_atoms:
            return "revised", "metadata-only revision, coordinates unchanged"
    url = structure_url(Path(rel), assembly)
    try:
        text = http_get(url).decode("utf-8")
    except Exception as e:  # noqa: BLE001
        return "ERROR", f"{url}: {e}"

    got = hashlib.sha256(text.encode()).hexdigest()
    if want and got != want:
        # A hash mismatch is not yet a problem: RCSB re-releases entries for
        # metadata reasons alone. Only a coordinate change invalidates a result.
        want_atoms = manifest.get(rel, {}).get("atoms_sha256")
        got_atoms = atoms_digest(text)
        if want_atoms and got_atoms == want_atoms:
            dest.write_text(text)
            return "revised", "metadata-only revision, coordinates unchanged"
        if not want_atoms:
            return "MISMATCH", (
                f"{url}: sha256 {got[:16]} != manifest {want[:16]}, "
                "and MANIFEST carries no atoms_sha256 to fall back on"
            )
        if url == RCSB_ENTRY.format(pdb=Path(rel).stem):
            try:
                archived = archived_copy(Path(rel).stem, want_atoms)
            except Exception as e:  # noqa: BLE001
                return "ERROR", f"{url}: coordinates differ, and the wwPDB archive failed: {e}"
            if archived:
                version, text = archived
                dest.write_text(text)
                return "archived", f"RCSB moved to a new major version; using wwPDB archive v{version}"
        return "MISMATCH", f"{url}: coordinates differ — RCSB revised the entry"

    dest.write_text(text)
    return "ok", ""


def fetch_msas(data: Path, work: dict, client: MsaClient, strategy: str, force: bool) -> list[tuple[str, str, str]]:
    """-> [(rel, status, note)] for one item's chains."""
    out: list[tuple[str, str, str]] = []
    paths = {
        rel: dataset_path(data, rel) for chain in work["chains"] for rel in (chain["unpaired"], chain["paired"]) if rel
    }
    todo = {id(c) for c in work["chains"] if force or not paths[c["unpaired"]].is_file()}
    for chain in work["chains"]:
        rel = chain["unpaired"]
        if id(chain) not in todo:
            out.append((rel, "cached", ""))
            continue
        try:
            a3m = client.unpaired(chain["sequence"], chain["index"])
        except Exception as e:  # noqa: BLE001
            out.append((rel, "ERROR", str(e)))
            continue
        paths[rel].parent.mkdir(parents=True, exist_ok=True)
        paths[rel].write_text(a3m)
        out.append((rel, "ok", f"{a3m.count('>')} seqs"))

    paired = [c for c in work["chains"] if c["paired"]]
    if paired and (force or not all(paths[c["paired"]].is_file() for c in paired)):
        try:
            blocks = client.paired([c["sequence"] for c in paired], strategy)
        except Exception as e:  # noqa: BLE001
            out.extend((c["paired"], "ERROR", str(e)) for c in paired)
            return out
        # strict: `paired` and `blocks` are the same length by construction --
        # MsaClient.paired refuses a response whose block count disagrees -- so a
        # silent short zip here would mean that check had been removed.
        for chain, a3m in zip(paired, blocks, strict=True):
            paths[chain["paired"]].parent.mkdir(parents=True, exist_ok=True)
            paths[chain["paired"]].write_text(a3m)
            out.append((chain["paired"], "ok", f"{a3m.count('>')} seqs"))
    elif paired:
        out.extend((c["paired"], "cached", "") for c in paired)
    return out


# ----------------------------------------------------------------------- main


def write_manifest(data: Path, structures: Iterable[str], msa_rels: Iterable[str]) -> dict:
    """Rewrite MANIFEST.json to describe the tree that is on disk now.

    A rebuilt tree is a DIFFERENT release, not a copy of the one it was checked
    against: its alignments came from a different server. Shipping it under the
    reference manifest would hand every user hashes that do not match the bytes
    beside them, and `--verify` would fail on a build that is fine. Same fields
    the reference build writes, so the two are readable side by side.
    """
    dest = dataset_path(data, "MANIFEST.json")
    files = {}
    for rel in sorted(list(structures) + list(msa_rels) + list(SPECS)):
        path = dataset_path(data, rel)
        if not path.is_file():
            raise SystemExit(f"cannot write a manifest: {rel} is missing")
        rec = {"sha256": sha256(path), "bytes": path.stat().st_size}
        if rel.endswith(".cif"):
            digest = atoms_digest(path.read_text())
            if digest:
                rec["atoms_sha256"] = digest
        elif rel.endswith(".a3m"):
            rec["depth"] = depth(path)
        files[rel] = rec
    out = {"files": files, "total_bytes": sum(v["bytes"] for v in files.values()), "file_count": len(files)}
    dest.write_text(json.dumps(out, indent=1) + "\n")
    return out


def depth(path: Path) -> int:
    with path.open() as fh:
        return sum(1 for line in fh if line.startswith(">"))


def check_depth(data: Path, manifest: dict, rels: Iterable[str]) -> list[tuple[str, int, int]]:
    """Alignments materially shallower than the reference build.

    Depth is the property the models are sensitive to, and it is the one a hash
    diff reports as a flat "differs". Losing it silently is the dangerous
    failure: the rebuild succeeds, the speedup still reproduces, and the accuracy
    numbers quietly come out lower — 7ROA's 32-row alignment is why OpenFold2's
    OSS lDDT reads 0.376 instead of 0.801.
    """
    out = []
    for rel in sorted(rels):
        want = manifest.get(rel, {}).get("depth")
        path = dataset_path(data, rel)
        if not want or not path.is_file():
            continue
        got = depth(path)
        if got < want * 0.9:
            out.append((rel, want, got))
    return out


def report(data: Path, manifest: dict, rels: Iterable[str]) -> dict:
    """sha256 of what is on disk now, next to what MANIFEST expected."""
    files, drift = {}, []
    for rel in sorted(rels):
        path = dataset_path(data, rel)
        if not path.is_file():
            files[rel] = {"sha256": None, "bytes": None}
            continue
        got = sha256(path)
        files[rel] = {"sha256": got, "bytes": path.stat().st_size}
        want = manifest.get(rel, {}).get("sha256")
        if want and want != got:
            drift.append(rel)
    return {"files": files, "drift": drift}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=HERE,
        help="directory holding spec_*.json + MANIFEST.json, and where "
        "msa/ templates/ ground_truth/ are written (default: %(default)s)",
    )
    ap.add_argument("--only", help="comma-separated input ids to build")
    ap.add_argument("--structures-only", action="store_true", help="RCSB downloads only — no API key needed")
    ap.add_argument("--msa-only", action="store_true")
    ap.add_argument("--verify", action="store_true", help="hash what is on disk against MANIFEST and exit")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat any difference from MANIFEST as an error, including "
        "alignments (use when reproducing a specific published run)",
    )
    ap.add_argument("--force", action="store_true", help="refetch files already present")
    ap.add_argument("--jobs", type=int, default=2, help="parallel MSA searches (default: %(default)s)")
    ap.add_argument("--msa-url", default=MSA_BASE)
    ap.add_argument("--pairing-strategy", default="greedy", choices=["greedy", "complete"])
    ap.add_argument("--max-msa-sequences", type=int, help="cap passed to the NIM; leave unset to take its default")
    ap.add_argument(
        "--write-manifest",
        action="store_true",
        help="after building, rewrite MANIFEST.json to describe the tree "
        "on disk — for publishing a rebuild as its own release",
    )
    ap.add_argument("--dry-run", action="store_true")
    args, unknown = ap.parse_known_args()
    if unknown:
        # Removed credential arguments must stay undisclosed.
        ap.error("unrecognized arguments; use --help")

    data = args.data.resolve()
    manifest = json.loads(dataset_path(data, "MANIFEST.json").read_text())["files"]
    items = load_specs(data)
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        missing = want - {i["id"] for i in items}
        if missing:
            raise SystemExit(f"unknown id(s): {', '.join(sorted(missing))}")
        items = [i for i in items if i["id"] in want]

    structures, msa_work = plan(items)
    msa_rels = [r for w in msa_work for c in w["chains"] for r in (c["unpaired"], c["paired"]) if r]

    for rel in list(structures) + msa_rels:
        dataset_path(data, rel)

    if args.verify:
        rep = report(data, manifest, list(structures) + msa_rels)
        missing = [r for r, v in rep["files"].items() if v["sha256"] is None]
        for rel in missing:
            print(f"  missing  {rel}")
        for rel in rep["drift"]:
            print(f"  differs  {rel}")
        print(
            f"\n{len(rep['files']) - len(missing) - len(rep['drift'])} match, "
            f"{len(rep['drift'])} differ, {len(missing)} missing"
        )
        return 1 if missing or rep["drift"] else 0

    print(f"inputs: {len(items)}   structures: {len(structures)}   alignments: {len(msa_rels)}")
    if args.dry_run:
        for rel, meta in sorted(structures.items()):
            print(f"  {rel:34s} <- {structure_url(Path(rel), meta['assembly'])}")
        for rel in sorted(msa_rels):
            print(f"  {rel:34s} <- {args.msa_url}{'/paired' if rel.endswith('_paired.a3m') else ''}/predict")
        return 0

    build_path = dataset_path(data, "BUILD.json")
    if args.write_manifest:
        dataset_path(data, "MANIFEST.json")

    api_key = os.environ.get("NGC_API_KEY", "")
    if not args.structures_only and msa_work and not api_key:
        raise SystemExit("NGC_API_KEY missing; set it or use --structures-only")

    failures: list[str] = []
    revised: list[str] = []
    archived: list[str] = []

    if not args.msa_only:
        print("\nstructures (RCSB)")
        for rel, meta in sorted(structures.items()):
            status, note = fetch_structure(data, rel, meta["assembly"], manifest, args.force)
            mark = {"ok": "+", "cached": "=", "revised": "~", "archived": "~"}.get(status, "!")
            print(f"  {mark} {rel:34s} {status}{'  ' + note if note else ''}")
            if status == "MISMATCH":
                failures.append(rel)
            elif status == "ERROR":
                failures.append(rel)
            elif status == "revised":
                revised.append(rel)
            elif status == "archived":
                archived.append(rel)

    if not args.structures_only and msa_work:
        client = MsaClient(api_key, args.msa_url, max_sequences=args.max_msa_sequences)
        print(f"\nalignments (MSA Search NIM, {', '.join(DATABASES)})")
        with futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            pending = {pool.submit(fetch_msas, data, w, client, args.pairing_strategy, args.force): w for w in msa_work}
            for fut in futures.as_completed(pending):
                for rel, status, note in fut.result():
                    note = note.replace(api_key, "[redacted]")
                    mark = {"ok": "+", "cached": "="}.get(status, "!")
                    print(f"  {mark} {rel:34s} {status}{'  ' + note if note else ''}")
                    if status == "ERROR":
                        failures.append(rel)

    # Report only on what this run was responsible for. Under --msa-only the
    # structures were never fetched, so their drift is not this run's news.
    touched = ([] if args.msa_only else list(structures)) + ([] if args.structures_only else msa_rels)
    rep = report(data, manifest, touched)
    build = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "structures": "rcsb",
        "msa": {
            "endpoint": args.msa_url,
            "databases": DATABASES,
            "pairing_strategy": args.pairing_strategy,
            "max_msa_sequences": args.max_msa_sequences,
        },
        "files": rep["files"],
    }
    build_path.write_text(json.dumps(build, indent=1) + "\n")

    if args.write_manifest and not failures:
        new = write_manifest(data, structures, msa_rels)
        print(f"rewrote MANIFEST.json — {new['file_count']} files, {new['total_bytes'] / 1e6:.1f} MB")

    # A `revised` or `archived` file can differ from MANIFEST — that is what the
    # coordinate check just cleared it for. Reporting it again as drift reads as
    # a second, unexplained problem.
    shallow = check_depth(data, manifest, msa_rels)
    msa_drift = [r for r in rep["drift"] if r.startswith("msa/")]
    other_drift = [r for r in rep["drift"] if not r.startswith("msa/") and r not in set(revised + archived)]

    print(f"\nwrote BUILD.json — {len(rep['files'])} file(s)")
    if revised:
        print(f"revised since MANIFEST, coordinates unchanged: {', '.join(revised)}")
    if archived:
        print(f"newer major version on RCSB, archived copy used: {', '.join(archived)}")
    if msa_drift:
        print(
            f"\n{len(msa_drift)} alignment(s) differ from MANIFEST — expected when "
            f"the server behind the endpoint is not the one that produced the "
            f"reference build. Depth, below, is what decides whether it matters."
        )
    if shallow:
        worst = min(g / w for _, w, g in shallow)
        print(
            f"\n  !! {len(shallow)} of {len(msa_rels)} alignment(s) are shallower "
            f"than the reference build, by up to {1 / worst:.0f}x:"
        )
        for rel, want, got in shallow[:8]:
            print(f"       {rel:32s} {want:7d} -> {got}")
        if len(shallow) > 8:
            print(f"       ... and {len(shallow) - 8} more")
        print(
            f"     Speedup will still reproduce; ACCURACY WILL NOT. The hosted "
            f"endpoint caps depth\n     at {UNPAIRED_CEILING} unpaired / "
            f"{PAIRED_CEILING} paired, below what this benchmark needs."
        )
    for rel in other_drift:
        print(f"  ! {rel} differs from MANIFEST")
    if shallow and args.strict:
        print(f"\n--strict: {len(shallow)} alignment(s) too shallow to reproduce the published accuracy")
        return 1
    if failures:
        print(f"\nFAILED on {len(failures)} file(s): {', '.join(failures)}")
        return 1
    if args.strict and rep["drift"]:
        print(f"\n--strict: {len(rep['drift'])} file(s) differ from MANIFEST")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
