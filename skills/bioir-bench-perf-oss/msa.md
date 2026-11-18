---
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
{}
---

# Mapping dataset MSAs onto an OSS input

How to get the alignments the dataset ships into whatever form
an OSS folding tree wants, and how to prove they arrived. Model
agnostic: the per-tree specifics are worked examples, and the
[procedure](#the-procedure) is what you re-run for a tree not listed
here.

Companion: [templates.md](templates.md). Both are inputs, both are
silent when dropped, and both must land on **either both sides or
neither**.

## What the dataset ships

Per protein polymer in `spec_full.json` / `spec_monomer.json`:

- `msas` — unpaired alignments, `[{"path": "msa/<id>_<n>.a3m",
  "format": "a3m"}]`, relative to `$DATASET_ROOT`
- `paired_msas` — paired alignments, same shape,
  `msa/<id>_<n>_paired.a3m`
- `boltz_yaml`, `boltz_msa_csv` — Boltz-native inputs on CASP items

RNA, DNA, and ligand polymers carry `msas: []`. That is correct data,
not a gap to fill: nothing about a ligand chain wants an alignment.

Two properties bite people. A listed A3M can be **tiny** — 37 bytes,
one sequence — and it still counts as attached; it is a real
alignment for a 30-residue chain, not a placeholder to skip. And a
single polymer can list **several** files, so a loader that takes
`paths[0]` looks like it works and quietly runs at a fraction of the
intended depth.

## The contract

1. Every alignment the spec lists for an in-scope item is loaded, on
   **both** sides.
2. No alignment is invented, and no search is run. No ColabFold or
   MSA-server call, no jackhmmer / hhblits, no network.
3. A protein run whose spec lists MSAs but whose features carry an
   empty MSA is a **hard failure**, not a fast row.
4. A listed path missing from disk is a blocker (`missing_file`),
   reported rather than skipped.
5. Depth and pairing are part of config parity: if one side caps MSA
   rows and the other does not, you are timing two different
   workloads.

`msa_status` per sample: `attached` | `none_declared` |
`missing_file` (blocker).

## The procedure

Work through this once per OSS tree, then record the answers in
`$WORKDIR/ref_data/` so a re-run does not re-derive them.

### 1. Find the input schema, not the README

Locate the entry point (`run_*.py`, a CLI, a `predict` subcommand)
and read the **schema object** it validates against — a pydantic
model, dataclass, or JSON schema. That file names the MSA fields
exactly; a README example may be stale or partial. Grep the tree for
`a3m`, `sto`, `m8`, `msa`, `alignment` to find both the schema and
the loader.

### 2. Classify each field by what the loader does with it

For every MSA-ish field, answer:

- unpaired or paired, and does this tree distinguish them at all?
- one path, a list of paths, or a directory per chain?
- what formats does the parser accept, and does it sniff or trust
  the extension?
- is the field per chain or per query?

An unpaired-only tree fed paired alignments (or the reverse) usually
does not error — it ignores the field.

### 3. Learn how the loader selects and keys files

Handing a loader a correct path is not enough: it decides **which files
to parse** and **which chain each file belongs to**, often from the
filename and directory rather than from the field you set. Both
decisions fail silently. Read the parser and answer:

- **Selection by stem.** Does the parser skip files whose stem is not in
  some registry of known alignment sources (a `max_seq_counts`-style
  dict, an `aln_order` list, a paired-only order list)? A dataset file
  named `<sample>_0.a3m` then parses to nothing, and the crash lands far
  away — often an `IndexError` on the first key of an empty dict. Rename
  through symlinks to a stem that satisfies **every** registry: one that
  gates parsing, and one that gates inclusion in the stack.
- **Chain identity from the path.** For directory-style layouts the
  chain's alignment key is frequently `path.parent.stem`, not the chain
  id you passed. Registration is usually first-write-wins, so if all
  chains share one flat folder they collapse onto a single key and every
  chain after the first inherits the first chain's alignment. Give each
  chain its own directory.
- **Per-file caps versus global caps.** A parse-time cap keyed by stem
  (`{"uniref90_hits": 10000, ...}`) is applied *before* dedup and
  ordering, while the other side may read the file whole and cap only at
  the end. Pick a stem whose cap cannot bite, and assert file depth
  against it so a deeper dataset fails loudly instead of comparing two
  different alignments.

Stage the result under `$WORKDIR/oss_data/msa/<sample>/<chain>/` with a
script that writes an index JSON of source, staged path, rows, and rows
actually used.

### 4. Find the enable flags

Pointing at files is not the same as using them. Trees commonly gate
alignments behind booleans (`use_msas`, `use_main_msas`,
`use_paired_msas`, `msa: true`) that default off or are derived from
some other flag. Set them explicitly from the data: whether the spec
listed unpaired files, paired files, or neither. Do not rely on a
default, and do not enable paired on a sample that has none.

### 4. Disable every search path

Find the auto-search switch and turn it off — `use_msa_server`,
`--msa_server_url`, ColabFold, or an implicit "no MSA file means
search". Prefer running with the network unavailable so a fallback
fails loudly instead of silently substituting a server alignment for
the dataset's.

### 5. Convert only when the parser cannot read A3M

If the tree needs another format or layout (CSV, STO, one file per
chain in a directory), convert from the shipped A3M and write the
result under `$WORKDIR/oss_data/msa/`. Never write into
`$DATASET_ROOT` (it is a verified artifact) or `$OSS_ROOT` (it is a
pinned checkout). Record what you generated and from what.

Prefer a symlink to a converted copy: renaming and regrouping is what
most trees actually need, and a link keeps both sides reading the same
bytes.

Rewrite stale paths rather than trusting shipped ones: a bundled
Boltz YAML may carry an absolute `msa:` path from the machine that
produced it, and it will resolve to something wrong or nothing at
all.

### 6. Verify on the features, then record

See [verification](#verification). Then write the resolved absolute
paths and `msa_status` onto every manifest and result row, so a
dropped alignment is visible in the artifact instead of hiding in a
latency number.

## Worked mappings

**BioIR processor path.** `MSARecord(path=..., format="a3m")` on
`Polymer.msas` / `Polymer.paired_msas`; see
[samples.md](samples.md#load-path-processor). Protein unpaired MSA is
required for Boltz-1/2 and OpenFold3 and for every AF2 / OF2 key
(`docs/ref/support-matrix.md`); paired is optional and used when the
spec lists it.

**OpenFold3.** Follow the pinned variant's complete MSA staging,
query-field, cap, bookkeeping, and reference-value contract in
[models/of3.md](models/of3.md#msas).

**Protenix-v2.** Follow the pinned release's query JSON, MSA path,
and per-entity assertions in
[models/protenix.md](models/protenix.md#convert-spec_fulljson-to-protenix-input).

**Boltz-1/2.** Native `boltz_yaml` + `boltz_msa_csv` from the spec.
Rewrite every CSV path to `$DATASET_ROOT/casp15/msa/<file>`; never
point at `examples/boltz2/...` or `examples/data/samples/`.

**AF2 / OF2 monomer.** One unpaired A3M per chain. Multimer builders
often additionally require a paired A3M on **every** chain — check
before assuming the monomer mapping transfers.

**OSS feature path (OSS pipeline + BioIR module).** The OSS featurizer loads
the alignments and both forwards consume that one feature dict, so
MSA parity is structural rather than something to re-verify per side
([no-pipeline.md](no-pipeline.md)). Verify once on the shared batch.

## Verification

Config inspection is not verification. Assert on the featurized
batch, per sample, before timing:

- the MSA tensor exists and has **more than one row** whenever the
  spec listed files (depth 1 usually means "query only", i.e. nothing
  loaded)
- the row count is consistent with the file you passed (parse the
  A3M and compare sequence counts, allowing for the tree's own
  dedup / cap)
- paired features are non-empty exactly when the spec listed paired
  files
- for a precomputed paired MSA, the final row count and
  `num_paired_seqs` follow the **pinned model profile's** semantics
- both sides agree on depth for the same sample; a large asymmetry is
  a capping or pairing difference, not a win

Cheap and decisive: count declared files, count attached files,
assert equality. Then diff the two sides' per-sample depth and stop
if they disagree.

## Failure modes seen in practice

- **Only the first file of a list attached.** Looks correct, runs
  shallow. Assert counts, not truthiness.
- **Every file skipped because its stem is unknown.** The parser
  returns an empty dict and the failure surfaces as an `IndexError` or
  a KeyError on a first-key lookup, nowhere near the cause.
- **All chains sharing one alignment** because the loader keyed them by
  a directory they all live in, and registration was first-wins. The
  run succeeds, the multimer was folded against one chain's MSA, and
  only a per-chain depth assert catches it.
- **Paired silently ignored** because the tree names the field
  differently, or because a `use_paired_msas`-style gate stayed
  false.
- **An adapter mirrors a different OSS revision's bookkeeping.** The files
  load and may even participate in main-row deduplication, but paired
  rows may not enter the final tensor and the paired count can remain
  unchanged. Re-derive bookkeeping whenever the checkpoint or
  revision moves; do not label either behavior correct without the
  model profile.
- **A server filled in the gap.** The row succeeds with an alignment
  that is not the dataset's, so it is neither reproducible nor
  comparable.
- **A dummy query-only MSA filled in the gap.** Some inference
  pipelines warn and continue when a staged file is missing. Treat
  depth one as a setup failure whenever the spec declared a deeper
  alignment.
- **Depth caps differ** between sides (`max_msa_seqs` and friends),
  making one side systematically cheaper on deep alignments.
- **A tiny A3M treated as empty** and skipped, which turns a valid
  short-chain sample into a no-MSA run.
- **RNA / ligand chains "fixed"** by inventing an alignment, which
  changes the workload and the prediction.
- **A separator byte read as a residue.** MMseqs2 terminates a
  per-query block with a NUL, and on a single-query response it lands
  one byte past the final newline — invisible in an editor, and the
  alignment is byte-identical without it. Left in, OpenFold2 dies in
  `make_msa_features` with `KeyError: '\x00'`; OpenFold3 does not
  raise there at all and instead fails later broadcasting `(1340,)`
  into `(1339,)`, one row longer than the query. Sixteen files shipped
  with it and cost a full matrix re-run.

  Two things made it survive review. The paired path *looked* like
  coverage — it strips NUL already, because it splits per-chain blocks
  on that byte — so only the unpaired path passed a response through
  whole. And the check that should have caught it did print the
  evidence: a per-chain core-length summary read `core_len=[1, 115]`,
  and the `1` was dismissed as a header artifact rather than a
  one-character row.

  So verify by reading **every character of every line**, not a
  summary of them, and treat any non-residue character as a hard
  failure rather than something to explain. A client that happens to
  strip it (boltz's own does) is not the same as a file that does not
  contain it.

## Recorded fields

On `sample_manifest.json` and both result files:

- `unpaired_msas`, `paired_msas` — resolved absolute paths
- `msa_status` — `attached` | `none_declared` | `missing_file`
- `boltz_yaml`, `boltz_msa_csv` — when the spec has them
- any generated file, with the source it was converted from

In `bench_config.json`, record the OSS field names and enable flags
you set, so the next run maps the same way without re-deriving it.
