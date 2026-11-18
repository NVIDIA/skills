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

# Dataset catalog — built, not downloaded

There is no dataset release to fetch. The builder ships **inside this skill**,
in [`dataset/`](dataset/) — a script and three JSON files, ~90 KB — and it
rebuilds the bench set on your machine from the two public sources it came
from:

```text
dataset/rebuild_dataset.py   33 KB   builds the tree
dataset/spec_full.json       26 KB   17 mixed complexes
dataset/spec_monomer.json    18 KB   15 protein monomers
dataset/MANIFEST.json        15 KB   sha256 + size + depth of every built file
```

`MANIFEST.json` describes what **this** script builds — alignments from the
MSA NIM, at the depths that endpoint returns. It is not the manifest of an
older release, so `--verify` on a correct build comes back clean rather than
reporting 46 alignments as drift.

Built output is 83 files, ~54 MB:

| what                      | where from                                  | licence            |
| ------------------------- | ------------------------------------------- | ------------------ |
| `ground_truth/*.cif` (27) | RCSB                                        | wwPDB, CC0 1.0     |
| `templates/*.cif` (8)     | RCSB                                        | wwPDB, CC0 1.0     |
| `msa/*.a3m` (46)          | NVIDIA MSA Search NIM, `Uniref30_2302` only | UniProt, CC BY 4.0 |

Shipping a builder rather than the bytes is the point: nothing is
redistributed, so no alignment licence has to be answered for, and a reader
can regenerate the inputs instead of trusting a tarball.

Do **not** use `examples/data/samples/` (demo / pipeline fixtures).
Do not invent a residue sweep or a second CASP dump.

**Sample ids are RCSB entry ids.** `7R1L`, not `T1152`; `5SBJ`, not
`5sbj-assembly1`. Every input is named for the PDB entry it *is*; the
benchmark it was selected from is recorded per item in `selected_from` as a
citation. An id carrying a suffix (`7WR3_A_C`) means the reference build took
the biological assembly rather than the asymmetric unit — and that suffix,
not the spec's `assembly` field, is what decides which RCSB URL is right
(`7PV5` carries `assembly: 1` and is still the plain entry file).

## Build

```bash
SKILL_DATASET=skills/bench-perf-oss/dataset
DATASET_ROOT=benchmarks/dataset
mkdir -p "$DATASET_ROOT" && cp "$SKILL_DATASET"/*.json "$DATASET_ROOT/"

python3 "$SKILL_DATASET/rebuild_dataset.py" --data "$DATASET_ROOT"
python3 "$SKILL_DATASET/rebuild_dataset.py" --data "$DATASET_ROOT" --structures-only
python3 "$SKILL_DATASET/rebuild_dataset.py" --data "$DATASET_ROOT" --verify
python3 "$SKILL_DATASET/rebuild_dataset.py" --data "$DATASET_ROOT" --only 7R1L --jobs 4
```

The three JSON files are copied next to the build rather than read in place:
the script writes `MANIFEST.json` under `--data` when asked to
(`--write-manifest`), and a build should never write back into the skill.

Stdlib only, Python 3.10+ (`zip(strict=)`). No install step.

**There is a second copy of this script**, at
`bioair-benchmark/tools/rebuild_dataset.py` in the benchmarking repo, which is
where it is developed and where the published dataset packages are built from.
The copy here is what a reader of this skill runs. They are the same file
today; if they ever disagree, the benchmarking repo is the source and this one
is stale. Re-copy rather than patching this one in place.

Credentials are **not** uniform across the two halves, and the split matters:

- **structures need nothing.** RCSB is anonymous; `--structures-only` runs on
  a machine with no credentials at all and produces 35 of the 83 files.
- **alignments need `NGC_API_KEY`** for the MSA NIM.
  Get one at [build.nvidia.com](https://build.nvidia.com/).
  The builder sends this token in the Authorization header to the HTTPS
  MSA service at `health.api.nvidia.com`, including its asynchronous status
  endpoint. Set `--msa-url` only to a service the user explicitly trusts to
  receive that token and the requested protein sequences. HTTP URLs and
  redirects are rejected; configure the final HTTPS endpoint directly.

Never write a token into the skill, a harness, notes, or a repo file.

Re-running is safe and cheap: a file already present and matching MANIFEST is
left alone, so an interrupted build resumes rather than restarting.

## Verifying a build — two rules, not one

A byte hash alone is the wrong test here, in both directions.

**Structures — byte hash, then coordinates.** RCSB re-releases entries for
metadata reasons alone, and a rebuild on a fresh machine has no local copy to
diff against. `templates/4KL8.cif` is the live case: revision 3.2 on
2026-08-12 added 11 KB of annotation over 14237 unchanged atoms. So MANIFEST
carries `atoms_sha256` per mmCIF — a hash of the ATOM/HETATM records with
annotation stripped — and the script reports:

- `ok` / `cached` — byte-identical
- `revised` — bytes differ, **coordinates identical**; accepted
- `archived` — RCSB moved the entry to a new major version, which moves
  atoms; the wwPDB versioned archive still holds the version MANIFEST was
  built from, and its coordinates match. `ground_truth/7PV5.cif` is the live
  case: version 3.0 on 2026-09-02 remediated a gold-modified cysteine, so the
  build takes archived version 2.1, byte-identical to MANIFEST
- `MISMATCH` — an atom moved and no archived copy matches; aborts, because
  that moves every lDDT for the target. Assembly files have no archived
  copies, so a major revision to one lands here

**Alignments — depth, not bytes.** A structure is a fixed deposition; an
alignment is a search result. It is reproducible for a fixed database version
and a fixed server, and the NIM pins its database (`Uniref30_2302`), but
nothing guarantees the server behind the endpoint is the one that produced
MANIFEST. So alignments are hashed into `BUILD.json` and diffed for
information, and MANIFEST carries `depth` per a3m. The script shouts when a
rebuild comes back materially shallower, because that is the failure that
would otherwise pass silently — the build succeeds, and the accuracy numbers
quietly come out lower.

`--strict` promotes any difference from MANIFEST to an error. That is the
right setting for reproducing a specific published run, and the wrong one for
a first build.

## Provenance — record the build, not a tag

A release tag used to be sufficient identification: same tag, same bytes. **A
built dataset does not work that way.** The same script on two machines can
produce different alignments if the endpoint moved underneath, so recording a
version string alone is a guarantee that is not being kept.

`BUILD.json`, written next to MANIFEST on every run, is what identifies a
build:

```json
{"built_at": "2026-08-31T12:33:50Z",
 "structures": "rcsb",
 "msa": {"endpoint": "https://health.api.nvidia.com/v1/biology/colabfold/msa-search",
         "databases": ["Uniref30_2302"],
         "pairing_strategy": "greedy",
         "max_msa_sequences": null},
 "files": {"msa/7R1L_0.a3m": {"sha256": "...", "bytes": 11762}, "...": {}}}
```

Record on `bench_config.json` and every result JSON: `dataset_root`,
`dataset_spec`, and the SHA-256 of **both** `MANIFEST.json` and `BUILD.json`.
Two runs are comparable when those two hashes match; a matching
`dataset_spec` alone proves nothing.

## Alignment depth is a property of the endpoint

The hosted MSA NIM caps a search at **101 unique sequences unpaired** (100
hits plus the query) and **500 paired**. Neither ceiling moves with
`max_msa_sequences`, `iterations` or `e_value` — the unpaired one truncates
silently, returning 200 with no indication it did.

Against alignments built from the public ColabFold endpoint (up to 10483 rows
unpaired, 44638 paired) that is **50x shallower overall**, and it moves both
halves of the benchmark, measured on an 11-GPU matrix:

- **lDDT falls** on every model: −0.008 to −0.044 mean, worst single sample
  0.752 → 0.345
- **speedup inflates** by 4-5% on Ampere / Ada / Hopper, and is within noise
  on Blackwell

Note the direction: this is truncation, not a worse search. Below the ceiling
the NIM finds *more* than the public endpoint did (`7ROA` 32 → 77, `7ZCX`
15 → 82), and those are exactly the shallow-alignment targets that collapse
OpenFold2's OSS accuracy.

So the depth recorded in MANIFEST is part of the measurement, not packaging
detail. Do not compare lDDT across two builds whose depths differ.

## Which spec

- Boltz-1/2, OpenFold3, Protenix-v2: `spec_full.json`
- AF2 / OF2 monomer: `spec_monomer.json` (protein only)
- AF2 multimer: `protein-protein` items from `spec_full.json`
  that contain only protein polymers. RNA / DNA / ligand rows are
  out of scope for this family
  ([models/of2.md](models/of2.md#protein-only)).

Do not silently mix specs. Write the chosen file on
`bench_config.json` as `dataset_spec`.

## Templates are in

Template-bearing items are **in scope**, and their templates get
attached. Templates are part of how these models are actually run, so
a bench that drops them measures a narrower workload than the one
users care about.

Template-bearing ids:

- `spec_full.json` — `7R1L`, `8B43`, `8SX8`
- `spec_monomer.json` — `7ROA`, `7QIH`, `8ORK`, `8FEF`, `7UTD`

The spec ships template **structures** only
(`templates/<entry>.cif`, optional `chain_id`). Attach every listed
file, resolved against `$DATASET_ROOT`, the same way MSAs are
attached.

**Templates on one side only is a hard failure**, worse than skipping
the sample: the bare side folds from less evidence, so its row looks
ordinary while its structure and lDDT answer a different question.
The two stacks may want templates in different forms, and a fixed
template-slot layout means shape checks cannot see a drop. Follow the
model profile rather than assuming a structure or alignment is enough.

How to map them onto an OSS tree, what to synthesize, how to verify
attachment, and how to handle the two sides disagreeing about which
templates are usable: [templates.md](templates.md).

Only when a side genuinely cannot consume the templates do you set
`in_scope: false`, with the specific reason (which side, which
missing capability) — and then that sample is excluded on **both**
sides. Never silently drop it, and never strip the templates and
time the rest of the item as a substitute.

## Manifest fields

Each `sample_manifest.json` entry must include:

- `sample_id` — spec `id`
- `spec` — `spec_full.json` or `spec_monomer.json`
- `chemical_class`, `seq_len`, `token_bin`
- `residues` — **sum of all protein / RNA / DNA chains**
  (`len(sequence) * n_copies` per polymer, then sum). Must
  match spec `seq_len`. Never chain A alone.
- `polymers[]` with `polymer_type`, `chain_id`, `n_res`
- `n_chains` — total chains including copies, and `has_ligand`.
  These decide whether DockQ applies and whether it needs
  `--small_molecule`
  ([measurement.md](measurement.md#quality-lddt-and-dockq)); a
  single-chain sample scores `dockq: null`, not `0.0`
- `unpaired_msas[]` — resolved absolute A3M paths, or empty
- `paired_msas[]` — resolved absolute A3M paths, or empty
- `msa_source` — `BUILD.json`'s `msa.endpoint` and `msa.databases`, so a
  row says which search produced the alignment it was folded from
- `msa_status` — `attached` or `none_declared`
- `gt_path` — absolute path from spec `gt`, or `null`
- `has_templates` — true when any polymer lists templates
- `templates[]` — resolved absolute template CIF paths (with
  `chain_id` when the spec sets one), or empty
- `template_status` — one of `none_declared`; `attached` (both sides
  took the shipped structures); `synthesized_alignment` (an alignment
  was generated for the OSS parser); or `unsupported` (a side cannot
  consume them; pair with `in_scope: false`)
- `in_scope` — `true` unless the model cannot accept the sample.
  Template-bearing items are in scope; see
  [templates](#templates-are-in)

## Load path (processor)

Do **not** call `examples/folding/run_demo.py:load_requests` on
`spec_full.json` (wrong wrapper: `items` + `id`, not a demo JSON).
Convert each in-scope spec item to an `InputRequest`, resolving
every `path` against `$DATASET_ROOT` — MSAs and templates alike.

```python
from pathlib import Path

from bionemo_ir.data.schemas import InputRequest, MSARecord, Polymer, Template

DATASET_ROOT = Path("benchmarks/dataset")


def _msas(entries, root: Path) -> list[MSARecord]:
    out = []
    for item in entries or []:
        path = root / item["path"]
        out.append(MSARecord(path=str(path), format=item.get("format", "a3m")))
    return out


def _templates(entries, root: Path) -> list[Template]:
    out = []
    for item in entries or []:
        path = root / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"template missing: {path}")
        out.append(
            Template(
                path=str(path),
                format=item.get("format", "cif"),
                chain_id=item.get("chain_id"),
            )
        )
    return out


def spec_item_to_request(item: dict, root: Path) -> InputRequest:
    polymers = []
    for polymer in item["polymers"]:
        polymers.append(
            Polymer(
                polymer_type=polymer.get("polymer_type", "protein"),
                chain_id=polymer["chain_id"],
                sequence=polymer["sequence"],
                msas=_msas(polymer.get("msas"), root),
                paired_msas=_msas(polymer.get("paired_msas"), root),
                templates=_templates(polymer.get("templates"), root),
            )
        )
    return InputRequest(input_id=item["id"], polymers=polymers)
```

`chain_id=None` lets BioIR auto-select the best-aligning chain of a
multi-chain template CIF, which is what the spec's `null` means.
Templates are protein-only; a template on an RNA / DNA / ligand
polymer is a manifest bug, not something to pass through.

Reuse `_load_msas` from `run_demo.py` only if `base_dir` is
`$DATASET_ROOT`.

## Load path (OSS)

Start from the OSS inference script and map each in-scope spec item
to its input. The per-tree field names, enable flags, conversions,
and verification live in [msa.md](msa.md) and
[templates.md](templates.md); the short form:

- **Boltz** — build the YAML from the spec's canonical `polymers` and write
  the A3M paths alongside, the same generic path every other model uses.
  There is no longer a native Boltz subtree to point at: the dataset used to
  ship `casp15/queries/*.yaml` and `casp15/msa/*.csv`, and those were dropped
  because the CSVs carry **no sequence identifiers** — so the databases behind
  them could not be stated, which is the one thing the alignments now have to
  be able to say. `boltz_yaml` and `boltz_msa_csv` no longer exist on any spec
  item. Do not point at `examples/boltz2/...` or `examples/data/samples/`.
- **Others** — convert spec polymers + resolved A3Ms to the script's
  FASTA / JSON / CSV layout. Still attach every A3M.

## MSA contract

Protein polymers that list A3Ms must load every unpaired and paired
file. RNA, DNA, and ligand polymers with empty `msas` are correct;
do not invent alignments. Mapping and verification:
[msa.md](msa.md).

`msa_status`:

- `attached` — at least one resolved A3M exists and was loaded
- `none_declared` — RNA / ligand only, or a protein whose spec lists
  `msas: []` and there is no A3M on disk for that id
- `missing_file` — a listed path is absent; **blocker**

A 37-byte A3M still counts as attached. Do not skip it — and do not treat it
as a setup failure either. `msa/5SBJ_0.a3m` is genuinely one sequence: a
30-mer flanked by `X`, for which the search returns no hits at all. Depth one
is a blocker only when MANIFEST's `depth` for that file says otherwise, which
is what makes the recorded depth worth checking rather than guessing at.

## Samples — `spec_full.json` (17)

`selected_from` is a citation, not a redistribution: the file shipped is the
RCSB entry, not anything authored by CASP15 or FoldBench.

| sample_id     | selected_from                  | class           | seq_len | token_bin | MSA (unpaired/paired) | templates |
| ------------- | ------------------------------ | --------------- | ------- | --------- | --------------------- | --------- |
| `8FZA`        | CASP15:R1117                   | ligand          | 29      | 16-256    | 0 / 0 (RNA+CCD)       | no        |
| `5SBJ`        | FoldBench:5sbj-assembly1       | monomer         | 30      | 16-256    | 1 / 0                 | no        |
| `7R1L`        | CASP15:T1152                   | protein-ligand  | 111     | 16-256    | 2 / 2                 | **11CI**  |
| `7ZJ4`        | CASP15:R1136                   | ligand          | 373     | 257-512   | 0 / 0 (RNA+CCD)       | no        |
| `7QSJ`        | FoldBench:7qsj-assembly1       | monomer         | 373     | 257-512   | 1 / 0                 | no        |
| `8AD2`        | CASP15:T1187                   | protein-ligand  | 330     | 257-512   | 2 / 2                 | no        |
| `7UWW`        | FoldBench:7uww-assembly1       | monomer         | 635     | 513-768   | 1 / 0                 | no        |
| `7UX8`        | CASP15:T1124                   | protein-ligand  | 766     | 513-768   | 2 / 2                 | no        |
| `7SPQ_A_A-2`  | FoldBench:7spq-assembly1_A_A-2 | protein-protein | 656     | 513-768   | 2 / 2                 | no        |
| `8K7X`        | FoldBench:8k7x-assembly1       | monomer         | 856     | 769-1024  | 1 / 0                 | no        |
| `8B43`        | CASP15:T1118v1                 | protein-ligand  | 775     | 769-1024  | 1 / 0                 | **6OVK**  |
| `7WR3_A_C`    | FoldBench:7wr3-assembly1_A_C   | protein-protein | 959     | 769-1024  | 2 / 2                 | no        |
| `8H2N`        | CASP15:T1125                   | monomer         | 1199    | 1025-1576 | 1 / 0                 | no        |
| `8SX8`        | CASP15:T1158v1                 | protein-ligand  | 1339    | 1025-1576 | 1 / 0                 | **8BWO**  |
| `8A8O_A_B`    | FoldBench:8a8o-assembly1_A_B   | protein-protein | 1142    | 1025-1576 | 2 / 2                 | no        |
| `8UXT`        | FoldBench:8uxt-assembly1       | monomer         | 1596    | 1577-2048 | 1 / 0                 | no        |
| `8IC7_A_B`    | FoldBench:8ic7-assembly1_A_B   | protein-protein | 1734    | 1577-2048 | 2 / 2                 | no        |

Timed default for Boltz / OpenFold3 / Protenix-v2: all 17 items, with
templates attached on the three that list them.

**`8FZA` and `7ZJ4` carry no alignment at all** — RNA plus a CCD ligand. That
makes them a free noise probe: between any two runs they measure run-to-run
scatter and nothing else, so a difference on them bounds what a difference
elsewhere is allowed to mean. See
[measurement.md](measurement.md#warmup-and-repeats).

## Samples — `spec_monomer.json` (15)

| sample_id | selected_from            | seq_len | token_bin | templates |
| --------- | ------------------------ | ------- | --------- | --------- |
| `7PV5`    | FoldBench:7pv5-assembly1 | 81      | 16-256    | no        |
| `7ROA`    | CASP15:T1104             | 115     | 16-256    | **7ROA**  |
| `7QIH`    | CASP15:T1106s1           | 120     | 16-256    | **8ARA**  |
| `7QSJ`    | FoldBench:7qsj-assembly1 | 373     | 257-512   | no        |
| `8FEF`    | CASP15:T1137s1           | 408     | 257-512   | **7AI2**  |
| `8ORK`    | CASP15:T1112             | 459     | 257-512   | **9SXB**  |
| `7UTD`    | CASP15:T1114s3           | 534     | 513-768   | **4KL8**  |
| `7UWW`    | FoldBench:7uww-assembly1 | 635     | 513-768   | no        |
| `8A8C`    | CASP15:T1129s2           | 639     | 513-768   | no        |
| `8K7X`    | FoldBench:8k7x-assembly1 | 856     | 769-1024  | no        |
| `8T41`    | FoldBench:8t41-assembly1 | 867     | 769-1024  | no        |
| `8WNJ`    | FoldBench:8wnj-assembly1 | 920     | 769-1024  | no        |
| `8H2N`    | CASP15:T1125             | 1199    | 1025-1576 | no        |
| `7ZCX`    | CASP15:T1154             | 1423    | 1025-1576 | no        |
| `8UXT`    | FoldBench:8uxt-assembly1 | 1596    | 1577-2048 | no        |

Every monomer item has one unpaired A3M. Timed default for AF2 / OF2 monomer:
all 15 items, with templates attached on the five that list them.

`7UTD`'s template is `4KL8` — the one entry RCSB has revised since the
reference build, and the reason MANIFEST carries `atoms_sha256`.

Overlap in both specs (built once, used twice): `7QSJ`, `7UWW`, `8K7X`,
`8H2N`, `8UXT`.

## Model filters

- **AF2 / OF2 monomer** — `spec_monomer.json`, all items.
- **AF2 multimer** — `chemical_class == protein-protein` from
  `spec_full.json` (`7SPQ_A_A-2`, `7WR3_A_C`, `8A8O_A_B`,
  `8IC7_A_B`). Drop RNA / ligand.
- **Boltz-1/2, OpenFold3, Protenix-v2** — `spec_full.json`, all
  items. Keep RNA, ligand, and protein–protein rows.

Filter on what the model cannot represent (an RNA chain on a
protein-only model), not on templates.

A filtered sample stays in the manifest with `in_scope: false` and a
one-line reason. It must not appear in either result JSON.
