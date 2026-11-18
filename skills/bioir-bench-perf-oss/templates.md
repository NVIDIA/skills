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

# Mapping dataset templates onto an OSS input

How to get the homolog templates the dataset ships into
whatever form an OSS folding tree wants, and how to prove they
arrived. Model agnostic: the per-tree specifics are worked examples,
and the [procedure](#the-procedure) is what you re-run elsewhere.

Templates are harder than MSAs in one specific way: the dataset ships
**structures**, while a tree may want an **alignment** that merely
points at structures. Handing it the structure then attaches nothing,
and nothing errors. Read [verification](#verification) before you
trust any template row.

Companion: [msa.md](msa.md).

## What the dataset ships

Per protein polymer, a `templates` list:

```json
"templates": [{"path": "templates/11CI.cif", "format": "cif",
               "chain_id": null}]
```

- Structures only — mmCIF, one file per template entry, under
  `$DATASET_ROOT/templates/`. No alignments, no search results, no
  preprocessed cache.
- `chain_id: null` means "pick the chain that best matches this
  query"; a string pins one chain of a multi-chain template.
- Protein chains only. A template on an RNA / DNA / ligand polymer is
  a manifest bug.
- A file may be shared by several query chains (a homodimer lists the
  same template on each), and the entry name is the file stem
  (`11CI.cif` -> entry `11CI`), which is not always a real PDB id.

Template-bearing ids: `T1152`, `T1118v1`, `T1158v1` in
`spec_full.json`; `T1104`, `T1106s1`, `T1112`, `T1137s1`, `T1114s3`
in `spec_monomer.json`.

## These are custom templates: include all of them

Everything the spec lists is a **custom template** — a structure the
caller chose and supplied. There is no search step and no candidate
pool to rank. The instruction is "use these", so the bench includes
**all** supplied templates for a chain, and the only legitimate
limiter is the **configured cap** (`n_templates` / `max_templates`
and equivalents), applied identically on both sides.

That makes every similarity heuristic irrelevant here, and worse than
irrelevant: a sequence-identity floor, a coverage minimum, a release
date cutoff, an e-value threshold, or a keep/drop rule that discards
a partially aligning template is a filter designed for search output.
Pointed at explicit user input, it silently overrides the caller. If
one side applies such a gate and the other does not, the two runs
featurize different inputs and the comparison is void, with both
sides reporting success.

So the bar is not "did some template attach" but **the same supplied
templates attached, in the same slots, on both sides**.

## The contract

1. Template-bearing items are **in scope**. Attach their templates.
2. **Include every supplied template**, up to the configured cap.
   Neutralize any selection or filtering gate that would drop one —
   by config where the tree exposes it, and by a
   [temporary patch](#temporary-patches-are-allowed) where it does
   not.
3. **Same set, same cap, same slots, same chain, both sides.** A
   templated forward compared against a bare or differently-filtered
   one is not a comparison.
4. Verify by **counting** what attached on the featurized batch, per
   side, per sample — not by checking config, and not merely by
   checking that something is non-zero.
5. Never write generated inputs into `$DATASET_ROOT` or `$OSS_ROOT`;
   patches to `$OSS_ROOT` are the one exception, and they are
   recorded and reverted.
6. Only if a side genuinely cannot represent templates at all does
   the sample run template-free on **both** sides, labelled.

`template_status` per sample is one of `none_declared`, `attached`,
`capped_by_config`, `synthesized_alignment`, `attached_via_patch`,
or `unsupported`.

## The procedure

### 1. Establish what the tree ingests

Grep for `template` in the entry-point schema and the data pipeline,
and classify the ingestion form:

- **structure-direct** — a path (or content) per query chain, parsed
  and aligned internally. Nothing to synthesize.
- **alignment-driven** — a query→template alignment (`sto` / `a3m` /
  `m8`) whose entries name structures resolved from a directory. The
  alignment is mandatory; the structure alone is inert.
- **preprocessed cache** — `.npz` arrays / precache entries built by
  a preprocessing script, with the raw structure as one possible
  input to that script.

Read the tree's own template documentation for the format details,
then confirm against the parser: docs describe the intended path,
code decides what actually loads.

### 2. Work out the entry-id to filename convention

For alignment-driven trees, the alignment's entry id is what locates
the structure, typically `<structure_directory>/<entry_id>.<fmt>`.
Name the entry after the shipped file's stem so it resolves, and
point the structure directory at `$DATASET_ROOT/templates` (or a
staging copy).

### 3. Kill the network fallback

Template pipelines often fetch missing structures from the PDB by
default (`fetch_missing_structures` and equivalents). Turn that off.
Otherwise a naming mistake becomes a silent download — or, for an
entry id that is not a real PDB code, a silent skip that leaves you
with a zero-filled template and no error.

### 4. Synthesize the minimal alignment, mirroring the other side

When the tree needs an alignment the dataset does not ship, generate
the smallest artifact its parser accepts: one record per template,
headed `<entry>_<chain>`, carrying the template chain's sequence
taken from the structure. Parsers that receive no explicit residue
range realign the sequence to the query themselves (Kalign, in the
OpenFold-family trees), which is exactly the behaviour you want.

Pick the **same template chain the other side picks**, and do it by
calling that side's own selection routine rather than reimplementing
its scoring. Chain choice changes the features, so choosing
independently invents an asymmetry that has nothing to do with the
code under test.

Write to `$WORKDIR/oss_data/templates/`, record it as
`synthesized_alignment`, and keep the per-sample chain and score in
`oss_data/index.json` so the report can state exactly what was
attached.

Send the preprocessor's own outputs to `$WORKDIR/oss_data/cache/` too.
Template pipelines commonly derive a cache, parsed-structure, or log
directory from an output root that defaults to a temp path, and they
`mkdir` all of them — left alone it either evaporates between runs or
lands next to the dataset.

### 5. Enable the flag

Templates are usually gated (`use_templates` or similar) and
frequently default off. Set it from the data, per sample: on for
template-bearing items, off for the rest. An installed dependency or
a populated field is not an enabled feature.

### 6. Neutralize every filter, and equalize the cap

Enumerate what each side can drop between "supplied" and
"featurized", then switch it off:

- similarity and coverage gates — sequence-identity ceilings or
  floors, minimum aligned length, minimum aligned fraction
- date gates — maximum release date, minimum query-template date gap
- quality gates — minimum resolved fraction
- keep/drop rules applied after alignment, which are the easiest to
  miss because they live in the featurizer rather than the config
- **the cap** — set `n_templates` / `max_templates` to the same
  number on both sides, and make it at least the largest supplied
  count so nothing is dropped for capacity reasons you did not intend

Prefer config. Where a gate is hard-coded, use a
[temporary patch](#temporary-patches-are-allowed). Record the final
value of every one of these knobs in `bench_config.json`, on both
sides, because "we left the defaults" is not a description of what
ran.

### 7. Walk the whole path when a slot stays empty

Correct inputs and an enabled flag still yield zero templates
surprisingly often, because a template crosses several hops —
alignment file, parse cache, per-chain id list, sampler, aligner,
keep/drop check, featurized slot — and every hop can drop it with no
log line. Do not bisect by guessing; instrument each hop once and
print what it received. Two hops account for most of it:

- **A directory argument that is never supplied.** Inference paths
  sometimes pass `None` where the training path passes a real cache
  directory, and the sampler treats that as "no templates" and
  returns empty — even though the per-chain cache path it needs is
  sitting in the data it was handed. Symptom: preprocessing reports
  success and writes a cache, and the aligner is then called with
  zero templates.
- **Gap rows on the template side.** An alignment map holds one row
  per aligned position, and rows where the *template* side is a gap
  may survive a filter that only drops gaps on the *query* side. The
  aligned-query count then exceeds the aligned-template count by
  exactly the number of such rows, and an equality-based keep/drop
  check discards the whole template. An off-by-a-few mismatch in that
  check is this, nearly every time.

A drop that reproduces without the model is much cheaper to chase:
run the preprocessor and the featurizer on one sample, with no
weights loaded, and print the counts each hop compared.

### 8. Verify by count, then time

Verify per side ([verification](#verification)) and confirm the two
sides attached the **same number** of templates, with the same entry
and chain per slot. Agreement is not the default; establish it.

## Temporary patches are allowed

Making both sides honour the supplied templates is worth a patch to
the OSS tree — the alternative is a comparison of two different
inputs. The pinned revision still governs; a patch is a recorded
delta on top of it, not a different checkout.

Rules:

- **Input selection only.** Patch what decides *which* templates are
  featurized: thresholds, filters, keep/drop predicates, caps.
  Never patch the compute path — no kernels, precision, layer math,
  recycling, or sampling. A patch that changes how the model runs
  invalidates the bench it was meant to fix.
- **Minimal and legible.** Prefer neutralizing one predicate over
  restructuring a function.
- **Recorded as a file.** Keep the diff at
  `$WORKDIR/ref_data/patches/<name>.patch`, apply it with
  `git -C "$OSS_ROOT" apply`, and list it in `bench_config.json`
  (`oss_patches`) with one line on what it changes and why.
- **Reverted at the end** (`git -C "$OSS_ROOT" apply -R`), leaving
  the pinned tree clean. Verify with `git -C "$OSS_ROOT" status
  --short` and record that too.
- **Disclosed in the report**, as a named caveat next to the numbers,
  not a footnote in the notes file.
- **Symmetric in intent.** If BioIR is the side that drops a supplied
  template, patch BioIR instead — the goal is both sides honouring
  the input, not making OSS match a BioIR quirk. A BioIR-side
  deviation from its reference implementation is also a bug worth
  filing, separately from the bench.

If a patch is the only way to attach templates at all on some side,
that sample's `template_status` is `attached_via_patch`.

## Worked mappings

**BioIR.** Structure-direct: `Template(path=..., format="cif",
chain_id=None)` on `Polymer.templates`, protein chains only. BioIR
runs no HHsearch / HMMsearch — pass hits you already have
(`docs/ref/support-matrix.md`, "Templates (caller-supplied CIF)").
`chain_id=None` auto-selects the best-aligning chain, matching the
spec's `null`. The parser stage loads file content, so a path is
enough. Snippet: [samples.md](samples.md#load-path-processor).

**OpenFold3.** Follow the pinned variant's complete CIF mapping,
selection-floor, cache, patch, and populated-slot contract in
[models/of3.md](models/of3.md#templates).

**Protenix-v2.** Pinned OSS needs a synthesized `templatesPath`
alignment per supplied CIF; follow
[models/protenix.md](models/protenix.md#caller-supplied-templates).

**OSS feature path (OSS pipeline + BioIR module).** One OSS-featurized batch
feeds both forwards, so template parity is structural; verify once on
the shared batch ([no-pipeline.md](no-pipeline.md)).

For any other tree, run [the procedure](#the-procedure) and add the
answers here.

## Verification

Shape checks cannot see a dropped template. Models with fixed template
slots can fill every unused slot with GAP-restype, all-zero-mask
placeholders, so tensors are identically shaped either way and the
forward simply runs on empty values.

Assert on **values and counts**, per sample, per side, outside the
timed window:

- **how many** template slots are populated —
  `(template_backbone_frame_mask.sum(dim=-1) > 0).sum()` or the
  equivalent per-slot reduction — equals
  `min(n_supplied, cap)`. This is the assertion that catches a
  filter; a plain `mask.sum() > 0` passes when three of four
  supplied templates were silently dropped
- the mask is all zero exactly when the spec listed none
- both sides report the same populated-slot count, and the same
  entry and chain per slot

The cheapest place to observe this is the feature generator itself:
wrap it, record the per-slot occupancy per sample, and assert after
the run. Featurization is outside the timing window, so instrumenting
it does not perturb the measurement — but make sure the recorder
actually fired, or a default of zero will read as a drop.

### Diff the two sides' decisions, then remove the disagreement

Each side applies its own **selection threshold** (is this template
similar enough?) and its own **keep/drop rule** (is the alignment
usable as featurized?). These are independent implementations, so
they disagree — most often on partial-coverage templates. For custom
templates both gates are inappropriate, so the response is to
neutralize them, not to accept the difference.

Fix or patch a deviating side so both featurize the supplied set,
record the change, and file the bug separately. Falling back to
template-free on both sides is the last resort, for when a side cannot
represent templates at all.

## Failure modes

- **Structure passed where an alignment was required on a legacy
  pin** — attaches nothing, errors nowhere. Conversely, synthesizing
  an alignment after a pin adds CIF-direct input creates needless
  chain-selection and cache-key differences.
- **Fixed slot padding read as success** — shapes match, values are
  zero.
- **Some supplied templates dropped by a similarity or coverage
  filter**, which `mask.sum() > 0` cannot see. Count slots.
- **Caps differing between sides**, so one side featurizes four
  templates and the other two.
- **Verification recorder never ran**, so "mass 0" meant "not
  measured" rather than "not attached".
- **Entry id did not match the filename**, so the loader skipped the
  template (or fetched a different structure from the PDB).
- **Different template chain per side** on a multi-chain CIF.
- **Threshold or keep/drop divergence** (above), the most expensive
  because both sides report success.
- **Generated inputs written into `$DATASET_ROOT`**, invalidating the
  verified dataset.

## Recorded fields

On `sample_manifest.json` and both result files:

- `has_templates`, `templates[]` — resolved absolute paths, with
  `chain_id` when the spec pins one
- `template_status` — the vocabulary above
- `n_templates_supplied` and `n_templates_attached` — the counts the
  comparison rests on, per side
- the selected entry and chain per slot, and any synthesized
  alignment path

In `bench_config.json`, record the ingestion form, the flags you set,
the structure directory, the cap and every filter value on **both**
sides, and `oss_patches` (plus any BioIR-side patch), so the next run
reproduces the same inputs without re-deriving them.
