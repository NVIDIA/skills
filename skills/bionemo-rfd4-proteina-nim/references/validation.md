# Validate and interpret a run

## Completion and artifacts

Read `transport.json` first. `prepared_not_submitted`, `pending`, and `failed`
are not completed inference. `source=saved_response` means extraction only.
For a live run, retain the endpoint, timestamps, request/response hashes,
HTTP status, and returned request IDs. Report actual generated artifacts,
not example output or a previous directory's results.

`summary.json` must have no `validation_errors`. The client checks nonempty
designs, task consistency, ordinary-sampling count, origins, model/runtime/
schedule objects, requested output presence, a basic mmCIF header, sequence
text, and `diagnostics.all_coordinates_finite is true`. It does not parse the
full CIF or independently prove physical validity. Failed output remains
available with `status=failed_validation`.

Open each structure with an mmCIF-aware parser. Check that atom-site coordinates
are finite and atom/residue counts are plausible. Preserve annotated mmCIF for
condition mapping; conversion to PDB can discard annotations and molecular
context. The [public API](https://build.nvidia.com/ipd/rfd4-proteina) defines the
response fields; the [model card](https://build.nvidia.com/ipd/rfd4-proteina/modelcard)
explains the scope of its own functional tests.

## Biological conditions

- Confirm that the target, framework, and held motif coordinates and identities
  were retained according to the actual input conditions.
- Use input/output annotations to identify mapped residues and insertions;
  generated chains can be renumbered. Do not derive chain identities from an
  opaque design ID or assume the first returned sequence is only the binder.
- For binders, establish target/binder chain mapping before interpreting contacts
  or chain-specific metrics. Check whether `chain_of_interest` names the intended
  designed chain.
- For antibody or nanobody design, verify framework retention and loop changes.
  Unresolved source atoms cannot be held at finite coordinates merely because
  the residue appears in the input; preserve identity separately where needed.
- For partial diffusion and inverse folding, inspect `input_ca_rmsd_angstrom`
  and `sequence_recovery` when available. Null values mean unavailable results,
  not zero RMSD or zero recovery. Compare requested and returned schedules and
  report inconsistencies without guessing their cause.

## Metrics and scientific quality

Report relevant geometry, clash, bond-length, motif, sequence-adherence, and
interface metrics without inventing universal thresholds. Finite coordinates
alone do not establish a physically sound or experimentally useful design.

An empty metrics dictionary despite requested tags deserves a warning. A null
metric is not a measurement. `aggregated_metrics[].n` counts contributing
values and may be smaller than the returned design count; include both counts.
A `.pass` aggregate from two of ten designs is not a ten-design success rate.

Keep final search survivors and retained intermediates separate. Compare search
rewards only under compatible metric/weight configurations. Selection by a
reward does not independently validate that same reward or experimental binding.

The documented API has no built-in refolding oracle. Inspect `model.oracle`
and actual predictor results before reporting refolding confidence. Independent
refolding, interface analysis, and experiments are separate evidence. Do not
infer pLDDT from a search reward, a generic score, or a B-factor column.

## Report and reproduce

Report request identity, destination and live/offline status, actual design
count, artifact paths, backend and schedule, warnings, failed replicates,
metric values with denominators, and whether independent validation ran.
Keep full sequences in FASTA and label any displayed truncation as a preview.

For reruns, preserve input bytes, weights, runtime, batch size, schedule, and
seeds. Search requires replaying the whole search. Save an edited request as a
new artifact and use a new output directory. Preserve pending/error diagnostics
and establish service state before deciding to submit again.
