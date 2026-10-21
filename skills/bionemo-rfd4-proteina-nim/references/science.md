# Scientific scope

The [public RFD4-Proteina model card](https://build.nvidia.com/ipd/rfd4-proteina/modelcard)
describes joint structure and sequence generation through latent flow matching.
The NIM accepts conditioning, samples or searches trajectories, and returns
annotated mmCIF, sequences, diagnostics, and requested metrics. Unlike a
backbone-only workflow, it does not require a ProteinMPNN handoff to produce
sequences. Independent sequence redesign or refolding can be separate steps.

Conditions compose: the task selects the sampler while the input identifies
fixed coordinates, sequences, motifs, hotspots, and designable regions.
A protein binder uses `generation` with target and binder conditions; an
antibody request must actually preserve its framework and any intended antigen.
A task label alone does not create those conditions.

The [Deploy tab](https://build.nvidia.com/ipd/rfd4-proteina/deploy) calls out two
input constraints: fixed side-chain coordinates require fixed sequence identity,
and unresolved source atoms must not be held at non-finite coordinates. Preserve
sequence where appropriate while allowing missing coordinates to be generated.

Schema acceptance establishes admissibility, not biological qualification.
Check motif/framework retention, conditioning adherence, geometry, sequence
composition, and the relevant interface. Search optimizes configured rewards;
a higher reward does not demonstrate affinity, expression, catalytic activity,
or experimental success. Evaluate antibody and nanobody cases on evidence
appropriate to each, rather than generalizing from a binder demonstration.

The public API's `model.oracle` field identifies refolding predictors active for
the request. The documented API has no built-in refolding oracle; an operator
can supply one on a configured deployment. Requesting refolding metric tags
cannot install an oracle. Do not infer pLDDT from a reward, a generic score, or
the mmCIF B-factor column. Quote predictor confidence only when the predictor
and its actual results are present.

For independent validation, retain target/binder chain mapping and run the
requested predictor and interface checks. Use task-specific thresholds backed
by relevant evidence. The model card's functional verification and the skill's
client tests do not constitute experimental design-success benchmarks.
