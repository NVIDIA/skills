# Cosmos video-supervision runtime data contract

The workflow has no dataset default. Every request supplies annotation and
media paths for training and validation. Preserve the submitted string and add
an accessible resolved path; never replace a missing input with another file.

## Structural family: video conversation

Each split has one JSON annotation array and one media root. Every item must
have a media field and at least two
conversation turns. Validation checks the complete manifest, all referenced
media, unique logical records, nonempty splits, train/validation overlap, and
record/media fingerprints. A directory name alone is not enough: annotation
and media mappings must be explicit in the generated backend spec.

## Structural family: task-aware video reasoning

Each split accepts one or more annotation files and one shared media root or
one media root per annotation. The canonical envelope is an object with
`format=tao-vl-reason-v1.0`, metadata containing the task, and an `items`
array. Task selection is optional but must produce at least one record.

Supported task names are `bcq`, `mcq`, `bcq_openended`, `mcq_openended`,
`open_qa`, `scene_description`, `video_summarization`,
`temporal_localization`, `temporal_description`, and `causal_linkage`.
Prompts and response targets come from the versioned task-aware adapter in the
selected backend runtime. Frame sampling and pixel budgets come from the
model/dataset profile or explicit user overrides.

Tasks whose metadata declares `accuracy` or `exact_match_accuracy` participate
in deterministic accuracy. Common binary-choice and multiple-choice task names
are recognized. Other tasks retain their declared text/task metrics and are
excluded from aggregate accuracy with a reason. The aggregate is
example-weighted over accuracy-defined records.

## Automatic profile discovery

Infer the structural family from annotation envelopes and record fields unless
the user explicitly supplies it. Record the inferred schema, record count,
unique media count, media reuse ratio, file extensions and byte-size summary.
When annotations contain width, height, FPS, or duration metadata, record their
sample counts and distributions. Use these characteristics—not a dataset name
or directory name—to select preprocessing, cache, and resource
profiles. If resolution metadata is absent, use a conservative model-safe
profile and require representative decoding in the training allocation before
the training child starts.

## Optional diagnostic subset and full materialization

Only an explicit user request may materialize a diagnostic subset under the
runtime-supplied results area and apply a sample limit. It records the source
manifest and fingerprint. The normal full plan reads the original runtime
annotations and rejects every sample-limit field. A diagnostic manifest is
never a full-run fallback or an automatic launch prerequisite.

Cosmos-RL may merge multiple task-aware annotations into a generated manifest while
preserving every original path and logical record fingerprint. Cosmos
Framework consumes the explicit annotation list natively. Both representations
must fingerprint to the same logical records and media before a comparison.

## Compute-node validation

Submission-host validation is insufficient for SLURM. The allocated-node
preflight repeats readability checks for annotations, media, prepared model,
cache, results, checkpoints, and SQSH through the exact container mounts. It
also decodes representative media with the selected GPU decoder. A missing
mount or unreadable file blocks the full allocation.
