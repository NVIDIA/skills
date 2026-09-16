# Site Data And Path Handling

Load this reference when a conversion must generate site partitions, resolve a
relative source data path, or give sites different data locations. The
always-applicable data-location invariants live in the "Data Location" section
of `conversion-common.md`; this reference adds the detailed mechanics.

## Site Data Partitioning

Preserve an existing user-provided site split. If no split exists, create
deterministic site-local training partitions unless the user explicitly asks all
sites to train on shared data. Prefer a seeded shuffle and use a stratified split
when classification labels are available. Do not default to stride or contiguous
splits because ordered data can produce biased sites.

For pandas `DataFrame` inputs, split positional row indices first, then build
each site frame with `df.iloc[positions]` or equivalent. Do not apply generic
array chunking directly to a `DataFrame`; library versions can return chunks
that no longer behave like data frames.

Make every array passed to an in-place shuffle writable. For label-stratified
partitioning, derive positional indices from the observed label column and copy
them before shuffling:

```python
positions = np.flatnonzero(frame[label_column].to_numpy() == label).copy()
rng.shuffle(positions)
site_frame = frame.iloc[positions]
```

Do not shuffle a possibly read-only array returned by `Index.to_numpy()`, and do
not pass positional indices to `DataFrame.loc`.

Report the split policy, seed, site count, and any reason stratification was not
used. Do not copy private site data into generated artifacts unless the user
explicitly requests it.

## Data Location

Apply the `conversion-common.md` "Data Location" invariants first: a configurable
`train_args` value or `per_site_config` entry, never a path hardcoded in
`client.py`, and never a path into the run workspace. Keep it site-overridable so
the conversion ports to real multi-site deployment, where each site's data lives
at a different location.

Classify each detected input before adapting `job.py`, preserving the source's
argument name and semantics:

- For a source-project-relative local file or directory, resolve it in `job.py`
  against an explicit absolute source-project root before Recipe construction.
  Hugging Face conversions use the maintained `resolve_source_local_path()`
  helper and pass that root through its `source_root` argument. Use `SOURCE_DIR`
  as that root only when the generated `job.py` is colocated with the inspected
  source project. When a read-only source root forces generated files elsewhere,
  pass the original inspected root through a configurable job-level value; never
  substitute the generated output directory or the caller's working directory.
- Preserve an absolute local file or directory path; the helper accepts it
  without prefixing the explicit source root.
- Pass a per-site path through `per_site_config`. Do not resolve it on the
  central authoring machine.
- Do not pass a Hugging Face Hub identifier or URL through a filesystem
  resolver.
- When the source has no file or directory argument, generate no path-specific
  option, helper call, or Recipe argument.

For example, if source inspection finds a local `dataset_path` argument, adapt
the maintained Hugging Face `job.py` asset without renaming it:

```python
parser.add_argument("--dataset_path", type=Path, default=DEFAULT_DATASET_PATH)
source_root = SOURCE_DIR  # Only when generated job.py is colocated with the source.
dataset_path = resolve_source_local_path(args.dataset_path, source_root=source_root)

recipe = build_recipe(
    ...,
    dataset_path=dataset_path,
)
```

For a read-only source root and a separate generated directory, expose or
otherwise preserve the inspected absolute source root in `job.py` and pass it as
`source_root`. Do not silently default it to the directory containing the
generated `job.py`.

Pass the resolved value through `train_args`; the client must consume it
unchanged. Do not reinterpret it relative to packaged `client.py`, the export
directory, or the simulator process working directory. Validate relative-path
conversions from a fresh caller working directory outside the source project.

When `per_site_config` supplies `train_args`, each site value completely
replaces recipe-level `train_args`; it is not a fragment to merge. Compose the
shared options and site data path into every override, then apply the validation
in `pytorch-family-recipe-construction.md` before a full simulation.
