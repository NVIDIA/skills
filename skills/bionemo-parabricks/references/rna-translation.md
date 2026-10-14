# STAR argument translation

Use `scripts/translate_star_to_parabricks.py` for Parabricks 4.7.0 or 4.7.1
alignment drafts. It reads a JSON request object or array and never opens the
FASTQs, invokes the decompressor, contacts a service, or launches Parabricks.
Requires Python 3.7 or newer; no third-party packages are needed.

## Request schema

| Field | Meaning |
| --- | --- |
| `version` | Explicit `4.7.0` or `4.7.1`; this is the software release, not an image tag |
| `star_args` | Original STAR argv as strings, including every flag/value; optional leading `STAR` |
| `ref` | Target reference FASTA |
| `output_dir` | Directory for STAR's generated outputs |
| `out_bam` | Separate final sorted BAM path |
| `name` | Optional identifier copied into the result |
| `mark_duplicates` | Optional boolean, default false to preserve STAR-only processing |

Read the supplied manifest and derive these fields from its actual values.
For a batch, normalize every job to this schema and pass the whole array once.
Do not relabel a requested release to bypass validation. Preserve paths with
spaces, comma-separated lane lists, read-group tokens, negative numbers and
ordered vector values. The helper retains the prefix basename while using the
new output directory. It keeps a decompression command as one argument and
leaves GPU worker-thread tuning at auto instead of copying STAR's CPU count.

## Example

Run from the skill directory with an existing request file:

```bash
python3 scripts/translate_star_to_parabricks.py --input requests.json
```

The CLI also accepts `--input -` for JSON on stdin. Use the helper to preserve
long argument lists, then review its output against the supplied manifest.

## Result and review

The output shape matches the input (object or array). Each result has `status`,
`draft_argv`, shell-quoted `command`, `notes`, `issues` and `untranslated`, plus
`name` when supplied. `draft` is a candidate; `needs_review` includes changed or
unsupported semantics; `invalid` has no command. Unknown STAR flags remain in
`untranslated` and are omitted from the draft, never silently passed through.

Map results to the user's requested artifact schema, retain every job ID, and
read back the written file. Check input modes, lane-to-read-group assignments,
decompressor text, output paths and every non-default setting against the source
manifest. Keep unresolved options visible. A valid JSON file alone does not
prove a correct translation. Exit 0 permits reviewable drafts; exit 2 means at
least one invalid request or an input/usage error.

The helper does not verify numeric ranges, index compatibility or biological
equivalence. Generated `command` strings are unvalidated drafts: the user must
review them before authorizing execution. See the RNA tool reference for STAR
output-contract limitations and index provenance. Single-cell and unlisted
options require manual review.
