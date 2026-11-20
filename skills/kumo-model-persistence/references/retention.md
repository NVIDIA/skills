# Retain and retrieve artifacts

Select one explicit fit run and the requested diagnostic files. Use
[diagnostics](../../kumo-training-diagnostics/SKILL.md) and
[explanations](../../kumo-model-explanations/SKILL.md) to identify trial/refit and stage.
Do not recursively upload all scratch: it can contain source data, other runs and credentials.

| Artifact | Retain as |
| --- | --- |
| Aggregate metrics and run metadata | JSON with metric names, run/split identity and finite values |
| Epoch histories and progress | Original JSON; optional normalized scalar CSV for analysis |
| Per-trial validation results | Original CSV and selected plan/selection metadata |
| Gradient scores | CSV, retaining table and column names |
| Column cohorts | Original nested JSON, preserving class series and population shares |
| Prediction and embedding selections | Every selected Parquet file plus selection/run metadata |
| Portable model | Complete bundle via registry or explicit local-bundle retention |

JSON, histories and embedding vectors are not interchangeable with scalar table rows. Use
an internal Snowflake stage to preserve original files. The
[output reference](../../kumoai-reference/references/outputs.md) explains prediction table serialization.

For a fresh fit, write aggregate metrics as an explicit selected file under the run directory.
This fails on nonfinite values rather than emitting nonstandard JSON; retain missing values
as null and investigate malformed metrics before export. Use a new filename to avoid replacement.

```python
import json
from pathlib import Path

metrics_path = Path(RUN_DIRECTORY) / "retained-metrics.json"
metrics_payload = json.dumps(training_result.metrics, allow_nan=False, indent=2)
with metrics_path.open("x") as handle:
    handle.write(metrics_payload + "\n")
```

## Create a manifest-backed archive

The packaged [archive helper](../scripts/retain_artifacts.py) accepts explicit local files
under one selected run root. It refuses symlinks, paths outside that root, duplicates, missing
files and replacement of an existing archive. Inspect the selection for sensitive data first.
The archive includes byte lengths and SHA256 digests; store its own digest in the retention
record. For a model bundle select all files in the complete bundle, including its manifests.

```python
# Import after adding the packaged skill scripts directory to sys.path.
from retain_artifacts import create_archive, restore_archive

digest = create_archive(RUN_DIRECTORY, SELECTED_RELATIVE_FILES, ARCHIVE_PATH)
print("archive SHA256:", digest)
```

`SELECTED_RELATIVE_FILES` is a list of actual relative paths, not wildcards. Preserve the full
relative path for identically named trial files. Use a new run-specific stage prefix and an
explicit archive name. Check the stage's location and access policy before transferring.

With the notebook's active Snowpark `session` and a selected internal `STAGE_PREFIX`:

```python
transfers = session.file.put(
    str(ARCHIVE_PATH), STAGE_PREFIX, auto_compress=False, overwrite=False,
)
if len(transfers) != 1 or transfers[0].status != "UPLOADED":
    raise RuntimeError(f"Archive upload did not create one new object: {transfers}")
```

A `SKIPPED` result is not proof that the existing object has identical bytes. Do not enable
overwrite to hide that conflict. Choose a new destination or verify the existing object.
Record the exact stage object path, digest, run identity and selected files outside ephemeral
scratch. Do not record authentication material.

## Retrieve an explicit archive

Use a fresh empty download directory and an exact stage object, not a stage-wide wildcard.
Download status must be successful. Then verify the saved archive digest and each file before
restoring into a new directory:

```python
received = session.file.get(STAGE_OBJECT, str(DOWNLOAD_DIRECTORY))
if len(received) != 1 or received[0].status != "DOWNLOADED":
    raise RuntimeError(f"Archive retrieval failed: {received}")
restore_archive(DOWNLOADED_ARCHIVE, RESTORE_DIRECTORY, EXPECTED_ARCHIVE_SHA256)
```

`DOWNLOADED_ARCHIVE` is the exact downloaded local file. The helper rejects malformed
manifests, archive traversal, duplicate members, unexpected files and checksum mismatches.
Set a storage budget before restoring large files. Once verified, point the diagnostics
readers at that explicit restored run. Transfer success alone is not a complete restore check.
