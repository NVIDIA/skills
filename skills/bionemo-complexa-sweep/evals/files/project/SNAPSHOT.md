# Sweep configuration review inputs

The original three configuration files are unmodified source snapshots from
Proteina-Complexa revision 708150f, supplied to both evaluation arms for offline
key lookup. Supplemental pipeline and generator source snapshots are identified
by revision and file hash in `source_manifest.json`; they make the command
interface and generated output paths available for inspection in both conditions.

These files are not a complete Hydra installation. They support preparation and
review of future commands, not a claim of tested configuration materialization
or GPU execution. The sibling CSVs are synthetic fixtures and do not measure
biological design quality.

The sibling `launch_review/` is an authored draft for an offline launch-plan
review task, not an upstream source snapshot or an executed campaign. The
`summary_review.csv` fixture represents a synthetic summary export with missing
and inconsistent data. Each task's declared inputs are supplied unchanged to
both evaluation conditions.
