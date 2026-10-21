# Target configuration fixtures

The target registries are synthetic minimal entries using the project schema.
No protein structures, validated motifs, or scientific results are supplied.
Preserve existing entries; write requested changes to the output copy.

Supplemental pipeline configurations are unmodified source snapshots, with
revision and file hashes in `source_manifest.json`. They support inspection of
the downstream consumer and schema in both evaluation conditions. This partial
snapshot does not contain the dependencies or inputs for running a campaign.
