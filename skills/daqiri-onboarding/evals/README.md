# Operational evaluation

These tasks use an external BYOE harness that provisions the read-only `input/` fixtures, matching source/native tools and an empty `solution/` directory. Hardware cases additionally receive a bounded lab tool through `DAQIRI_LAB_CHECK`.

The submitted `eval-result.json` records the evaluated commit, frozen dataset/input identities, attempt policy and measured baseline/skill results. Raw captures, transcripts and host inventories are retained outside this repository. CI imports the supplied report and regenerates publication reports and signatures.
