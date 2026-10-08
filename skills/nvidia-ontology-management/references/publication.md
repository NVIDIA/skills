<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Governed publication, readback, and rollback

Use this workflow to apply a reviewed semantic model or make validated
analytical results reusable. Publication is a durable governed write followed
by exact readback. A chat response, chart, local receipt, or successful request
is not publication.

MCP is read-only and cannot publish. Use only an approved REST or source writer
with the caller's own permissions. Publication may expose a recommendation; it
must not imply approval, commitment, or execution without a separate authorized
event.

## Choose an honest publication mode

- **Receipt only:** retain validated artifacts locally when no approved writer
  exists. Report explicitly that nothing was published.
- **Native semantic model:** use scoped Auto Ontology model export/import for Terms,
  attributes, represented tables, and semantic relationships.
- **Reusable definition:** use a validated CustomAnalysis for supported SQL or a
  PqlAnalysis for supported PQL. These store definitions, not arbitrary result
  rows.
- **Governed result table:** write immutable, keyed result partitions to an
  approved source table outside Auto Ontology, then ingest and model that table for reuse.
  Auto Ontology does not provide a generic MCP or REST endpoint for arbitrary result-row
  writeback.

Do not use conversation persistence as a result store. If none of the modes can
meet the requested durability, identity, or permission requirements, stop at a
receipt.

## Pre-publication contract

Bind and validate:

- source, catalog, and semantic model identity;
- exact physical and semantic IDs and approved joins;
- run, result, entity, scenario, and parent identities;
- population, measure definition, unit, grain, and time basis;
- producer/model/solver and input/definition/result hashes;
- evidence class, uncertainty, validity, status, and supersession;
- writer identity, authorized scope, destination, and rollback target.

Keep observed, computed, structural, modeled, predicted, recommended, and
committed evidence distinct. Store run totals at run grain and complete scenario
memberships when downstream joins require the full set.

## Native semantic changes

1. Export the exact database-ID scope and retain the original YAML plus SHA-256
   as the rollback artifact.
2. Preserve live physical IDs; generated IDs are not natural-name upserts.
3. Validate the proposed diff and affected queries.
4. Apply first in an isolated environment through the public API.
5. Re-export the same scope and compare exact IDs, structure, properties,
   relationships, and counts.
6. Run positive and negative readiness, answerability, SQL, row, and answer
   probes.
7. Promote only the reviewed artifact; on mismatch, quarantine it and restore
   and verify the prior state where supported.

Fail closed around known product limitations:

- [issue #250](https://github.com/NVIDIA/auto-ontology/issues/250): replacement may
  retain an omitted semantic FK;
- [issue #265](https://github.com/NVIDIA/auto-ontology/issues/265): replacement may
  retain existing properties instead of updating them;
- [issue #267](https://github.com/NVIDIA/auto-ontology/issues/267): a different physical
  ID with the same natural database name may surface as HTTP 500.

Do not bypass these failures with direct Postgres edits. A `success: true`
response and payload-oriented summary counts do not prove persisted parity.

## Governed analytical results

For a governed result table, use stable `(run_id, result_id)` and business entity
keys, append-only run partitions or transactional staging, explicit units and
grain, source/parent lineage, status, validity, and supersession. Verify row
count, key uniqueness, hashes, and full-precision values before ingestion.

After ingestion, map results back to the original business entity with
compatible typed keys. Compile the semantic layer and prove fresh readback from
the published destination. Test invalid, stale, superseded, unknown-run, and
ambiguous-measure cases; they must not silently fall back to the latest row.

## Required receipt

Retain the requested scope, semantic input identity, writer class, permission
result, destination or definition ID, payload/artifact hash, exact before/after
diff, row/key/hash verification, compilation identity, post-write SQL and rows,
and rollback target. Never retain tokens, passwords, or connection strings.

Current Auto Ontology certification fields are not an atomic review/promotion system.
Until that product lifecycle exists, describe the reviewed state accurately and
do not claim atomic certification, promotion, supersession, or rollback.
