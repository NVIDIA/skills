---
name: nvshmem-docs
description: Find version-aware official NVSHMEM and NVSHMEM4Py documentation for releases, installation, APIs, runtime settings, transports, containers, and troubleshooting.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA NVSHMEM Team <nvshmem@nvidia.com>
  tags:
    - nvshmem
    - docs
---

# NVSHMEM Documentation

## Purpose

Find and summarize version-aware official NVSHMEM and NVSHMEM4Py documentation. Fetch and cite the most specific live pages; do not make technical claims from memory, search snippets, or the bundled topic map.

## Prerequisites

Require [topic-map.md](references/topic-map.md) and a retrieval tool that can open public NVIDIA HTTPS documentation. No API key or account is required. Support only NVSHMEM versions whose major version is 3 or later; reject a requested version below 3 before looking up documentation.

## Instructions

Read the topic map for every request to resolve the topic and candidate pages. Use it only for routing; it is not evidence.

### Choose the documentation edition

Normalize an optional leading `v`. Use `latest` when no version is requested. For an exact `MAJOR.MINOR.PATCH` version, first use its archive. Replace the literal `VERSION` in each archive URL below with the normalized NVSHMEM version (for example, `3.7.1` becomes `https://archive.docs.nvidia.com/nvshmem/api/3.7.1/index.html`):

| Content | Latest | Exact-version archive |
| --- | --- | --- |
| API reference, concepts, usage, and runtime settings | `https://docs.nvidia.com/nvshmem/api/latest/` | `https://archive.docs.nvidia.com/nvshmem/api/VERSION/index.html` |
| Release notes, installation, best practices, and support | `https://docs.nvidia.com/nvshmem/release-notes-install-guide/` | `https://archive.docs.nvidia.com/nvshmem/VERSION/index.html` |

For a `MAJOR.MINOR` value, ask for the patch only when an exact archived edition is necessary. Start at the selected root and follow its table of contents because internal paths can differ between editions. For cross-cutting questions, use both documentation sets from the same edition.

When an exact archive is unavailable, inspect the relevant `latest` root and its official release/version information. If it identifies the requested version as the current release, use that latest documentation and state that the requested version is served from `latest` rather than an archive. If `latest` identifies a different version, report that the requested archive is unavailable; offer the different latest documentation only as clearly labeled current context. Never silently use latest documentation for another version.

### Retrieve and answer

1. Infer the topic and optional version from the user or invoking skill, then route it with the topic map.
2. Open the appropriate root above and the relevant topic page in that edition. If an exact archive is unavailable, resolve whether `latest` is the requested version before using it as evidence.
3. Answer concisely with direct official links. State the documentation version when one was requested or applicability matters.
4. If an exact archive cannot be retrieved and latest is a different version, say so. Offer clearly labeled latest context only if useful.

Keep answers user-facing. Do not disclose internal retrieval mechanics or attempted-source details; report only conclusions, applicable versions, evidence links, and material uncertainty.

If live pages cannot be retrieved, return the appropriate root links, state that their contents were not verified, and avoid documentation-backed technical claims.

### Handle NVSHMEM4Py

For an NVSHMEM4Py request with no version, use the latest documentation. If an NVSHMEM version is requested, use that NVSHMEM documentation edition without resolving the NVSHMEM4Py package version. Only when an NVSHMEM4Py version or version compatibility is requested, resolve the independent version pair from the live compatibility guide—never infer or hard-code it—and use the resolved NVSHMEM version to select an archive. Ask only when a bare version is ambiguous or multiple NVSHMEM releases match.

Unprefixed topic-map gates apply to NVSHMEM. Prefix a Python-only gate with `nvshmem4py:`, compare it only with the NVSHMEM4Py version, and name the version axis in the answer.

### Check version gates

When a routed topic-map row has `introduced_in` or `removed_in`, verify the gate from its live official evidence and always report it. Compare NVSHMEM gates by `MAJOR.MINOR` and `nvshmem4py:` gates by the full package version. Report a mismatch when the requested version is before `introduced_in` or at/after `removed_in`; otherwise report no mismatch. Use a compact form such as `Version gate: introduced in 3.7; requested 3.5 — mismatch.`

Do not perform a broader availability search merely because a version was supplied. Only do so when the user explicitly asks whether an ungated feature or exact symbol is available, introduced, removed, or supported. Then inspect the requested archive and matching release notes. Absence from a page or search result is inconclusive; if evidence is not decisive, say availability could not be established.

For NVIDIA HPC SDK containers, use the official NGC catalog and HPC SDK documentation for container procedures. Do not infer the bundled NVSHMEM version from an image tag unless an official page states it.

## Limitations

- Documentation does not prove installed binaries, container contents, configuration, runtime behavior, or performance.
- A missing page, symbol, or search result does not prove unavailability.
- Technical answers require retrieved live official evidence; the topic map is only a routing index.

## Troubleshooting

| Error | Cause | Solution |
| --- | --- | --- |
| Topic-map route fails | Documentation moved | Navigate the selected edition's live table of contents. |
| Exact archive fails | The edition is unavailable, is the current unarchived release, or retrieval failed | Inspect `latest` to identify its version. Use it only when it matches the request; otherwise report the failure and offer it only as clearly labeled current context. |
| Requested NVSHMEM version is below 3.0 | Unsupported documentation edition | Reject the request; do not search archive or latest documentation. |
| Version is ambiguous | Component was not named | Ask whether it is an NVSHMEM or NVSHMEM4Py version. |
| Compatibility is unresolved | No unique live compatibility row exists | Report the candidates or uncertainty; do not infer a pair. |

## Examples

```text
Use $nvshmem-docs to explain NVSHMEM_THREAD_SERIALIZED.
Use $nvshmem-docs to find NVSHMEM4Py installation documentation for NVSHMEM 3.5.21.
Use $nvshmem-docs to check whether NVSHMEM4Py 0.2.2 is compatible with NVSHMEM 3.6.5.
```
<!-- ci trigger -->
