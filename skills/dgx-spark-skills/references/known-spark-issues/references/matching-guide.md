# Matching guide

## Current source set

Refresh the relevant pages at use time and record the retrieval date:

- [DGX Spark Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) — active documented issues, expected behavior, and workarounds.
- [DGX Spark Release Notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html) — current Founders Edition component versions and issue resolution history. OTA packages are available to all devices when NVIDIA publishes an update; OEM partner firmware is released independently and may lag.
- [OS and Component Update Guide](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html) — Founders Edition update method and safeguards.
- [DGX Dashboard](https://docs.nvidia.com/dgx/dgx-spark/dgx-dashboard.html) — current Dashboard behavior and update access.
- [NVIDIA Sync User Guide](https://docs.nvidia.com/sync/latest/) — current Sync concepts and supported flows.
- [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) — supported GB10 device, release, node-count, topology, and readiness conditions.
- [ConnectX-7 Networking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html) — physical and interface context for multi-Spark issues.
- [DGX Spark support](https://docs.nvidia.com/dgx/dgx-spark/support.html) — support and field-diagnostic routing.

For supplemental forum leads, use the search and author-labeling guidance in [get-spark-support/WORKFLOW.md](../../get-spark-support/WORKFLOW.md).

## Applicability matrix

Use this matrix to select the relevant user-evidence fields for the report.
Use `Unknown` for absent evidence; retain what the user actually said.

| Dimension | User state | Source scope | Match rule |
| --- | --- | --- | --- |
| Model | Founders Edition, recognized OEM GB10, unknown | Named model or stated system class | Apply shared software guidance where relevant; do not extend model-specific firmware, update procedures, or recovery claims to other systems. |
| Software | DGX OS/OTA plus affected component versions | Introduced, affected, fixed, or current version | Compare the installed version with this issue's documented boundary; fix inclusion alone does not establish a symptom match. |
| Symptom | Exact error and observed behavior | Source symptom or documented expected behavior | Prefer exact errors; distinguish expected behavior from defects. |
| Operating mode | Local, display-attached, headless | Display/session/headless condition | Do not infer a headless workaround from a display issue or forum anecdote. |
| Connection path | Client application, SSH target/alias, direct/managed/tailnet route | Documented connection mechanism | Application and route are separate facts: terminal SSH can use a managed alias. An unopened UI does not establish that its background connection is unused. |
| Multi-Spark | Node count, per-node versions, topology, failing node | Supported count/topology/version and affected component | Evaluate every node; one mismatched node can make cluster-wide applicability uncertain. |

For each individual candidate:

1. Extract its documented symptom, model/operating scope, and affected or fixed
   version boundary. Use `unknown` for information the source does not establish.
2. Compare the user's installed version with that issue's boundary, not with the
   latest release. Report `Unknown`, `Predates fix`, `Includes fix`, or
   `Not comparable`. Predating a fix is only potential version applicability;
   do not invent an affected range or compare unrelated release branches.
3. Assess symptom and operating-condition correspondence separately against the
   report's user-evidence fields. Mark each material condition `Established`,
   `Contradicted`, or `Unknown`. A contradiction must cite a supplied user fact,
   not an assumption about a missing field. Retain a documented condition with a
   plausible symptom connection when its scope is unknown; keep indirect symptom
   connections conditional. An unrelated condition is not a candidate.
4. Select the result label below, then use the workflow's Report format. Keep
   each issue's version condition attached to it in both the row and summary.
   Never use one latest-release cutoff for issues fixed in different releases.

For example, “Linux terminal; Sync running; drops without opening Sync” establishes
an application and timing, not an unmanaged route. SSH target/alias and network
path remain `Unknown`. Normal terminal SSH can resolve Sync-managed entries through
SSH configuration; see [client-access.md](../../use-sync/references/client-access.md).
Conversely, “direct LAN IP, no alias/proxy/tunnel/Tailscale” supplies route evidence
that can contradict a managed-tunnel condition. Do not request device inspection
to fill these fields without consent.

Use these result labels:

- `Current documented match` — all material applicability fields match the user's system and installed versions, even if only release history documents the issue and a newer release fixes it.
- `Resolved historical match` — symptom and scope correspond to the documented historical issue, and the user's installed release includes its fix. This does not establish the cause of a symptom persisting on that release.
- `Expected behavior` — NVIDIA documents the observation as normal for the platform.
- `Possible match — evidence needed` — a concrete symptom connection remains, no known fact excludes the candidate, and material applicability evidence is missing.
- `Excluded candidate` — a known user fact contradicts a documented required condition; state both.

Derive the overall finding after assessing candidates and source coverage:

- Report established matches using their labels above; preserve uncertainty for
  any remaining candidates. A historical match does not establish a current cause.
- With no established match, use `No confirmed match yet` if a relevant candidate
  remains unresolved or a required source could not be checked. Name the specific
  missing evidence or source, even if the candidate is described as a condition
  or lead rather than given a result label.
- Use `No NVIDIA-documented match found in the checked sources` only when the
  relevant sources were checked and no relevant candidate remains unresolved:
  either no documented behavior has a concrete connection to the symptom, or
  known facts exclude the candidates. Name the checked scope. Unrelated unknowns
  and merely imaginable causes do not block this finding.

## Risk review examples

Risk depends on the proposed action, not on whether it appears on a Known Issues page.

- Reading a displayed error or user-provided version: no execution.
- Running a version query: `Observe`; still obtain diagnostic consent.
- Saving logs: `Collect`; define files, duration, destination, redaction, and retention before permission.
- Flushing caches, restarting an application, changing cables, or rebooting: `Change`; explain disruption and obtain permission.
- Package, driver, kernel, firmware, UEFI, update, or recovery operations: `Change` or `Destructive`; use the current NVIDIA procedure, expose prerequisites and data risk, and obtain explicit permission immediately before action.

If a source presents a command without rollback, say `Rollback not documented in the cited source`. Do not manufacture one.
