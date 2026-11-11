# Known Spark Issues

Find applicable current issues; do not recite a cached list. Read [matching-guide.md](references/matching-guide.md) and refresh the relevant sources for every request.

## Gather relevant context

Search using the symptom and context already supplied, then ask only for missing fields needed to distinguish plausible matches:

- exact symptom or error, impact, onset, and last relevant change;
- exact model and whether it is NVIDIA DGX Spark Founders Edition, a recognized OEM GB10 system, or unknown;
- DGX OS/OTA, driver, kernel, firmware, NVIDIA Sync, framework, and container versions relevant to the symptom;
- local, display-attached, or headless operation;
- for multi-Spark issues: node count, models, software state per node, cabling/topology, whether NVIDIA Sync Cluster Assistant was used, and which node fails.

With explicit permission, use the `identify-spark-system` workflow for local identity. Do not run commands or inspect a device without diagnostic consent.

## Find and rank matches

1. Read the relevant NVIDIA sources listed in the matching guide; fetch only those needed for the symptom.
2. Use the guide's applicability rules to compare the user's symptoms, installed versions, model, and operating conditions, then rank the matches.
3. If forum leads would help, follow the search and author-labeling guidance in [get-spark-support/WORKFLOW.md](../get-spark-support/WORKFLOW.md). Forum posts provide leads; they do not establish an NVIDIA-documented match.

Select sources for every plausible connection path, including paths not yet
confirmed by the user. For SSH or remote-access symptoms, when NVIDIA Sync
involvement is confirmed or unknown, check the relevant Sync connection
documentation and Sync release history alongside Spark Known Issues and Release
Notes. Missing connection details limit applicability; they do not exclude that
source family.

Bound negative findings to the sources actually checked. If relevant sources
remain unchecked or unavailable, state that limitation and retain uncertainty
rather than concluding that no NVIDIA-documented issue exists.

Select up to three individual documented issues. Omit lower-ranked candidates rather than merge different fixes to meet this limit; omission is not exclusion. Derive the overall finding using the matching guide. Never fill a gap with a remembered issue, community workaround, or speculative convenience advice.

## Permission and command review

Before presenting a workaround command or action, self-review it as:

- `Observe` — reads state only;
- `Collect` — captures logs, dumps, files, or conversation context;
- `Change` — installs, updates, edits, restarts, reboots, power-cycles, changes cables, or changes firmware/UEFI;
- `Destructive` — can erase, overwrite, reimage, purge, or materially risk data.

State the classification, scope, prerequisites, side effects, and rollback if the NVIDIA source provides one. Require explicit permission before diagnostics, collection, or any device-changing action. A request to identify an issue is not permission to execute its workaround. Never invent a rollback.

## Report

Write the report in this order: user evidence, candidate comparisons, then one
finding with the remaining questions. Derive the finding from the completed
comparisons; do not prepend a separate yes/no answer or add a second summary.

First show a compact evidence table for the fields used in the comparisons:

| Field | User evidence |
| --- | --- |
| Relevant version, symptom, or operating condition | Short quotation from the user's report, or Unknown |

For SSH symptoms, distinguish the client application, SSH target/alias, and
network path in separate fields. If a power/shutdown condition is considered,
include adapter and device power/reboot state. A source describes what could
happen; it cannot supply a missing fact about this user's system.

Then compare up to three relevant individual issues, one per row. Keep different
fix boundaries separate. For version-independent conditions use `Not comparable`
in the version comparison, and assess their scope in the same way.

| Issue and source | Fixed in | Installed | Version comparison | Scope comparison | Result |
| --- | --- | --- | --- | --- | --- |
| Individual issue and direct source link | Fix version, Unknown, or N/A | User's version, or Unknown | Unknown / Predates fix / Includes fix / Not comparable | Established / Contradicted / Unknown: documented condition and supporting user evidence | Matching-guide label |

`Contradicted` requires both a documented required condition and contrary user
evidence from the first table. A missing field is `Unknown`, not a contradiction.
If the source does not establish a symptom connection, say it is not established;
do not invent a narrower scope to exclude it. Omit unrelated issues rather than
listing them solely to reject them.

Finish with the matching guide's overall finding. If any relevant scope row is
`Unknown` or a required source remains unchecked, preserve that uncertainty in
the finding. Ask only for facts that could change a retained row or source gap.
An included fix alone establishes neither a historical symptom match nor the
cause of a persistent symptom. If an action is relevant, summarize its official
resolution, risk, and required permission without executing it.
