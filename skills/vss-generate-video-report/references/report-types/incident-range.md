# Incident range report (Mode B)

Numbered steps for Mode B, loaded on demand from [`SKILL.md`](../../SKILL.md) (`$SKILL_DIR/SKILL.md`); routing, the shared setup and the generic prerequisite probes live there (Mode B has no mode-specific gate) and the skeleton is defined in `SKILL.md` § Report types.

- **Trigger phrases:** `SKILL.md` § Examples — the Mode B rows.
- **Prerequisite gate:** `SKILL.md` § Runtime prerequisites — the *Mode-by-mode checklist* row(s) for Mode B; no HITL step.
- **Inputs to resolve:** Step 1 below.
- **Template:** Step 3 below.
- **Output contract:** `SKILL.md` § Instructions → *Output contract for evaluators*, the Mode B lines.
- **Failure modes:** `SKILL.md` § Error Handling, plus the rules stated in the steps below.

---

## Mode B — Report on incidents in a time range

### Step 1 — Resolve the time range and (optionally) sensor

- `start_time` / `end_time` must be ISO 8601 UTC (`YYYY-MM-DDTHH:MM:SS.sssZ`). Resolve relative phrases ("last hour", "today") against the current host clock.
- If the user names a sensor, capture it as `source` + `source_type=sensor`. Otherwise leave both unset for an all-sensors query.

### Step 2 — Fetch incidents via `/vss-query-analytics`

Hand off to `/vss-query-analytics`, which configures and invokes the project-local
VSS CLI. It uses `vss analytics incidents`; use this command shape after its
bootstrap:

```bash
vss analytics incidents \
  --source "<sensor-id>" --source-type sensor \
  --start-time "<ISO>" --end-time "<ISO>" \
  --vlm-verdict all \
  --limit 100 \
  --include objectIds --include info --include category --include place
```

`--source` and `--source-type` go together: keep both for a sensor scope, omit
BOTH for an all-sensors query. Always keep `--vlm-verdict all`, including a
latest-incident lookup: Alerts stores the incidents used by this report in the
VLM-verified incident index, including both confirmed and rejected verdicts.
Never omit `--vlm-verdict all` (that queries the raw incident index). Do not
add unsupported sort flags; the API already returns newest-`end` first. For
“last / latest / most recent incident”, omit the time bounds unless the user
named a range, use `--limit 1`, and generate the report only from that returned
incident. If the command exits 0 with `count` 0 / `incidents` `[]`, STOP and
return the Mode B empty-range line; never invent an incident, safety concerns,
severity, or corrective actions, and never fall back to raw Elasticsearch,
VA-MCP, Mode A, or fabricated data.

Read-only boundary (mandatory):
- Mode B is strictly read-only analytics retrieval. Never write, seed, backfill, or mutate Elasticsearch/VA data.
- Forbidden examples: indexing synthetic incidents, replaying fixture payloads into ES, calling write/update/delete APIs to "make data available" for the report.
- If no incidents exist for the requested range/scope, handle as empty results (see below); do not fabricate data.

For each incident keep: `id`, `sensorId`, `timestamp`, `end`, `category`,
`place.name`, `info.verdict`, `info.reasoning`, `objectIds`, and the clip URL
(`info.videoSource` — the VST storage URL the alert enrichment writes; fall back
to any other clip-pointer field the response carries). **Apply the
browser-playable rewrite (see *Clip URLs: VLM input vs browser report link* in
SKILL.md — `VSS_PUBLIC_URL` on Kubernetes, or
`$VSS_PUBLIC_HOST:$VSS_PUBLIC_PORT` on Docker) to every clip URL before pasting
it into the report** — the raw value is often a private `HOST_IP:30888` URL the
user's browser cannot reach.

### Step 3 — Fill the Incident Range Report template

Load the matching template from [`$SKILL_DIR/references/report-templates/incident-range-report.md`](../report-templates/incident-range-report.md). Treat the template as read-only — copy its structure, then group by sensor (or by category if no sensor scope), tally verdicts, and list each incident with timestamp / category / verdict / reasoning. Fill all placeholders before returning markdown. Never leave template instructions, placeholder tokens, or internal-only URLs in user output. Every incident clip value must be a rewritten browser-playable URL (SKILL.md § Clip URLs: `RAW_URL` in → `BROWSER_CLIP_URL` out, one block per URL); omit the clip line when the incident carries no clip URL or when the rewrite printed an empty value, and say why in the accompanying chat response. The `Scope` row is one value: the sensor id, or `all sensors`.

For non-empty results, rendered output MUST start exactly with:
- `# Incident Range Report`
- `## Basic Information`
- a pipe table containing rows: `Report Identifier`, `Range`, `Scope`, `Total Incidents`, `Confirmed / Rejected / Unverified`

The result contains `count`, `incidents`, and `has_more`. When `has_more` is
true, `count` is a page size, not a total: say at least N (query capped) or
rerun with a larger positive `--limit`. Zero results means `count` is `0`,
`has_more` is false, and `incidents` is `[]`. A nonzero CLI exit or output
without an `incidents` key is
a failure (SKILL.md § Error Handling), never an empty range. On zero results,
STOP and return exactly this one-line sentence shape (single line only):
`No incidents found for scope <scope> in range <start_time> to <end_time>.`

When zero results:
- Do not render `# Incident Range Report`.
- Do not render `## Basic Information`.
- Do not render any markdown table, bullets, or summary section.
- Do not invent incidents, do not seed test data, and do not fall back to Mode A.
