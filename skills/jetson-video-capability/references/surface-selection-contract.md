# Surface Selection Contract

This capability skill owns natural-language surface classification for
capability requests. Operation-owning controllers keep their own documented
natural-language mapping, then apply the same pure four-value routing truth
table locally while resolving one `--request`; they do not inherit the
capability-only unnamed preference below. No capability script imports routing
code from setup or another sibling skill. Routing itself performs no probe,
command, filesystem, package, or network operation. The orchestrating candidate
gates provide its eligibility inputs; the documented unnamed-request preference
below is explicit and never permits an unauthenticated fallback.

Within a capability request, a genuinely bare “video SDK” phrase with no
product qualifier is ambiguous: ask whether the customer means native Video
Codec SDK, PyNvVideoCodec, or both, then stop before probing. Report-only intent
or a request to probe does not authorize `--runtime both`.

A capability support/catalog request that names no SDK surface or product
phrase uses native-preferred fallback selection, whether broad, exact, or a
bounded subset; it is not an `auto` request. This is a pre-routing rule: when
native is eligible, feed explicit `native` to the four-value routing contract.
Do not evaluate Py; if the response displays that unselected peer, report it as
`not_evaluated` with reason `surface_not_selected`. Only when native is
ineligible, evaluate the Py candidate through the authority order below and
feed explicit `pynvc` to the routing contract when that route is eligible. If
neither is eligible, return a blocked capability response outside the surface
plan, preserve both typed reasons, and provide setup remediation without asking
for a surface choice. Carry a successfully resolved surface explicitly into
any authorized downstream operation; the receiving controller does not
reclassify it from the original wording. Only explicit “auto”, “whichever”,
“best available”, “choose for me”, or equivalent wording that expressly
delegates the SDK choice is genuine `auto`; it is never `both`.
Naming Python or PyNvVideoCodec is explicit `pynvc`. Naming Video Codec SDK,
`native`, `AppEncCuda`, or `AppDec` is explicit `native` and never probes
PyNvVideoCodec or resolves its registry.

## Authority boundary

The surface plan is routing-only and non-authorizing. `requested_surface` is
exactly one of `native`, `pynvc`, `auto`, or `both`. `eligibility` maps each
surface to a boolean or an
`{"eligible": bool, "reasons": [...]}` object derived from validated live evidence.
The returned plan contains exactly:

```json
{
  "requested_surface": "auto",
  "classification": "ready|blocked|selection_required",
  "selected_surfaces": ["native"],
  "eligible_surfaces": ["native"],
  "reasons": []
}
```

Do not treat `classification: ready` or an `eligible_surfaces` entry as setup
readiness, operation proof, live availability, package approval, or command authorization. Derive
each surface's eligibility from the authenticated inputs or local prerequisites
required by that released route. Setup-produced evidence is optional for a
capability selected-surface query: its absence alone does not make a surface
ineligible. With no setup evidence, the native query uses only its explicit
fresh-workspace package-build route and the Py query uses only the exact
invoking interpreter and wheel. The owning controller authenticates its fixed
native or Python route names and builds the plan from current facts rather than
a copied readiness claim or an imported sibling constant.

If setup evidence is supplied, validation is mandatory. A stale, malformed,
mismatched, or changing artifact fails closed and cannot be discarded to make
the local route eligible. For unavailable selected local prerequisites, apply
the shared missing-prerequisite rule in capability `SKILL.md` step 3.

Select the Python authority before the capability query. Use a valid supplied
setup environment or exact interpreter first. For explicit `pynvc`/`both`, a
genuine `auto` candidate gate, or the unnamed fallback after native is
ineligible, if neither is supplied and
`jetson-video-setup` is installed, the agent invokes setup's public read-only
probe and inspects its fresh environment artifact: `--runtime pynvc` for
explicit `pynvc` or the unnamed fallback, and `--runtime both` for
`both`/`auto`. Capability code never reads the registry or imports setup. Only
a live, GPU-matched artifact with
`pynvc.installed=true` and `pynvc.identity.status=verified` is Py authority.
Only when that handoff is absent, stale, unreadable, invalidly bound, or cannot
launch does an explicit `pynvc` or `both` request become `input_required` for
one exact canonical absolute `pynvc_interpreter`. For `auto`, use
`setup_probe_unavailable` when setup is absent; otherwise preserve the probe's
exact typed failure reason and keep PyNvVideoCodec `not_evaluated`. An eligible
native branch may still run. A healthy registered Py candidate must participate
in the truth table below and can never be reported unevaluated for lack of an
interpreter. For the unnamed fallback, preserve an unusable Py candidate's
typed reason beside the ineligible native reason and provide setup remediation.
Do not scan for or guess a venv.

The private `nvcodec-pynvc-runtime-binding` is deliberately narrower than the
benchmark/pipeline `nvcodec-local-runtime-binding`: it authenticates only the
current PyNvVideoCodec capability-query process and is never a request input or
a substitute for setup evidence.

## Resolution truth table

The table begins after natural-language classification and contains no fifth
“unnamed” request value. Canonical surface order is `native`, then `pynvc`.
Explicit requests never fall back.

| Request | Classification / selection |
|---|---|
| `native` | `ready` with `[native]` when eligible, else `blocked`. Never fall back. |
| `pynvc` | `ready` with `[pynvc]` when eligible, else `blocked`. Never fall back. |
| `both` | `ready` only when both eligible; otherwise `blocked`, with each eligible branch selected for independent execution and an explicit blocked operation plus reasons for every ineligible peer. |
| `auto`, exactly one eligible | `ready`, selects that surface. |
| `auto`, both eligible | `selection_required`, selects none. |
| `auto`, zero eligible | `blocked`, selects none. |

On `selection_required`, the controller runs no branch command and reserves no
branch output; ask the user to choose `native`, `pynvc`, or `both`. For `both`,
`selected_surfaces` lists the eligible branches that may launch, while the result
still represents every requested branch. Each controller executes eligible branches
independently with distinct outputs and records an explicit blocked outcome and its
reasons for every ineligible peer. It never hides a failed, blocked, or unknown peer
merely because another branch succeeds.

## Aggregation

Do not translate branch outcomes into a synthetic cross-domain vocabulary. Preserve
the owning domain's statuses, including `operation_verified`, `operation_failed`,
`blocked`, and `unknown`, and roll them up only at the request result. A branch may
be `operation_verified` only after its authenticated operation succeeds. Missing,
extra, or invalid branch outcomes are contract failures, not evidence that readiness
changed after selection.
