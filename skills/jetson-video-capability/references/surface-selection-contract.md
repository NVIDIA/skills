# Surface Selection Contract

This capability skill owns natural-language surface classification for
capability requests. For other operations, the agent invokes the named owning
skill through public skill dispatch, uses that skill's documented
natural-language mapping, and applies the same pure four-value routing truth
table in that stage; those stages do not inherit the capability-only unnamed
preference below. Capability policy does not import routing code from setup or
another sibling skill. Routing itself performs no probe, command, filesystem,
package, or network operation. The owning skill's candidate gates provide its
eligibility inputs; the documented unnamed-request preference below is explicit
and never permits an unauthenticated fallback.

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
any authorized downstream operation; pass it as data through public skill
dispatch so the receiving skill does not reclassify it from the original
wording. Only explicit “auto”, “whichever”,
“best available”, “choose for me”, or equivalent wording that expressly
delegates the SDK choice is genuine `auto`; it is never `both`.
Naming Python or PyNvVideoCodec is explicit `pynvc`. Naming Video Codec SDK,
`native`, `AppEncCuda`, or `AppDec` is explicit `native` and never probes
PyNvVideoCodec.

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
readiness, operation proof, live availability, package approval, or command
authorization. Capability discovery depends on `jetson-video-setup`; invoke its
public read-only workflow through skill dispatch and use its fresh readiness
result as the eligibility input. If setup is unavailable, return
`dependency_required`; do not import its files or invent a substitute probe.

Validation is direct and current. Require the requested Jetson and GPU. Native
eligibility requires one installed, package-verified
`nvidia-video-codec-sdk`, one package-owned Samples root, and the tools and
runtime libraries required by the selected report binary. Py eligibility
requires the exact interpreter selected by setup, one importable
PyNvVideoCodec distribution whose loaded module is inside that environment,
and a clean `pip check`. These facts establish installation eligibility only,
never an operation or product-support verdict.

Inspect only Py for explicit `pynvc` or the unnamed fallback, and inspect both
for `both` or genuine `auto`. Capability never scans for a venv, changes an
environment, or uses API/test outcomes to make an ineligible candidate
eligible. For `both`, an eligible peer may still run while the ineligible peer
retains its blocked outcome; for `auto`, apply the truth table below.

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

On `selection_required`, the agent runs no branch command and reserves no branch
output; ask the user to choose `native`, `pynvc`, or `both`. For `both`,
`selected_surfaces` lists the eligible branches that may launch, while the result
still represents every requested branch. The agent executes each eligible branch
through its owning skill with distinct outputs and records an explicit blocked
outcome and its reasons for every ineligible peer. It never hides a failed,
blocked, or unknown peer merely because another branch succeeds.

## Aggregation

Do not translate branch outcomes into a synthetic cross-domain vocabulary. Preserve
the owning domain's statuses, including `operation_verified`, `operation_failed`,
`blocked`, and `unknown`, and roll them up only at the request result. A branch may
be `operation_verified` only after its authenticated operation succeeds. Missing,
extra, or invalid branch outcomes are contract failures, not evidence that readiness
changed after selection.
