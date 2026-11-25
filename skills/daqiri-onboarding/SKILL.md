---
name: daqiri-onboarding
license: Apache-2.0
description: Guide DAQIRI setup and configuration generation or repair. Diagnose DAQIRI initialization failures, packet drops, RX-worker/application deficits, buffer errors and saved benchmark/counter evidence, including analysis-only requests with an existing runtime. Explain DAQIRI memory paths, select polling and RX reorder modes, and handle explicitly requested customer demos.
---

# DAQIRI Onboarding

Reuse the matching checkout, supplied facts and runtime; read only unresolved details. The [documentation map](references/documentation-map.md) locates sources when needed.

For an understood scalar edit to a valid template, unless generation or rendering is explicitly requested, copy it to the requested output, change the requested scalars and necessary dependent references, and validate the output with the supplied native validator. Preserve unrelated values. Give changed fields and requested commands. Skip the generator, configuration reference, input revalidation and setup discovery unless dependencies are unclear or validation fails. Mode, engine, memory and reorder changes need their dependency checks. A future smoke command does not authorize running traffic.

| Request | Read / do |
|---|---|
| Explain memory, bursts, streams or engines | Relevant section of `docs/concepts.md`; system-configuration tutorial only for missing platform details. |
| New YAML, unclear dependencies, validation failure | [Configuration](references/configuration.md). |
| Saved evidence | Use structured CSV/JSON observations directly. Read [evidence](references/evidence.md) for unresolved accounting/path/rate semantics or log parsing. Its helper supports raw benchmark and `INTEGRITY_RESULT` logs; stop on unsupported formats instead of retrying guessed inputs. |
| Latency or polling mode | [Polling modes](references/polling-modes.md). |
| RX ordering, NIC placement or reorder capability failure | [Reorder selection](references/reorder-selection.md). |
| Initialization, allocation or recycling errors | [Troubleshooting](references/troubleshooting.md). |
| Prepared benchmark | [First run](references/first-run.md); setup only for missing prerequisites. |
| Missing checkout/build/runtime | [Setup](references/setup.md), at the first missing prerequisite. |
| Explicit customer demo | [Demo routing](references/customer-demo.md). |

Keep these boundaries:

- Honor transport, file preservation and execution scope. Compile as the current user in the project container; hardware/network actions need existing authorization. Historical diagnostics need no host discovery.
- Validation, NIC reception, application delivery, payload integrity and throughput are separate claims. Report application deficits exactly; counters alone do not establish their cause. Even equal counts do not certify unique delivery or payload identity.
- Preserve counter baselines and deltas: positive unchanged values are not absolute zeros. Draining cannot recover discarded events; counts do not identify missing sequences or explain a residual gap.
- Report actual commands/statuses and evidence limits; identify missing prerequisites when blocked. Keep small repairs and diagnoses concise.

For a persistent raw-ibverbs worker deficit or partial-burst tail, use [DevX RX gaps](references/ibverbs-rx-gaps.md) for private-counter coverage and publication boundaries; establish the new run's cause from its evidence.
