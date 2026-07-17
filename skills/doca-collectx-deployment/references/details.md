# doca-collectx-deployment — reference detail

Moved out of `SKILL.md` to keep the loader under the per-file size budget. This is supporting detail, not routing logic.

## Example questions this skill answers well

The CLASSES of collector-deployment questions this skill is built
to answer, each with one worked example. The class is the
load-bearing piece; the worked example is one instance.

- **"People keep saying 'DOCA telemetry' as if it's one thing —
  which surface am I actually deploying?"** — worked example: *"I
  see `clx`, `doca-telemetry`, `doca-telemetry-exporter`, and
  'DOCA Telemetry Service' all referenced; which one is the
  collector I'm standing up?"*. Answered by the four-surface
  decomposition in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  + the scope-boundary section in [`SKILL.md`](../SKILL.md) — the
  agent draws the clx-collector / reader-library /
  publisher-library / productized-DTS split BEFORE any config.
- **"My collector daemon starts but no schema rows appear for a
  provider."** — worked example: *"the collector is running but
  the provider I enabled produces nothing downstream"*. Answered
  by the no-provider-rows layer in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  + the gate-before-commit rule in
  [`CAPABILITIES.md ## Safety policy`](../CAPABILITIES.md#safety-policy):
  the device does not expose the counter under the chosen
  dimensions — the canonical silently-dropped metric.
- **"How do I turn on the Prometheus endpoint / ship to Fluent
  Bit / export NetFlow from my collector?"** — worked example: *"I
  want a Prometheus scrape target for my BlueField telemetry
  collector"*. Answered by the export-backend table in
  [`CAPABILITIES.md ## Capabilities and modes`](../CAPABILITIES.md#capabilities-and-modes)
  (class level: pull vs push vs local) + the
  [`TASKS.md ## configure`](../TASKS.md#configure) step 5 / run-side
  verification — with the exact flag and config-path spelling
  routed to the live config and the public DTS / Telemetry guide.
- **"The collector is running but my downstream Prometheus /
  Grafana shows nothing — where do I start?"** — worked example:
  *"exporter looks up, consumer is empty"*. Answered by the
  layered ladder in
  [`CAPABILITIES.md ## Error taxonomy`](../CAPABILITIES.md#error-taxonomy)
  (exporter-silent → downstream-skew → transport) + the
  end-to-end smoke in [`TASKS.md ## test`](../TASKS.md#test): a
  running daemon is not proof of deployment.
- **"Is the counter family I want even exposed on this device
  before I commit it to the collector config?"** — worked
  example: *"will this PCIe-diagnostic counter family work on my
  BlueField-3 under these property dimensions?"*. Answered by the
  gate-before-commit rule in
  [`CAPABILITIES.md ## Safety policy`](../CAPABILITIES.md#safety-policy)
  + the per-device support probe discipline shared with
  [`doca-telemetry-utils`](../../tools/doca-telemetry-utils/SKILL.md).
- **"I think I actually need the DOCA Telemetry Service container,
  not a hand-rolled collector — where do I go?"** — worked
  example: *"I want the turnkey DTS service auto-started on my
  BlueField"*. Answered by the scope boundary in
  [`SKILL.md`](../SKILL.md): DTS-as-deployed is externally
  productized (Non-goal #7); the agent routes to the
  [public DTS guide](../../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route)
  rather than synthesizing DTS config.

## What this skill deliberately does not ship

This skill is **agent guidance**, not a templates / sample-config
bundle. To keep the boundary clean, it deliberately does not
contain — and pull requests should not add:

- **The doca-telemetry library API.** The hardware-counter reader
  API (per-domain `doca_telemetry_<domain>` contexts on a
  `doca_dev`) is owned by
  [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md). This skill
  routes there for any reader-API question; it does not
  re-document the per-domain lifecycle, the cap-query rule, or the
  reader error taxonomy.
- **The doca-telemetry-exporter library API.** The application-side
  publisher API (schema / source / type, register-before-emit) is
  owned by
  [`doca-telemetry-exporter`](../../libs/doca-telemetry-exporter/SKILL.md).
  This skill routes there for any "how do I emit a counter from my
  program" question.
- **The productized DTS container.** The DOCA Telemetry Service
  as-deployed — its packaged config schema, its built-in provider
  set, its kubelet manifest, its NGC image tag — is externally
  productized (per
  [`AGENTS.md` Non-goal #7](../../../AGENTS.md#non-goals-questions-the-agent-should-recognize-and-refuse-politely)).
  The agent refuses to synthesize DTS config file names, provider
  knob names, or paths and routes to the
  [public DTS guide](../../doca-public-knowledge-map/SKILL.md#externally-productized-doca-software--not-in-this-bundle-but-here-is-where-to-route).
- **Invented clx symbols, provider names, schema field names,
  exporter flag names, or config paths.** All of these are
  install-bound and docs-bound; the authoritative sources are the
  live collector config on the target and the public DOCA
  Telemetry / DTS guides reached through
  [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md).
  Quoting a clx provider name or a config path from memory as
  authoritative is the load-bearing hallucination failure for this
  skill.
- **Pre-baked collector configs, sample exporter configs, sample
  systemd units, or sample pod-specs.** Collector deployment is
  site-specific (which providers, which export sink, which launch
  posture); the safe answer is to *derive* the config against the
  live install and the public docs, with the launch posture routed
  to [`doca-bare-metal-deployment`](../../doca-bare-metal-deployment/SKILL.md)
  or [`doca-container-deployment`](../../doca-container-deployment/SKILL.md).
- **A `samples/`, `templates/`, `config/`, or `reference/`
  subtree of any kind.** A mock or incomplete artifact in this
  skill's tree, even one labeled "reference", is misleading:
  operators will read it as production-ready.

## Related skills

- [`doca-telemetry`](../../libs/doca-telemetry/SKILL.md) — the
  hardware-counter **reader** library. When a source the
  collector samples is the operator's own program reading device
  counters, the reader API lives there. This skill owns the
  collector runtime; that skill owns the reader API.
- [`doca-telemetry-exporter`](../../libs/doca-telemetry-exporter/SKILL.md)
  — the application-side **publisher** library. A DOCA program
  that feeds the collector as an external source publishes through
  this API. This skill owns the collector / sink side; that skill
  owns the publisher / source side.
- [`doca-telemetry-utils`](../../tools/doca-telemetry-utils/SKILL.md)
  — the operator-side support CLI that enumerates the
  diagnostic-counter schema, translates name ↔ Data ID, and probes
  per-device counter support before a config commits. This skill
  reuses its gate-before-commit discipline for the collector
  config.
- [`doca-setup`](../../doca-setup/SKILL.md) — env preparation and
  install verification. This skill assumes its preconditions are
  satisfied (DOCA installed and healthy on the collection target).
- [`doca-version`](../../doca-version/SKILL.md) — canonical DOCA
  version-handling rules. This skill's `## Version compatibility`
  cross-links the four-way match and adds the
  collector-vs-source-vs-consumer schema-version alignment overlay.
- [`doca-hardware-safety`](../../doca-hardware-safety/SKILL.md) — the
  cross-cutting meta-policy for any change touching DPU / NIC
  hardware state. The collector is read-only against the device;
  any mutating step (a privileged-data daemon needing a
  hardware-state change, an `mlxconfig set`, a firmware burn)
  leaves this skill for that meta-policy.
- [`doca-bare-metal-deployment`](../../doca-bare-metal-deployment/SKILL.md)
  and
  [`doca-container-deployment`](../../doca-container-deployment/SKILL.md)
  — the two launch-posture skills. The collector's *launch shape*
  (foreground / service-supervised on bare metal, or container /
  kubelet) follows those skills; this skill owns the collection
  pipeline that runs inside whichever posture the operator picks.
- [`doca-public-knowledge-map`](../../doca-public-knowledge-map/SKILL.md)
  — the routing table to the public DOCA Telemetry guide, the
  public DTS guide (externally-productized row), and the installed
  DOCA layout. This skill does not duplicate URLs; it points at the
  map and adds the collector-deployment overlay. The productized
  DTS service is reached through that map (Non-goal #7).
- [`doca-debug`](../../doca-debug/SKILL.md) — the cross-cutting
  layered debug ladder. Collector-deployment-specific debug
  (no rows, exporter silent, downstream skew) layers on top of the
  cross-cutting host / network ladder there.
