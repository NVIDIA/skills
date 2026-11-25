# nvidia-doca-plugin Plugin

## Description:

NVIDIA DOCA skills for developing, deploying, operating, and debugging applications on BlueField DPUs and ConnectX networking devices.

Owner-declared readiness (not a verification result):

This plugin is ready for commercial or non-commercial use.

This plugin is ready for external use.

## Owner:

NVIDIA

### License/Terms of Use:

**Package license:** Apache-2.0 AND CC-BY-4.0. Component-card declarations that differ are listed under Plugin Components.

## Use Case:

External developers, operators and platform teams use these skills to develop, deploy, operate and debug DOCA applications on BlueField DPUs and ConnectX devices. Sources: developer use case and operator use case.

### Compatible Agents:

| Agent | Declared minimum version | Compatibility basis |
| --- | --- | --- |
| Claude Code | 2.1.150 | Expected by package/specification conformance; host compatibility not independently established |
| Codex | 0.137.0 | Expected by package/specification conformance; host compatibility not independently established |
| Cursor | 2.5.0 | Expected by package/specification conformance; host compatibility not independently established |

Source: declared host requirements. Evaluation observations below do not establish compatibility with every host/version.

### Release Management:

| Channel | Location | Release date / status |
| --- | --- | --- |
| GitHub | N/A | Release date not supplied |

No marketplace/OCI release URL or date is declared in the supplied evidence.

### Deployment Geography for Use:

Global, as declared in all 53 bundled skill cards; see the bundled skill declaration.

## Plugin Package:

**Package format:** Agent Plugins manifest (declared schema 1.0.0), with Claude Code, Codex and Cursor host manifests.

**Manifest locations:** `plugin.json`, `.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`, `.cursor-plugin/plugin.json`.

**Package unit:** Directory.

**Client extension namespaces:** None declared. Host manifest directories are not custom client namespaces.

**Directory layout** (shared skill companion files are abbreviated):

```text
nvidia-doca-plugin/
├── plugin.json
├── .claude-plugin/plugin.json
├── .codex-plugin/plugin.json
├── .cursor-plugin/plugin.json
├── .eval-results/
├── assets/
├── evals/
├── skills/<skill>/
│   ├── SKILL.md
│   ├── skill-card.md
│   ├── skill.oms.sig
│   └── ...
├── LICENSE
├── README.md
├── SBOM.cdx.json
├── plugin-lock.json
├── provenance.intoto.jsonl
├── BENCHMARK.md
├── plugin-card.md
└── plugin.oms.sig
```

## Plugin Components:

**Agents:** None declared. **MCP servers:** None declared. **Hooks:** None declared. **Standalone commands/rules/client-specific executable components:** None declared. **Bundled third-party runtime dependencies:** None declared. The package declaration describes a skills-only bundle; the SBOM inventories the 53 skill components. Runtime prerequisites are listed below.

Skills are listed for each selected host. Claude and Codex rows can reference the same packaged skill; they do not represent additional bundled copies. Component presence is separate from the incomplete static-analysis coverage described under Testing Completed.

### Skills:

| Skill | Signature verification |
| --- | --- |
| doca-aes-gcm (codex) | Verified |
| doca-aes-gcm (claude) | Verified |
| doca-argp (codex) | Verified |
| doca-argp (claude) | Verified |
| doca-argus (codex) | Verified |
| doca-argus (claude) | Verified |
| doca-bare-metal-deployment (claude) | Verified |
| doca-bare-metal-deployment (codex) | Verified |
| doca-bench (codex) | Verified |
| doca-bench (claude) | Verified |
| doca-bf3-deployment (codex) | Verified |
| doca-bf3-deployment (claude) | Verified |
| doca-bf4-deployment (codex) | Verified |
| doca-bf4-deployment (claude) | Verified |
| doca-caps (codex) | Verified |
| doca-caps (claude) | Verified |
| doca-comch (codex) | Verified |
| doca-comch (claude) | Verified |
| doca-comm-channel-admin (claude) | Verified |
| doca-comm-channel-admin (codex) | Verified |
| doca-common (claude) | Verified |
| doca-common (codex) | Verified |
| doca-compress (codex) | Verified |
| doca-compress (claude) | Verified |
| doca-container-deployment (codex) | Verified |
| doca-container-deployment (claude) | Verified |
| doca-debug (codex) | Verified |
| doca-debug (claude) | Verified |
| doca-devemu (codex) | Verified |
| doca-devemu (claude) | Verified |
| doca-dma (claude) | Verified |
| doca-dma (codex) | Verified |
| doca-dms (claude) | Verified |
| doca-dms (codex) | Verified |
| doca-dpa (codex) | Verified |
| doca-dpa (claude) | Verified |
| doca-dpdk-bridge (codex) | Verified |
| doca-dpdk-bridge (claude) | Verified |
| doca-dts (codex) | Verified |
| doca-dts (claude) | Verified |
| doca-erasure-coding (codex) | Verified |
| doca-erasure-coding (claude) | Verified |
| doca-eth (codex) | Verified |
| doca-eth (claude) | Verified |
| doca-firefly (codex) | Verified |
| doca-firefly (claude) | Verified |
| doca-flow (claude) | Verified |
| doca-flow (codex) | Verified |
| doca-flow-tune (claude) | Verified |
| doca-flow-tune (codex) | Verified |
| doca-gpunetio (codex) | Verified |
| doca-gpunetio (claude) | Verified |
| doca-hardware-safety (codex) | Verified |
| doca-hardware-safety (claude) | Verified |
| doca-mgmt (claude) | Verified |
| doca-mgmt (codex) | Verified |
| doca-ngauge (claude) | Verified |
| doca-ngauge (codex) | Verified |
| doca-pcc (codex) | Verified |
| doca-pcc (claude) | Verified |
| doca-pcc-counters (claude) | Verified |
| doca-pcc-counters (codex) | Verified |
| doca-perftest (codex) | Verified |
| doca-perftest (claude) | Verified |
| doca-pipeline-language-developer-tools (claude) | Verified |
| doca-pipeline-language-developer-tools (codex) | Verified |
| doca-programming-guide (codex) | Verified |
| doca-programming-guide (claude) | Verified |
| doca-public-knowledge-map (claude) | Verified |
| doca-public-knowledge-map (codex) | Verified |
| doca-rdma (codex) | Verified |
| doca-rdma (claude) | Verified |
| doca-rmax (claude) | Verified |
| doca-rmax (codex) | Verified |
| doca-setup (claude) | Verified |
| doca-setup (codex) | Verified |
| doca-sha (claude) | Verified |
| doca-sha (codex) | Verified |
| doca-sha-offload-engine (codex) | Verified |
| doca-sha-offload-engine (claude) | Verified |
| doca-socket-relay (claude) | Verified |
| doca-socket-relay (codex) | Verified |
| doca-spcx-cc (claude) | Verified |
| doca-spcx-cc (codex) | Verified |
| doca-sta (codex) | Verified |
| doca-sta (claude) | Verified |
| doca-structured-tools-contract (codex) | Verified |
| doca-structured-tools-contract (claude) | Verified |
| doca-telemetry (codex) | Verified |
| doca-telemetry (claude) | Verified |
| doca-telemetry-exporter (codex) | Verified |
| doca-telemetry-exporter (claude) | Verified |
| doca-telemetry-traceback-application (claude) | Verified |
| doca-telemetry-traceback-application (codex) | Verified |
| doca-telemetry-utils (claude) | Verified |
| doca-telemetry-utils (codex) | Verified |
| doca-upgrade (codex) | Verified |
| doca-upgrade (claude) | Verified |
| doca-urom (codex) | Verified |
| doca-urom (claude) | Verified |
| doca-urom-svc (claude) | Verified |
| doca-urom-svc (codex) | Verified |
| doca-verbs (codex) | Verified |
| doca-verbs (claude) | Verified |
| doca-version (claude) | Verified |
| doca-version (codex) | Verified |

**Component license declarations that differ from the package manifest:**

| Text in component card | Components |
| --- | --- |
| Apache-2.0 | doca-argus, doca-comm-channel-admin, doca-debug, doca-dpdk-bridge, doca-upgrade, doca-version |
| Apache 2.0 | doca-container-deployment, doca-dma, doca-perftest |

These are the component cards' declared terms; the package LICENSE remains linked above.

## Requirements / Dependencies:

**Requires API key or external credential:** Optional / workflow-dependent. Reading the skills needs no target credential; BF4, DTS, remote perftest, Rivermax and some DMS workflows require the credentials or license listed below.

**Credential types:** Basic auth for BF4 management and applicable DMS modes; mTLS client certificate for DMS; SSH key for remote perftest; Root/sudo host access or Kubernetes (DPF) cluster credentials for DTS; Separately obtained Rivermax software license file. Sources: doca-bf4-deployment, doca-dms, doca-dts, doca-perftest, doca-rmax.

**Credential supplied by:** User/operator or deployment administrator for the chosen target/service. The actual account and secret source are deployment-specific.

**Runtime environment:** No DOCA installation is needed to read the skills. Live workflows require the matching Linux Arm/x86 DOCA environment and supported BlueField/ConnectX hardware; DOCA 3.6 is the documented baseline, with workflow-specific version requirements. Service workflows may require containers and BlueField kubelet/static-pod setup; DPF workflows may require cluster access. No `PLUGIN_ROOT`, `PLUGIN_DATA` or other plugin-specific environment-variable requirement is declared in the supplied manifests.

**Network access required:** Workflow-dependent: NVIDIA documentation and software download sites; linux.mellanox.com and mellanox.github.io are explicitly allowed in the evaluation sandbox. Operator-selected SSH, BMC/Redfish, DMS and workload-peer endpoints; Argus can forward findings to an operator-selected SIEM. These examples are not a complete production egress allowlist.

**Subprocess execution:** The package contains instructions, not bundled startup scripts. Following the instructions can execute shell commands, build/install tools, SSH, hardware-management utilities and container/service commands under the host agent or operator permissions.

**Filesystem read:** /opt/mellanox; /etc/nvidia; /etc/mlnx-release; /sys/class/infiniband; /sys/class/net; /sys/bus/pci/devices; /dev/infiniband; /dev/mst; /proc/driver; Workflow-selected source/configuration files and logs

**Filesystem write:** Workflow-selected build/output files, configuration, service manifests and logs; Privileged install, kernel/device configuration and hardware-management workflows can write system state; scope depends on the selected task and host permissions. The evaluation sandbox read-only paths do not make all runtime workflows read-only.

Do not include secrets in prompts, logs, output or package files. Use least-privilege credentials and rotate keys as appropriate.

## Capability and Permission Summary:

**Operations Allowed:** Subject to host permissions and the approved task.

- **Create** Build outputs, configuration files and service manifests.
- **Read** Source files, configuration, logs and device information from operator-selected systems.
- **Update** Build and configuration files, device settings, firmware/BFB and managed services.
- **Delete** Managed files or resources in authorized DMS workflows.
- **Execute** Build, installation, diagnostic, hardware-management and service commands, including reboots where approved.
- **Search** Source, configuration and device information.
- **Web Search** No package declaration. Access to online documentation depends on the host's tools and permissions.

**Operation scope:** DOCA development, installation, diagnostics, device configuration and service operation on operator-selected hosts, DPUs and NICs. Documented workflows include destructive DMS operations, firmware/BFB changes, reboots and configuration writes. Host permissions and the approved task bound effective authority. See the bundled hardware safety instructions.

**Changes Host Agent Autonomy:** No. The package declares no automatic hooks or event triggers.

**Autonomy effect:** The skills guide agent decisions and tool use, including routing to other skills and confirmation/refusal gates. The package declares no automatic hooks or MCP servers. It does not establish an independently enforced autonomy boundary or grant extra host permissions.

**Configured / component-declared operation lower bound:** Not established

The aggregate must cover the most permissive component. This lower bound is not an exhaustive permission list; host policy determines effective authority.

## Known Technical Limitations:

- Live checks require access to the actual target host; otherwise the skills instruct the agent to provide commands and treat unseen values as unknown.
- Hardware capabilities, DOCA/firmware versions and deployment shape constrain applicability. Documentation alone does not prove that a particular target supports an operation.

## Known Risks and Mitigations:

| Risk | Mitigation |
| --- | --- |
| Skills and tool descriptions bundled in a plugin can introduce incorrect, misleading or malicious guidance into the agent's context. | Review and scan the plugin and every component before deployment. |
| Following bundled instructions can invoke tools with the user or agent authority. | Apply host permissions, sandboxing, egress controls and human approval appropriate to the selected workflow. |
| Incorrect cross-skill routing or incompatible DOCA/firmware guidance can lead to disruptive device or service changes. | Confirm target/version compatibility and follow the hardware-safety workflow before applying changes; require approval and recovery planning for disruptive actions. |

## Disable and Revoke:

**Disable / uninstall:** Disable or remove the plugin through the consuming host's plugin controls; the exact procedure and effect have not been verified here. This does not undo hardware/configuration changes or stop separately launched services. Measured time to effect remains unknown.

**Revoke credentials:** Revoke or rotate the relevant SSH keys, management accounts, client certificates or cluster credentials with their issuing system. No plugin-specific credential revocation service is declared; Rivermax license handling follows its license provider.

**Measured time to effect:** Not measured (target: 300 seconds or less)

## Data Flow and Security:

**Logging and traceability:** Host/agent command output and workflow diagnostics; DMS documents frontend/backend logs and Argus documents container findings plus optional forwarding. Retention, rotation and access policy depend on the deployment.

**Persisted data:** Selected workflows persist builds, configuration, service manifests, logs and hardware settings. The package declares no plugin-managed persistence store; storage locations and retention are deployment-specific.

**Data sent to external services:** Documentation/download requests and target-management/workload traffic depend on the selected workflow. Argus can forward security findings to a configured SIEM; operators choose its endpoint and authentication. Agent/model data handling follows the chosen host configuration and is not specified by this package.

**Database name:** No plugin-managed database is declared. Downstream services or SIEM deployments may use separate storage.

**Database type:** None for plugin-managed storage. Downstream database type is deployment-specific.

**Authorization / human approval:** Confirm the agent is on the intended host before host checks. Hardware-mutating workflows require preflight inventory, applicable approval, maintenance planning, rollback and verification. Potentially link-breaking changes require a named, reachable out-of-band recovery path; refuse when required recovery conditions are absent. Destructive DMS operations require explicit target-bound confirmation. See cross-cutting safety rules.

**Security controls:** Documented least-privilege and secret-handling instructions; do not place credentials in prompts/logs/output. Hardware safety and destructive-operation confirmation/refusal instructions. DMS documents allowed_users and dmsgroup controls and authentication modes including credentials and mTLS. These are configured service controls; instruction text alone is not proof of runtime enforcement.

**Exposure categories** (documented workflow capabilities; actual endpoints depend on deployment):

- [x] External API/connector/service — documentation/downloads and selected external targets.
- [x] Internal API/connector/service — operator-selected management, workload and SIEM endpoints.
- [ ] Agent-to-agent inbound — none declared by the plugin.
- [ ] Unattended / scheduled execution — no automatic plugin trigger declared; separately launched services may continue independently.

Details: Argus data flow and DMS access controls.

**Exceptions:** None declared

## Testing Completed:

**CLI package validation:** PASSED

**CLI blocking findings:** 0

The CLI validation decision includes package policy checks. The native scanner result below describes its separate analysis scope; neither result grants signing approval.

**CLI validation warnings:** 589 recorded warnings. The main reported limitations are:

| Warning group | What remains unverified |
| --- | --- |
| Unsupported file formats | Dedicated validation for `assets/doca.svg` and `provenance.intoto.jsonl`. |
| Static parser limit | An expression in `skills/doca-dpdk-bridge/TASKS.md` exceeded the parser size limit. |
| Incomplete analysis | Some security checks have degraded or unavailable evidence. |
| Split source batches | Some manifest, cross-skill, authority and shared-state relationships were not compared together. |
| Dynamic targets | Some runtime/PATH-selected programs, routing targets and cycle/contract checks could not be resolved statically. |

This is a presentation summary, not a resolution of those warnings. Full warning records remain authoritative.

The groups above summarize recurring warning themes; all 589 warning records remain in the linked full evidence.

- [ ] Plugin static analysis — outcome: PASS

A checkmark records completed work, not approval. Incomplete analysis is unchecked; findings and review outcomes remain separate.

**Native scanner analysis:** PASS

**Scan execution:** 0 check execution errors

Analysis coverage is incomplete; unresolved checks are listed below.

Full evidence · Native scan · Declared requirements and policies

| Native check state | Recorded count |
| --- | --- |
| COMPLETE | 7739 |
| NOT_APPLICABLE | 4214 |
| NOT_REQUESTED | 124 |
| WARNING | 1299 |

124 check records were not requested; no result is claimed for them. Reasons and policy scope remain in the native scan.

**Native scanner findings retained in evidence:** 2917

**CLI findings retained in evidence:** 2972

| Severity | Check | Finding | Location |
| --- | --- | --- | --- |
| MEDIUM | coverage.limitation (claude) | Verification risk: The provider required smaller requests. All parts were scheduled, but relationships between separate source or context parts remain unverified | .claude-plugin/plugin.json:None |
| MEDIUM | coverage.limitation (claude) | Verification risk: manifest.authority: the comparison of .claude-plugin/plugin.json grants and default behavior against these skills was not performed. The manifest was supplied in a separate batch from this neighborhood. | .claude-plugin/plugin.json:None |
| MEDIUM | coverage.limitation (claude) | Verification risk: manifest.authority, manifest.external_sources, security.SDI-3 and security.TP4 cannot be completed because the primary manifest .claude-plugin/plugin.json (and plugin.json, plugin-lock.json, provenance.intoto.jsonl) was supplied in a different evidence split. The unverified comparison is between that manifest&#x27;s declared permissions, tools and source locations and the instructions in the supplied skill files. | .claude-plugin/plugin.json:None |
| MEDIUM | coverage.limitation (claude) | Verification risk: manifest.authority, manifest.external_sources, security.SDI-3 and security.TP4: the comparisons against .claude-plugin/plugin.json (permissions, launch settings, external sources, tool declarations) are unverified because that manifest was assigned to a different batch. | .claude-plugin/plugin.json:None |
| MEDIUM | coverage.limitation (claude) | Verification risk: Supporting baseline sources README.md and plugin.json were also batched elsewhere, so the comparison of the manifest description against these skills&#x27; behavior remains unverified. | .claude-plugin/plugin.json:None |
| MEDIUM | coverage.limitation (claude) | Verification risk: The primary manifest .claude-plugin/plugin.json was supplied in a separate batch, so its declared permissions and launch settings were not compared with these skills. The supplied SKILL.md frontmatter declares no permissions or launch settings. | .claude-plugin/plugin.json:None |
| MEDIUM | coverage.limitation (claude) | Verification risk: The manifest URL, package and executable declarations are in the omitted .claude-plugin/plugin.json, and the omitted plugin-lock.json and provenance.intoto.jsonl hold the provenance evidence. Skill docs here cite only public documentation URLs, which this check does not evaluate as declarations. | .claude-plugin/plugin.json:None |
| MEDIUM | coverage.limitation (claude) | Verification risk: manifest.declarations and manifest.identity_conflicts: .claude-plugin/plugin.json, plugin.json and .cursor-plugin/plugin.json were in a different batch. Comparing their declared skill paths and names against the supplied skills/\* sources remains unverified. | .claude-plugin/plugin.json:None |
| MEDIUM | coverage.limitation (claude) | Verification risk: manifest.declarations: .claude-plugin/plugin.json, .cursor-plugin/plugin.json and plugin.json were supplied in a different batch. The comparison of their component declarations and paths against the skills in this neighborhood was not performed here. | .claude-plugin/plugin.json:None |
| MEDIUM | coverage.limitation (claude) | Verification risk: manifest.declarations and manifest.identity_conflicts: .claude-plugin/plugin.json, .cursor-plugin/plugin.json and plugin.json were in a different batch. Their component declarations and identities were not compared against the SKILLCARD and SKILL.md declarations in this neighborhood. | .claude-plugin/plugin.json:None |

Showing 10 of 2972 CLI findings; 0 of 0 blocking findings shown. All findings are retained in the linked full evidence.

| Review | Recorded status / applicability | Basis / reviewer | Reference |
| --- | --- | --- | --- |
| Plugin static analysis / scan | Not supplied | Not supplied | Native scan |
| MCP server security review | Not applicable to the declared skills-only package | No MCP server declared | None supplied |
| Dependency / SBOM review | Not supplied | Not supplied | None supplied |
| Network security | Not supplied | Not supplied | None supplied |
| Product security | Not supplied | Not supplied | None supplied |

- [ ] MCP server security review — Not applicable: no MCP server declared
- [ ] Dependency / SBOM review — No review evidence supplied
- [ ] Network security — No review evidence supplied
- [ ] Product security — No review evidence supplied

Review statuses are owner assertions, not authenticated approvals; automated scan and test results are reported separately.

Evaluation and publication assessment: BENCHMARK.md.

The linked benchmark reports the quality/lift assessment and its coverage. The recorded task and metric summary below preserves the benchmark grouping and coverage limits.

**Plugin-targeted evaluation:** The submitted result explicitly targets `plugins/nvidia-doca-plugin` as a `PLUGIN`. The summary below describes those recorded tasks; individual skill evaluation details remain in the linked skill cards.

Recorded evaluation: original results.

**Recorded execution status:** completed

**Evaluation scope:** The plugin dataset `evals/evals.json`, executed with and without the plugin on Claude Code and Codex. The export reports NeMo Gym 0.7.0 and completed execution. It does not establish full coverage of every bundled skill, all supported host versions or production hardware.

**Evaluation observations:** 24 task cases, each recorded with and without the plugin on two agent harnesses: **96 observations** in total. Examples include AES-GCM offload, DMS deployment, hardware-safety checks, and out-of-scope BFB/BSP requests. The complete task list and per-attempt measurements remain in the linked original results.

The source export records 23 unchanged task cases from the original evaluation and a revised `no-host-access` case from a later evaluation. Those are the 23-task and 1-task groups below; no additional evaluation was run for this card.

| Agent harness | Recorded model | Without-plugin observations | With-plugin observations |
| --- | --- | --- | --- |
| Claude Code | aws/anthropic/bedrock-claude-sonnet-5-5 | 24 | 24 |
| Codex | openai/openai/gpt-6.1-sol | 24 | 24 |

**With-plugin metric summary** — copied from the retained benchmark. The 23-task and 1-task groups are shown separately, as recorded; no new combined average is calculated.

| Agent / recorded group | Runtime security | Correctness | Discoverability | Effectiveness | Efficiency |
| --- | --- | --- | --- | --- | --- |
| Claude Code — 23 tasks | 100/100 | 73.19/100 | 60.87/100 | Partial 73.53/100 | 98.81/100 |
| Claude Code — 1 task | 100/100 | 100/100 | 0/100 | Not applicable | 100/100 |
| Codex — 23 tasks | 100/100 | 93.48/100 | 95.65/100 | 95.24/100 | 99.18/100 |
| Codex — 1 task | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 |

Scores are out of 100. Runtime security uses the minimum recorded score; other applicable dimensions use the benchmark's reported aggregation. Discoverability is the recorded routing score. For Claude's 23-task group, effectiveness excludes 6 inapplicable tasks and remains **partial**: 4 of the 17 usable runs have partial observation coverage. Codex effectiveness excludes 2 inapplicable tasks from its 23-task group. Not-applicable values are not zero scores.

**Baseline comparison** — the benchmark's comparable scores use correctness, efficiency and security. Discoverability and unavailable dimensions are excluded; these values are not overall pass rates.

| Agent / recorded group | Without plugin | With plugin | Change |
| --- | --- | --- | --- |
| Claude Code — 23 tasks | 44.29/100 | 90.67/100 | +46.38 points |
| Claude Code — 1 task | 66.67/100 | 100/100 | +33.33 points |
| Codex — 23 tasks | 49.61/100 | 97.55/100 | +47.95 points |
| Codex — 1 task | 100/100 | 100/100 | +0 points |

These observations do not establish complete coverage of all 53 skills, every supported host, or production hardware. Token usage, turn counts, result identifiers and repeated model fields are retained in the original evidence rather than repeated in this card. See BENCHMARK.md for metric definitions, grouping and coverage details.

**Planned result coverage:** unknown

Component tests do not establish whole-plugin behavior. Detailed results retain missing values and errors.

## Plugin Version(s):

**Version (manifest):** `1.0.0`

**Signing identifier:** Pending a new whole-plugin signature for this edited card. Individual skill signature statuses are listed in the Skills table.

**Content digest (declared artifact and algorithm):** Pending recomputation for the edited package during signing. The previous package digest is not the digest of this repaired package.

**Signature / provenance:** New whole-plugin signature pending. Existing build provenance describes the upstream bundle and is not a signature over this edited card.

## Reference(s):

- Declared DOCA skills upstream repository
- Baseline signature verification

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal team to ensure this plugin meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report quality, risk, security vulnerabilities or AI concerns through [NVIDIA's reporting route](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
