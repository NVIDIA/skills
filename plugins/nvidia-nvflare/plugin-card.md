# nvidia-nvflare Plugin

> **Evaluation: PASS · Partial coverage** · Best evaluated agent: **Codex, 80.66/100**.
>
> **Static validation: Incomplete coverage**.

## Description

NVIDIA FLARE skills for federated learning conversion, job diagnosis, federated statistics, and Auto-FL optimization and reporting. The bundle contains **nine
skills: eight user workflows and one shared support skill**. Install the complete bundle to preserve references
between members.

## Owner

**NVIDIA FLARE Team** · [federatedlearning@nvidia.com](mailto:federatedlearning@nvidia.com)

### License / Terms of Use

**Apache-2.0 AND CC-BY-4.0:** [Apache 2.0](LICENSE) applies to code, scripts,
configuration and code examples; [CC BY 4.0](LICENSE-CC-BY-4.0) applies to
instructions and other NVIDIA-authored skill content. See the
[scope and attribution policy](SKILLS-LICENSE.md); third-party terms remain applicable.

## Use Case

Developers and ML engineers use the skills to convert training code into
federated workflows, diagnose jobs, compute statistics across sites, and optimize
or report Auto-FL campaigns.

### Compatible Agents

| Agent | Declared entry point | Evaluation scope |
| --- | --- | --- |
| Claude Code | `.claude-plugin/plugin.json` | Bundled-skill tasks |
| Codex | `plugin.json` and `.codex-plugin/plugin.json` | Bundled-skill tasks |
| Cursor | `plugin.json` or `.cursor-plugin/plugin.json` | No Cursor evaluation supplied |

Minimum agent versions are not declared. The Codex manifest requires an executor
on the consuming host. Native plugin loading and compatibility across all host
versions have not been evaluated.

### Release Management

| Channel | Location | Version / release information |
| --- | --- | --- |
| GitHub source | [NVIDIA/NVFlare](https://github.com/NVIDIA/NVFlare) | Plugin manifest version 1.0.0; plugin release date not declared |
| Documentation | [NVIDIA FLARE](https://nvidia.github.io/NVFlare/) | Framework documentation |

The plugin version is independent of the NVFlare Python package version.
Package-wide commercial-use readiness and marketplace availability are not declared.

### Deployment Geography for Use

No package-wide geographic restrictions or deployment declaration are supplied.

## Plugin Package

**Format:** Agent Plugins manifest, schema 1.0.0, with Claude Code, Codex and
Cursor manifests. **Package unit:** directory.

| Manifest | Purpose |
| --- | --- |
| [plugin.json](plugin.json) | Portable plugin identity |
| [.claude-plugin/plugin.json](.claude-plugin/plugin.json) | Claude Code discovery |
| [.codex-plugin/plugin.json](.codex-plugin/plugin.json) | Codex discovery and interface |
| [.cursor-plugin/plugin.json](.cursor-plugin/plugin.json) | Cursor discovery |

The package includes complete `skills/` directories, evaluation fixtures,
license texts, [README.md](README.md), [BENCHMARK.md](BENCHMARK.md) and this card.
Generated evaluation exports are retained separately from the signed inventory;
authored evaluation datasets and fixtures remain package content.

## Plugin Components

### Skills

| Skill / member card | Workflow | Member signature |
| --- | --- | --- |
| [nvflare-orient](skills/nvflare-orient/skill-card.md) | Route open-ended NVIDIA FLARE questions | [skill.oms.sig](skills/nvflare-orient/skill.oms.sig) |
| [nvflare-convert-pytorch](skills/nvflare-convert-pytorch/skill-card.md) | Convert PyTorch training to federated jobs | [skill.oms.sig](skills/nvflare-convert-pytorch/skill.oms.sig) |
| [nvflare-convert-lightning](skills/nvflare-convert-lightning/skill-card.md) | Convert Lightning training to federated jobs | [skill.oms.sig](skills/nvflare-convert-lightning/skill.oms.sig) |
| [nvflare-convert-huggingface](skills/nvflare-convert-huggingface/skill-card.md) | Convert Hugging Face Trainer and TRL workflows | [skill.oms.sig](skills/nvflare-convert-huggingface/skill.oms.sig) |
| [nvflare-diagnose-job](skills/nvflare-diagnose-job/skill-card.md) | Diagnose failed or stalled jobs | [skill.oms.sig](skills/nvflare-diagnose-job/skill.oms.sig) |
| [nvflare-fed-stats](skills/nvflare-fed-stats/skill-card.md) | Compute federated statistics | [skill.oms.sig](skills/nvflare-fed-stats/skill.oms.sig) |
| [nvflare-autofl](skills/nvflare-autofl/skill-card.md) | Optimize an existing federated job | [skill.oms.sig](skills/nvflare-autofl/skill.oms.sig) |
| [nvflare-autofl-report](skills/nvflare-autofl-report/skill-card.md) | Report a stopped or interrupted Auto-FL campaign | [skill.oms.sig](skills/nvflare-autofl-report/skill.oms.sig) |
| [nvflare-shared](skills/nvflare-shared/skill-card.md) | Shared references and templates used by the other skills | [skill.oms.sig](skills/nvflare-shared/skill.oms.sig) |

`nvflare-shared` supplies references and templates for the other members rather
than a separate user workflow. Some hosts may still list it as a skill.
Individual member signatures and the whole-plugin signature cover different
inventories; member files must be reverified after any member-content change.

### MCP Servers, Hooks and Other Components

The package guide declares no MCP server, hook or automatic installer. Scanner
discovery remains incomplete for some component types; these declarations are
source descriptions rather than a runtime compatibility test.

### Bundled Dependencies

No vendored runtime dependency is declared. NVFlare and the selected workflow's
Python frameworks must be supplied by the consuming environment.

## Requirements / Dependencies

| Requirement | Scope |
| --- | --- |
| NVFlare 2.9.0 or later | Documented workflow prerequisite |
| Python and workflow frameworks | Dependencies vary by selected conversion or analysis workflow |
| Host tools and permissions | File editing, subprocess execution and job submission depend on the authorized task |
| Deployment credentials | Supplied for the selected environment where required; no package-wide credential mechanism is declared |

Installing the plugin does not install NVFlare, download models, provision GPUs
or authorize job submission. Keep credentials outside prompts, logs and package files.

## Capability and Permission Summary

Skills guide the agent to read selected code, configuration, data and logs;
create converted jobs and reports; and run workflow scripts or federated jobs
when authorized. Effective filesystem, subprocess and network permissions come
from the consuming host and the approved task. The plugin does not grant
additional host authority or establish an independently enforced autonomy policy.

## Known Technical Limitations

- Native plugin loading and all host/version combinations were not evaluated.
- The evaluation covers five bundled-skill task cases, with partial execution and grading coverage; it does not establish complete coverage of every bundled skill.
- Static analysis has incomplete archive, encoded-content and connection coverage. Some authored evaluation metadata is outside the native scanner's default selection.
- Install all nine members to preserve shared references and templates.

## Known Risks and Mitigations

| Risk | Mitigation / boundary |
| --- | --- |
| Instructions or generated changes may be incorrect. | Review generated jobs, configurations and reports against the intended workflow. |
| Conversion and Auto-FL can modify files or launch jobs. | Confirm the target project and execution scope; apply the consuming host's permissions and deployment controls. |
| Evaluation fixtures contain adversarial instructions and payloads. | Treat fixtures as test data, not authorization to execute their embedded requests. |
| Partial analysis leaves some behavior unverified. | Review the retained findings and coverage before deployment. |

## Disable and Revoke

Disable or remove the plugin through the consuming host's plugin controls.
Stop independently launched jobs separately and revoke deployment credentials
through their issuing systems where applicable. Package-specific disable and
revocation timing has not been measured.

## Data Flow and Security

| Data | Documented flow |
| --- | --- |
| Inputs | Selected source projects, datasets, job configurations, logs and campaign state |
| Outputs | Converted jobs, configuration updates, plots, statistics and reports |
| Network access | Depends on the chosen workflow, external data/model sources and deployment |
| Persistence | Outputs and campaign state are written to workflow-selected storage |

Storage locations, retention, access controls and logging policies depend on the
selected deployment; no package-wide assurance is supplied. Do not include
private datasets, credentials or sensitive traces in the distributable package.

## Testing Completed

### Evaluation results

| Agent | With-plugin quality score | Score coverage |
| --- | --- | --- |
| Claude Code | **77.94/100** | Partial |
| **Codex** | **80.66/100** | **Partial** |

The recorded evaluation-quality verdict is **PASS**. Codex is the highest-scoring
agent and meets the retained passing-agent score requirement. Its matched-baseline
lift is **+37.27 points**; Claude Code's is **+6.63 points**.
The saved cross-agent aggregate is **77.94/100**, distinct from Codex's individual
score. All scores follow the unchanged retained assessment.

The export contains **20 task slots**: five cases, two agents, and with-plugin
and baseline variants. **17 executions completed; 12 slots were fully graded.**
Evaluation covers bundled skills; native plugin loading was outside its scope.
Efficiency decreased on matched tasks by **10.11 points on Claude Code** and
**10.42 points on Codex**. See [BENCHMARK.md](BENCHMARK.md) for dimension scores,
paired comparisons and coverage.

### Static validation

| Check | Result |
| --- | --- |
| Analysis coverage | **Incomplete coverage** |
| Native analysed files | 178 in the retained scan snapshot |
| Total recorded findings | 762 raw / 531 unique |
| Validation warnings | 201 |
| Bandit, Gitleaks and Semgrep | Completed |
| Scanner suppressions applied | None |

Archive inspection, encoded-content analysis and connection analysis have
incomplete coverage. The full validation report retains the findings and detailed
diagnostics; no scanner suppressions have been applied.

### Security reviews

No accepted product-security, dependency/SBOM or network-security review record
is supplied. Evaluation results, static validation and signature verification are
separate checks; this card does not grant publication approval.

## Plugin Version(s)

| Item | Value |
| --- | --- |
| Plugin manifest version | 1.0.0 |
| Evaluation tools | NeMo Gym 0.7.0; Claude Code and Codex |
| Validation tools | CLI 1.3.1435; Plugin Verifier 0.1.10 |
| Whole-plugin signature filename | `plugin.oms.sig` |
| Content digests | Recorded in the signed resource inventory in `plugin.oms.sig` |
| Signing profile | External |

The detached signature must be verified against the approved external certificate
chain and the complete package inventory. Generated evaluation exports and member
signature sidecars are excluded from the parent signing inventory; authored
plugin and skill content remains included. Existing member signatures require
independent certificate and inventory verification.

## Reference(s)

- [Package guide](README.md)
- [Evaluation benchmark](BENCHMARK.md)
- [License scope and attribution](SKILLS-LICENSE.md)
- [NVIDIA FLARE source](https://github.com/NVIDIA/NVFlare)
- [NVIDIA FLARE documentation](https://nvidia.github.io/NVFlare/)

## Ethical Considerations

Review the selected workflow for applicable privacy, data-governance and
industry requirements before use. Report security vulnerabilities or AI
concerns through [NVIDIA's reporting route](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail).
