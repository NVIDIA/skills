## Description: <br>
Use this skill when integrating NVIDIA NeMo Fabric into a consumer application, service, evaluation harness, or platform through the typed Python SDK — translating the consumer's own application, job, or deployment config into an in-memory FabricConfig, choosing the single-invocation convenience API or an explicitly started runtime, validating with plan and doctor, and consuming normalized results, artifacts, and telemetry. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to integrate NVIDIA NeMo Fabric into their own applications, services, evaluation harnesses, or platforms through the public Python SDK, mapping their configuration to a typed FabricConfig, selecting a runtime lifecycle, validating with plan and doctor, and consuming normalized results, artifacts, and telemetry. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Mapping Consumer Config To FabricConfig](references/config-mapping.md) <br>
- [Results, Evidence, And Errors](references/results-and-errors.md) <br>
- [SDK API Inventory](references/sdk-api-inventory.md) <br>
- [NeMo Fabric Python SDK](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/python.mdx) <br>
- [NeMo Fabric Installation Guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/getting-started/install.mdx) <br>
- [NeMo Fabric Overview](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/about-nemo-fabric/overview.mdx) <br>
- [Hermes Integration Guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/integrations/harness/hermes.mdx) <br>
- [System Stats](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/system-stats.mdx) <br>
- [nemo_fabric.client API Reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.client.md) <br>
- [nemo_fabric.runtime API Reference](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/reference/api/python-library-reference/nemo_fabric.runtime.md) <br>
- [NeMo Fabric Adapter Catalog](https://github.com/NVIDIA/NeMo-Fabric/blob/main/sdk/python/nemo-fabric-adapter-catalog/README.md) <br>
- [code_review_agent Example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/code_review_agent) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline Python and shell code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative), 1 attempt per task, each run in an isolated sandbox pod in a `k8s-sandbox` environment; evaluator version `1.6.4`, evaluated 2026-10-29. Overall verdict: PASS — recommended for publication. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal (100%). <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal (100%). <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` (tool-call productivity) and 50% `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 71.7% — baseline ran, but no comparable score was available; uplift unavailable | 74.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 66.7% (-33.3 points) | 83.3% → 83.3% (±0.0 points) |
| Correctness | 33.3% → 90.0% (+56.7 points) | 40.0% → 63.3% (+23.3 points) |
| Discoverability | 88.8% — baseline ran, but no comparable score was available; uplift unavailable | 88.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 33.5% → 35.9% (+2.4 points) | 43.1% → 63.1% (+20.0 points) |
| Efficiency | 77.0% — baseline ran, but no comparable score was available; uplift unavailable | 74.6% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.7.0 (source: Cargo.toml workspace version; pyproject.toml reports placeholder 0.0.0; git SHA 886e692, committed 2026-10-29) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
