## Description: <br>
Model and publish semantic definitions in Auto Ontology, covering terms, relationships, measures, imports, and governed results (not deployment or querying). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data engineers use this skill to have an agent inspect and change the Auto Ontology semantic layer: editing terms and SQL attributes, defining source-grounded concepts, relationships, and measures, importing or exporting models with exact readback, and publishing governed definitions or results. <br>

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
- [Ontology write API (index)](references/write-api.md) <br>
- [Source-grounded ontology modeling](references/modeling.md) <br>
- [Governed publication, readback, and rollback](references/publication.md) <br>
- [Runtime contract](assets/runtime-contract.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [API Calls, Analysis, Configuration instructions] <br>
**Output Format:** [Markdown with inline HTTP request examples against the Next.js /api gateway] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [MCP access is read-only; writes need the token owner's permissions, and unsupported publication stops at an explicit receipt] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 positive evaluation tasks, one attempt per task, each run in an isolated k8s sandbox pod and compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? (scored from `security`) <br>
- Correctness: Is the answer correct? (scored from `accuracy`) <br>
- Discoverability: Was the right skill loaded when needed? (scored from `skill_execution`) <br>
- Effectiveness: Did the skill help complete the task? (50% `goal_accuracy` + 50% `behavior_check`) <br>
- Efficiency: Did it avoid wasted tool calls and token usage? (50% `skill_efficiency` + 50% `token_efficiency`) <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.9% (no comparable baseline; uplift unavailable) | 85.3% (no comparable baseline; uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 84.6% → 76.9% (-7.7 points) |
| Correctness | 29.2% → 100.0% (+70.8 points) | 47.7% → 89.2% (+41.5 points) |
| Discoverability | 93.7% (no comparable baseline; uplift unavailable) | 85.8% (no comparable baseline; uplift unavailable) |
| Effectiveness | 42.4% → 89.3% (+46.9 points) | 33.9% → 81.1% (+47.2 points) |
| Efficiency | 91.4% (no comparable baseline; uplift unavailable) | 93.3% (no comparable baseline; uplift unavailable) |

## Skill Version(s): <br>
0.2.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
