## Description: <br>
Omniverse Content Services guides USD content collaboration and content access workflows on existing infrastructure, covering USD content versioning, 3D creation pipelines, and data access across distributed and scale-out compute, and routes implementation questions to the ovstorage, ovpackage, ovlive, usd-content-workspace, and usd-caches experts. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers, engineers, and agent harnesses building USD workflows and 3D content pipelines use this skill to discover workflow requirements (source authority, collaboration, versioning, compute locality, permitted caching) and to plan integrations that keep existing systems as the source of truth across manufacturing, automotive, AEC, media, robotics, digital-twin, and physical AI simulation use cases. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Repository routes](references/repositories.md) <br>
- [Discover the content workflow](references/workflow-discovery.md) <br>
- [Release record](references/releases.md) <br>
- [NGC prebuilds and documentation](references/prebuilt-artifacts.md) <br>
- [ovstorage](https://github.com/nvidia-omniverse/ovstorage) <br>
- [usd-content-workspace (release/v0.1.0)](https://github.com/nvidia-omniverse/usd-content-workspace/tree/release/v0.1.0) <br>
- [ovpackage](https://github.com/nvidia-omniverse/ovpackage) <br>
- [ovlive](https://github.com/nvidia-omniverse/ovlive) <br>
- [usd-caches](https://github.com/nvidia-omniverse/usd-caches) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions] <br>
**Output Format:** [Markdown] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Workflow plans and expert handoffs record expert name, expert version, source repository, revision, and entrypoint; credentials are never included in reports.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
10 evaluation tasks (8 positive, 2 negative), 1 attempt per task, each run in an isolated sandbox pod in a `k8s-sandbox` environment and compared against a no-skill baseline. Evaluator version `1.5.6`; evaluated 2026-10-24. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of `skill_efficiency` and `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
Overall verdict: PASS — Recommended for publication.

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.2% (uplift unavailable) | 80.0% (uplift unavailable) |
| Security | 100.0% → 85.0% (-15.0 points) | 100.0% → 85.0% (-15.0 points) |
| Correctness | 64.0% → 100.0% (+36.0 points) | 74.0% → 92.0% (+18.0 points) |
| Discoverability | 95.0% (uplift unavailable) | 83.1% (uplift unavailable) |
| Effectiveness | 45.3% → 86.3% (+41.0 points) | 50.7% → 69.4% (+18.7 points) |
| Efficiency | 74.6% (uplift unavailable) | 70.4% (uplift unavailable) |

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | PASSED WITH OBSERVATIONS | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | PASSED | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | PASS | 2 agent(s); 10 task(s) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
