# nvidia-skills Plugin

## Description:
A single-skill plugin that checks NVIDIA's live skills catalog, recommends a matching NVIDIA skill for the task, and gives an install command plus a first prompt.

This plugin is ready for commercial or non-commercial use.

This plugin is ready for external use

## Owner:
NVIDIA

### License/Terms of Use:
Apache-2.0 AND CC-BY-4.0

## Use Case:
Discover and optionally install NVIDIA agent skills for NVIDIA-related work such as GPU acceleration, CUDA, AI agents, training, inference, robotics, Omniverse, and simulation. Generic non-NVIDIA tasks are out of scope.

### Compatible Agents:
Claude Code, Codex

### Release Management:
N/A

### Deployment Geography for Use:
Global

## Plugin Package:
**Package Format:** Claude Code plugin, Codex plugin, Cursor plugin
**Manifest Location(s):** plugin.json; .claude-plugin/plugin.json; .codex-plugin/plugin.json; .cursor-plugin/plugin.json
**Package Unit:** Directory
**Client Extension Namespace(s):** Not established from the supplied package.
**Directory Layout:**
```
nvidia-skills/
├── .claude-plugin/
├── .codex-plugin/
├── .cursor-plugin/
├── assets/
├── evals/
├── plugin.json
├── skills/  # 1 skill; listed below
└── …  # Other package files
```

## Plugin Components:

### Skills:
| Skill | Verified |
|---|---|
| nvidia-skill-finder | Verified |

### Bundled Dependencies:
No code or libraries are bundled. Runtime prerequisites are Node.js plus the skills CLI downloaded from npm through npx. In Hermes it needs Hermes 0.22 or newer for --source, plus curl and jq, with python3 as a fallback.

## Requirements / Dependencies:
Requires API Key or External Credential: No
Credential Type(s): None; no credentials are described or stored by the plugin.
Credential Supplied By: None; the plugin states it stores no credentials, and no credential provider is described.

Do not include secrets in prompts/logs/output or in the package (`plugin.json`, `mcp.json`, headers, env); use least-privilege credentials; rotate keys as appropriate.

**Runtime Environment:** Runs as an agent skill in Claude Code, Codex, Cursor, and Hermes, with Claude Code, Codex and Cursor manifests plus a portable Agent Plugins manifest used by Hermes. In Hermes it uses Hermes's own catalog and install commands and never calls npx.
**Network Access Required:** Makes outbound requests to npm to download the skills CLI, and to NVIDIA's GitHub repository, the GitHub raw content host, and the NVIDIA build site for catalog data. Installs fetch skills from the main branch of NVIDIA/skills.
**Subprocess Execution:** When shell access is available, the skill instructs the agent to run subprocesses: npx with the downloaded skills CLI, curl, jq or python3, and hermes. These are skill instructions, not bundled executables.
**File System Access:** Writes installed skill files into the agent's global skills directory or the active Hermes profile. The Codex manifest declares a Write capability, and the plugin bundles no code of its own.

## Capability and Permission Summary:
**Operations Allowed:**

| Operation | Scope |
|---|---|
| Create | After the user agrees, it installs new NVIDIA skills into the agent's global skills directory, or into the active Hermes profile. This adds files and changes the agent's capabilities. |
| Read | Reads the remote NVIDIA skills catalog listing. It also checks whether a relevant skill is already installed or already in context. |
| Update | Not established from the supplied package. |
| Delete | Not established from the supplied package. |
| Execute | Instructs the agent to run shell commands: npx skills add to list the catalog, npx skills add with --global --yes to install after the user agrees, and npx skills find for wider search. In Hermes it uses curl with jq or python3, plus hermes skills search, install, and inspect. |
| Search | Matches the request against NVIDIA catalog skill names, descriptions, product groups, and skill cards. If nothing matches, it may suggest npx skills find, which searches outside NVIDIA's catalog. |
| Web Search | No general web search is described. As a fallback, the skill reads the catalog from specific NVIDIA and GitHub sites. |

**Changes Host Agent Autonomy:** No hooks are included, but implicit invocation is enabled, so the skill can activate on NVIDIA-related requests without being named. Installs run only after the user agrees, and the npx install then passes --yes.

## Known Technical Limitations:
The skill must not invent catalog entries or guess skill names, and it is out of scope for generic non-NVIDIA tasks. Hermes --source needs version 0.22 or newer, and Hermes search uses substring matching that truncates at 25 results by default. A client restart or reload may be needed after install.

## Known Risks and Mitigations:
Risk: Skills and tool descriptions bundled in a plugin can introduce incorrect, misleading, or malicious guidance into the agent's context.
Mitigation: Review and scan the plugin and every component before deployment.

Risk: Skills installed through the `npx` workflow use the current content of NVIDIA/skills' `main` branch, so later installations may retrieve updated content. The CLI is downloaded from npm at runtime. After the user agrees, the npx install passes --yes. Wider search can surface third-party skills from outside NVIDIA's catalog, and no integrity checks outside Hermes are described.
Mitigation: The source instructs human confirmation before any install. Hermes installs go through Hermes's own security scan and confirmation prompt without --yes, and uppercase NVIDIA identifier casing is required to resolve to the first-party source. These are instructions, not enforced controls.

## Disable and Revoke:
Not established from the supplied package.
Measured Time to Effect: Not measured

## Data Flow and Security:
**Logging and Traceability:** The plugin states it sends no telemetry, and no logging is described. The external skills CLI follows its own policies.
**Data Persisted by Plugin (PLUGIN_DATA or equivalent):** Installed skills persist in the agent's global skills directory or the Hermes profile. The plugin states it stores no credentials.
**Data Sent to External Services:** Retrieves NVIDIA catalog metadata and skill content from GitHub and NVIDIA sites, and downloads the skills CLI from npm. Wider search sends the user's query through the skills CLI and can surface third-party skills. The plugin states it sends no telemetry, but the external CLI follows its own policies.

**Database Name:** No database is described; the plugin contains no code and only reads a remote catalog listing.
**Database Type**
**☐ Commercial**
**☐ Confidential**
**☐ Internal**
**☐ Open-Source**
**☐ Proprietary**
**☐ Public**
**☐ Regulated**
**☑ None**

**Security Controls**
The source describes these controls as instructions: installs require human confirmation; Hermes installs use Hermes's own security scan and confirmation prompt without --yes; and uppercase NVIDIA identifier casing selects the trusted first-party source over community mirrors.

**Exposure to Threats**
**☑ External API/Connector/Service**
**☐ Internal API/Connector/Service**
**☐ Agent-to-Agent Inbound**
**☐ Unattended / Scheduled Execution**
**Exceptions:** Not established from the supplied package.

## Testing Completed:
**☑ Plugin Static Analysis / Scan**
**☐ MCP Server Security Review**
**☐ Dependency / SBOM Review**
**☐ Network Security**
**☐ Product Security**

Evaluation: partially completed; 18 tasks; 36 with-plugin attempts; 36 baseline attempts.

4 with-plugin attempts have incomplete grading.

Validation: passed; 422/676 checks completed; 0 blocking findings; 152 advisory findings.

## Plugin Version(s):
**Version:** 1.5.0
**Signing Identifier:** Not recorded for this generated card.
**Content Digest:** N/A
**Signature / Provenance:** The whole-plugin signature is included as `plugin.oms.sig`.

## Reference(s):

- [Plugin overview](README.md)
- [NVIDIA Skill Finder](skills/nvidia-skill-finder/SKILL.md)
- [Bundled skill card](skills/nvidia-skill-finder/skill-card.md)
- [Benchmark](BENCHMARK.md)
- [Hermes catalog lookup and installation](skills/nvidia-skill-finder/references/hermes-catalog.md)
- [NVIDIA Skills catalog](https://github.com/NVIDIA/skills/tree/main/skills)

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal team to ensure this plugin meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report quality, risk, security vulnerabilities or NVIDIA AI concerns through the [NVIDIA vulnerability disclosure program](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail), as linked in the bundled skill card.
