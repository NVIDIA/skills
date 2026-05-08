# VSS Skills

Skills for working with NVIDIA Video Search & Summarization (VSS). Each subdirectory under `skills/` is a self-contained skill following the [agentskills.io](https://agentskills.io/specification) specification, with `name`, `description`, `version`, and `license` declared in its `SKILL.md` frontmatter.

## Catalog

| Skill | Description |
|---|---|
| [alerts](alerts/SKILL.md) | Skill to add, manage, and monitor alerts on streamed video. |
| [deploy](deploy/SKILL.md) | Skills to deploy, debug, or tear down any VSS profile using a docker compose-centric workflow. |
| [report](report/SKILL.md) | Skill to produce video analysis reports by querying the VSS agent's `/generate` endpoint. |
| [rt-vlm](rt-vlm/SKILL.md) | Skill to use the real-time VLM microservice on stored or streamed video - captions, alerts, streams, or OpenAI-compatible completions. |
| [video-analytics](video-analytics/SKILL.md) | Skill for querying video analytics data and metrics from Elasticsearch via the VA-MCP server. |
| [video-search](video-search/SKILL.md) | Skills for searching video archives using natural language, multi-embedding fusion, and VLM critique. |
| [video-summarization](video-summarization/SKILL.md) | Skill for summarizing a video through chunking, dense captioning, and aggregation functions using the Long Video Summarization (LVS) microservice. |
| [video-understanding](video-understanding/SKILL.md) | Skill for using video understanding tool to answer text questions about video content using a VLM. |
| [vios](vios/SKILL.md) | Skill for video and stream management, recording timelines, clip extraction, snapshots (and more) using the Video IO and Storage microservices. |
| [vss-frag](vss-frag/SKILL.md) | Skill for deploying and integrating the `video_search_frag` extension and generate video summary reports — Long Video Summarization, Enterprise RAG context, HITL parameter collection. |

Skills with `eval/*.json` specs are exercised automatically by the Skills Eval CI workflow on every PR that touches `skills/**` — see [`.github/skill-eval/AGENTS.md`](../.github/skill-eval/AGENTS.md) for harness behavior.

## Install (recommended: ask your coding agent)

Open this repository in your coding agent (Claude Code, Codex, Cursor, or any other agentskills.io-compatible host) and paste the following prompt:

> Read `skills/README.md` and every `SKILL.md` file under `skills/`. For each skill in the catalog, install it for this host so I can invoke it from a shell or chat session. Use the host's standard skills directory:
>
> - Claude Code: `~/.claude/skills/<name>/`
> - Codex: `~/.codex/skills/<name>/`
> - Hosts that follow the agentskills.io universal path: `~/.agents/skills/<name>/`
>
> Symlink each skill folder rather than copying it so a `git pull` here keeps every install up to date. Skip skills that are already installed and pointing at this checkout. When you're done, list the skills you registered and which directory you used.

The agent will read the frontmatter of each `SKILL.md`, create the symlinks, and confirm what's installed. The skills become invokable in the next agent session.

### Single-skill install

To install skills individually, paste the following prompt:

> Install only `skills/<name>/` for this host the same way.

### Update

After `git pull`, the symlinks already point at the updated content — nothing to do unless skills were added or renamed. To pick up new skills use the following prompt:

> Re-read `skills/README.md` and add any new skills missing from this host's skills directory.

### Uninstall

To uninstall skills, paste the following prompt:

> Remove every VSS skill symlink you previously created under this host's skills directory.

## Source of truth

This `skills/` directory is the canonical source. Skills published to the public catalog at `github.com/nvidia/skills` are mirrored from here at sync time per [`components.yml`](https://github.com/NVIDIA/skills/blob/main/components.yml).
