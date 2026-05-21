# Advanced: Agent Teams for PR Review

You can use Claude Code's **agent teams** (tmux/teammate mode) to run multiple specialized review agents in parallel. Each agent gets its own context window and they communicate via a shared task list and direct messaging.

This is useful any time you want interactive parallel agents that communicate with each other during review — not just for large PRs. Note that agent teams use ~3x more tokens than a single-agent review.

## Setup

### 1. Enable agent teams

Add to your Claude Code settings (`.claude/settings.json` or `~/.claude/settings.json`):

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

### 2. Choose a display mode

In `~/.claude.json`:

```json
{
  "teammateMode": "tmux"
}
```

Options:
- **`in-process`** (default): all teammates in the main terminal. Use `Shift+Down` to cycle between them.
- **`tmux`**: each teammate gets its own tmux pane. Click a pane to interact directly.
- **`auto`**: uses tmux if available, falls back to in-process.

tmux mode requires tmux to be installed. Not supported in VS Code integrated terminal.

## Example: Launch a Review Team

Start Claude Code in the repo, then:

```
Create a review team for PR #123 with these teammates:
1. Bug detection agent — focus on logic errors, null refs, race conditions, security issues
2. Guidelines agent — check all coding guideline skills in .claude/skills/
3. Memory agent — review .claude/review-memory/ for recurring patterns, suggest promotions
4. CI agent — when review is done, trigger CI via /ok-to-test-rl
Have them coordinate findings and report back.
```

Claude will:
- Spawn each teammate as an independent Claude Code instance
- Each teammate loads `CLAUDE.md` and the skills automatically
- Teammates claim tasks from the shared list and communicate directly
- The lead synthesizes findings before posting

## Example: Focused Team

For a smaller, more targeted review:

```
Create a 2-agent review team for PR #456:
1. Code reviewer — run /review-pr 456 and go through findings
2. Test reviewer — check if the PR has adequate test coverage for all changed files
Have them share findings when done.
```

## Tips

- **Start with 2-3 teammates** — more agents means more coordination overhead
- **Assign independent work** — agents work best when they own separate concerns
- **Monitor progress** — in tmux mode, watch each pane; in in-process mode, use `Shift+Down`
- **Direct interaction** — click into any teammate's pane (tmux) or cycle to it (in-process) to ask questions or redirect

## Limitations

- **Experimental feature** — may have rough edges
- **No session resumption** — if you close the terminal, teammates are lost
- **Higher token cost** — each teammate is a full Claude instance
- **One team per session** — clean up before starting a new team
- **No nested teams** — teammates cannot spawn their own teams
