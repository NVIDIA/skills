---
name: review-pr
description: Interactive code review for NVIDIA-NeMo/RL pull requests. Checks out PR locally, reads existing comments, applies coding guidelines from skills, previews findings, and posts review comments. Also supports reviewing the current branch locally.
when_to_use: Reviewing a PR; '/review-pr <number>'; 'review this PR', 'check PR', 'code review for PR'.
argument-hint: [<pr-number>] [update] [--deep]
allowed-tools: [AskUserQuestion, Bash, Read, Glob, Grep, Agent, mcp__github__pull_request_read, mcp__github__pull_request_review_write, mcp__github__add_comment_to_pending_review, mcp__github__add_reply_to_pull_request_comment, mcp__github__add_issue_comment]
---

# Interactive PR Review — NVIDIA-NeMo/RL

Review a pull request or local branch interactively, applying the project's coding guidelines.

## Parse Arguments

- If `$ARGUMENTS` contains a number → **PR mode** with that PR number
- If `$ARGUMENTS` is empty or has no number → **LOCAL mode** (review current branch vs main)
- If `$ARGUMENTS` contains `update` → **UPDATE** mode (PR mode only)
- If `$ARGUMENTS` contains `--deep` → use parallel subagents for deeper review

**Repo**: `NVIDIA-NeMo/RL`

**Examples:**
- `/review-pr 123` — review PR #123, single agent
- `/review-pr 123 --deep` — review PR #123 with parallel subagents
- `/review-pr 123 update` — follow up on existing threads for PR #123
- `/review-pr` — review current branch vs main, terminal output only

---

## Phase 1: Setup

### PR mode
```bash
git fetch origin pull/$PRNUM/head:pr-$PRNUM-review
git checkout pr-$PRNUM-review
```

### LOCAL mode
Already on the correct branch. Determine the merge base:
```bash
git merge-base main HEAD
```

---

## Phase 2: Gather Context

### PR mode — parallel MCP fetches

All with `owner=NVIDIA-NeMo`, `repo=RL`, `pullNumber=$PRNUM`:

1. `mcp__github__pull_request_read` method=`get` → PR title, description, author, base branch, labels, head SHA
2. `mcp__github__pull_request_read` method=`get_diff` → full diff
3. `mcp__github__pull_request_read` method=`get_files` → list of changed files
4. `mcp__github__pull_request_read` method=`get_review_comments` → existing review threads
5. `mcp__github__pull_request_read` method=`get_reviews` → existing reviews
6. `mcp__github__pull_request_read` method=`get_comments` → general PR comments

### LOCAL mode — git diff

```bash
git diff $(git merge-base main HEAD)..HEAD
git diff $(git merge-base main HEAD)..HEAD --name-only
```

No MCP calls needed.

### Both modes — local reads

7. Read @CLAUDE.md from repo root for review philosophy
8. Read all `.claude/skills/*/SKILL.md` files (except `review-pr`) for guideline rules
9. Glob `.claude/review-memory/*.md` — if any exist, read them for learned patterns

---

## Phase 3: Analyze

### Single-agent mode (default)

Analyze all changes yourself. Return a list of candidate issues — each with file, line, category, and description. Do NOT score them yet; scoring happens in the validation step (Phase 3b).

### Deep mode (`--deep`) — parallel subagents

Launch 3 opus subagents in parallel using the `Agent` tool. Provide each with the diff, PR description (if PR mode), and the guideline skills content. Each returns a list of candidate issues (file, line, category, description). Do NOT ask them to score — scoring happens in Phase 3b.

**Subagent 1 — Guideline compliance:**
Review the diff against all guideline skills (code-style, config-conventions, error-handling, testing, copyright, docs). For each violation, return the file, line, description, and which skill it violates.

**Subagent 2 — Bug scan (diff only):**
Scan for obvious bugs in the diff without reading surrounding context. Flag syntax errors, type errors, clear logic errors, missing imports, unresolved references.

**Subagent 3 — Contextual bug scan:**
Read surrounding code and git history for each changed file. Look for bugs that only become apparent with context: incorrect API usage (especially megatron-bridge, megatron-lm, automodel, gym), race conditions, broken assumptions.

After all subagents return: merge results and deduplicate (same file+line+issue = one finding). Then proceed to Phase 3b.

### Analysis rules (all modes)

#### NEW mode
- Analyze the diff against all guideline skills
- For each changed file, read surrounding context locally using `Read` and `Grep` to understand the change in context
- Cross-reference existing review comments (PR mode only, from step 4) to avoid duplicating points already raised by other reviewers
- Also apply any patterns from review memory files
- Categorize findings:
  - **[BUG]** — Logic errors, null refs, race conditions, syntax errors
  - **[TEST]** — Missing or insufficient test coverage
  - **[GUIDELINE]** — Violations of coding guidelines from skills
  - **[DOC]** — Outdated or missing documentation

#### UPDATE mode (PR mode only)
- Review all unresolved review threads on the PR
- For each thread, determine if action is needed:
  1. **We disagree with the response** → draft a comment like "We can resolve this thread because XYZ"
  2. **The author asked a question** or made a comment that needs a response → draft an answer
- CAN also create new comments if you notice something warranted while reviewing threads
- Skip threads that are resolved or where no response is needed

---

## Phase 3b: Validate & Score

For each candidate issue from Phase 3, launch a **separate opus validation subagent** using the `Agent` tool. Launch these in parallel (batch all at once).

Each validation subagent receives:
- The candidate issue (file, line, category, description)
- The relevant code context (the diff hunk + surrounding lines)
- The PR title and description (if PR mode)
- The specific guideline skill content (if it's a guideline violation)

The validation subagent's job:
1. **Independently verify** the issue is real by examining the actual code — e.g., if the issue says "variable is not defined", check that it's actually undefined; if it says a CLAUDE.md/skill rule is violated, confirm the rule applies to this file
2. Assign a **confidence score (0-100)**:

| Score | Meaning |
|-------|---------|
| 0 | Not confident, likely false positive |
| 25 | Somewhat confident, might be real |
| 50 | Moderately confident, real but minor |
| 75 | Highly confident, real and important |
| 100 | Absolutely certain, definitely real |

3. Return: validated (yes/no), confidence score, and optionally a refined description

**Filter:** discard any issue scoring below **80**. These are the false positives we want to avoid.

---

## Phase 4: Preview & Confirm

Display findings to the user with confidence scores:

### PR mode
```
PR #<number>: <title> (by <author>)
Files changed: <count>

--- Findings (scored ≥80) ---
[BUG 95] path/to/file.py:42 — <brief description>
  Suggested: "<the comment text>"

[GUIDELINE 85] path/to/other.py:15 — <brief description>
  Suggested: "<the comment text>"

--- Filtered (scored <80) ---
N low-confidence issues omitted

--- Skipped (already covered by other reviewers) ---
- <brief list>

--- Thread Responses (<count>) --- (UPDATE mode only)
Thread on file.py:10 (by <author>) — <planned action>
  Suggested reply: "<the reply text>"
```

Then use `AskUserQuestion`:
- Options: **(1) Post all** — post everything as shown, **(2) Discuss individually** — go through each item one by one, **(3) Cancel** — do nothing
- If user picks "Discuss individually": iterate through items. For each, ask if they want to approve, edit the text, or skip.

### LOCAL mode
```
Branch: <branch-name> (vs main)
Files changed: <count>

--- Findings (scored ≥80) ---
[BUG 95] path/to/file.py:42 — <brief description>
[GUIDELINE 85] path/to/other.py:15 — <brief description>

--- Filtered (scored <80) ---
N low-confidence issues omitted
```

No posting step — terminal output only. Ask if the user wants to discuss any findings.

---

## Phase 5: Post Review (PR mode only)

### New comments (NEW and UPDATE mode)

1. Create a pending review: `mcp__github__pull_request_review_write` with `method=create`, `commitID=<head_sha>` (no `event` — creates pending review)
2. For each approved comment: `mcp__github__add_comment_to_pending_review` with:
   - `path`: relative file path
   - `line`: line number on the RIGHT side of the diff
   - `side`: `RIGHT`
   - `subjectType`: `LINE` (or `FILE` if the comment is file-level)
   - `body`: the comment text
3. Submit the review: `mcp__github__pull_request_review_write` with `method=submit_pending`, `event=COMMENT`, `body=<one-line summary of findings>`

If a line number cannot be mapped from the diff, fall back to `subjectType: FILE`.

### Thread replies (UPDATE mode)

For each thread response: `mcp__github__add_reply_to_pull_request_comment` with:
- `commentId`: the ID of the comment being replied to
- `body`: the reply text

---

## Phase 6: Update Review Memory

After the review (both PR and LOCAL mode), for each comment/finding the user approved or discussed:

1. Glob `.claude/review-memory/*.md` for existing patterns
2. Check if a file already covers this pattern (by reading titles)
3. **If a matching memory file exists:**
   - Add the new occurrence to the `## Occurrences` section
   - Ask user: "Pattern '<name>' has come up before. Promote to the `<skill-name>` skill? (yes/no)"
   - If yes → add as a new subsection in the appropriate skill's `SKILL.md`, then delete the memory file
   - If no → keep in memory
4. **If no matching memory file exists:**
   - Create a new flat file in `.claude/review-memory/` with this format:

```markdown
# <Pattern Name>

**Do:** <what to do instead>
**Don't:** <the anti-pattern>

## Occurrences
- PR #<number>: <file>:<line> (<date>)
```

Create the `.claude/review-memory/` directory if it does not exist (`mkdir -p`).
