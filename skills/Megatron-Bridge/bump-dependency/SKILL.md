---
name: bump-dependency
description: Bump a pinned dependency (TransformerEngine, Megatron-LM, NRX, etc.), regenerate the lockfile, open a PR, and drive it to green by attaching a watchdog to the "CICD NeMo" workflow and quarantining failing functional tests as flaky until the run is green.
when_to_use: Bumping a dependency pin in `pyproject.toml` or `uv.lock` and shepherding the PR to green. 'bump TE', 'bump transformer-engine', 'update TE pin', 'bump submodule', 'update lock file', 'bump dependency PR', 'watch CI for a bump', 'quarantine flaky tests after bump', 'run all tests for this bump'.
---

# Bump Dependency

End-to-end workflow for shipping a dependency bump in Megatron Bridge.
Optimised for the case where TE, MCore, or another GPU-heavy pin moves
forward — which often surfaces flakes that have to be quarantined before
the PR can land.

The pipeline is always: **edit → relock → push → /ok to test → watchdog →
quarantine on red → re-trigger → repeat until green**.

## When to reach for this skill

- Bumping a git-source pin in `pyproject.toml` `override-dependencies`
  (e.g. `transformer-engine @ git+...@<ref>`).
- Bumping the `3rdparty/Megatron-LM` submodule.
- Any change that touches `uv.lock` and needs the full L0 + L1 matrix to
  prove out before merge.

For pure dep additions/removals without a CI loop, the
`build-and-dependency` skill is enough.

## Required context

Read first, then follow the steps below:

- @CONTRIBUTING.md — PR title/label policy, DCO sign-off
- @skills/build-and-dependency/SKILL.md — `uv lock` mechanics, container choice
- @skills/cicd/SKILL.md — how `copy-pr-bot` and `/ok to test` work
- @skills/testing/SKILL.md — `active/` vs `flaky/` directory layout, `git mv` quarantine recipe

## Step 1 — Worktree and edit

Create a worktree off `main` per @CLAUDE.md. Then, **before any `uv lock`**:

```bash
git submodule update --init 3rdparty/Megatron-LM
```

The submodule must be initialised in the worktree or `uv lock` errors
with "not a Python project" on the MCore path.

Edit the pin. For TE the canonical knob is the override line in
`pyproject.toml`:

```toml
override-dependencies = [
    ...
    "transformer-engine @ git+https://github.com/NVIDIA/TransformerEngine.git@<new-ref>",
    ...
]
```

Use a **branch name** (`release_v2.15`) only when you want to track a
moving tip; use a full SHA for reproducibility. TE branches use
`release_vX.Y` (underscore), not `release/vX.Y`. Verify with
`git ls-remote https://github.com/NVIDIA/TransformerEngine.git`.

## Step 2 — Regenerate the lockfile

Run `uv lock` inside the project container per
@skills/build-and-dependency/SKILL.md "Regenerating uv.lock". Then
confirm only the intended packages moved:

```bash
git diff --stat pyproject.toml uv.lock
```

If the diff carries changes you didn't ask for (transitive movements you
can't explain), stop and investigate before pushing. Note that
`override-dependencies` carries CVE floors that float — unrelated
packages bumping by a patch version is expected; accept those, don't
revert them.

## Step 3 — Commit and push

Sign-off + signed-commit + PR title format per @CONTRIBUTING.md and
@skills/cicd/SKILL.md "Commit and PR Workflow". For a bump:

```bash
git add pyproject.toml uv.lock
git commit -S -s -m "[build] chore: bump <package> to <ref>"
git push -u origin <branch-name>
```

A signed commit (`-S`) lets `copy-pr-bot` trigger CI without manual
`/ok to test` for the first push — but you'll still post `/ok to test`
on every subsequent SHA in this loop (Step 5).

## Step 4 — Open the PR

Title and labels per @CONTRIBUTING.md. Two bump-specific requirements:

- Apply `needs-more-tests` — **mandatory** for a bump; expands the matrix
  from L0 to L0+L1.
- For a high-blast-radius bump (TE, MCore submodule, anything that
  touches CUDA kernels), also apply `full-test-suite` to pull L2 into
  the PR run. L2 covers VL models, checkpoint conversion, and heavy
  quantization which otherwise only run on schedule.

The PR body template — this is the durable record of the bump:

```markdown
<details><summary>Claude summary</summary>

## What
- Bump <package> to <ref>.
- Regenerate `uv.lock`.

## Lockfile delta
```
Updated <package> <old> -> <new>
```

## Test plan
- [ ] L0 CI green
- [ ] L1 CI green (label `needs-more-tests` applied)

## Quarantined tests (this bump)
_None yet — will be appended as flakes are identified during CI iteration._

</details>
```

To update the PR title or body later, use `gh api -X PATCH
"repos/NVIDIA-NeMo/Megatron-Bridge/pulls/<N>" -F "body=@/tmp/pr-body.md"`
— never `gh pr edit`.

## Step 5 — Trigger CI on the exact SHA

Trigger mechanics live in @skills/cicd/SKILL.md "How CI Is Triggered".
For this loop the rule is simple: **on every new SHA you push, post
`/ok to test $(git rev-parse HEAD)`** as a PR comment, even if your
commits are signed. This guarantees the run targets the SHA you actually
want exercised and re-fires anything that got cancelled or cached.

## Step 6 — Attach the watchdog (always; never a cronjob)

For a bump PR you want a single live process that emits per-job state
changes for the **CICD NeMo** workflow only. Other workflows (docs,
wheel, copyright, install-test) are noise here — the gate that decides
green-or-red for a bump is `CICD NeMo`.

**Always attach a watchdog with the Monitor tool. Never schedule wakeups
or cronjobs for this loop.** A watchdog gives you:

- Sub-minute reaction time on every job transition.
- A single live process — no scattered scheduled-wakeup state to reason
  about.
- Natural early termination via `TaskStop` once the run is green.

### Watchdog script

Save to `/tmp/watchdog-<PR>.sh` and chmod +x:

```bash
#!/usr/bin/env bash
# Watchdog: monitor "CICD NeMo" runs on pull-request/<PR> and emit
# per-job state changes. Stays alive across re-runs (new commits).
set -u
PR=<PR>
REPO=NVIDIA-NeMo/Megatron-Bridge
BRANCH="pull-request/$PR"

prev_run_id=""
declare -A prev_state

emit() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

while true; do
  run_json=$(gh run list --repo "$REPO" --workflow "CICD NeMo" \
    --branch "$BRANCH" --limit 1 \
    --json databaseId,status,conclusion,headSha 2>/dev/null || echo "[]")
  run_id=$(echo "$run_json" | jq -r '.[0].databaseId // empty')
  run_status=$(echo "$run_json" | jq -r '.[0].status // empty')
  run_conclusion=$(echo "$run_json" | jq -r '.[0].conclusion // empty')
  run_sha=$(echo "$run_json" | jq -r '.[0].headSha // empty')

  if [[ -z "$run_id" ]]; then
    sleep 30; continue
  fi

  if [[ "$run_id" != "$prev_run_id" ]]; then
    emit "RUN ${run_id} STARTED sha=${run_sha:0:8} status=${run_status}"
    prev_run_id="$run_id"
    unset prev_state
    declare -A prev_state
  fi

  jobs_json=$(gh run view "$run_id" --repo "$REPO" --json jobs 2>/dev/null || echo "{}")
  while IFS=$'\t' read -r name status conclusion; do
    [[ -z "$name" ]] && continue
    cur="${status}/${conclusion}"
    if [[ "${prev_state[$name]:-}" != "$cur" ]]; then
      case "$status" in
        completed)
          emit "JOB ${name} -> ${conclusion}" ;;
        in_progress)
          if [[ -z "${prev_state[$name]:-}" || "${prev_state[$name]}" == "queued/" ]]; then
            emit "JOB ${name} -> in_progress"
          fi ;;
      esac
      prev_state[$name]="$cur"
    fi
  done < <(echo "$jobs_json" | jq -r '.jobs[]? | [.name, .status, (.conclusion // "")] | @tsv')

  if [[ "$run_status" == "completed" ]]; then
    emit "RUN ${run_id} COMPLETED conclusion=${run_conclusion}"
  fi

  sleep 60
done
```

### Arming the watchdog

```text
Monitor(
  description="CICD NeMo run state changes on PR <N>",
  command="bash /tmp/watchdog-<N>.sh",
  persistent=true,
  timeout_ms=3600000
)
```

`persistent: true` keeps it alive across re-runs (you'll push more
commits when quarantining flakes). Stop it with `TaskStop(<task-id>)`
once the run is green.

### Why never a cronjob / scheduled wakeup

- Cronjobs run blind — they fire on a clock, not on an event. You'll
  either over-poll (cache miss every wake-up) or miss long stalls.
- Wakeups can't easily fan out to "tell me whenever a job transitions"
  — they only resume the agent on a fixed interval.
- A persistent Monitor surfaces every job edge in real time and exits
  cleanly when the work is done.

## Step 7 — Quarantine on red, then iterate

When a `JOB <name> -> failure` event fires:

1. **Triage the failure — is it the bump or a flake?** Skim the logs:

   ```bash
   RUN_ID=<from "RUN ... STARTED" event>
   gh run view "$RUN_ID" --repo NVIDIA-NeMo/Megatron-Bridge --log-failed > /tmp/run.log
   wc -l /tmp/run.log
   tail -200 /tmp/run.log
   ```

   This is the bump-specific judgement call: only quarantine if the
   failure reproduces on `main` or is clearly unrelated infrastructure.
   If the failure is caused by the bump (real regression), **stop
   quarantining** — fix the underlying issue or revert the bump.
   Quarantining a real regression hides the very signal the bump PR
   exists to surface.

2. **Move the launch script to `flaky/`** per @skills/testing/SKILL.md
   "Moving a Test to Flaky". Map a CI job name to its launch script via:

   - prefix `gb200_` → `gb200/active/`, otherwise `h100/active/`
   - the rest is the script's basename without `.sh`

3. **Append to the PR body's Quarantined tests section** with a one-line
   reason and a follow-up tracking link if you have one. This is the
   durable record of what this bump deferred — the section exists
   precisely so a reviewer can see at a glance which flakes were
   side-stepped to land the bump.

4. **Commit, push, retrigger**:

   ```bash
   git commit -S -s -m "[ci] chore: quarantine flaky <test> for <package> bump"
   git push
   gh pr comment <N> --repo NVIDIA-NeMo/Megatron-Bridge \
     --body "/ok to test $(git rev-parse HEAD)"
   ```

5. **Update the PR body** via `gh api PATCH` so the quarantine list
   stays current.

The watchdog is persistent — it picks up the new run automatically and
emits `RUN <id> STARTED` for the new attempt. Loop back to step 1.

## Step 8 — Stop when green

`RUN <id> COMPLETED conclusion=success` is the exit condition. Then:

```bash
gh pr checks <N> --repo NVIDIA-NeMo/Megatron-Bridge | awk '{print $2}' | sort | uniq -c
TaskStop(<watchdog-task-id>)
gh api -X PATCH "repos/NVIDIA-NeMo/Megatron-Bridge/pulls/<N>" -F "body=@/tmp/pr-body.md"
```

## Common pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Wrong TE branch ref (`release/v2.15`) silently resolves nothing | TE uses `release_vX.Y` with an underscore | Verify with `git ls-remote` before locking |
| Lockfile diff includes unrelated CVE-pinned packages | `override-dependencies` carries floors that float | Re-run lock and accept; don't try to revert those |
| Signed first push triggers CI but later pushes don't | `copy-pr-bot` re-trusts on each new SHA only via `/ok to test` once you're past the first signed commit in this loop | Always re-post `/ok to test $(git rev-parse HEAD)` per Step 5 |
| Watchdog goes silent for 30+ min | `gh` rate-limited or auth expired | Bump poll interval; `gh auth status`; restart Monitor |
| Job name doesn't map to a script in `active/` | `gb200_` prefix is the hardware indicator, not part of the filename | Strip `gb200_` and look in `gb200/active/` |

## Anti-patterns

- **Cron / scheduled wakeups for this loop.** Always Monitor.
- **Polling all workflows.** Filter to `CICD NeMo` — the rest are noise
  for a bump.
- **Quarantining a real regression** to "make CI green." That defeats
  the purpose of the bump PR. Only quarantine if the failure reproduces
  on `main` or is clearly unrelated infrastructure.
- **`gh pr edit`** for title/body. Use `gh api PATCH`.
- **HEREDOC in `gh pr create --body`.** Always go through a tmpfile +
  `--body-file`.
- **Bundling unrelated changes** (feature work, refactors) into a bump
  PR. Bumps should stay surgical so CI failures attribute cleanly.
