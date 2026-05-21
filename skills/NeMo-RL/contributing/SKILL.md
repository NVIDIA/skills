---
name: contributing
description: Contribution conventions for NeMo-RL. Covers PR title format, commit sign-off, and CI triggering.
when_to_use: Opening a PR; writing a commit message; triggering CI; 'PR title format', 'sign-off', 'conventional commits', 'how do I trigger CI', during code review of PR process.
---

# Contributing Conventions

## PR Title Format

PR titles **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) spec. This is enforced by the `semantic-pull-request` CI check.

```
<type>[optional scope]: <description>
```

Allowed types:

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `ci` | CI/CD changes |
| `docs` | Documentation only |
| `refactor` | Code restructuring without behaviour change |
| `test` | Adding or fixing tests |
| `chore` | Maintenance (deps, configs, tooling) |
| `perf` | Performance improvement |
| `build` | Build system changes |
| `revert` | Reverts a previous commit |

**Do:**
```
ci: retry apt-get installs to handle mirror sync failures
feat(grpo): add dataclass config defaults infrastructure
fix: preserve RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
```

**Don't:**
```
[ci] fix: retry apt-get installs   ← area tags are not part of this convention
Update stuff
Fix bug
```

## Commit Sign-off

All commits must be signed off with `-s`:

```bash
git commit -s -m "fix: correct reward normalization"
```

## CI Triggering

After pushing, trigger CI with:

```
/ok to test <full-commit-sha>
```

Use `git rev-parse HEAD` (not the short form) to get the full SHA.
