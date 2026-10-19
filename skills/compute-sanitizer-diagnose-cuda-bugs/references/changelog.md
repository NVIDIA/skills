# Compute Sanitizer — Version & Changelog Triage

> Reference sub-skill of the `compute-sanitizer` skill. Read this when a result
> looks like a **false positive/negative**, a **flag/feature is missing**, or a
> user asks "why does this happen / how do I fix this" and the answer may be
> **"upgrade your CUDA Toolkit."**
> Detects nothing itself — it maps a symptom to a known upstream fix/feature.

## Why this matters

Compute Sanitizer ships **inside the CUDA Toolkit** and is improved on a fast,
per-release cadence. A large fraction of "bugs" users report are already fixed
upstream, and features users think are missing were added in a specific version.
Recommending an upgrade (with the exact release-notes line) is often the correct,
fastest fix — and it is something a remote agent can determine without the user's
GPU.

## Step 1 — Identify what the user is running

```bash
compute-sanitizer --version     # tool version, e.g. "2026.2.1" (calendar: YYYY.minor[.patch])
nvcc --version                  # toolkit / compiler version (should match the sanitizer)
nvidia-smi                      # driver version — some fixes require a minimum driver
```

Notes:
- The Compute Sanitizer version tracks the **CUDA Toolkit** it shipped with, *not*
  the CUDA runtime/driver version. Quote the sanitizer version, not `nvidia-smi`'s
  "CUDA Version".
- A few fixes also require a **minimum driver** (the release notes call this out —
  e.g. coredump generation for synccheck/initcheck on shared memory needs driver
  ≥ 595).

## Step 2 — Check the changelog

- **Compute Sanitizer Release Notes** (tool-specific, authoritative):
  <https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html>
  Organized as `Updates in <version>` with `Added …` / `Fixed …` lines.
- **CUDA Toolkit Release Notes** (broader toolkit/compiler context):
  <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>

Search the page for the user's symptom keywords (the tool name plus the PTX
instruction, API, or error text — e.g. `cp.async.bulk`, `wgmma`, `racecheck
cluster`, `stream-ordered`, `initcheck shared`).

## Step 3 — Decide and advise

- **Symptom matches a `Fixed …` entry in a version newer than the user's** →
  advise: *"This is a known false positive/bug fixed in `<version>`; upgrade the
  CUDA Toolkit to `<version>` or later."* Quote the release-notes line.
- **A flag/feature the user wants matches an `Added …` entry newer than theirs** →
  advise: *"`<feature>` was added in `<version>`; it isn't available on your
  `<older version>` — upgrade to use it."*
- **Nothing matches** → treat it as a genuine issue and continue with the normal
  tool workflow (it is not a version artifact).

Always cite the specific release-notes entry so the recommendation is verifiable —
do not guess a version.

## Worked examples (from the release notes)

Use these as patterns; confirm the current wording/version on the release-notes
page since it updates over time.

| User symptom | Changelog entry | Advice |
|---|---|---|
| "I want compile-time instrumentation (`-fdevice-sanitize=memcheck`) but nvcc rejects it" | **Added** in **2025.4** | Upgrade CTK to 2025.4+ |
| "initcheck won't check `__shared__` memory / `--initcheck-address-space` unknown" | shared-memory initcheck + `--initcheck-address-space` **Added** in **2025.4** | Upgrade to 2025.4+ |
| "memcheck flags a `cp.async.bulk` OOB that isn't real" | false positive with `cp.async.bulk` `ignore_oob` **Fixed** in **2026.2.1** (earlier `cp_mask` FP fixed in 2026.2) | Upgrade to 2026.2.1+ |
| "racecheck reports a race across a cluster barrier that looks wrong" | cluster/indirect-barrier racecheck false positives **Fixed** in **2026.1** | Upgrade to 2026.1+ |
| "New CUDA API error reports appeared after upgrading" | extended CUDA API error reporting **Added (on by default)** in **2026.1** | Expected; silence with `--report-api-errors no` or `explicit` |
| "racecheck on Windows hangs at program exit" | rare Windows exit hang **Fixed** in **2026.2.1** | Upgrade to 2026.2.1+ |

## Caveats

- The changelog is the source of truth — **quote it**, and prefer the exact
  version string over "a newer version."
- An upgrade recommendation is about the **toolkit/sanitizer**, which the user can
  install independently of the driver in most cases; only mention a driver bump
  when a release-notes entry requires it.
- Upgrading can also *introduce* newly-surfaced reports (e.g. extended API error
  reporting in 2026.1) — mention this so the user isn't surprised.

## References

- [Compute Sanitizer Release Notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
- [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [Compute Sanitizer docs](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
