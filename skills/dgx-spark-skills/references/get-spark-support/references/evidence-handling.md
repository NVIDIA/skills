# Support Evidence Handling

This reference describes safe consent, redaction, and review boundaries for
preparing DGX Spark support evidence. Consult the current sources before
collection:

- [DGX Spark support options](https://docs.nvidia.com/dgx/dgx-spark/support.html)
- [DGX Spark Field Diagnostic Software](https://docs.nvidia.com/dgx/dgx-spark/support.html#field-diagnostic-software)
- [NVIDIA guidance for `nvidia-bug-report.sh`](https://docs.nvidia.com/deploy/xid-errors/working-with-xid-errors.html)
- [DGX Spark forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)

## Consent proposal

Before collection, present a concrete proposal such as:

```text
Proposed support evidence
- system state: release/OS/kernel/GPU versions
- logs: service X from START through END
- diagnostic logs: one timestamped run
- crash data: metadata only, or one specifically named core
- conversation: messages START through END
- staging computer and path: COMPUTER, PRIVATE_PATH; private directory and redacted copies

No upload or submission will occur. Approve, narrow, or decline each category.
```

Do not hide privileged commands or broad collection behind a generic approval.

## Private staging

Choose and disclose the staging computer and private path in the
evidence-preparation proposal, honoring any user preference. The computer may be
the Spark or the user's laptop that is connecting over SSH; do not infer it from where 
the chat is displayed. Create the directory only after approval, with owner-only
permissions. Avoid repositories, shared folders, cloud-synced folders, and
world-readable temporary locations. Do not expose collected content in terminal
output, command arguments, or chat.

Record source and destination paths without embedding credentials. Refuse to
overwrite existing files. Keep unredacted originals separate from redacted
copies and delete neither without explicit approval.

## Evidence selection

Select the minimum evidence that tests the issue hypothesis:

| Issue | Prefer | Avoid by default |
| --- | --- | --- |
| GPU/driver failure | exact error, narrow journal window, approved `nvidia-bug-report.sh`, relevant diagnostic summary | whole-home scans or unrelated application logs |
| Service/application failure | service status, bounded service journal, application log around reproduction | complete journal history |
| Crash | timestamp/executable metadata, selected service log, specifically requested core | all coredumps or arbitrary process memory |
| Field Diagnostics failure | one run's `summary.json`, `output.log`, `run.log`, relevant per-test log | every historical diagnostic run |
| Reproduction/question | minimal system versions, steps, expected/actual result, approved conversation excerpt | hidden prompts, tool metadata, unrelated history |

These examples are not pre-authorized commands. After permission, adapt the time
window and unit to the issue:

```bash
journalctl --since 'YYYY-MM-DD HH:MM:SS' --until 'YYYY-MM-DD HH:MM:SS' --unit UNIT
coredumpctl list --since 'YYYY-MM-DD HH:MM:SS'
coredumpctl info PID
```

Exporting journal text or a selected core creates a file and needs approval.
`nvidia-bug-report.sh` collects kernel and driver information into
`nvidia-bug-report.log.gz`; NVIDIA says to run it as root and notes that it can
rarely take up to an hour. Explain this before requesting permission. Never run
it merely because it is available.

## Text redaction helper

Before invoking the helper with `--output`, tell the user where temporary output
will be created and how to handle an interrupted session. Include this notice
with the existing consent for local writes; no additional approval is needed
when those writes are already authorized. Replace the placeholder with the
actual absolute output directory:

> This will create a private temporary directory and file in `<output directory>`. Treat it as
> sensitive. Normal completion and handled errors remove it, but a forced
> shutdown may leave a `.redaction-*` directory there. If this session stops
> unexpectedly, check that directory and remove any leftover temporary directory
> from this run once you've confirmed it is no longer in use.

This notice is not needed for `--check`, which creates no output file.

The text redactor can run on Linux, macOS, or Windows on the approved staging
computer. First check that Python 3.10 or newer is available there (`python3
--version`, or `python --version` on Windows). If it is missing, ask before
installing it or agree on another staging computer; do not silently change hosts.
The system-identification helper still runs on the Linux Spark/GB10 target.

Run the helper from the skill directory with Python bytecode disabled (use
`python` in place of `python3` on Windows):

```bash
python3 -B references/get-spark-support/scripts/redact-support-text.py INPUT --output OUTPUT.redacted
python3 -B references/get-spark-support/scripts/redact-support-text.py --check OUTPUT.redacted
```

The helper creates an owner-only output and refuses to overwrite it. On Windows
it applies a protected owner-only ACL before writing; if that fails, it stops.
Use a local filesystem supporting Windows ACLs and hard links (such as NTFS). It redacts
common authorization/cookie headers, sensitive key-value assignments and CLI
flags, secret-bearing URL parameters, private-key blocks, JWT-like values, and
well-known credential prefixes. It prints only replacement counts.

The helper streams regular UTF-8 text files with limits of 64 MiB per file,
64 KiB per line (including its newline), 1 MiB per private-key block, and 30
seconds of worker execution per invocation (plus process setup and cleanup). It retains complete lines and private-key boundaries
across reads; no agent-managed chunking is needed. Unknown, nested, mismatched,
or unfinished private-key markers cause failure. Output is published only after
successful processing, with temporary output removed on handled failure or
interruption. A supervising process terminates a worker that exceeds the timeout
before publishing its output. Forced termination (for example, SIGKILL) can leave an owner-only
`.redaction-*` temporary directory; treat it as sensitive and follow the deletion
consent rules above.

On a limit or malformed-input error (exit 2), do not treat the input as redacted.
Select a smaller relevant excerpt preserving complete records and private-key
blocks, then rerun the helper. Do not split a key block to bypass a limit.
Ordinary credential matching is line-based; wrapped secrets without supported
private-key markers still require manual review.

Automated matching is incomplete and may also mask useful values. Manually
review the redacted copy and the proposed attachment list. Never paste a
suspected secret into chat to ask whether it is secret.

## Binary and compressed artifacts

- Expand compressed text only inside the approved private staging directory;
  redact the expanded copy and do not modify the original archive.
- Treat a core dump as process memory that may contain credentials, user data,
  prompts, model inputs, and proprietary content. Text redaction does not make a
  binary core safe.
- Exclude binaries by default. Include one only when the receiving NVIDIA
  channel needs it and the user explicitly approves that named file after seeing
  its metadata and risk.
- Do not create a public link, upload service, case database, or automatic
  submission workflow.

## Required user review

Give the user a manifest and the draft without uploading either. For every item,
record:

- filename and purpose;
- collection source and time window;
- byte size and SHA-256;
- whether it is text-redacted, manually reviewed, opaque/binary, or excluded;
- known redactions and remaining disclosure risks.

Require the user to review every prepared item. A manifest entry is not proof
that opaque contents are safe; require a separate inclusion decision or exclude
the artifact. Keep the package marked `NOT APPROVED FOR SHARING` until the user
approves the final material.

## Channel and source labels

- Content in current NVIDIA documentation or an NVIDIA support article:
  **NVIDIA guidance**.
- A forum reply by an NVIDIA employee: **NVIDIA staff forum guidance**, with the
  exact link and date; cross-check against current docs.
- A community forum reply: **community suggestion**, never an official fix;
  if affiliation cannot be verified, label it **affiliation unknown**.
- Agent synthesis: **inference**, with supporting evidence.

Use the support page's live links for ticket/chat/entitlement routing. Do not
copy login URLs into automation or assume an entitlement. Do not expose security
vulnerabilities in the public forum; route them to NVIDIA PSIRT from the support
page.
