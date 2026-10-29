# Get Spark Support

Help users find relevant DGX Spark forum discussions and support channels.
When needed, prepare the smallest useful redacted support package. Stop before
every external submission.

Read [evidence-handling.md](references/evidence-handling.md) before
collecting or preparing any evidence.

## Set the authority boundary

Use current NVIDIA product documentation and support articles as authoritative.
The [DGX Spark forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719)
contains both NVIDIA staff guidance and community suggestions. Label forum
material by author/source and verify proposed fixes against current NVIDIA
documentation; never present an unverified community workaround as NVIDIA
guidance.

Identify NVIDIA employees by the forum's Employee affiliation, including
moderators who carry it. If extracted text omits badges, check the topic JSON
at `/t/<topic-id>.json` for the author's `primary_group_name` or `flair_name`
equal to `Employee`. The `staff` and `moderator` flags describe forum privileges,
not employment; `false` does not exclude an NVIDIA employee. If affiliation
cannot be verified, label it affiliation unknown rather than guessing.

## Search existing forum discussions

For an unresolved product question or suspected known issue, search existing
threads before drafting a new post. Build a compact query from the exact error
when available, the affected component and symptom, and the system model or
relevant OS, kernel, driver, or firmware version.

Scope the search to the DGX Spark / GB10 category with this URL pattern:

```text
https://forums.developer.nvidia.com/search/query?term=<URL-ENCODED-TERMS>%20%23accelerated-computing%3Adgx-spark-gb10
```

For example, search for `cx7 problem` with:

```text
https://forums.developer.nvidia.com/search/query?term=cx7%20problem%20%23accelerated-computing%3Adgx-spark-gb10
```

Open the most relevant threads rather than relying on search snippets. Read the
full discussion, including later corrections, and check the system model,
software or firmware versions, dates, and author/source before applying it.
Prefer exact-error and currently applicable matches over view or reply counts;
forum activity is not incidence data. Return direct links to every thread used.
If no relevant match is found, say so and continue with current NVIDIA
documentation or the appropriate support route. If the search fails, explain
the limitation rather than claiming no relevant discussions exist.

Do not upload files, post to a forum,
open or update a support case, send messages, or build a case/upload database or
service. Offer to help draft a forum post or reply when useful; draft only when
the user requests it or accepts the offer, and do not publish it.

For forum-search requests, return relevant thread links and summarize their
applicability. Continue to evidence preparation only when requested.

## Obtain scoped permission

Explain what each category may contain, then obtain explicit user permission
before any of these actions:

1. inspect or collect system and application logs;
2. list, read, copy, or export crash dumps;
3. copy or summarize conversation history;
4. create local staging files or run a redaction helper.

Permission must name the categories and reasonable time/component scope. Consent
to troubleshoot, run diagnostics, or collect one log is not blanket consent.
Before creating staging files, name the computer and private staging path in
the permission proposal.
If the user declines a category, omit it and record the omission rather than
substituting broader evidence.

## Prepare minimal evidence

After permission:

1. Create the private staging directory on the computer approved in the
   evidence proposal, using restrictive permissions. Never write evidence to a
   shared or repository directory.
2. Record a concise user-provided summary: symptom, expected versus actual
   behavior, impact, onset/time zone, reproduction steps, and changes already
   attempted. Separate observations from hypotheses.
3. Collect only the system state, narrow log window, diagnostic run, or selected
   application data relevant to the issue. Prefer exact commands and timestamps
   over an unbounded "collect everything" request.
4. For a GPU/driver issue, offer the official `nvidia-bug-report.sh` collection
   only after explaining its breadth and receiving specific permission. For a
   Field Diagnostics issue, use only the selected matching run directory from
   the `run-spark-diagnostics` workflow.
5. For a crash, inspect only approved metadata first. Stage a selected binary
   core only when NVIDIA Support specifically needs it and the user accepts that
   process memory can contain secrets and cannot be reliably text-redacted.
6. Include conversation context only from the approved range. Exclude system
   prompts, hidden/tool metadata, internal reasoning, unrelated messages, and
   other people's personal information.

## Redact and review

Keep originals private and create separate redacted working copies. For textual
evidence, use
[`redact-support-text.py`](scripts/redact-support-text.py) after the user
approves local processing. It masks common credentials without printing matched
values and refuses binary input or overwriting an output file. It is a first
pass, not a guarantee.

Then manually review every prepared item for credentials, cookies, private keys,
tokens, URLs with credentials/query secrets, email addresses, usernames, host
names, IP/MAC addresses, internal URLs, customer or project names, file paths,
and unrelated personal data. Retain an identifier only when it is necessary for
diagnosis and the user approves sharing it.

Compressed text must be expanded into the private staging area before textual
redaction. Do not run the text redactor on binary crash dumps, database files,
or opaque archives. Exclude an artifact that cannot be safely reviewed, or list
it separately with its risk and require an explicit user decision.

Prepare a manifest with filename, purpose, source time range, size, checksum,
redaction status, and exclusions. Show the user the draft and manifest and
require their review of **all** prepared material. If they have not reviewed and
approved it, mark it `NOT APPROVED FOR SHARING`.

## Choose an NVIDIA channel

Use the current [DGX Spark support page](https://docs.nvidia.com/dgx/dgx-spark/support.html)
to choose among:

- **DGX Spark hardware support ticket/chat:** suspected hardware failure, a
  Field Diagnostics `FAIL`/`RETEST`, or an RMA pre-check;
- **NVIDIA AI Enterprise—DGX Spark support:** software-stack cases only when the
  user's entitlement certificate explicitly includes that product;
- **DGX Spark forum:** product questions, reproducible issues, and community or
  NVIDIA-engineer discussion that do not require private evidence;
- **NVIDIA PSIRT:** suspected security vulnerabilities; do not disclose a
  vulnerability or secrets in a public forum.

For an OEM GB10 system, explain that software guidance may overlap but do not
invent or direct the user to an OEM-specific recovery or support procedure. If
the appropriate NVIDIA route cannot be established, say so and stop at the
prepared draft.

## Draft without submitting

Use this structure as applicable for a forum or case draft; keep replies focused
on the existing discussion:

```text
Title:
System and software state:
Problem and impact:
Expected behavior:
Actual behavior:
Reproduction steps and timing:
Troubleshooting already attempted:
Relevant diagnostic result or exact error codes:
Prepared attachments (redacted and user-reviewed):
Questions for NVIDIA:
```

Mark each claim as observed, NVIDIA-documented, community-suggested, or an
inference when the distinction matters. Provide official source links. If
evidence was prepared, report the staging computer and path, included and excluded
evidence, redaction limitations, selected channel, and user-review status.
Do not include an upload or submission command.
