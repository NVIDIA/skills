<!-- SPDX-License-Identifier: Apache-2.0 -->

# Prepare an NVAT bug report

Prepare a report when the user asks to submit a bug or package a reproducible
failure. Use existing captures first. Rerunning collection can change device
evidence, and rerunning appraisal can retrieve different reference manifests.

## Describe the failure

Include these details:

```text
Summary:
Expected behavior:
Observed behavior:
Reproduction steps:
Command or API call (credentials redacted):
Process exit code and SDK error, if available:
UTC time of the failure:
OS, architecture, SDK/CLI version and driver version:
GPU model and known responder, if relevant:
Execution environment (host, container, or CVM):
Workflow and input format (legacy/TCG or CoRIM/CMW):
Policy and reference-manifest versions or digests, if relevant:
Attached files and any redactions:
```

Record missing information as unavailable. Do not infer successful collection,
appraisal, or policy acceptance from an incomplete report. Preserve the exact
error and command options apart from redacted credentials and sensitive paths.

## Collect environment details

Run read-only checks in the environment where the failure occurred. Useful
commands on a Linux target include:

```sh
uname -s
uname -r
uname -m
if [ -r /etc/os-release ]; then cat /etc/os-release; fi
nvattest version
```

If the CLI is missing, record that fact. For a source build, include the source
revision and relevant build options. Record installed CLI and native-library
package versions using the target's package manager when available.

If GPU access is relevant and the utility is installed, collect the model and
driver version without requesting device UUIDs or serial numbers:

```sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

Record container/CVM context and whether device access differs from the host.
Record any missing utility in the report. Do not install software or change
the driver while collecting a report. Do not dump all environment
variables or copy arbitrary configuration directories.

## Choose and review attachments

Use the command and capture procedure for the failing workflow. See
[diagnostic collection](diagnostics.md) for CoRIM appraisal logs and legacy
input handling. Keep stdout/result JSON and stderr logs separate, and record
the original process status.

Keep original captures local. Prepare reviewed copies in a separate directory:

- `report.txt`: the failure description and reproduction steps;
- `environment.txt`: the relevant environment/version output;
- `failure.log`: the logs that demonstrate the failure.

Include result JSON, evidence, reference manifests, or a policy only when
needed to reproduce or explain the failure. Review their contents and sharing
requirements first. Remove service keys, passwords, private signing keys,
authorization headers, and unnecessary device identifiers or sensitive paths
from shared copies. Check for known credential values as well as field names;
a filename or extension alone does not establish that a file is safe to share.

Label modified artifacts as redacted. A redacted signed token or evidence
file is not a valid input for cryptographic verification; retain the original
locally if further verification is needed.

## Package the reviewed files

After preparing the three reviewed files above, create a new archive in a
fresh temporary directory. This avoids retaining entries from an older ZIP:

```sh
archive_dir=$(mktemp -d)
zip -j "$archive_dir/nvat-bug-report.zip" \
  ./reviewed/report.txt \
  ./reviewed/environment.txt \
  ./reviewed/failure.log
unzip -l "$archive_dir/nvat-bug-report.zip"
```

Before running the command, confirm that `zip` and `unzip` are installed and
review all listed files. Use regular files, not symlinks. Add any other reviewed
files by explicit filename; avoid directory recursion, wildcards, home directories,
and whole repository archives. If these utilities are unavailable, use the
agent's archive tool with the same explicit file list.

Inspect the archive's file list and contents before sharing it. Tell the user
where the archive is stored and what it contains. Leave uploading, issue
creation, and sharing to the user's explicit request.
