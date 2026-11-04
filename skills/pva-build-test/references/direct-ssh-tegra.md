<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Direct-SSH Tegra procedure

This procedure is a skill workflow.
It is not an NVIDIA SDK procedure.

Use it only for a Tegra that the current host can reach directly.

## Contents

- [Inputs](#inputs)
- [Probe](#probe)
- [Transfer](#transfer)
- [Run](#run)
- [Prohibited actions](#prohibited-actions)

## Inputs

Set these values from user-supplied inputs:

```bash
PROJECT_ROOT=<absolute-project-root>
RESULT_DIR=<absolute-result-directory>
DUT_HOST=<tegra-address>
DUT_USER=<tegra-user>
DUT_PORT=<ssh-port>
DUT_KNOWN_HOSTS=<absolute-known-hosts-file>
REMOTE_DIR=<unique-absolute-target-directory>
DEPLOY_ARCHIVE=<absolute-local-archive>
BUILD_SDK_MAJOR_MINOR=<major.minor>
BUILD_SAFETY_VARIANT=<safety-or-non-safety>
```

Validate each path and value before use.
Do not put a password in a command, environment variable, log, or report.

Apply these minimum checks:

```bash
[[ "$PROJECT_ROOT" == /* ]]
[[ "$RESULT_DIR" == /* ]]
[[ "$DUT_KNOWN_HOSTS" == /* ]]
[[ "$DEPLOY_ARCHIVE" == /* && -f "$DEPLOY_ARCHIVE" ]]
[[ "$DUT_PORT" =~ ^[0-9]+$ ]]
(( DUT_PORT >= 1 && DUT_PORT <= 65535 ))
[[ "$DUT_USER" =~ ^[A-Za-z_][A-Za-z0-9_-]*$ ]]
[[ "$DUT_HOST" =~ ^([A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?|[0-9A-Fa-f:]+)$ ]]
[[ "$REMOTE_DIR" =~ ^/tmp/[A-Za-z0-9._/-]+$ ]]
[[ "$REMOTE_DIR" != *..* ]]
[[ "$BUILD_SDK_MAJOR_MINOR" =~ ^[0-9]+\.[0-9]+$ ]]
[[ "$BUILD_SAFETY_VARIANT" =~ ^(safety|non-safety)$ ]]

if [[ "$DUT_HOST" == *:* ]]; then
  DUT_SCP_HOST="[$DUT_HOST]"
else
  DUT_SCP_HOST="$DUT_HOST"
fi
```

Validate each test argument against its expected type.
Do not permit shell operators, command substitutions, or redirections in a test argument.

## Probe

Use key or SSH-agent authentication.
Use a persistent known-hosts file.
Disable proxy commands and jump hosts for this connection.

Inspect the effective SSH configuration before the probe.
Require both `proxycommand none` and `proxyjump none`.

```bash
mkdir -p "$(dirname "$DUT_KNOWN_HOSTS")"

ssh \
  -p "$DUT_PORT" \
  -o BatchMode=yes \
  -o ProxyCommand=none \
  -o ProxyJump=none \
  -o ConnectTimeout=10 \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="$DUT_KNOWN_HOSTS" \
  "$DUT_USER@$DUT_HOST" \
  'uname -m; uname -r; cat /etc/nv_tegra_release 2>/dev/null || true'
```

Require an Arm64 architecture and a compatible BSP.

Check the cuPVA runtime without changing the target:

```bash
ssh \
  -p "$DUT_PORT" \
  -o BatchMode=yes \
  -o ProxyCommand=none \
  -o ProxyJump=none \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="$DUT_KNOWN_HOSTS" \
  "$DUT_USER@$DUT_HOST" \
  'dpkg-query -W -f="\${Package} \${Version}\n" "pva-sdk-*-l4t" &&
   /sbin/ldconfig -p | grep -E "libcupva_host|libcupva_host_utils"'
```

Compare the installed package major and minor version with `BUILD_SDK_MAJOR_MINOR`.
Confirm that the installed package variant matches `BUILD_SAFETY_VARIANT`.
Stop if the version, variant, or required libraries are absent or incompatible.

Use the supplied account for transfers and tests without elevation where possible.
Require administrative permissions only for requested commands that need them,
such as clock configuration, allowlist updates, or privileged deployment. Accept
an already prepared target or a user-approved noninteractive method scoped to
those commands. Do not require unrestricted passwordless sudo.

Check permissions for the actual executable and arguments. `sudo -n true` does
not establish whether a different, command-scoped operation is allowed. Do not
run a state-changing command merely to probe access. If a required operation
cannot run with the approved permissions, return `BLOCKED` and name that operation.
Do not request a password or change sudoers, ownership, or device permissions.

## Transfer

Create a unique directory.

```bash
ssh \
  -p "$DUT_PORT" \
  -o BatchMode=yes \
  -o ProxyCommand=none \
  -o ProxyJump=none \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="$DUT_KNOWN_HOSTS" \
  "$DUT_USER@$DUT_HOST" \
  "mkdir -p '$REMOTE_DIR'"

scp \
  -P "$DUT_PORT" \
  -o BatchMode=yes \
  -o ProxyCommand=none \
  -o ProxyJump=none \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="$DUT_KNOWN_HOSTS" \
  "$DEPLOY_ARCHIVE" \
  "$DUT_USER@$DUT_SCP_HOST:$REMOTE_DIR/input.tar.gz"
```

Extract the archive and check dynamic dependencies:

```bash
ssh \
  -p "$DUT_PORT" \
  -o BatchMode=yes \
  -o ProxyCommand=none \
  -o ProxyJump=none \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="$DUT_KNOWN_HOSTS" \
  "$DUT_USER@$DUT_HOST" \
  "cd '$REMOTE_DIR' && tar -xzf input.tar.gz && ldd <relative-target-executable>"
```

Stop if `ldd` reports a missing required library.

## Run

Run the supplied test command in the remote directory.

```bash
ssh \
  -p "$DUT_PORT" \
  -o BatchMode=yes \
  -o ProxyCommand=none \
  -o ProxyJump=none \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="$DUT_KNOWN_HOSTS" \
  "$DUT_USER@$DUT_HOST" \
  "cd '$REMOTE_DIR' && <validated-target-test-command>"
```

Do not interpolate an unvalidated operator name, test filter, or file path into the remote command.

Copy result files back with `scp` and `DUT_SCP_HOST`.
Preserve the remote directory unless the user asks you to remove it.

## Prohibited actions

- Do not guess credentials.
- Do not disable host-key checking.
- Do not use an SSH proxy, relay, or jump host.
- Do not place secrets in logs.
- Do not change target packages without user approval.
- Do not run a performance test before correctness passes.
- Do not report host wall time as PVA execution time.
