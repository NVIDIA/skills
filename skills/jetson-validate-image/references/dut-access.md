# DUT access contract

Detailed reference for `../SKILL.md` — the full `dut_access:` profile
schema, resolution order, connection probe, file-transfer rules,
sudo invocation matrix, UART implementation contract, and security
notes. The top-level skill keeps only the high-level summary; the
deep details live here so the SKILL.md stays under the agent-routing
token budget.

## Profile schema (extends `platform_template.yaml`)

The active target profile gains an optional `dut_access:` block.
The block is **only required for on-target validation**; static
checks ignore it. When the block is absent and the user requests
on-target validation, the skill prompts interactively for every
required field and offers (once per profile) to persist the
answers — never silently invents values.

```yaml
# WHERE — how to reach the DUT after flash. Optional; required only
# for on-target validation.
dut_access:
  transport: "<REQUIRED: ssh | uart>"           # default ssh

  # SSH transport — primary path. Used by jetson-validate-image's
  # on-target leg AND for scp file transfers (test scripts +
  # artifacts).
  ssh:
    host:        "<REQUIRED: IP or DNS resolvable from the host running the skill>"
    user:        "<OPTIONAL: login user — default ubuntu>"
    port:        "<OPTIONAL: TCP port — default 22>"
    auth:        "<REQUIRED: key | password | prompt>"
    key_file:    "<OPTIONAL: absolute path to private key — required when auth=key>"
    password_env: "<OPTIONAL: name of host env var holding the password — required when auth=password; never inline the password itself>"

  # UART transport — full peer to ssh. Supports login, command
  # execution, sudo, and base64-over-tty file transfer. Used when
  # the DUT has no network configured, ssh is broken, or the user
  # wants serial-console access (e.g. for boot-time inspection
  # AND post-boot queries from the same channel). File transfer
  # over UART is byte-banged at ~10 KB/s on 115200 baud — fine
  # for small artifacts (test scripts, log slices), slow for
  # anything >100 KB; the implementation warns and recommends
  # the ssh transport in that case. See the "UART implementation
  # contract" section below for the state-machine semantics.
  uart:
    tty:                "<REQUIRED if transport=uart: e.g. /dev/ttyACM0>"
    baud:               "<OPTIONAL: default 115200>"
    login_user:         "<OPTIONAL: default ubuntu>"
    login_password_env: "<REQUIRED if transport=uart and DUT login is password-protected: env var name; never inline the password itself>"
    shell_prompt:       "<OPTIONAL: regex matched against the DUT shell prompt. Default: implementation sets a uuid-tagged PS1 sentinel at login so the prompt is uniquely matchable. Override only when the DUT's PS1 cannot be modified.>"
    lock_strategy:      "<OPTIONAL: refuse | wait | steal — what to do when another process (minicom, picocom, getty) holds the tty. Default: refuse (safe). `wait` polls until released; `steal` is not implemented (would require sending control characters that may corrupt the holder's state).>"

  # Privilege escalation on the DUT. Many validation queries
  # (dmesg with dmesg_restrict=1, modprobe, systemctl restart)
  # require root. Resolve once; reuse across all on-target
  # commands.
  sudo:
    method:      "<REQUIRED: nopasswd | password | prompt | none>"
    password_env: "<OPTIONAL: env var holding sudo password — required when method=password>"

  # Writable scratch directory on the DUT for pushed test scripts
  # and collected artifacts. Created if missing on first connect.
  workdir:   "<OPTIONAL: absolute path on DUT — default $HOME/jetson-validate>"
```

`auth: prompt` and `sudo.method: prompt` mean "ask the user at
run time and do not store" — the right choice for interactive
shared hosts where password persistence is unacceptable.
`password_env` decouples the value from the YAML so the profile
can be committed to a repo; the skill reads the env var at run
time and refuses if it's unset.

## Resolution order

For each field, in order:

1. **Active profile** `dut_access:` block — read at skill start.
2. **Host environment variable** (`password_env` / `login_password_env`).
3. **Interactive prompt** — used when the profile says `prompt`
   or the field is missing. The prompt offers a one-shot persist
   into `dut_access:` (subject to per-field policy: never persist
   raw passwords, only env-var *names*).

Refusal triggers:

| Condition | Refusal |
|---|---|
| On-target scope requested AND `dut_access.transport` unresolvable after all three steps | Refuse with "no DUT transport configured." Route the user to author `dut_access:` or pick a different validation scope. |
| `transport=ssh` AND `host` empty after all three steps | Refuse. |
| `auth=password` AND `password_env` unset on host | Refuse: "$PASSWORD_ENV is unset." Do not fall back to prompt unless `auth=prompt`. |
| `auth=key` AND `key_file` does not exist on host | Refuse with the resolved path. |
| `sudo.method=password` AND `sudo.password_env` unset | Refuse with the same shape. |
| `transport=uart` AND `uart.tty` empty or device file does not exist | Refuse. |
| `transport=uart` AND `uart.login_password_env` set but the env var is unset on host | Refuse. (Same shape as the SSH `password_env` case.) |
| `transport=uart` AND `pyserial` not importable in the skill's Python | Refuse with install hint (`apt install python3-serial` / `pip install pyserial`). |
| `transport=uart` AND `lock_strategy=refuse` AND another process holds `uart.tty` (via `fuser`) | Refuse with the holding PID(s) so the user can stop the conflicting process (e.g. close minicom). |

## Connection probe (mandatory before any validation query)

Once transport + credentials resolve, run a low-cost probe and
refuse on failure. The probe shape is transport-specific but the
output validation rules are identical.

**SSH probe:**

```bash
sshpass -e ssh \
  -o StrictHostKeyChecking=accept-new \
  -o UserKnownHostsFile="$WORKSPACE/target-platform/<profile-stem>.known_hosts" \
  -o LogLevel=ERROR \
  -o ConnectTimeout=10 \
  "$SSH_USER@$SSH_HOST" \
  'uname -r; cat /etc/nv_tegra_release 2>/dev/null || true'
```

Pass the resolved password via `SSHPASS=…` env (never as
`-p <pw>`, which would leak in process listings). The
known_hosts file is per-profile so different DUTs at different
IPs don't fingerprint-collide.

**UART probe** (via the helper script — see "UART implementation
contract" below):

```bash
"$DUT_UART_PASSWORD_ENV"=... \
scripts/uart_session.py \
  --tty "$UART_TTY" \
  --baud "$UART_BAUD" \
  --user "$UART_LOGIN_USER" \
  --password-env "$DUT_UART_PASSWORD_ENV" \
  probe
```

The `probe` subcommand performs login, runs `uname -r` and
`cat /etc/nv_tegra_release`, and exits — identical observed
output to the SSH probe so the post-probe validation logic is
shared. The script's exit code propagates probe success / failure
(login timeout, prompt-match failure, tty unavailable, etc.).

**Output validation** (same for both transports):

- `uname -r` should match `bsp_image.version`'s kernel release
  (e.g. an R38.x BSP ships a `6.8.12-tegra` kernel). Warn (not
  refuse) on mismatch — the user may have intentionally booted an
  older image for comparison.
- `/etc/nv_tegra_release` should advertise the active BSP version
  and GPU stack (e.g. `INSTALL_TYPE=openrm` for Thor / T264).
  Mismatch surfaces "DUT not booted from the just-flashed BSP."

## File transfer

Per transport:

| Transport | Push | Pull | Throughput | Size policy |
|---|---|---|---|---|
| `ssh` | `scp` over the same resolved credentials | `scp` | wire-rate (typically tens of MB/s on local LAN) | no warning |
| `uart` | `base64`-encoded body, sent over the tty inside a heredoc; DUT runs `base64 -d > $dst` | `base64` on DUT, decode on host | ~10 KB/s effective on 115200 baud (the baud rate × 8/10 framing overhead - protocol round-trips) | **WARN** when source > 100 KB; recommend switching to `ssh` transport |

Workdir target: push test scripts into `dut_access.workdir`; pull
collected logs into
`<workspace>/target-platform/<profile-stem>/results/<timestamp>/`.

UART size-warning template (emitted to stderr before xfer starts):

```
WARNING: pushing 524288 bytes over base64-over-tty at 115200 baud.
  Estimated time: ~52 sec (at ~10 KB/s effective throughput).
  For files >100 KB, consider scp via the ssh transport instead.
```

The skill does **not** abort on size — it proceeds after the
warning.

## Sudo invocation on the DUT

Per `sudo.method` × transport:

| Method | SSH invocation | UART invocation |
|---|---|---|
| `nopasswd` | `sudo <cmd>` — assumes login user has NOPASSWD in `/etc/sudoers.d/`. | `sudo <cmd>` — same assumption. |
| `password` | `echo "$DUT_SUDO_PWD" \| sudo -S <cmd> 2>/dev/null` — pwd resolved from `sudo.password_env`. | Send `sudo <cmd>\r`; expect-detect `[sudo] password for <user>:\s*`; send `$DUT_SUDO_PWD\r`; resume capture until prompt sentinel. The expect detector must run only until the FIRST sudo prompt of the command — subsequent occurrences are not re-fed. |
| `prompt` | Interactive ask at first need; cache for the rest of the session. | Same caching rule; the prompt-detection logic is the only thing that differs. |
| `none` | Run commands as login user only; refuse if the validation step requires root. | Same. |

Wrap reliably-needed-root commands in a helper so the method
choice is centralized.

## UART implementation contract

Reference implementation: `scripts/uart_session.py`. The skill
invokes it as a black box; alternative implementations (Tcl
`expect`, Go, etc.) are acceptable as long as they honor the
contract below.

**Dependencies:** Python 3.6+, `pyserial`. No `pexpect` — the
state machine is rolled directly on top of `serial.Serial` to
keep the dependency surface minimal. Validated against the
default `python3` on Ubuntu 18.04 (3.6) through 24.04 (3.12);
install pyserial via `apt install python3-serial` (works on all
of those) or `pip install pyserial`.

**Subcommands:** `probe`, `exec`, `push`, `pull`. Each performs
its own login at session start and closes the tty at exit (no
session reuse across invocations — keeps the contract stateless
between calls; the cost is one login per command, ~1–2 sec).

**CLI shape:**

```
uart_session.py --tty <dev> [--baud <n>] --user <name>
                --password-env <ENVVAR>
                [--sudo-password-env <ENVVAR>]
                [--shell-prompt <regex>]
                [--lock-strategy refuse|wait]
                <subcommand> [args...]

  probe                    — login, run uname -r + nv_tegra_release.
  exec [--use-sudo] <cmd>  — run cmd; propagate remote exit code.
  push <src> <dst>         — base64-over-tty upload.
  pull <src> <dst>         — base64-over-tty download.
```

Resolved values come from the active profile's `dut_access:`
block. Passwords are read from the named env vars at run time;
the script refuses if any required env var is unset.

**State machine (login):**

1. Open the tty (raw, 8N1, baud per `--baud`).
2. **Lock check**: shell `fuser <tty>`; if held and
   `lock_strategy=refuse`, exit 3 with the holding PIDs.
3. Drain the read buffer (consume kernel-printk noise that
   accumulated while the tty was idle).
4. Send `\r` to elicit a login prompt or shell prompt.
5. Wait (timeout) for the first match of:
   - `login:` → already-logged-out shell. Send user, wait for
     `Password:`, send password, wait for default shell prompt
     (`[\$#]\s*`).
   - `[\$#]\s*$` → already-logged-in shell. Skip auth.
6. Replace PS1 with a uuid-tagged sentinel so the prompt is
   uniquely matchable for the rest of the session:
   `export PS1='__PROMPT_<uuid>__> '`. Drain the echo + first
   sentinel-marked prompt.

**State machine (exec):**

1. Generate a per-call exit-code marker
   (`__EXIT_<uuid>__<rc>\n`).
2. Send the wrapped command: `<cmd>; printf "__EXIT_<uuid>__%d\\n" "$?"\r`.
3. Stream the tty into a buffer until the exit-code marker
   appears, with interleaved sudo-prompt detection (only if
   `--use-sudo` was passed):
   - If `\[sudo\] password for [^:]+:\s*` matches **before** the
     exit-code marker, send the sudo password once and continue.
4. Once the marker matches, split the buffer:
   - Output = bytes before the marker, minus the leading
     command-echo line, minus ANSI escapes
     (`\x1b\[[0-9;?]*[a-zA-Z]`).
   - Exit code = capture group from the marker regex.
5. Consume the trailing prompt sentinel so the buffer is clean
   for the next call (defensive, even though sessions close
   per invocation).
6. Print stdout to the script's stdout; exit the script with
   the captured remote exit code.

**State machine (push / pull):**

- **Push**: read local file → base64-encode → wrap in heredoc:
  ```
  base64 -d > <dst> <<EOF_<uuid>
  <base64 in 76-char lines>
  EOF_<uuid>
  ```
  Send via `exec`. Timeout scaled by file size
  (`max(60, size / 5000)` seconds, ~5 KB/s safety margin).
- **Pull**: `exec("base64 < <src>")` → decode → write local file.
  Same timeout scaling.
- **Both**: if `size > 100 * 1024`, write the WARNING template
  to stderr before transfer.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | success (or remote exit 0 for `exec`) |
| 1..127 | for `exec`: remote command's exit code |
| 128 | login failed (no login or shell prompt within timeout) |
| 129 | command timed out |
| 130 | sudo prompt detected but no sudo password configured |
| 2 | env var required by `--password-env` / `--sudo-password-env` is unset |
| 3 | tty held by another process and `lock_strategy=refuse` |
| 4 | pyserial import failed |

**Robustness notes:**

- The uuid-tagged PS1 sentinel removes prompt-matching ambiguity
  from custom `PS1` values, ANSI color codes, and OSC-escape
  decorations.
- Kernel printk noise emitted mid-session can interleave with
  command output. The state machine drains the buffer pre-login
  and relies on the per-call exit-code marker to delimit
  command boundaries, so printk lines are captured into the
  command's stdout (visible to the caller, but don't break
  parsing).
- The per-call session policy (login → exec → close) means a
  validation pass running N commands pays N × login cost. For
  N > 10, prefer the SSH transport; UART is intended for
  diagnostic and bring-up scenarios, not high-volume runs.

## Security notes

- **Never inline a password in the profile YAML.** The schema
  carries `password_env` (the *name* of the env var), not the
  value. Skills must refuse to read a literal password field if
  one is ever added by hand.
- **Conversation-log exposure** is the user's responsibility when
  they pick `auth=prompt` or `sudo.method=prompt` in an
  AI-assisted session — passwords typed in response to prompts
  appear in the transcript. Document this in the prompt itself
  before reading the password back.
- **SSH `StrictHostKeyChecking=accept-new`** (not `no`) accepts
  on first connect and pins thereafter. A subsequent
  fingerprint change is then a refusal — usually a sign that the
  DUT was reflashed and SSH host keys regenerated, or that the
  IP was reassigned. Prompt the user to remove the per-profile
  known_hosts entry rather than auto-accepting.
