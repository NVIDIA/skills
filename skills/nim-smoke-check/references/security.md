# Security: requests, programs, environment and files

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
-->

Read this when the user asks what the smoke check sends, runs, reads or writes. The scripts are
read-only by construction, and `tests/test_nsc_security.py` checks each rule below. No setting of
the scripts changes where requests go or which programs run. The settings of Python, `kubectl` and
`docker` are trusted inputs (see Trust).

## Requests go only to the user's URL

| what | rule |
|---|---|
| host | only the base URL from `--url`, or the URL `--docker` or `--k8s` resolves to (a published port, or the script's own `kubectl port-forward` on `127.0.0.1`); `--health-url` for the health routes |
| methods and routes | `GET` on `/v1/health/live`, `/v1/health/ready`, `/health`, `/v1/models`, `/v1/models/ID` (hosted), `/v1/version`, `/v1/metadata`; `POST` on `/v1/chat/completions` only |
| request fields | the allow-list in `scripts/nsc_checks.py` (`REQUEST_FIELDS`); never `nvext`, `guided_*`, `chat_template` or `structured_outputs` |
| redirects | never followed: a 3xx is reported as `redirect_refused`, and the host it names gets no request |
| key | sent as `Authorization: Bearer`, added unredirected, only to the scheme, host and port of the user's URL, with the host as written (`gw.example.com.`, with a final dot, is another host than `gw.example.com`): a `--health-url` (or a published `NIM_HEALTH_PORT`) on another host, port or scheme gets its health `GET`s without it; `http://integrate.api.nvidia.com`, in any spelling the name lookup reads as it, is refused (`bad_input`) before the key is read, and so is a host with a `%` (percent-encoding), which the connection would decode into another name; a key that would go over plain `http://` to a host that is not this machine (the name `localhost`, 127.x, `::1`, a port-forward, and `0.0.0.0` or `::`, which connect to it; not a `*.localhost` name or a host with a final dot, which the resolver may look up in DNS) stops the run before any request (`key_over_plain_http`, exit 2), unless the user agreed to `--allow-plain-http-key`, and then the warning is in `facts.warnings` however the run ends, and in `results.json` and `report.md` when it writes them |
| other hosts | none: no telemetry, no update check, no download, no NGC or registry request |
| the target URL | taken only from the command line, never from an environment variable |

Every request goes through `nsc_common.open_url`: no redirect handler, TLS verification always
on (the system CA store, or `--ca-file` alone), and no proxy for a loopback URL (also a 127.x
address written in a short or numeric form, such as `127.1`). Each URL decides for itself: a
`--health-url` on another host keeps the proxy settings even when `--url` is loopback. One
deadline, the request's timeout, covers connecting, the TLS handshake, sending and every read of
the answer, so headers or a body that keep arriving slowly still end then (`timeout`). Only the
name lookup has no limit of its own.

## Programs come from PATH

The scripts run only `kubectl` and `docker`, never a shell, with stdin closed and a timeout on
every call. Each program is found in the absolute folders of `PATH`. An empty or relative entry
(`::`, a trailing `:`, `.`) means the current folder, often a project someone else wrote, so it
is skipped. No variable picks the program. `nsc_access.run_tool` refuses (`refused_command`, exit
5) any command outside this list before it starts it:

| program | commands |
|---|---|
| `kubectl` | `auth can-i create pods --subresource=portforward`; `get nimservice NAME -o json`; `get service NAME -o json`; `get services -l SELECTOR -o json`; `get pods -l SELECTOR -o json`; `get pod NAME -o json`; `port-forward pod/POD [LOCAL]:PORT`; `config get-contexts -o name` |
| `docker` | `container inspect NAME`; `port NAME`; `image inspect --format '{{json .Config.Env}}' IMAGE`, where IMAGE is the ID of the image the container runs or the pod's `REPO@sha256:` digest, never a tag; `context inspect --format '{{.Endpoints.docker.Host}}'` |

Every `kubectl` command but `config get-contexts` carries `--context CTX -n NS`. No Secret is
read. Of a container's or a pod's environment, only `NIM_SERVER_PORT`, `NIM_HEALTH_PORT`,
`NIM_SSL_MODE` and `NIM_PASSTHROUGH_ARGS` are kept; facts show only whether
`NIM_PASSTHROUGH_ARGS` is set, never its value. The port-forward is stopped in a `finally`
block, at exit, and on SIGTERM or SIGHUP.

## Python runs isolated

SKILL.md runs the scripts as `python3 -I`. Python then ignores `PYTHONPATH`, the other `PYTHON*`
variables and the user site folder: a `sitecustomize.py` on `PYTHONPATH` could otherwise rewrite
every request. Each script adds its own folder to `sys.path`.

## What the scripts read from the environment

| variable | used by | what for |
|---|---|---|
| the variable named by `--api-key-env` | `run` | the key, sent as a bearer token; `--api-key-env` takes a name, and a value that looks like a key (also one whose short prefix ends in `_`, such as `gsk_...`) is refused (`literal_secret`) without being echoed; a variable that is not set is `bad_input`, and the error does not repeat the name. A flag counts only when written in full: `--api-key=KEY` is an unknown flag (`bad_input`), never a short form of `--api-key-env`, and a usage error names an unknown flag without its value |
| `NVIDIA_API_KEY` | `run` | the key, only for `https://integrate.api.nvidia.com` and only when `--api-key-env` is not given; unset or empty, a run of chat checks there is `bad_input` before any request |
| `PATH` | `run` (`--docker`, `--k8s`) | finds `kubectl` and `docker` (absolute folders only) |
| `DOCKER_HOST` | `run` (`--docker`) | a remote daemon is refused (`docker_remote`): its port is not on this machine |
| `HOME` | `run`, `report` | the default `--out` folder, `~/.cache/nim-smoke-check/`, and a path that starts with `~` |

A key is registered with the masker as soon as it is read. A key shorter than 6 characters, too
short to be masked, is `bad_input` before any request, and its value is not shown. A key is
masked in stdout, stderr (a traceback too) and every file, and so are:

- well-known token shapes (`nvapi-`, `hf_`, `glpat-`, `gh?_`, `sk-`, `AKIA`) and bearer tokens;
- the value of a credential flag: `--api-key X`, `--hf-token=X`, and the item after the flag in
  a list, `["--api-key", "X"]`;
- the value of a credential name in a header, an assignment or a JSON field: `x-api-key: X`,
  `NGC_API_KEY=X`, `{"password": "X"}`, `?access_token=X`, `Authorization: Basic X` (shown as
  `Basic ***`). A credential name holds `api-key`, `access-key`, `private-key`, `secret`,
  `passw` or `authorization`, or ends in a token name such as `HF_TOKEN` or `x-auth-token`.

Known gap: a value of no known shape after a bare `key`, `auth` or `token` (a reasoning
switch's `key`, a log-probability's `token`) or after another name (`credentials:` starts
kubectl's own errors) is masked only when it is the key the run read. A name given to an `-env`
flag (`--api-key-env NAME`) is a variable name and stays readable. Text from the endpoint is
masked before it is cut (an error message to 200 characters, an evidence field to 2,000), and so
is the output of `kubectl`, `docker` and a credential plugin (a message to 300 characters), so a
key that any of them echoes is never kept in part.

## What Python and the programs read themselves

| reader | variables |
|---|---|
| Python | `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY` (not for a loopback URL); `TMPDIR` for the port-forward's temporary folder |
| OpenSSL | `SSL_CERT_FILE`, `SSL_CERT_DIR` (the system CA store, when no `--ca-file` is given) |
| `kubectl` | `KUBECONFIG` (the cluster and credentials of `--context`; a credential plugin in it runs), proxy variables |
| `docker` | `DOCKER_HOST`, `DOCKER_CONTEXT`, `DOCKER_CONFIG`, `DOCKER_CERT_PATH`, `DOCKER_TLS_VERIFY` |

Each program gets the script's environment. A program on `PATH` runs as the user and can read the
user's files anyway, so giving it fewer variables would protect nothing. `PATH` is the control.

## Files

| file | mode | what |
|---|---|---|
| `--out` folder (default `~/.cache/nim-smoke-check/<UTC time>-<model>/`) | `0700` | created when missing; must be new or empty. Checked before any request (for the default folder, its parent), and again before the files are written |
| `results.json` | `0600` | the run: `target`, `access`, `counts`, every result (status, reason, message, evidence paths, sources, hint, known issues, warnings, attempts, latency and details), `warnings` and `handoff`; also for a run that stopped early (a lost port-forward, Ctrl-C, an unexpected error), with the checks that did not run in `not_run`. The plan is not kept; `run --dry-run` prints it |
| `report.md` | `0600` | the summary table; no prompt or answer text. Text from the endpoint or the command line in it (a model id, a message, a warning) is one line, without control, line-separator or bidi characters, with backslash, backtick, pipe, square and angle brackets escaped, so it cannot add a line, a heading, a cell, a link or HTML |
| `evidence/<check>-<n>.json` | `0600` | one request and its answer: `Authorization` masked, text cut to 2,000 characters, image bytes replaced by SHA-256 and size |
| port-forward output | `0700` folder from `tempfile.mkdtemp` | kubectl's output, removed when the forward stops |

A folder, or one above it, that belongs to another user (root aside), or that other users can
write without the sticky bit of `/tmp`, is refused (`out_not_private`, exit 2). A group write
counts as other users unless the group is the user's private group: the primary group, named
after the user, with no other member and no other account that has it as its primary group (of
the accounts the system lists). So a `0775` folder under user private groups (umask `002`) is
accepted, and one whose group is a shared `users` group is refused. A symbolic link in the
`--out` path as written must belong to the user or root (else `out_not_private`), and the folder
that holds it, with the folders above, is checked like the others. The files go to the folder
the check found, so a link changed after the check cannot move them. Each file is
written to a new temporary file that then replaces the name, so a symlink planted there is
replaced, never followed. `--dry-run` writes no file. `report DIR` only reads `DIR/results.json`
(a regular file, not a symlink).

## Trust

- The environment is a trusted input. `PATH` picks the programs. `KUBECONFIG` and the Docker
  settings pick the cluster and the daemon, and can run plugins. The proxy and CA settings can
  route or inspect the requests. Run the scripts only from an environment you trust.
- The endpoint is trusted to answer, not to be honest: its answers are data. A check judges them
  and saves them as evidence; nothing in an answer is run or followed. JSON nested deeper than 64
  levels counts as not JSON, and a value nested deeper than 128 levels is replaced in anything
  printed or saved, so no answer can end a run by exhausting Python's recursion limit. Any
  other error that no check expected ends the run with one JSON object (`internal_error`, exit
  4), never a bare traceback, and so does Ctrl-C (`interrupted`, exit 130); the checks that ran
  keep their files, and the port-forward is stopped first.
- `model_features.json` is reviewed data in this repository. Its sources are public pages only.
- `kubectl` uses the credentials of the `--context` you name. The scripts send no credential but
  the key you name.

## Reporting a security problem

Do not open a public issue for a security problem. Report it to NVIDIA PSIRT, as described on
<https://www.nvidia.com/en-us/security/>, or by email to `psirt@nvidia.com`.

## Tests

Only test code points the scripts at local fake servers and fake `kubectl` and `docker`
(`tests/nsc_testkit.py`). `tests/test_nsc_security.py` parses every script to check that URLs
are opened only in `nsc_common.open_url`, programs start only in `nsc_access.run_tool`, and the
environment variables read are the ones in the table above. At run time it checks that a
redirect is not followed, that the key never reaches another host or any output (also when the
endpoint echoes it across a cut), that a `--health-url` on another port, or on the same host
with a final dot, gets no key, that `http://` to the hosted API is refused, that a
percent-encoded host is refused before any name is looked up, that a key over plain `http://` to
another host (also one `--docker` gives, a `*.localhost` name or a host with a final dot) stops
the run before any request unless `--allow-plain-http-key` is given, which keeps the warning
however the run ends, that a key passed as a value or after a short form of a flag
(`--api-key=KEY`) is refused, that a key too short to be masked is refused, that `--out` refuses
a shared folder, that JSON nested too deeply or a model id that is not text never ends a run,
and that the scripts run under `python3 -I`. It also checks that no case in `evals/evals.json`
holds a key-shaped value.

The tests write their runs under pytest's temporary folder. When that folder fails the
`--out` check (a sandbox or a user namespace can show `/tmp` as owned by `nobody`),
`tests/conftest.py` stops the run with one message (exit 1) instead of one `out_not_private`
failure per test: rerun with `--basetemp` set to a new folder under one you own.
