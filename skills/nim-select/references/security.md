# Security: endpoints, programs and the environment

Where the scripts send requests, which programs they run, what they read from the environment and
where they write. No `NIM_SELECT_` variable or other setting of the scripts changes the hosts they
send requests to or the programs they run. The settings of Python and of the tools the scripts run
are trusted inputs (Trust, below).

## Requests go to built-in endpoints
Every request is a read-only, anonymous GET to one of these, and no environment variable changes
them:

| what | endpoint |
|---|---|
| NGC API: catalog search, repositories, image lists, NIM Metadata API | `https://api.ngc.nvidia.com/v2` |
| TGC catalog | `https://tgc.turing-agi.com` |
| TuringCM catalog | `https://nim.turingcm.com` |
| Harbor API of the CND image hosts | `https://tgcr.turing-agi.com` (for `tgcr-gz.turing-agi.com` images too), `https://resource.turingcm.com`, `https://io.chancloud.com` |
| location check (`locate` only) | `https://www.google.com/generate_204` |
| country lookup (`country` only, run after the user agrees) | `https://www.cloudflare.com/cdn-cgi/trace`: Cloudflare sees this machine's public IP address and names its country; the script keeps only the country code and prints nothing else of the answer |

No redirect is followed. These endpoints send none, so a redirect is a proxy or a portal answering:
NGC reports it as `ngc_bad_response`, a CND host as not readable, `locate` as not reachable, and `country` as `country_unknown`. The
host a redirect names gets no request.

`NIM_SELECT_NGC_BASE`, `NIM_SELECT_HARBOR_BASE`, `NIM_SELECT_TGC_BASE`, `NIM_SELECT_TURINGCM_BASE`
and `NIM_SELECT_LOCATE_URL` have no effect. An image on a registry mirror is still checked as the
NGC image with the same name (SKILL.md, Inputs).

## Another API host
A user may ask to send the NGC requests to another API host, such as their gateway
`https://api-proxy.example.com/v2`. Say that the endpoint is built in and that no setting of the
scripts changes it; edit or copy no script, and set no `NIM_SELECT_` variable (none has an effect).
Then run resolve, inspect and select (or list) against the built-in endpoint as usual, and never
stop first to ask how to reach NGC. A forward proxy (`host:port`) the user names may be passed as
`HTTPS_PROXY`, and the requests still go to the hosts above (Trust, below); an API gateway URL such
as `https://api-proxy.example.com/v2` is not a forward proxy. `fit --metadata-file FILE --tag TAG`
asks NGC first too, and uses the file alone only when NGC cannot be reached: it is a fallback for a
host that cannot reach NGC, not a way to keep requests away from it. On `ngc_unreachable`, show the
error and check the network or the proxy settings (SKILL.md, On failure 2).

## Programs come from PATH
Inspection runs `nvidia-smi`, `docker` and `kubectl`, read-only, found in the absolute folders of
`PATH`. Of `kubectl` it runs `config get-contexts -o name`, which reads only the kubeconfig (a
context the user names that it lacks stops inspect with `context_not_found`), then `get nodes` and
`get resourcequota` with that context. An empty or relative `PATH` entry (`::`, a trailing `:`,
`.`) means the current folder, often a project someone else wrote, so it is skipped. The host
memory is read from `/proc/meminfo`. `NIM_SELECT_NVIDIA_SMI`, `NIM_SELECT_DOCKER`,
`NIM_SELECT_KUBECTL` and `NIM_SELECT_MEMINFO` have no effect.

Each program gets the whole environment of the script and reads its own settings:
- `kubectl`: `KUBECONFIG` defines the `--context` you name: its cluster, and its credentials, which
  kubectl sends to that cluster. A credential (exec) plugin in it runs.
- `docker`: `DOCKER_HOST`, `DOCKER_CONTEXT` and the Docker config (`DOCKER_CONFIG`) pick the
  daemon, and `docker info` runs the CLI plugins of the Docker config. Docker facts come only from
  a local daemon (`unix://`, `npipe://`): inspect asks `docker context inspect` which daemon is
  used, and leaves out the facts of a remote one (`docker_remote`), whose GPUs are not this host's.

A program on `PATH` runs as the user: it can read the environment of its parent and the user's
files anyway, so giving it fewer variables would protect nothing. `PATH` is the control.

## Python runs isolated
SKILL.md runs the scripts as `python3 -I`. Python then ignores `PYTHONPATH` and the other
`PYTHON*` variables, the user site folder and `SSLKEYLOGFILE`: a `sitecustomize.py` on `PYTHONPATH`
could otherwise rewrite every request, and `SSLKEYLOGFILE` writes the TLS keys of each request to a
file. The scripts need none of them; each adds its own folder to `sys.path`.

## What the scripts read from the environment
| variable | used by | what for |
|---|---|---|
| `TZ`, `LC_ALL`, `LANG` | `locate` | what the location hint measures |
| `PATH` | `inspect` | finds `nvidia-smi`, `docker` and `kubectl` (absolute folders only) |
| `DOCKER_HOST` | `inspect` (host) | a remote daemon: its GPUs are not this host's, so the Docker facts are left out (`docker context inspect` covers `DOCKER_CONTEXT` and the Docker config) |
| `HOME` | `inspect` (host), `list`, `--out`, `--env` | the default NIM cache folder, `~/.cache/nim`, for the free-disk figure; the list's cache, `~/.cache/nim-select/ngc-cache.json`; a file path that starts with `~` |
| `NIM_SELECT_RETRY_DELAY` | NGC requests | the first retry delay in seconds (default 1, at most 10; a value that is not a number is the default); it changes only how long a retry waits |

Python reads the proxy settings (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`) for every request, and
OpenSSL the CA settings (`SSL_CERT_FILE`, `SSL_CERT_DIR`).

## Files
`--out` writes only where no other user can change the file. Each missing folder on the way is
created private (`0700`), one level at a time, so a umask of `002` never makes a new `~/.cache`
group-writable. A folder, or one above it, that belongs to another user (root aside), or that other
users can write without the sticky bit of `/tmp`, is refused (`out_not_private`). Each folder is
checked as it is, never through a symlink, so another user cannot swap one of theirs for a symlink
to a folder of the user's while it is checked. For a folder of
the user's, the fix names it and the `chmod g-w FOLDER` (`go-w` when others can write) to ask the
user to run; a file in another folder only the user can write works too. Root gets no chmod in its
fix: it owns the system folders, and a group shares some of them on purpose (`/var/local` is
`root:staff 2775` on Debian), so the fix sends root to a folder of its own. A folder a group can
write is refused even when the group is the user's own private group: whether another account is
in a group cannot be verified on every host (directory services such as SSSD or LDAP, the
`initgroups` sources), so no group is trusted, and the user removes the access with the chmod in
the fix. With a POSIX ACL, the group bits show the ACL's mask, which bounds every user and group
the ACL names, so the same check covers them, and `chmod g-w` takes the write access out of the
mask. An ACL of another kind, as on macOS, is not read. The
JSON goes to a new temporary file that then replaces the named file, so a symlink planted at that
name is replaced, never followed. SKILL.md keeps the files in `~/.cache/nim-select/`. `--env` and
`--metadata-file` read the file the user names, such as an environment file from another host.

`list` keeps the NGC answers it reads in `~/.cache/nim-select/ngc-cache.json`, written the same
way as an `--out` file, with mode `0600`. It reads the file back only when it is a regular file of
this user that neither the group nor others can write, not a symlink, in a folder no other user can
change, and opened without following a symlink; otherwise the list runs without it and says why in
`facts.cache.problem`. What the cache keeps, for how long, and `--no-cache`:
`references/list-mode.md`, The cache.

## Trust
The environment is still a trusted input: `PATH` picks the programs, `KUBECONFIG` and the Docker
settings pick the cluster and the daemon and can run plugins, and the proxy and CA settings can
route or intercept the requests. Run the scripts only from an environment you trust. The scripts
send no credentials: no NGC API key, no registry login, no token. `kubectl` uses the credentials of
the `--context` you name.

## Tests
Only test code points the scripts at local servers, fake programs, a test memory file, a test
cache folder or a shorter list deadline: `tests/ns_testkit.py` patches the scripts' modules, and its
runner applies the same patches before it runs a script. `tests/test_security.py` sets each variable listed above as having no effect and
checks that no request and no program changes, checks that a redirect is not followed and that
`--out` creates its folders private and refuses a shared folder, and parses every script to check
that the environment variables it reads are the ones in the table above.
