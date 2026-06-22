# SLURM SSH Setup, Credentials, And Storage

SSH preflight detail, prerequisite setup, the full credential list, backend details, storage rules, and the SSH failure remediation prompt. If this reference conflicts with `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the compact/current source wins.

## Preflight

```bash
# 1. SSH to the login node works without a password prompt
SLURM_HOST="${SLURM_HOSTNAME%%,*}"
[ -n "$SLURM_USER" ] && [ -n "$SLURM_HOST" ] || {
  echo "MISSING: export SLURM_USER and SLURM_HOSTNAME (comma-separated for failover) in your shell before launching."
  exit 1
}
ssh -o BatchMode=yes -o ConnectTimeout=10 "${SLURM_USER}@${SLURM_HOST}" "true" 2>/dev/null || {
  echo "MISSING: passwordless SSH to ${SLURM_USER}@${SLURM_HOST} not working. See the Prerequisites section."
  exit 1
}

# 2. Optional: TAO SDK wrapper for Job handles + S3 wrapping.
# nvidia-tao-sdk is on public PyPI; pin lives in versions.yaml (wheels.tao_sdk_slurm).
PIN=$("${TAO_SKILL_BANK_PATH:?}/scripts/resolve_versions_key.py" wheels.tao_sdk_slurm)
python -c "import tao_sdk" 2>/dev/null || {
  echo "Installing missing Python requirement: $PIN"
  python -m pip install "$PIN"
}
python -c "import tao_sdk"

# 3. Enroot credentials on the cluster for private nvcr.io images.
# Pyxis on the compute nodes invokes enroot to import the Docker image. Enroot
# does NOT read NGC_KEY from the SLURM job env — it requires persistent
# credentials in ~/.config/enroot/.credentials on the login/compute nodes.
# Without this, anonymous pulls of nvcr.io/nvstaging/* (or any auth-gated
# repo) fail with "Could not process JSON input" at job startup. Skip if the
# image is from a public repo.
if [ -n "$NGC_KEY" ]; then
  REMOTE_CRED_OK=$(ssh -o BatchMode=yes "${SLURM_USER}@${SLURM_HOST}" \
    'test -s ~/.config/enroot/.credentials && echo OK || echo MISSING' 2>/dev/null)
  if [ "$REMOTE_CRED_OK" != "OK" ]; then
    echo "MISSING: ~/.config/enroot/.credentials not set on ${SLURM_HOST}."
    echo "After user approval, install it from NGC_KEY (no value echoed):"
    echo "  printf 'machine nvcr.io login \$oauthtoken password %s\\nmachine authn.nvidia.com login \$oauthtoken password %s\\n' \"\$NGC_KEY\" \"\$NGC_KEY\" \\"
    echo "    | ssh -o BatchMode=yes \"\${SLURM_USER}@\${SLURM_HOST}\" '"
    echo "        mkdir -p ~/.config/enroot && umask 077 && cat > ~/.config/enroot/.credentials && chmod 600 ~/.config/enroot/.credentials"
    echo "      '"
    exit 1
  fi
fi
```

If a check fails, the agent prompts the user to authorize the install/fix via Bash. Pip-installable Python requirements are the exception: install them automatically, then rerun preflight.

The enroot-credentials step (#3) only needs to run **once per (cluster, user)** —
subsequent SLURM sessions inherit the file. Use the `printf | ssh` heredoc
pattern above so the `NGC_KEY` value never lands in shell history, intermediate
files, or chat output. Do not `cat` or `echo` the value at any step. After the
file is in place, both the SDK's SQSH pre-conversion job (which runs on
`sqsh_conversion_partition`) and the actual training job's Pyxis pull will
authenticate as `$oauthtoken` against `nvcr.io`.

## Prerequisites

Before any SLURM job can be submitted or any runner script is generated, the
host running the TAO service or SDK must be able to log in to at least one host
from `SLURM_HOSTNAME` over SSH **without an interactive password prompt**. The
handler runs `sbatch`, `squeue`, `sacct`, `scancel`, and log tails
non-interactively, so password or 2FA prompts will fail the job at submit or
status time.

Set this up once per (host, login node, user) tuple:

1. Ensure an SSH keypair exists for the service user (e.g. `~/.ssh/id_ed25519`).
   Create one with `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519` if it is
   missing. The handler defaults to the same locations described under
   `SSH_KEY_PATH` in [Credentials](#credentials).
2. Install the public key on each login node:

   ```bash
   ssh-copy-id -i ~/.ssh/id_ed25519.pub <SLURM_USER>@<login-host>
   ```

   This is the only step that requires the user's password; run it interactively
   once per login host listed in `SLURM_HOSTNAME`. If `ssh-copy-id` is not
   available, append the public key manually:

   ```bash
   cat ~/.ssh/id_ed25519.pub | ssh <SLURM_USER>@<login-host> \
     'mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
      cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'
   ```
3. Trust the host key so SSH does not stall on the "authenticity of host" prompt
   inside the handler. Either log in once interactively to accept the prompt,
   or pre-populate `~/.ssh/known_hosts` with `ssh-keyscan -H <login-host> >> ~/.ssh/known_hosts`.
4. Verify the result is fully non-interactive for at least one listed login
   host:

   ```bash
   ssh -o BatchMode=yes -o PreferredAuthentications=publickey \
     <SLURM_USER>@<login-host> 'hostname && squeue -u $USER -h | head -n 1'
   ```

   `BatchMode=yes` forces failure if SSH would otherwise prompt; this command
   must succeed before the SLURM platform is usable.
5. When the service runs in a container (microservices deployment), mount the
   private key into the container at the path referenced by `SSH_KEY_PATH`, with
   `chmod 600` and matching ownership for the in-container user. The handler
   refuses keys with world-readable permissions.

For convenience, a per-host alias in `~/.ssh/config` lets you reference a short
name everywhere:

```text
Host slurm-login
    HostName <login-host>
    User <SLURM_USER>
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking accept-new
```

If a site enforces 2FA on every SSH connection, passwordless key auth alone is
not enough; coordinate with the cluster admin to allow key-only auth from the
service host or use an SSH agent with cached credentials and expose it to the
handler via `SSH_AUTH_SOCK`.

## Credentials

- **SLURM_USER** (required): SSH username for the login node. In microservices
  workspace metadata this is `cloud_specific_details.slurm_user`.
- **SLURM_HOSTNAME** (required): Comma-separated login hostnames for failover.
  Microservices schema stores this as the list field
  `cloud_specific_details.slurm_hostname`.
- **SLURM_PARTITION** (required): Partition list for GPU job submission. Ask
  for this in the mandatory SLURM intake list. The packaged default is
  `polar,polar3,polar4,grizzly`, which are treated as 4-hour queues.
- **SSH_KEY_PATH** (preferred and expected before launch): private key path for
  non-interactive public-key auth to the login node. If passwordless SSH fails,
  ask the user for `SSH_KEY_PATH=/path/to/private_key` and show the setup steps
  below; do not bury this behind several alternate choices.
- **SSH_AUTH_SOCK** (advanced fallback): SSH agent socket with an accepted key
  already loaded. Prefer `SSH_KEY_PATH` in user-facing remediation prompts.
- **SLURM_BASE_RESULTS_DIR** (optional): Base shared filesystem path. Default
  convention from `tao-core` is `/lustre/fsw/portfolios/edgeai/users/<your-dir>` (your per-user results directory on Lustre).
- **SLURM_ACCOUNT** (usually required by site policy): Account charged by
  `#SBATCH --account`.

Do not ask for `SLURM_ACCOUNT` or `SLURM_BASE_RESULTS_DIR` in the initial
intake unless the user says their site requires an account, wants a custom
results root, or the workflow cannot proceed without overriding defaults.

## Backend Details

Use `backend_details.backend_type = "slurm"` when routing a job to this
platform. Supported backend details from the microservices schema:

```json
{
  "backend_type": "slurm",
  "partition": "polar,polar3,polar4,grizzly",
  "cluster_name": "optional-name"
}
```

Runtime metadata is stored under `backend_details.slurm_metadata`, especially
`slurm_job_id` and `job_dir`. Do not invent these values. They are written
after `sbatch` returns a scheduler job id.

## Storage

SLURM jobs run on the cluster, so local paths from the API host are not valid
dataset paths. Prefer shared filesystem URIs:

- Use `lustre:///absolute/path` for user-provided datasets on Lustre.
- `slurm://` paths may appear in microservices metadata and are converted to
  actual Lustre paths before the container starts.
- Avoid bare `/local/path` and `file://` dataset URIs for SLURM. Validation in
  `tao-core` rejects local and file paths for remote backends.

Accept either dataset roots or direct spec-key paths:

- Root mode: `/lustre/.../<model>/train`, which model skills map to required
  files such as `<root>/annotations.json` and `<root>` as media path.
- Direct spec mode: exact fields such as
  `custom.train_dataset.annotation_path=/lustre/.../train.json` and
  `custom.train_dataset.media_path=/lustre/.../videos.tar.gz`.

After passwordless SSH succeeds and before generating scripts, validate each
required dataset file/path from the login host:

```bash
ssh -o BatchMode=yes <SLURM_USER>@<working-login-host> \
  'test -e /lustre/.../annotations.json && test -e /lustre/.../media_or_archive'
```

If the remote `test -e` fails, stop and ask for corrected paths or for the data
to be staged onto shared cluster storage. Do not create runner scripts that will
fail inside the first training job.

## SSH Failure Remediation Prompt

When passwordless SSH fails, use this concise prompt:

```text
SLURM is blocked on passwordless SSH. Please provide:

SSH_KEY_PATH=/path/to/private_key

If you have not set up passwordless access yet:
1. Create a key if needed:
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
2. Install the public key on one login host:
   ssh-copy-id -i ~/.ssh/id_ed25519.pub <SLURM_USER>@<login-host>
3. Trust the host key:
   ssh-keyscan -H <login-host> >> ~/.ssh/known_hosts
4. Lock private-key permissions:
   chmod 600 ~/.ssh/id_ed25519
5. Verify it works without prompts:
   ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 <SLURM_USER>@<login-host> 'hostname'

After that, rerun with SSH_KEY_PATH=~/.ssh/id_ed25519.
```

Results default to:

```text
/lustre/fsw/portfolios/edgeai/<your-dir>/results/<job_id>
```

`<your-dir>` is your per-user directory on the cluster's Lustre share.

(`<your-dir>` is your per-user directory under the Lustre portfolio path.) The runner sets `TAO_API_RESULTS_DIR` to the parent results directory because
container code appends the job id when writing status and artifacts.
