---
name: ngc
description: Install, configure, or verify NVIDIA NGC CLI and API key access. Use when NGC CLI is missing, the NGC API key needs to be set or tested, or NGC resource access fails.
---

# NGC CLI — Install, Configure, Verify

Manages NVIDIA NGC CLI setup and API key access. Required before deploying any VSS profile.

## When to Use

Use this skill when:

- NGC CLI is not installed (`ngc: command not found`)
- NGC API key is missing or needs to be verified
- An NGC resource pull fails with auth errors
- User asks to set up or reconfigure NGC access

## Check Current State

```bash
# Is NGC CLI installed?
ngc --version

# Is key in environment?
echo "NGC_CLI_API_KEY: ${NGC_CLI_API_KEY:+SET}${NGC_CLI_API_KEY:-NOT SET}"
```

---

## Install NGC CLI (if missing)

**AMD64 Linux:**

```bash
curl -sLo /tmp/ngccli.zip \
  https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.10.0/files/ngccli_linux.zip
sudo mkdir -p /usr/local/lib
sudo unzip -qo /tmp/ngccli.zip -d /usr/local/lib
sudo chmod +x /usr/local/lib/ngc-cli/ngc
sudo ln -sfn /usr/local/lib/ngc-cli/ngc /usr/local/bin/ngc
ngc --version
```

**ARM64 Linux:**

```bash
curl -sLo /tmp/ngccli.zip \
  https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.10.0/files/ngccli_arm64.zip
```

_(then same install steps as above)_

---

## Configure NGC API Key

If the user doesn't have a key yet, guide them:

1. Go to https://ngc.nvidia.com → sign in
2. Top-right → **Setup** → **API Keys** → **Generate Personal Key**
3. Set permissions: **NGC Catalog**
4. Copy the key immediately (shown only once)

Once they have the key:

```bash
export NGC_CLI_API_KEY='<key>'
# Optionally persist in shell profile:
echo "export NGC_CLI_API_KEY='<key>'" >> ~/.bashrc
```

> Do not store the raw key in `TOOLS.md` or any workspace file.

---

## Verify Access

```bash
ngc registry resource info nvidia/vss-developer/dev-profile-compose:3.0.0
ngc registry image info nvidia/vss-core/vss-agent:3.0.0
```

Both should return resource info without errors.

**Common error:** `Missing org — If Authenticated, org is also required.`
→ Fix: run `ngc config set` and ensure the org matches the one selected when generating the key.

---

## Quick Config via ngc CLI

```bash
ngc config set
# prompts for API key, org, team, format
```
