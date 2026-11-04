# Standalone CLIP skill setup

When `tao-finetune-clip` is installed individually, its folder does not
include the shared bank helpers used for launch preflight, image resolution,
and job records. Before running `tao-setup` or submitting a CLIP action, set
`TAO_SKILL_BANK_PATH` to a complete TAO skill-bank checkout, not to the
individual installed skill directory. Verify it with:

```bash
export TAO_SKILL_BANK_PATH="${TAO_SKILL_BANK_PATH:-$HOME/tao-skill-bank}"
for helper in check_tao_launch_preflight.py resolve_tao_image.py resolve_tao_model.py tao_job_record.py redact_secrets.py; do
  [ -f "$TAO_SKILL_BANK_PATH/scripts/$helper" ] || {
    printf 'Missing bank helper: %s/scripts/%s. CLIP requires a complete bank checkout.\n' "$TAO_SKILL_BANK_PATH" "$helper" >&2
    exit 1
  }
done
[ -f "$TAO_SKILL_BANK_PATH/versions.yaml" ] && [ -d "$TAO_SKILL_BANK_PATH/skills" ] || {
  echo 'Incomplete TAO skill-bank checkout; stop before CLIP launch.' >&2
  exit 1
}
```

If no complete checkout exists, obtain approval to clone
`https://github.com/NVIDIA-TAO/tao-skill-bank.git` into a user-selected
location, then set the variable and rerun the check. Carry the verified root
into every shell call that uses helpers; exports may not persist between
agent tool calls. Once the check passes, run `tao-setup` for host preflight,
credentials, and cross-skill discovery, then follow `tao-launch-workflow`.

For CLIP image and action configuration, read this skill's packaged
`references/skill_info.yaml`, `references/spec_template*.yaml`, and
`schemas/<action>.schema.json` when present. For a manual CLIP image lookup,
use `references/skill_info.yaml`; do not use the generic launch workflow's
`config.json` fallback. This skill does not package `config.json`,
`defaults.json`, or `references/model_info.yaml`.

If the checkout check fails, stop before launch. A manual image lookup or
Docker invocation does not replace the shared helpers required to bind the
results directory to a job record before submission. This prerequisite is
scoped to CLIP launches from this skill.
