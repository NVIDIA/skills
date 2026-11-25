# Standalone CLIP skill setup

When `tao-finetune-clip` is installed individually, its folder does not
include the shared bank helpers used for launch preflight, image resolution,
and job records. Before running `tao-setup` or submitting a CLIP action, set
`TAO_SKILL_BANK_PATH` to the `nvidia-tao-toolkit` plugin folder, which holds
`skills/`, `scripts/`, `catalog/` and `versions.yaml`, not to the individual
installed skill directory and not to the root of a repository checkout. When
the variable is not already set (no session hook exported it, as in Codex),
set it to the plugin folder that contains this skill, the folder two levels
above this skill's folder, when that folder passes the check. The first lines
of the block do that once `<this-skill-folder>` is replaced with the folder
that holds this skill's `SKILL.md`. Verify it with:

```bash
# TAO_SKILL_BANK_PATH not set? Set it to the plugin folder that contains this skill: two levels above this skill's folder.
skill_dir="<this-skill-folder>"   # the folder that holds this skill's SKILL.md
if [ -z "${TAO_SKILL_BANK_PATH:-}" ] && [ -f "$skill_dir/../../catalog/skill-layers.json" ]; then
  TAO_SKILL_BANK_PATH="$(cd -P "$skill_dir/../.." && pwd -P)"
fi
export TAO_SKILL_BANK_PATH="${TAO_SKILL_BANK_PATH:-$HOME/tao-skill-bank/plugins/nvidia-tao-toolkit}"
for helper in check_tao_launch_preflight.py resolve_tao_image.py resolve_tao_model.py tao_job_record.py redact_secrets.py; do
  [ -f "$TAO_SKILL_BANK_PATH/scripts/$helper" ] || {
    printf 'Missing bank helper: %s/scripts/%s. Set TAO_SKILL_BANK_PATH to the nvidia-tao-toolkit plugin folder that contains this skill (two levels above the skill folder); a repository checkout without plugins/nvidia-tao-toolkit is not a bank.\n' "$TAO_SKILL_BANK_PATH" "$helper" >&2
    exit 1
  }
done
[ -f "$TAO_SKILL_BANK_PATH/versions.yaml" ] && [ -f "$TAO_SKILL_BANK_PATH/catalog/skill-layers.json" ] || {
  echo 'Incomplete TAO skill bank; a repository checkout without plugins/nvidia-tao-toolkit is not a bank. Stop before CLIP launch.' >&2
  exit 1
}
```

If no such folder exists, obtain approval to install the complete
`nvidia-tao-toolkit` plugin, then set the variable to the installed plugin
folder and rerun the check. A checkout of the repository is a bank only when
it contains `plugins/nvidia-tao-toolkit`,
and the bank is then that folder. Carry the verified root
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
