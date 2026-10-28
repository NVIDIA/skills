<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# Maintain the generated SDK references

`SKILL_ROOT` is the absolute installed Doctor skill directory, after resolving
symlinks; `PVA_SDK_ROOT` is the selected SDK. `SDK_REFERENCES` is the absolute
workspace corpus path recorded in Doctor's report, normally
`$WORK_DIR/references/sdk`. Generated references are reproducible output, but
`sync` replaces that directory. Setup creates or refreshes references within
the authorized workspace. Doctor runs read-only checks and recommends Setup
when references are missing, outdated, or damaged; it never runs `sync`.

Keep the output outside installed skill directories and the SDK documentation
tree, and away from their ancestors and the input manifest. Before writing,
`sync` resolves symlinks and checks these protected paths. Runtime references
belong in the operator workspace, so installation contents stay unchanged.

## Versions and upgrades

These are independent package revisions, not SDK release numbers:

| Counter | Current value | Changes when |
| --- | ---: | --- |
| Manifest `schema_version` | 5 | The accepted input manifest structure changes. |
| Lock `schema_version` | 5 | The generated lock structure changes. Revision 5 makes `converter_version` an integer. |
| `converter_version` | 10 | Discovery or generated Markdown behavior changes. Revision 10 handles HTML-extension case and uses the materializer's new name. |

All three materializer counters are integers. They advanced during package
development; resetting them to 1 could make an old cache appear compatible.
The input manifest stays at 5 because its structure has not changed. Other
contracts have independent versions:

| Contract | Current `schema_version` |
| --- | --- |
| Operator specification | Integer `2` |
| Evidence | Integer `2` |
| Test matrix | Integer `1` |
| OpGen handoff | Integer `3` |
| Build-Test result | String `"2.0"` |

Regenerate older specification, evidence, and handoff files through Spec using
the current templates and the approved requirements and test cases. The
finalizer rejects unsupported versions; manually changing a version number
does not migrate a file. New Build-Test runs use the current result template.
Keep historical reports in their original formats. These workflow revisions
do not require regenerating an otherwise valid SDK-reference corpus.

`verify` rejects an old lock or converter revision. Run `pva-setup` for the
selected SDK and work directory, then rerun Doctor. Setup uses the current
script and manifest rather than editing a lock or its hashes by hand. It checks
Python 3.9+ and `pdftotext` before running these materializer commands:

```bash
mkdir -p "$(dirname "$SDK_REFERENCES")"
python3 "$SKILL_ROOT/scripts/materialize_sdk_references.py" sync \
  --manifest "$SKILL_ROOT/references/sdk-doc-sources.json" \
  --sdk-doc-root "$PVA_SDK_ROOT/share/doc" \
  --output-dir "$SDK_REFERENCES"

python3 "$SKILL_ROOT/scripts/materialize_sdk_references.py" verify \
  --manifest "$SKILL_ROOT/references/sdk-doc-sources.json" \
  --sdk-doc-root "$PVA_SDK_ROOT/share/doc" \
  --output-dir "$SDK_REFERENCES"
```

After regeneration, rerun Doctor to record the verified location and lock hash.
If Spec already finalized a handoff, regenerate it through Spec because the
Doctor report is a hashed input. For older reports without `sdk_reference_root`
and `sdk_reference_lock_sha256`, run Doctor in the operator workspace; do not
reuse an unrecorded corpus under an old skills checkout.

## Skill-name upgrades

The `jetson-pva-*` skills were renamed to `pva-*`. Update `SKILL_ROOT` to the
installed `pva-doctor` directory and keep `SDK_REFERENCES` pointed at the existing
workspace corpus. The renamed `skill_routes` keys change the manifest hash, so
`verify` rejects a corpus generated with the old names even when the SDK is unchanged.

Run Setup to regenerate the index and lock with the new route names using
the commands above. Rerun Doctor afterward and regenerate any
Spec handoff that hashes the old Doctor report. This rename does not change the
manifest or lock schema versions.

## Recover an interrupted sync

The replacement step temporarily renames the old corpus to
`$WORK_DIR/references/.sdk.backup-<pid>`. Conversion uses a separate
`$WORK_DIR/references/.sdk.staging-<suffix>` directory. Both are siblings of
`SDK_REFERENCES`; a non-default corpus basename changes their names accordingly.
A crash may leave either. Setup uses the following recovery steps; Doctor
reports the failure and leaves these paths unchanged.
Only a backup matching the current process ID blocks the replacement check;
a leftover from a different PID does not block every later sync.

1. Confirm no materializer is still running against this output directory.
   Inspect the exact destination and leftover paths; do not remove them by glob.
2. If `SDK_REFERENCES` exists, run `verify`. A passing corpus can be used even when
   an older backup remains. Preserve the backup until the recovery is confirmed.
3. If `SDK_REFERENCES` is absent and a known backup is the intended corpus, restore
   that exact backup to `SDK_REFERENCES` and verify it. With multiple candidates,
   do not guess which one is authoritative; preserve them and regenerate from
   the selected SDK instead.
4. If verification fails or a same-PID backup blocks replacement, preserve the
   leftover backup at an explicitly selected unused path outside `SDK_REFERENCES`.
   Run the authorized `sync` and `verify` commands above. Never treat an abandoned
   staging directory as a complete corpus.
5. After verification succeeds, retained backup/staging directories can be
   archived or removed as part of the authorized cleanup. Leave SDK sources
   and unrelated files untouched.
