# External platform skills — conformance and inferred mappings

There is no registry and no interface file: **a platform skill declares the
contract by documenting it, and you verify it by reading.** Any installed skill
(in-bank or external — e.g. a `tao-run-on-kratos` from another repo) is a
conformant TAO platform iff its SKILL.md documents, for its native CLI:

1. **submit** that opens the job record BEFORE launching (record-then-launch)
   and names/labels the backend object after the id;
2. **status** that maps native states to the fixed vocabulary
   `PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`;
3. **logs** and **cancel** (cancel must also tear down orphans and mark the
   record).

Before first use of an unfamiliar platform skill, read its execution section
and check those four points. The record's `--platform` field is an **open,
format-validated slug** (the skill's short name, e.g. `kratos`) — never an
allowlist to be edited.

**Not explicitly conformant? You may INFER the mapping** from the skill's
native primitives (create/run/inspect/stop), under three rules:

1. The bank's invariants still bind: open the record first, **name/label the
   backend object after `$JOB_ID`** (what makes inferred status/cancel findable
   later), and enumerate teardown — including what stops **billing** — before
   you submit.
2. State the inferred mapping in the launch review (`submit=…, status=…→vocab,
   logs=…, cancel/teardown=…`) so the user confirms it with the launch, and
   smoke a cheap job before real GPU-hours.
3. Persist what worked — write the mapping into the run workspace and propose
   it as a skill addition after first success — so later sessions reuse it
   instead of re-deriving it differently.

If a verb has **no native equivalent** (no way to observe or stop the work),
do not run tracked jobs there. This inference path is how `tao-run-on-brev`
came to exist: instance-lifecycle docs only, mapping inferred once, validated,
then codified.
