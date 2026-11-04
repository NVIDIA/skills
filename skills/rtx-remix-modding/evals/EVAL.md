# Evaluating `rtx-remix-modding`

`evals.json` holds 14 cases: **11 that must activate the skill** and **3 that must not**.

Every case is derived from text this skill ships today, and each tests what the agent *says and
routes to* rather than whether a USD write landed. Nothing here needs a GPU, a running Remix or
a capture — which is the point, because the harness that scores it has none of them.

`skillevaluator` reads this directory from beside `SKILL.md` and publishes the
with-skill-versus-without lift on the skill card. That measured lift is what the dataset exists
to produce.

## What each case defends

| # | Rule under test | Failure it catches |
| --- | --- | --- |
| 1 | Edits are per captured asset, and the shared scope must be disclosed | Confirming the one door the user pointed at, and letting them find the other five on their own |
| 2 | There is no undo tool, and no hand-simulated substitute for one | Authoring `references = None` or emptying the replacement prim — a blocked reference composes to nothing, which deletes the asset from the game with no error anywhere |
| 3 | Remix cannot create geometry, by any route | Reaching for `CreatePrim`, which reports success and renders nothing |
| 4 | An empty result is the whole answer; never re-run an uninvalidated query | Running the same search again, spending the remaining turn budget on an answer already held |
| 5 | What a capture does not contain cannot be reached | Dropping to raw `pxr` traversal to look again — same stage, same nothing |
| 6 | Relay a tool's refusal with its remedy, and invent no cause | Reporting "impossible" when the tool said "do X and this works", or guessing at the user's setup |
| 7 | Missing `remix_*` tools are a gap to name, not to work around | Hand-editing `.usda`, which lands in a layer the runtime ignores — looks like success, renders nothing |
| 11 | Unsupported prim deletion routes to the Toolkit GUI, including after selection | Omitting the refusal, or substituting raw USD/filesystem edits, hide/off-world/empty-reference tricks, or whole-layer removal — same disappearing geometry, bypassed validation/undo, or collateral edits |
| 12 | A write that resolves to more than one target needs the user's confirmation of the set | Replacing all five disco-ball maps because "all" plus exact 4K twins looked unambiguous — no undo, and the user never saw the list |
| 13 | One named target needs no confirmation (control for 12) | Asking a needless scope question for a single named map, which would mean case 12's fix over-reached |
| 14 | A large set is summarized, not listed, and still confirmed | Writing forty materials' maps because the list was too long to show, or burying the question under every target |

These eleven are not a sample of the skill; they are the rules that cost real debugging to get
right. `.agents/rules/agent-skill-rules.md` records why the wording of each
one is load-bearing, and six of these cases sit directly on top of it: case 5 because three
prohibition-shaped phrasings were walked past before a factual one held, case 4 because the
repeat-query rule only survives with its exception stated second, case 1 because a competing
"be brief" instruction silently swallowed the disclosure, cases 3 and 11 because a generic
"decline when no tool exists" never fires when the agent can always reach raw USD and *do*
something, and case 12 because an exact replacement for every member read as the user's
confirmation.

Cases 8, 9 and 10 are the negatives: an LLM deployment question, a CUDA tiling question, and a
generic USD composition question. They carry `"expected_skill": null` and assert only that the
skill does **not** activate. They guard the `description` frontmatter, which is long and broad —
an overreaching description pulls the skill into unrelated conversations and costs context on
every turn. Tier 1's quality check flags the same risk statically; these three catch it in
behaviour.

## Passing

A case passes when every string in its `assertions` array holds for the agent's answer. The
assertions are natural language, so grading needs an LLM judge rather than string matching, and
`expected_output` is the reference answer that judge compares against.

Negative cases pass when the skill is not selected at all. They can only be graded by a harness
that performs skill *selection* — loading the skill and then asking whether it applies is a
different question, and will not catch an over-broad description.

## Running

The [GitLab CI configuration](../../../.gitlab-ci.yml) includes NVCARPS downstream skill
validation with the `external` profile and automatic signing enabled. Run the MR pipeline,
then follow its NVCARPS downstream validation report and job artifacts.
The external validation and signature-verification triggers use `needs: []` to start without
waiting for build, test, or publishing stages; their existing MR rules still apply.

Review the Tier 1 content checks, Tier 2 deduplication, Tier 3 agent evaluation, and SkillCritic
advisories separately. For agent evaluation, compare results with and without the skill and
inspect individual agents and cases rather than relying only on the best agent's summary.
Record the evaluated commit when reporting results.

For local reproduction, use the evaluator version, commands and environment from the CI job
trace and that version's documentation.

## Keeping it current

- Update cases in the same change as the skill or tool behavior they describe. Hypothetical
  observations must match the current API's responses and supported parameters.
- Keep routing expectations separate from the quality of answers to unrelated requests.
- Rerun downstream validation after edits. Existing generated `BENCHMARK.md`, `skill-card.md`,
  reports and `skill.oms.sig` do not establish results or a valid signature for edited content.
  Let the pipeline regenerate those artifacts; do not edit them by hand or claim improved
  scores before the new results exist.
