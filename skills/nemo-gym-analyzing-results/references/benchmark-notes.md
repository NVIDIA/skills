# Benchmark Notes

Known differences between Gym benchmarks and the published methods they follow, and traps seen when comparing their numbers. The installed nemo-gym's code and config win over these notes and over each benchmark's README (on GitHub under `benchmarks/<name>/`), which can lag behind the code.

## Terminal-Bench 2.1 (benchmark `terminal_bench_2_1`)

- After 0.7.0, the `terminus_2` and `opencode` presets run the published task folders through the Harbor environment server, 8 rollouts per task. Each task's agent budget and resources come from its own `task.toml`, capped by the harness's 10,800-second limit. The deprecated `*_legacy_agent` presets run the same tasks through the resources server; scores on the two paths are not interchangeable.
- On the Harbor path, once the task's tests have run, a missing reward file scores 0. Infrastructure endings (a model-server or transport failure the harness reports, a sandbox or provider failure, a lost sandbox) are retried or masked. A model request the endpoint rejects with HTTP 400, 413 or 422 is graded as the model's result. A sandbox killed for running out of memory is a masked or failed sample, not a 0.
- `qemu-startup` and `qemu-alpine-ssh` score 0 on every rollout because their published images fail their own tests: 2 of 89 tasks. When comparing with a source that repairs them, exclude them on both sides (`--exclude-tasks qemu-startup,qemu-alpine-ssh` on the Harbor presets) and say so.
- Timeouts are wall-clock, so throughput and concurrency change scores. Artificial Analysis publishes its own timeout and episode limits; compare them with the run's settings before comparing scores.

## Terminal-Bench 4 (benchmark `terminal_bench_4`)

- After 0.7.0 the benchmark is Terminal-Bench 4.1: the 66 tasks of 4.0 with new verifiers. 4.1 scores and `partial_credit` are not comparable with 4.0, which is what 0.7.0 ran.
- The profile follows Artificial Analysis's prompts, 500-step limit and 30-second command timeout; Artificial Analysis's methodology names 4.0. It defaults to 1 repeat. Set `++num_repeats=3` to follow Artificial Analysis's protocol of averaging pass@1 over three runs; the official leaderboard uses five attempts.
- Long command output is cut to its first and last 5,000 characters. Runs from before NVIDIA-NeMo/Gym#3624 showed full output, so they are not comparable with later ones.
- After 0.7.0 the default `miniswe` preset runs the task folders through the Harbor environment server (NVIDIA-NeMo/Gym#5697); the deprecated `miniswe_legacy_agent` preset keeps the resources server's relay. Treat runs on the two paths as separate measurements.
- A verifier that runs out of its own time budget, or a missing, empty or malformed reward file, scores 0 and stays in the denominator. On the Harbor path these rows carry `failure_kind` `harbor:verifier_timeout`, `harbor:missing_reward` or `harbor:invalid_reward`; on the legacy path, `evaluation_completed: false`. After 0.7.0, a test script that exits 75 without a valid reward is masked as `harbor:verifier_unavailable` instead, unless it ran in the agent's own sandbox. Provider or socket timeouts and sandbox startup failures are infrastructure failures and leave the score. Runs from before NVIDIA-NeMo/Gym#5038, 0.7.0 among them, left verifier timeouts out of the denominator, so they are not comparable with later runs.

## tau2 and tau3 (benchmark `tau2`)

- `banking_bm25_grep_artificial_analysis.yaml` runs the 97 banking tasks for 8 repeats, with a `gpt-5.4-mini-2026-03-17` user simulator at medium reasoning and at most 200 steps.
- Prepared rows drop `NL_ASSERTION` from the reward basis, so there is no NL-assertion judge. Only one banking task uses it, but scores are not strictly identical to Artificial Analysis.
- Agentic tau runs vary by several points between runs. Check repeats before reading a gap.

## GDPval (benchmark `gdpval`)

- Rubric mode is the default; the reward is the judge's overall score.
- Comparison mode reports an Elo against reference models, with `normalized_elo = (elo - 500) / 2000`. Compare only with numbers computed against the same reference models and the same reference outputs.
- The judge is sampled from a panel per call. The default is 1 repeat.
- In comparison mode, the submission is judged against every saved output of every reference model for the task (one matchup each; a reference with none for the task is skipped silently), `num_comparison_trials` times per matchup. If any judge call in a matchup fails, or none of its replies can be read, the whole matchup is dropped and listed in `judge_response.ref_errors`, and the row is scored from the other matchups. So runs with the same completed rollouts can rest on different numbers of judge votes. A complete row has empty `ref_errors`, `total_invalid` (unreadable replies) of 0, `total_judged` equal to `num_comparison_trials` × `ref_repeat_count`, and no `judge_response.manual_imputation` (`eval_missing_as_loss` or `model_failure_as_loss` mark a loss scored without a judgment).
- After 0.7.0, when every matchup fails for a reason a retry can fix, the row becomes a masked judge failure that keeps the saved answer, and recovery grades it again; a payload the judge will always refuse becomes a terminal `permanent_judge_error`. In 0.7.0 too, a payload that no judge can take becomes `transport_ineligible`, and `--judge-failed-only` refuses multi-stage rows; resume the multi-stage collection instead.
- An agent that runs out of wall time before it delivers is scored as the model's result: 0 in rubric mode and a loss against every reference in comparison mode, so it stays in the score and the Elo; recovery reruns a timeout whose window held a server-side fault. Runs from before NVIDIA-NeMo/Gym#6077 left these timeouts out of the score, so they are not comparable with later runs.

## HLE (benchmark `hle`)

- 2,158 text-only questions, 1 repeat. The dataset revision is not pinned.
- The judge is the policy model unless the config overrides it, and no judge temperature is set. Ask which judge produced each number before comparing.
- The server emits `mean/reward` and a judge parsing-issue rate. An unparseable verdict counts as wrong; a failed judge call becomes `judge_failed`. Runs from before NVIDIA-NeMo/Gym#5865, 0.7.0 among them, searched for the exact label `Judgement: yes` or `Judgement: no`, so a verdict in other capitalisation or with markup inside the label (`Judgement: **yes**`) counted as wrong, while markup around the whole label was read; they are not comparable with later runs.
- HLE-Diamond (benchmark `hle_diamond`, after 0.7.0) is a 1,000-question subset of HLE, 272 of them with images; its text presets run 728. Compare it only with HLE-Diamond numbers over the same rows.

## LiveCodeBench (benchmark `livecodebench`)

- v5 covers problems dated 2024-08-01 to 2025-03-01 (279 problems); v6 covers 2024-08-01 to 2025-06-01. Both run 3 repeats, from a dataset pinned to a Hugging Face PR ref.
- Think tags left in the final answer, or more than one `<think>` or `</think>` in the reasoning, replace the reward with `reasoning_format_penalty` (default 0), so a correct solution scores 0.
- Grader fixes changed scores: stdin handling (NVIDIA-NeMo/Gym#2508, NVIDIA-NeMo/Gym#2824), and `sys.stdout.buffer` writes and a 1e-6 float tolerance (NVIDIA-NeMo/Gym#2956), so runs from before them are not comparable with later ones. NVIDIA-NeMo/Gym#3051 also accepts any capitalisation of a Yes/No answer.
- Other LCB harnesses may still fail correct solutions that use fast I/O (`sys.stdin.buffer`, `sys.stdout.buffer`).

## GPQA (benchmark `gpqa`)

- GPQA-Diamond, with options shuffled by an md5(question) seed, the Artificial Analysis-style four-choice prompt, `Answer: X` grading, `max_steps: 1` and 8 repeats.
- Other implementations differ in prompt, option order and answer parsing. Compare like with like.
- The `gpqa_diamond` resources server is a separate training environment with a different verifier.

## SWE-bench Verified and Pro (benchmark `swebench`)

- Reward is whether the instance resolved. After 0.7.0, a Verified evaluation that fails for a reason other than the model's patch (a sandbox error, a patch-extraction command that failed to execute, or a test run that did not finish) is masked as a `verifier_error` and leaves the score. An empty patch from a repository the agent broke is the model's 0, and so is a patch the harness could not apply, although that one has `evaluation_completed: false`. Pro has its own rules; for example a lost patch capture becomes `session_lost`. Pro first re-runs an inconclusive verification on a new sandbox up to 2 more times, each attempt capped at 1,200 s and all of them at 2,700 s. After 0.7.0, one still inconclusive after that is masked as a `verifier_error`; 0.7.0 scored it 0, so its Pro scores are not comparable with later ones.
- Patch application falls back from `git apply` variants to `patch` with fuzz 5.
- The Verified dataset is not pinned; Pro (731 instances) is. Repeat counts differ by profile (3 for the OpenCode profiles, 1 for others such as Hermes and NOOA), so check the profile.
- Validate the environments with golden patches before chasing a gap to a published number, and check that identical patches get identical rewards across runs.

## AA-Omniscience (benchmark `omniscience`)

- Everything in this section is after 0.7.0 (NVIDIA-NeMo/Gym#5891, NVIDIA-NeMo/Gym#5783, NVIDIA-NeMo/Gym#6541). 0.7.0 ran the unpinned v1 public set with `openai/gpt-oss-20b` as judge (2,048-token cap). When a judge reply was not a bare letter, it took A (correct) if a capital A appeared anywhere in it, then B, C or D, so a reply such as `NOT_ATTEMPTED` counted as correct. Its numbers are not comparable with later ones.
- Uses the AA-Omniscience v1.1 public set (300 questions). Scores are not comparable with runs on the v1 public set (600 questions, a different question per id), and Artificial Analysis publishes scores on its 3,000-question set.
- The dataset revision is pinned. The default judge is GPT-5.6 Luna at medium reasoning effort and temperature 0, with a 32,768-token judge output cap and 8 repeats. A different judge model gives different scores. The earlier default, `openai/gpt-oss-20b`, has been retired on `integrate.api.nvidia.com`; its numbers are not comparable with Luna's.
- The grader expects the judge to reply with a single letter. It also accepts a bare letter on the last line and an explicit verdict at the start of a line (`Grade: A`, `NOT_ATTEMPTED`, `C: PARTIAL_ANSWER`) followed by nothing, closing punctuation or a short reason. A bare letter on the last line wins over earlier grades, unless leftover think tags are present. Without one, conflicting explicit verdicts, a verdict line whose other letters contradict the grade, or no accepted verdict make a `judge_failed` row: it is masked and graded again once from the saved answer at the end of the run, not counted as incorrect. A row that fails again stays out of every metric, so the run is incomplete: `gym eval run` exits 0 with `complete: false` in the coverage file, and a strict run (`gym eval submit` by default) stops with no aggregate. Many `judge_failed` rows mean the judge does not follow the prompt; read the rejected replies in `failure_reason` of `<stem>_failures.jsonl` (their first 200 characters) before trusting the score.

## AA-LCR (benchmark `aalcr`)

- 100 tasks × 16 repeats. The default (v1.0) judge is Qwen3-235B-A22B-Instruct-2507-FP8. `config_v1_1` is the recommended config: the v1.1 grading update that Artificial Analysis introduced, with a newer dataset revision and a different judge. Compare it only with v1.1 numbers.
- An empty answer scores 0 without a judge call.
- v1.0 counts a malformed judge reply as wrong; v1.1 marks it `judge_failed`, which leaves the score. An empty judge reply is `judge_failed` in both.

## MMLU-Pro (benchmark `mmlu_pro`)

- A 10-choice prompt with `Answer: X` grading. Other prompts or answer extraction give different numbers, and MMLU-Pro is not MMLU.
