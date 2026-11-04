## Tune across runs

Default to **three completed runs** when meaningful hypotheses and resources permit: one baseline and up to two single-group changes. After each run, inspect metrics and video, formulate one promising hypothesis, change only that group, then launch the next run. If no credible change remains, or a prerequisite/error blocks safe progress, stop early and explain why. Respect an explicit user budget even when it permits fewer than three runs.

Follow the guide's order. Do not move to a later group before checking earlier groups relevant to the observed failure:

1. **Asset and physics quality:** inspect plug/socket geometry, contact, collision meshes, solver behavior, grasp, and whether the final 1–2 cm are physically reachable. Do not treat a reward change as a cure for impossible contact.
2. **Curriculum depth and anneal:** adjust `at_goal_depth_range`, `approach_depth_range`, `at_goal_prob`, `at_goal_prob_final`, or `anneal_end_iter`. If success collapses as near-goal starts disappear, slow the anneal or narrow approach depth. Keep the curriculum's step divisor matched to PPO rollout length.
3. **Linear versus exponential reward:** compare `plug_socket_keypoint_tracking.weight`, `plug_socket_keypoint_tracking_exp.weight`, and `kp_exp_coeffs` at a matched curriculum stage. Use terminal success and contact video to choose.
4. **Socket pose domain randomization and observation noise:** change only one of these at a time, after a stable baseline, and record the realism versus convergence trade-off.
5. **Action scale:** tune when motion is too coarse, too slow, or cannot seat the plug; inspect contact and action-rate penalties.
6. **PPO training length:** extend `max_iterations` if learning still improves, or adjust learning settings only with a specific instability hypothesis.

Compare runs at the **same reset distribution or a fixed evaluation condition**. Training success under 80% near-goal resets is not directly comparable to success under 0% near-goal resets. Prefer `Metrics/terminal_success_rate` over a single dense success reading; also report mate-point error, termination causes, and reward curves. For hyperparameter selection, start each candidate from scratch by leaving `parent_checkpoint` unset, and match the seed, training budget, and evaluation conditions. This isolates the changed hyperparameter from pretraining history. If the explicit question is fine-tuning a trained policy, branch all candidates from the same parent checkpoint and label those results as fine-tuning. Never compare a warm-started policy with a fresh policy as if their training cost or starting capability were the same. Disclose stochastic variation and never select a winner solely from a different curriculum mix or one video.

To test whether more training helps, first evaluate intermediate checkpoints from one long run; this gives a direct learning curve without another training initialization. A checkpoint continuation is optional, but RSL-RL interprets `max_iterations` as **additional** updates and resumes update labels at the checkpoint's saved `iter` value. The supervisor reads that value, verifies the shifted final checkpoint, and reports the parent. The simulator and reset curriculum are newly created when continuing, so explicitly set the new anneal schedule or final reset distribution; continuing the model does not continue the old environment state.

## Independent validation

After selecting a checkpoint, use the supervisor's
[independent validation workflow](supervisor.md#independent-validation-with-fresh-seeds)
to evaluate it on unused seeds within the available evaluation budget. A request
for validation alone reuses that checkpoint without starting another training run.
Keep independent results separate from the screening/tuning results used to select it.

## Final report

Report the repository/version checks, exact run commands and log link, a table of changed hyperparameters and outcomes for every completed run, failed/restarted attempts, best checkpoint and validated videos/exports, elapsed time, and the remaining limitation. Distinguish a promising simulation result from demonstrated full-approach insertion and hardware performance.
Deliver the report to the user when the series finishes. State actual training time separately from idle gaps and wall time, and mark any unavailable trainer exit status instead of silently treating artifact verification as process success.
