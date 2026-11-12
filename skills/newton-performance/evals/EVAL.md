# Evaluation Scope

This smoke suite measures routing and the measurement workflow of the
performance skill for the Newton wheel. It covers explicit, implicit,
contextual, multi-turn, and negative prompts. Tasks request reviews, plans, or
instructions only; they do not require a GPU or execute simulations. These
cases do not establish measured speedups or a supported Newton version range.

## Behaviors

- Separate timing scopes such as startup, warm stepping, and rendering, and
  keep fidelity checks alongside every measurement.
- Use bounded coupled sweeps when settings interact.
- Report missing workloads or prerequisites, and label proposed experiments
  separately from collected measurements.
- Find graph-capture and host-synchronization problems in a step loop and give
  a correct fix; any fix supported by the Newton documentation receives credit.
- Size collision and solver buffers with the correct per-world or total scope.
- Leave setup, solver and API questions, correctness debugging, and Warp-only
  kernel work to other skills, including after a conversation moves out of
  performance work.

## Notes

Run paired trials with and without the skill in otherwise identical environments
on both Claude Code and Codex. Use the same public documentation access in both
arms. Repeat trials to measure variance. Report task correctness, routing,
total token consumption including skill loading and retries, and elapsed time.
Assertions judge the task outcome; routing is measured separately through
`expected_skill`, and the baseline is judged on the same outcome assertions.

The multi-turn cases encode the earlier turns in the prompt. The startup and
coupled-sweep cases are adapted from the `newton-optimize-simulation` proposal.
No live trial results exist yet for this version of the skill. Keep larger
executable performance evaluations separate from this routing suite.
