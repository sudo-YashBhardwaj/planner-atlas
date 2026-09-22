# Replay-mixture experiment (mechanistic, not confirmatory)

Recorded 2026-09-21, before any replay-mixture branch was trained.

Question: does the planner-targeted repair trade planner-region reliability against broader
predictive quality and closed-loop control as a function of the replay weight alpha on newly
acquired data?

- Streams: the existing confirmatory B=40,000 random and planner streams, seeds 4..15. No new
  environment interaction; uncertainty is not part of this experiment.
- New branches: alpha = 0.125 (112 base + 16 acquired per batch) and alpha = 0.250 (96 + 32), for
  random and planner, all 12 seeds: 48 branches.
- Reused, because produced at the same commit 2d6364d with the identical training and evaluation
  protocol: alpha = 0.000 (continued, 128 + 0) and alpha = 0.500 (64 + 64) from
  runs/confirm2/grid.jsonl.
- Fixed: base checkpoint, 1000 steps, AdamW lr 5e-5 wd 1e-3, clip 1.0, batch 128, training seed
  equal to the stream seed, frozen normalization statistics, dropout, live-replay latent cache,
  held-out subsample seed 909, confirmatory manifest 5e945883 (seed 307), P3 evaluation planner,
  all metric definitions.
- Outputs: per alpha and strategy, seed-level mean and uncertainty of planner-region regret, q0
  regret, MPC semantic success, held-out prediction loss, |optimism|, rollout error, terminal
  error; and the relationship between planner-region reliability and MPC across alpha.
- Rule: no alpha is selected post hoc as a result; the object is the shape of the trade-off. No
  follow-up experiment is launched from this one.
