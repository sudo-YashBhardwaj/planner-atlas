# Confirmatory experiment: planner-selected vs random acquisition at B=40,000

Pre-registered 2026-09-20, before any confirmatory branch was trained or evaluated.

## Hypothesis (primary, one comparison)

At B = 40,000 true environment transitions, planner-selected acquisition produces **lower
planner-region top-choice regret** than random acquisition after identical repair compute.

- Primary comparison: `planner B=40k` vs `random B=40k`
- Primary metric: planner-region top-choice regret (mean over the 48 manifest cases)
- Unit of replication: independent acquisition/training seed
- Seeds: 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 (12 seeds, none used by the exploratory grid,
  which used seeds 0, 1, 2 and remains reported separately as exploratory)
- Direction: planner minus random is expected to be negative

## Branches per seed

`continued B=0`, `random B=40k`, `uncertainty B=40k`, `planner B=40k`.
`continued` and `uncertainty` are secondary controls and are not part of the confirmatory claim.
The untouched released checkpoint is evaluated once as a reference, not per seed.
No 2,500 or 10,000 budget branches.

## Frozen (unchanged from the exploratory protocol)

Model architecture; frozen visual representation; frozen BatchNorm running statistics
(`frozen_rep_v2`); action normalization; uncertainty definition (population variance across the
5 fixed members of predicted terminal goal cost); Random Shooting / CEM; P2 acquisition planner
and P3 evaluation planner; PushT semantic task; replay reconstruction; acquisition selectors;
AdamW lr 5e-5, weight decay 1e-3, grad clip 1.0, batch 128, 1000 optimizer steps; 50/50 base/new
replay mixture for positive-budget branches; B=40,000 budget; all evaluation metric definitions.

## Evaluation manifest

Fresh, frozen before results: 48 cases, manifest seed 202, drawn only from validation-split
episodes (episode split seed 0, validation fraction 0.1). 48 distinct episodes, zero episode
overlap with acquisition (which draws only from training-split episodes) and zero episode overlap
with the exploratory manifest. Digest:
`86ed349fab722bdcff4b6708089964169ebbc30acb103207517ba6cb01c42fef`.
Identical across every confirmatory branch; its digest is recorded in every output row.

## Analysis plan (fixed in advance)

Per seed, average the primary metric over the 48 cases, then take the paired difference
`planner - random`. Report: the 12 paired differences, mean, SD, SE, 95% confidence interval
(t critical value 2.201 for 11 degrees of freedom), a two-sided exact paired permutation test over
all 2^12 sign flips, Cohen's d_z = mean / SD, and how many of the 12 seeds carry the expected
negative sign.

Secondary metrics, reported descriptively and explicitly not as confirmatory findings: q0 regret,
MPC semantic score, absolute optimism, signed optimism, rollout error, terminal error, held-out
prediction loss.

## Stopping rule

Report the result whatever it is, including a null or reversed one. No follow-up study (in
particular no replay-mixture study) is launched from this experiment.
