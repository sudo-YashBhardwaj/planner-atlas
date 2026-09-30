# Scientific ledger

The project's claims by status, as of 2026-09-24 (corrected 2026-09-30). Each entry names its evidence and its standing:
**confirmatory** (pre-registered test), **descriptive**, or **exploratory**. Numbers are PushT with
the released LeWM checkpoint unless stated.

## A. Confirmed findings

1. **Planner-selected repair improves matched-planner ranking relative to random repair.**
   *Confirmatory.* At 40,000 transitions and equal repair compute, planner − random planner-region
   top-choice regret = −0.00826, 95% CI [−0.01176, −0.00476], 11/12 seeds, exact sign-flip
   p = 0.0015 (pre-registered). Produced by `2d6364d` (tag `results-confirm2`); `docs/report.md`.
   Scope: the acquisition planner and the evaluation pool are the same P2 CEM configuration.
2. **Optimization selects high-error, optimistic trajectories.** *Descriptive, every seed.*
   Planner-selected acquisition has 2.6× random's rollout error and signed optimism +48.4 against
   +8.8; uncertainty selection finds comparable error but less optimism and task relevance. Stage 1
   shows the same concentration in the predicted-good tail (top-1% − bottom-50% optimism +29.0,
   41/48 cases, d_z 1.05). Its G2 found no excess over an exchangeable-error null in how decision
   regret grows with search (random shooting over a fixed proposal; CEM was not tested).
3. **Random repair improves closed-loop MPC more than planner repair.** *Secondary, descriptive.*
   MPC semantic success: random +0.099 over the continued control (sign-flip p = 0.0015); planner
   +0.010, CI spanning zero.
4. **Held-out prediction loss does not track either downstream metric.** *Descriptive.* It orders
   the repaired branches differently from both planner-region regret and MPC success.
5. **Deeper search improves true task outcomes even as latent regret rises.** *Descriptive, Stage 1.*
   J_selected improves with fixed-q0 search depth (slope CI excludes zero); adaptive CEM beats
   compute-matched random shooting in true task cost by 0.096 at 1,536 evaluations.

## B. Falsified or unsupported hypotheses

1. **Planner-conditioned reliability (E1): closed before any outcome.** Two pre-declared validity
   gates for a CEM-P2 vs RS-1024 matched/cross matrix failed on the consumed manifest: the
   `q_final` proposal gate (median per-dimension Bhattacharyya 0.909, threshold ≤ 0.88) and the
   decision-set gate (median normalized energy distance 0.0995, threshold ≥ 0.10). In the 50-d
   action space the planners differ in search depth, not in where they search. These calculations
   were outcome-free and base-model-only; they were run in-session and are not stored on disk.
2. **Optimization-depth decision-reliability failure (Stage-1 ICML thesis): NO-GO.**
   Pre-registered (`ebdfa8a`, `8b47d24`), executed at `65d3f00`. G1 failed: latent-regret slope
   +4.02, 28/48 positive (needed 34), d_z 0.29, Monte-Carlo p 0.051. G2 (excess over the
   exchangeable null) −2.2, 20/48; G3 (task regret) 28/48, d_z 0.23.
   `docs/results/optimization_depth_stage1.md`.
3. **Structured exploitation beyond the ordinary optimizer's curse.** Not supported (Stage-1 G2).
4. **Task-level harm from deeper search.** Not supported (Stage-1 G3; selected task cost improves).
5. **Cross-planner generalization of the repair ranking gain.** Not established: planner − random
   on each branch's own P3 plan is −0.0039, p = 0.25 (exploratory).
6. **Forgetting explains the repair control gap.** Not supported (exploratory replay-weight
   analysis: held-out loss recovers at lower planner weight while planner MPC stays level).

## C. Open questions

1. **Why does random repair improve MPC?** Unresolved. Random data is less task-concentrated and
   drifts further from the base latent distribution; an untested hypothesis is that it covers
   off-nominal recovery states MPC needs.
2. **Replay-weight behaviour** (*exploratory*, run after the confirmatory result against its stopping
   rule): more planner-data weight steadily improves planner-region ranking with MPC level; more
   random-data weight raises MPC with ranking flat. Not confirmed.
3. **Budget dependence of the repair effect.** Established only at 40,000 transitions; smaller
   budgets appear only in a superseded exploratory grid.
4. **Generality.** All repair and Stage-1 findings are one environment (PushT), one released model
   family and CEM/random-shooting planners.
