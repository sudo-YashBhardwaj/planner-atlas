# Stage 1 — optimization depth and decision reliability (exploratory go/no-go)

Pre-registered before any Stage-1 outcome was computed. Written at repository state `c9aae44`
(tag `project-final`) and committed on top of it; the implementation commit follows this one.

## 1. Study identity

| item | value |
| --- | --- |
| environment | PushT (`swm/PushT-v1`), 5 action blocks = 25 raw steps |
| evaluation manifest | **consumed** confirmatory manifest, seed 307, 48 cases, 48 distinct validation episodes |
| manifest digest | `5e9458839dccb124ccd94394853a8592b79b27c94a11c4819bbe040dd8b5d746` |
| base checkpoint | `weights.pt`, sha256 `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607` |
| dynamics protocol | `frozen_rep_v3` |
| repository state at writing | `c9aae44`, tag `project-final` |
| exact result-producing historical tag | `results-confirm2` → `2d6364d` (unchanged by this study) |

Explicit statements of scope:

- This is an **exploratory Stage-1 go/no-go study**, not a confirmatory experiment.
- **No fresh confirmatory manifest is consumed or generated.** The 48 cases are re-derived
  deterministically with seed 307 and verified against the digest above.
- **No model retraining and no repair training occur.** The released checkpoint is used unchanged.
- **No uncertainty analysis is performed.** The canonical five-member ensemble is `frozen_rep_v2`
  and HEAD correctly refuses non-`v3` checkpoints. **No compatibility shim will be added**, and the
  ensemble will not be retrained. This omission is deliberate and is recorded here.
- The already-confirmed PushT repair result (`results-confirm2`) is untouched prior evidence.

## 2. Scientific question

**Primary.** With the candidate proposal distribution fixed, does increasing search depth cause the
world model to become progressively less reliable at selecting the best available decision?

**Secondary.** Does any such degradation exceed what is expected from ordinary exchangeable-error
optimizer's curse?

**Task relevance.** Does the same degradation appear in true PushT task ranking, not only in the
latent planning objective?

**We are not testing whether optimism increases with N as a primary scientific claim.** Under
best-of-N selection with a non-degenerate error distribution, `E[optimism | selected]` increases
with N as a matter of order statistics, and the project's existing data already shows it (unselected
q0 plans average +9.0 optimism; CEM-selected plans at 1,024 evaluations average +48.0). Optimism
curves are therefore **sanity / characterization** results (§9), not gates.

## 3. Immutable RS candidate bank

For each of the 48 cases, one deterministic bank of **N_max = 4096** action sequences is drawn
i.i.d. from the planner's own initial proposal q0: `initial_proposal(horizon=5)` = N(0, I) over the
[5, 10] normalized action-block space, clamped to `model_action_bounds` (±4.759…4.870), sampled by
`sample_proposal` on the planning device with `torch.Generator(device).manual_seed(20260923 + c)`
for case index c = 0..47.

**Every RS-N condition is an exact prefix of this one bank.** Independent banks are never drawn for
different N.

- Full ladder (descriptive): **N ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}**
- Primary upper-ladder reliability range: **N ∈ {256, 512, 1024, 2048, 4096}**
- Additional exact prefixes for the compute-matched CEM analysis: **{256, 512, 768, 1024, 1280, 1536}**

The bank is immutable once written. Analysis consumes the cache and never reruns a planner.

## 4. True outcome cache

Every candidate is executed **once** in the simulator (reconstructing the case start by replaying
its episode, as everywhere else in this project) and the following are cached:

candidate action sequence; candidate index; model predicted terminal latent cost; true realized
terminal latent cost; signed optimism; absolute optimism; true PushT task cost; position and angle
error; rollout error; terminal error; semantic success flags; terminal executed latent (for the
geometric diagnostics); geometric/OOD diagnostics already canonical in the project; simulator
outcome cache identity.

**Metric conventions (verified against code and against existing result rows):**

| quantity | definition |
| --- | --- |
| predicted latent cost `Ĉ` | `Σ` over the 192 latent dimensions of (ẑ₅ − z_goal)² |
| true realized latent cost `C` | the same **sum**, with z₅ encoded from the executed final frame |
| signed optimism `o` | `C − Ĉ`; **positive = model was over-optimistic** |
| absolute optimism | `|o|` |
| rollout error | **mean** squared latent error over blocks 1..5 (not a summed cost) |
| terminal error | **mean** squared latent error at block 5 |
| true task cost `J` | position error / 512 + wrapped angle error / π |
| semantic success | block within 20 px and π/9 of its goal pose |

Note the deliberate asymmetry: costs are **sums** over 192 dimensions, rollout/terminal errors are
**means**, a factor of 192 apart.

## 5. Decision-reliability metrics

For case c and prefix N, with `î(N) = argmin_{i ≤ N} Ĉ_i`:

```
R_latent(N) = C_{î(N)}  −  min_{i ≤ N} C_i
R_task(N)   = J_{î(N)}  −  min_{i ≤ N} J_i
```

Both are ≥ 0 by construction. Reported separately, never conflated with ranking regret:

- selected task cost `J_selected(N) = J_{î(N)}`
- oracle-best task cost `J_oracle(N) = min_{i ≤ N} J_i`

`R_latent` is the primary decision-reliability endpoint: prediction and oracle use the *same* latent
objective, so the only difference is learned prediction versus true execution.

## 6. Primary case-level slopes

Over the upper ladder (N ≥ 256; x = log2 N ∈ {8, 9, 10, 11, 12}), per case, by ordinary least
squares:

```
beta_latent[c] = OLS slope of R_latent(N) on log2(N)
beta_task[c]   = OLS slope of R_task(N)   on log2(N)
```

Supporting endpoint contrasts (interpretable, not primary):

```
Delta_latent[c] = R_latent(4096) − R_latent(256)
Delta_task[c]   = R_task(4096)   − R_task(256)
```

Inference unit is the **physical case**, n = 48. The 8 (or 5) ladder points within a case are
repeated measures and are never treated as independent replicates.

## 7. G1 — latent decision-reliability gate

G1 passes only if **all** hold:

- mean `beta_latent` > 0
- **≥ 70% of cases with `beta_latent` > 0, i.e. ≥ 34 of 48**
- Monte-Carlo sign-flip p < 0.05
- Cohen's d_z ≥ 0.5

**If G1 fails: NO-GO.**

Statistics (identical machinery for every case-level statistic in this protocol):

- **Monte-Carlo sign-flip test**, B = 100,000 draws, `np.random.default_rng(20260923)`. The same
  ±1 draw matrix is used for every test, which is intentional and keeps each p-value independently
  reproducible. Report the two-sided p̂ and its Monte-Carlo SE `sqrt(p̂(1−p̂)/B)`. Exact enumeration
  is impossible at n = 48 (2⁴⁸ ≈ 2.8e14); **this test is never called exact.**
- **Exact binomial sign test** (two-sided) on the non-zero cases.
- **Exact Wilcoxon signed-rank** by dynamic programming over integer rank sums, valid when |d| has
  no ties; with ties, average ranks are used and the result is reported as approximate.
- Also: mean, median, SD, SE, 95% t interval, sign proportion, Cohen's d_z.

## 8. Empirical exchangeable-error null

For each case, with `o_i = C_i − Ĉ_i`, replicate b:

1. permute the observed residuals across candidate identities: `o_i^(b) = o_{π_b(i)}`;
2. synthetic predictions `Ĉ_i^(b) = C_i − o_i^(b)`;
3. for every nested prefix N: `î^(b)(N) = argmin_{i ≤ N} Ĉ_i^(b)`;
4. compute null selected optimism, `R_latent_null(N)`, and descriptively `R_task_null(N)`.

**10,000 residual permutations per case**, `np.random.default_rng([20260923, c])` for case index c.
No simulator execution or model inference is required, so the null is cheap.

### 8.1 Null caveats (recorded before any outcome)

**A. Null regret slopes may be negative.** Fixture B produced a null slope of −3.17 while the
observed slope was +0.45. A positive excess can therefore arise from the null falling rather than
the observation rising. **G1 and G2 are complementary and neither is interpretable alone:** G1 asks
whether observed ranking regret actually grows with search; G2 asks whether that degradation exceeds
the exchangeable-error baseline.

**B. The permutation preserves** the exact candidate bank, the exact true latent-cost values, the
exact empirical residual marginal (including its scale and tails) and the exact search ladder.
**It destroys** the coupling between residual and candidate identity, structured candidate-dependent
error, and heteroscedastic or local error structure tied to candidate identity.

**C. The null does not preserve predicted-cost variance**, because
`Var(Ĉ) = Var(C) + Var(o) − 2·Cov(C, o)` and the permutation drives the covariance term to zero.
Therefore `corr(C, o)` is recorded and reported **per case as a descriptive diagnostic only, never
as a gate**. Positive `Cov(C, o)` makes the null noisier and G2 potentially conservative; negative
`Cov(C, o)` makes it potentially anti-conservative.

## 9. G2 — structured-exploitation gate

```
E_null_beta_latent[c] = mean null R_latent slope over the 10,000 permutations
excess_beta[c]        = beta_latent_observed[c] − E_null_beta_latent[c]
```

G2 passes only if **all** hold:

- mean `excess_beta` > 0
- **≥ 65% of cases with `excess_beta` > 0, i.e. ≥ 32 of 48**
- Cohen's d_z ≥ 0.5
- Monte-Carlo sign-flip p < 0.05

**G2 PASS:** decision-reliability degradation exceeds what an exchangeable optimizer's-curse
baseline with the same empirical residual marginal explains. **G2 FAIL:** the search-depth effect is
adequately explained by textbook regressional Goodhart. **If G1 passes but G2 fails: NO-GO for the
ICML extension.**

A ratio such as `observed / null` is explicitly **not** used, because ratios are unstable when the
null slope is near zero. The paired excess is used directly.

## 10. G3 — true task-relevance gate

On `beta_task`, G3 passes if:

- mean `beta_task` > 0
- **≥ 60% of cases with `beta_task` > 0, i.e. ≥ 29 of 48**
- Cohen's d_z ≥ 0.3

The Monte-Carlo sign-flip p is reported, but statistical significance alone is not the gate, because
task cost is noisier than the latent objective.

Frozen interpretations:

| | selected true task cost | `R_task` | reading |
| --- | --- | --- | --- |
| **A** | worsens | worsens | outright decision harm |
| **B** | improves | worsens | search remains useful, but the model captures progressively less of the additional opportunity exposed by deeper search |
| **C** | improves | falls | deeper search genuinely improves model-mediated decisions; weakens or falsifies the reliability-failure framing |
| **D** | — | flat | latent ranking degradation is not clearly task-relevant |

For a Stage-2 GO we want **A or B**.

## 11. Overall GO / NO-GO matrix (frozen)

| case | G1 | G2 | G3 | verdict |
| --- | --- | --- | --- | --- |
| 1 | PASS | PASS | PASS | **STRONG GO** — later proceed to an independently preregistered second-environment replication |
| 2 | PASS | PASS | FAIL | structured latent failure, weak downstream relevance → **default NO-GO** for the current ICML thesis; potential TMLR/workshop framing |
| 3 | PASS | FAIL | — | effect explained by ordinary optimizer's curse → **NO-GO** |
| 4 | FAIL | — | — | no meaningful decision-reliability degradation → **NO-GO immediately** |
| 5 | PASS | PASS | PASS, with absolute selected task cost still improving | **not contradictory**: search remains useful while ranking efficiency degrades relative to growing candidate opportunity |
| 6 | `R_latent` and `R_task` both decrease with N | | | deeper search improves model-mediated decisions → **NO-GO** |

**No reinterpretation after results.** A NO-GO is reported as a NO-GO and is not rescued with
another planner, another metric, another instrument or another environment.

## 12. Sanity / characterization results (explicitly not gates)

- **S1** selected signed optimism vs log2 N
- **S2** absolute optimism vs log2 N
- **S3** top predicted 1% vs bottom 50% signed optimism (paired, case-level)
- **S4** full rank-conditioned reliability curves

Rank bins are by base-model predicted cost, rank 1 = best predicted, **disjoint and exhaustive** over
the 4096-candidate bank:

| bin | ranks | size |
| --- | --- | ---: |
| 0–1% | 1–41 | 41 |
| 1–5% | 42–205 | 164 |
| 5–10% | 206–410 | 205 |
| 10–25% | 411–1024 | 614 |
| 25–50% | 1025–2048 | 1024 |
| 50–100% | 2049–4096 | 2048 |

Per bin: signed optimism, absolute optimism, true latent cost, predicted latent cost, true task cost,
rollout error, terminal error, success fraction, and a ranking statistic (pairwise rank agreement of
predicted against true cost) where meaningful.

These are expected to show optimizer's-curse behaviour. Their success proves little novelty; their
failure would indicate a bug, unusual model behaviour, or a problem with the assumed mechanism.

## 13. Geometry / OOD diagnostics (descriptive, not a gate)

The causal control is already in the design: **the candidate generator q0 is fixed across all RS-N**,
so only selection pressure varies. The selected order statistic may of course move geometrically;
that movement is measured honestly rather than denied.

Descriptive diagnostics, all already canonical in this project: latent nearest-base distance and
Mahalanobis-to-base against the 100,000 training-split latents of the live cache (rng seed 0, as in
`summarize_results.coverage`); action norm / q0 negative log-density (½‖x‖² up to a constant);
existing coverage metrics where meaningful.

Permitted framing if supported: *reliability changes with selection pressure while the
candidate-generating proposal is fixed.* **Not permitted:** "there is no distribution shift."

## 14. CEM secondary analysis (no new gate)

CEM population 256, iterations I ∈ {1, 2, 3, 4, 5, 6}, one deterministic trace per case such that
**CEM-I is an exact prefix of CEM-(I+1)**, with all evaluated candidates retained. Cumulative model
evaluations at depth I are exactly 256·I, compared against the compute-matched RS prefixes
I=1→RS-256, 2→512, 3→768, 4→1024, 5→1280, 6→1536.

Descriptive questions only: does CEM show the same reliability degradation; does adaptive refinement
amplify it relative to best-of-N at matched compute; how much of CEM behaviour is explained by
ordinary selection depth. **The load-bearing Stage-1 result is fixed-q0 RS-N.**

## 15. Fixture validation history

Protocol design was validated on synthetic fixtures **before any real Stage-1 outcome was computed**.

| fixture | construction | result |
| --- | --- | --- |
| A | exchangeable i.i.d. error | observed slope −0.92 inside the null's 90% band [−1.95, +2.29]; excess ≈ 0 ✓ |
| B | "trap": large favourable error concentrated on the 1% genuinely worst candidates | observed +0.45 vs null mean −3.17 → **excess +3.62**, clearly separated from A ✓ |
| C | perfect predictor | `R_latent(N) = 0` at every N; null slopes all zero; excess 0 ✓ |
| D1 | hand-built curves: selected task cost improves, oracle improves faster, regret rises | classified **"B: search useful, ranking efficiency degrades"**, not "optimization failed" ✓ |

Rejected constructions and criteria, recorded for transparency:

- Two earlier B constructions were rejected. `o = 0.9·(C − C̄)` leaves `Ĉ` increasing in `C`, so the
  predictor retains signal and regret falls; a rank-reversing variant pairing a symmetric Gaussian
  residual with a skewed Gamma cost makes `Ĉ` U-shaped in `C`, so selection lands mid-range and
  regret goes flat. The trap construction was adopted because it tests exactly the structured
  coupling G2 targets, with the residual marginal held identical between observation and null.
- An earlier per-case "observed exceeds the null's 99th percentile" pass criterion was rejected
  because it is not the statistic G2 uses; G2 is the paired `excess_beta` across cases.
- A generative D2 instance was not adopted: one-case selected-cost curves are dominated by
  extreme-draw noise (a single favourable draw at N = 4096 flipped the classification), which is why
  inference aggregates over 48 cases and why G3's thresholds are looser than G1's.

## 16. Tests to be written before the run

1. RS-N sets are exact prefixes of the same 4096 bank.
2. The selected plan equals `argmin` predicted cost over the prefix.
3. Selected predicted cost is non-increasing with N.
4. `R_latent ≥ 0` for every case and N.
5. `R_task ≥ 0` for every case and N.
6. Perfect-predictor fixture gives zero regret at every N.
7. Residual permutation preserves the residual multiset exactly.
8. Residual permutation changes the candidate→residual assignment.
9. The null uses the same candidate bank and the same N ladder.
10. Fixture A yields `excess_beta` ≈ 0.
11. Fixture B yields clearly positive `excess_beta`.
12. Fixture D1 classification is correct.
13. CEM traces are exact prefixes across depths.
14. Cumulative CEM evaluation count is exactly 256·I.
15. Compute-matched RS uses exactly the corresponding prefix.
16. The optimism sign convention matches existing confirmatory rows.
17. Manifest digest and base-checkpoint digest are enforced on load.
18. Cache lookup is deterministic and returns identical values on repeat.
19. No fresh evaluation manifest is generated.
20. No uncertainty/ensemble code path is reachable from Stage 1.

## 17. Provenance

Every cache records: git commit; **this protocol's commit and digest**; environment; manifest digest;
case ID; episode; start step; base checkpoint digest; dynamics protocol; q0 sampler identity;
candidate-bank seed; N_max; horizon; action bounds identity; objective identity; candidate index;
planner identity where applicable; CEM iteration where applicable; simulator cache identity; metric
definition version.

Candidate banks are immutable after construction. Analysis scripts consume caches and never rerun
planners.

## 18. Figure plan (pre-declared)

1. RS search depth versus `R_latent`, with the signed-optimism sanity curve beside it.
2. Selected true task cost versus oracle-best true task cost versus `R_task`.
3. Observed `beta_latent` versus exchangeable-null beta, and the `excess_beta` distribution.
4. Rank-conditioned reliability curves.
5. CEM depth versus compute-matched RS.

Geometry/OOD diagnostics are a supporting figure or appendix unless unexpectedly central.

## 19. Compute

Measured on this machine: ~24 ms per candidate (execute ≈ 20 ms, encode ≈ 3.5 ms, predict ≈ 0.4 ms)
at start_step ≈ 90. Projected: A1 (196,608 candidates) ≈ 55 min; A2 CEM traces (73,728 candidates)
≈ 21 min; **total ≈ 1.3 h on one RTX 3090, cache ≈ 275 MB.** N_max is not reduced.

## 20. Stopping rule and failure policy

- The result is reported whatever it is, including a NO-GO.
- No follow-up experiment is launched from this study. In particular, **no Cube/Wall second
  environment is started regardless of the verdict**; a Stage-2 replication would be independently
  preregistered.
- No post-hoc change of estimand, gate, threshold, ladder, null or inference unit.
- Analysis code is written and unit-tested on synthetic fixtures before any real Stage-1 row exists.
- A candidate execution that fails is rerun once with identical inputs; a case that still fails is
  dropped whole and the drop is reported with its reason, never after inspecting its slopes.
- Determinism is verified by re-running one case's bank and comparing all cached rows bit for bit.

## 21. What this study cannot establish

It cannot establish that planner-targeted repair fixes what optimization exposes (that is the
existing `results-confirm2` evidence, on a different instrument), nor that any finding generalizes
beyond PushT, beyond this released checkpoint, or beyond q0-proposal search. It measures one fixed
model's ranking reliability as a function of search depth on 48 held-out cases.
