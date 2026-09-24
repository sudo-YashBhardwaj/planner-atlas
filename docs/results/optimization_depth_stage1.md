# Stage 1 — optimization depth and decision reliability: result

**Verdict: NO-GO — G1 failed** (frozen decision matrix, case 4). The proposed optimization-depth
ICML thesis is not supported and is closed. No experiment follows from this result.

## Provenance

| item | value |
| --- | --- |
| preregistration | `ebdfa8a` (`docs/protocols/optimization_depth_stage1.md`) |
| amendment before implementation | `8b47d24` (section 22: all 48 cases required, Monte-Carlo sign-flip arithmetic, determinism case 0) |
| implementation | `65d3f00`, tag `optimization-depth-stage1-frozen` |
| environment / cases | PushT, consumed confirmatory manifest seed 307 (`5e945883…`), 48 cases |
| base checkpoint | `48938400…` (released LeWM PushT), dynamics protocol `frozen_rep_v3` |
| candidates | per case: one immutable 4096-plan q0 bank (RS-N = prefixes) and one 6-iteration CEM trace (population 256) |
| run | 2026-09-23, one RTX 3090; build 1 h 41 m, no failures or retries; 48/48 RS and 48/48 CEM caches complete |
| determinism | case 0 rebuilt: RS and CEM caches bitwise identical in every dataset and metadata key (0 mismatches). The pass was printed by `verify-determinism` and is recorded here; the command writes no file |
| outputs (local disk) | `runs/optimization_depth_stage1/analysis/summary.json` sha256 `618e51af…`, `per_case.jsonl` `3644969…`, `null_beta.npy` `90f48922…` |
| protocol deviations | **none** |

Implementation choices fixed in `65d3f00` before execution: bank and CEM trace both seeded
`20260923 + case` (CEM iteration 1 is exactly RS-256; it selected the same candidate in 48/48
cases); null replicate b is the b-th successive `rng.permutation(4096)` of
`default_rng([20260923, case])`; one frozen 100,000 × 48 sign matrix from seed 20260923;
`semantic_success` is final-state success; geometry uses the terminal executed latent.

## RS-N (fixed q0 proposal), mean ± SE over 48 cases

Latent costs are sums over 192 dimensions; optimism = realized − predicted (positive = optimistic).

| N | predicted C | realized C | signed opt | abs opt | R_latent | J_selected | J_oracle | R_task | success |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 169.2 ± 14.4 | 196.6 ± 16.8 | 27.3 ± 8.4 | 33.0 ± 8.0 | 18.5 ± 7.2 | 0.238 ± 0.026 | 0.192 ± 0.022 | 0.046 ± 0.012 | 0.29 |
| 64 | 148.8 ± 14.3 | 169.2 ± 15.8 | 20.5 ± 4.6 | 27.1 ± 3.8 | 12.3 ± 4.0 | 0.217 ± 0.025 | 0.166 ± 0.020 | 0.050 ± 0.012 | 0.33 |
| 128 | 134.8 ± 13.9 | 157.5 ± 15.1 | 22.7 ± 5.5 | 29.5 ± 4.8 | 18.9 ± 5.7 | 0.202 ± 0.022 | 0.149 ± 0.019 | 0.053 ± 0.010 | 0.35 |
| 256 | 121.2 ± 12.8 | 153.0 ± 15.2 | 31.8 ± 7.1 | 38.4 ± 6.4 | 24.0 ± 5.9 | 0.193 ± 0.022 | 0.137 ± 0.018 | 0.055 ± 0.010 | 0.35 |
| 512 | 103.0 ± 11.8 | 147.9 ± 15.9 | 44.9 ± 10.4 | 51.3 ± 9.7 | 35.8 ± 9.0 | 0.191 ± 0.025 | 0.119 ± 0.016 | 0.072 ± 0.014 | 0.40 |
| 1024 | 93.3 ± 11.0 | 134.2 ± 15.6 | 40.8 ± 9.7 | 47.1 ± 9.0 | 29.4 ± 6.8 | 0.176 ± 0.024 | 0.114 ± 0.016 | 0.062 ± 0.013 | 0.40 |
| 2048 | 77.1 ± 10.4 | 135.6 ± 16.2 | 58.5 ± 12.0 | 63.8 ± 11.4 | 44.1 ± 8.4 | 0.178 ± 0.024 | 0.100 ± 0.015 | 0.078 ± 0.014 | 0.42 |
| 4096 | 69.1 ± 10.3 | 119.6 ± 16.2 | 50.5 ± 11.5 | 57.6 ± 10.8 | 39.9 ± 7.5 | 0.163 ± 0.024 | 0.086 ± 0.014 | 0.077 ± 0.015 | 0.44 |

## Gates (case-level slopes over N = 256…4096 against log2 N, n = 48)

| gate | statistic | mean [95% CI] | positive | d_z | MC sign-flip p (SE) | result |
| --- | --- | --- | --- | --- | --- | --- |
| **G1** | `beta_latent` | +4.02 [−0.04, +8.08] | 28/48 (need ≥ 34) | 0.29 (need ≥ 0.5) | 0.0513 (0.0007) (need < 0.05) | **FAIL** |
| G2 | `excess_beta` | −2.21 [−6.34, +1.93] | 20/48 (need ≥ 32) | −0.16 | 0.290 (0.0014) | FAIL |
| G3 | `beta_task` | +0.0049 [−0.0012, +0.0110] | 28/48 (need ≥ 29) | 0.23 (need ≥ 0.3) | 0.116 (descriptive) | FAIL |

Supplementary: G1 exact binomial p 0.028, exact Wilcoxon p 0.050; G2 binomial 0.31, Wilcoxon 0.21
(exact); G3 binomial 0.14, Wilcoxon 0.088 (exact). G1 fails on count and effect size regardless of
the borderline p-value.

**Exchangeable-error null (G2).** Mean observed `beta_latent` +4.02; mean null slope across cases
+6.23 (median per-case null SD 8.6). The observed degradation does not exceed what the same
residuals produce when exchanged across candidates. `corr(C, o)` per case: median +0.175, IQR
[−0.118, +0.326], range [−0.447, +0.841], positive in 67% of cases, which by protocol caveat 8.1 C
makes the null potentially conservative. This is a diagnostic only and did not alter G2.

**Task curves (G3).** Over N = 256…4096, J_selected 0.193 → 0.163 (slope −0.0073, CI
[−0.0135, −0.0010]); J_oracle 0.137 → 0.086; R_task 0.055 → 0.077 with no consistent case-level
trend. Frozen reading: **D — task regret flat; latent degradation not clearly task-relevant.**

## Characterization (not gates)

- **S1** selected signed optimism slope: +5.10 [+0.07, +10.13], 23/48 positive.
- **S2** absolute optimism slope: +5.08 [+0.19, +9.96], 27/48 positive.
- **S3** top-1% minus bottom-50% signed optimism: **+28.95 [+20.9, +37.0], 41/48, d_z 1.05.**
- **S4** rank-conditioned reliability (bank sorted by predicted cost, mean over cases):

| bin | signed opt | abs opt | rollout err | terminal err | true J | success | rank agr. (latent / task) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0–1% | 32.3 | 40.0 | 0.097 | 0.203 | 0.202 | 0.35 | 0.26 / 0.19 |
| 1–5% | 21.1 | 31.1 | 0.074 | 0.149 | 0.240 | 0.26 | 0.25 / 0.12 |
| 5–10% | 15.7 | 26.6 | 0.062 | 0.117 | 0.260 | 0.20 | 0.17 / 0.05 |
| 10–25% | 12.0 | 22.9 | 0.051 | 0.092 | 0.277 | 0.11 | 0.31 / 0.08 |
| 25–50% | 8.5 | 19.2 | 0.042 | 0.071 | 0.293 | 0.06 | 0.35 / 0.05 |
| 50–100% | 3.3 | 17.8 | 0.040 | 0.066 | 0.311 | 0.03 | 0.54 / 0.11 |

## CEM vs compute-matched RS (secondary, descriptive)

| I (evaluations) | predicted C, CEM / RS | signed opt | R_latent | R_task | J_selected |
| --- | --- | --- | --- | --- | --- |
| 1 (256) | 121.2 / 121.2 | 31.8 / 31.8 | 24.0 / 24.0 | 0.055 / 0.055 | 0.193 / 0.193 |
| 2 (512) | 61.0 / 103.0 | 55.8 / 44.9 | 47.0 / 35.8 | 0.081 / 0.072 | 0.163 / 0.191 |
| 3 (768) | 38.0 / 96.2 | 53.7 / 44.9 | 50.6 / 34.2 | 0.081 / 0.062 | 0.128 / 0.178 |
| 4 (1024) | 21.2 / 93.3 | 56.4 / 40.8 | 59.0 / 29.4 | 0.079 / 0.062 | 0.105 / 0.176 |
| 5 (1280) | 15.9 / 87.9 | 50.6 / 50.3 | 52.4 / 36.5 | 0.072 / 0.066 | 0.091 / 0.176 |
| 6 (1536) | 12.5 / 81.5 | 50.6 / 58.4 | 52.0 / 42.7 | 0.066 / 0.072 | 0.081 / 0.177 |

At matched model evaluations CEM reaches much better true task cost (CEM − RS at I = 6:
−0.096 ± 0.021) while its latent regret is on average larger (+9 to +30, SEs 10–14) and its task
regret shows no consistent difference.

## Geometry / OOD (descriptive)

The candidate-generating proposal q0 is fixed while selection pressure changes. For the selected
candidate from N = 32 to 4096: latent nearest-base distance 3.38 → 3.49 (slope +0.03 ± 0.06),
Mahalanobis to base 13.69 → 13.36 (+0.02 ± 0.03), action norm 7.09 → 7.49 (+0.048 ± 0.034 per
doubling, about 0.07 bank SD), q0 negative log-density 25.5 → 28.3. Selection drifts slightly
toward larger actions; latent OOD indicators stay essentially flat. This is not a claim that there
is no distribution shift.

## Conclusions

**Supported**

- Prediction error and optimism are strongly concentrated in the model's predicted-good tail (S3,
  S4), as ordinary best-of-N selection predicts.
- Deeper fixed-q0 search lowers predicted cost and, on average, somewhat increases selected
  optimism and latent regret.
- That latent-regret increase is heterogeneous across cases and fails the preregistered reliability
  gate (G1).
- It does not exceed the exchangeable-error optimizer's-curse baseline (G2).
- Task-ranking degradation is not established (G3; reading D).
- Absolute selected task performance improves with deeper search.
- Adaptive CEM improves true task performance substantially over compute-matched random shooting,
  even while its latent regret can be larger.

**Not supported**

- A consistent optimization-depth-induced decision-reliability failure.
- Structured exploitation beyond the ordinary optimizer's curse.
- Task-level harm from deeper search.
- The proposed optimization-depth ICML thesis.

**Larger latent-model error under optimization is not equivalent to worse downstream decisions.**
Here, the searches that selected plans with larger latent optimism and regret also selected plans
with better true task outcomes.

## Relation to the confirmed repair study

The confirmed study (`results-confirm2`, see `docs/report.md`) found that planner-selected repair
improved P2 candidate ranking relative to random repair (pre-registered, sign-flip p = 0.0015),
while random repair improved closed-loop MPC more (secondary, descriptive). Stage 1 adds that even
rising latent ranking regret under search need not come with worse task outcomes. Together these
reinforce that latent prediction/ranking metrics and closed-loop control are distinct downstream
notions of model quality. This combined observation is consistent with known optimizer's-curse and
Goodhart effects and is not claimed to be novel enough for an ICML contribution.

## What Stage 1 cannot establish

The absence of an effect: it failed to meet a preregistered bar on 48 consumed PushT cases, one
released checkpoint and q0-proposal search, and says nothing about other environments, models or
planners.
