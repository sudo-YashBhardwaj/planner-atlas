# Planner Atlas

**When a planner searches through a learned world model, it favours plans whose cost the model
underestimates. Is the planner's own experience the best data for repairing the model?**

This project tests that question on the PushT manipulation task, using the released
[LeWM](https://github.com/lucas-maes/le-wm) latent world model. Three repair strategies get the
same environment budget (40,000 transitions) and the same training compute:
- random plans;
- the plans a model ensemble disagrees on most;
- the planner's own plans.

**Short answer.** For ranking the planner's own candidate plans, repair on planner-selected data
beat repair on random data: 14% lower regret (pre-registered, 12 seeds, p = 0.0015). It was not
distinguishable from uncertainty-selected data. Random data gave the larger gain in closed-loop
control, a secondary metric. Held-out prediction loss ranked the models differently from both.

![The study in three panels: planning inside the world model, the optimism of the plans search selects, and repair with 40,000 transitions per strategy](docs/figures/schematic.png)

*The study in one picture.*
1. *The world model encodes the start and goal frames and rolls a plan of 5 action blocks forward
   in latent space. It scores the plan by the predicted distance to the goal, and CEM searches for
   the lowest predicted cost.*
2. *The plans that search selects are over-optimistic. Their realized latent cost exceeds the
   prediction far more for the planner's plans than for random ones.*
3. *Each strategy's 40,000 transitions fine-tune the model's dynamics under identical compute. The
   repaired models are then tested on candidate ranking and on closed-loop control.*

**Read more:** [full write-up](docs/writeup.md) · [paper-style report](docs/report.md) ·
[claims ledger](docs/results/scientific_ledger.md) · [pre-registrations](docs/protocols/)

## Key findings

| status | finding | evidence |
| --- | --- | --- |
| **confirmed** (pre-registered) | Repairing on planner-selected data makes the model rank *that same planner's* candidate plans better than repairing on random data | 14% lower top-choice regret: −0.0083 [−0.0118, −0.0048]; 11 of 12 seeds; exact sign-flip p = 0.0015 |
| descriptive, every seed | Search selects plans that are over-optimistic, high-error and more often successful | against random plans: optimism +48.4 vs +8.8; 2.6× the rollout error; open-loop success 0.67 vs 0.08 |
| secondary | Random data, not planner data, improves closed-loop control | MPC success over the continued control: +0.099 with random data, +0.010 with planner data |
| descriptive | Held-out prediction loss tracks neither ranking nor closed-loop control | the model with the best held-out loss ranks the planner's candidates worst |
| exploratory | The gain is not established for another planner configuration | task cost of the plan each repaired model's own higher-pressure CEM picks: −0.0039 [−0.0109, +0.0031], p = 0.25 |
| pre-registered, **NO-GO** | In the later optimization-depth study (Stage 1), choosing among more candidates neither consistently worsened the model's choice nor worsened it faster than the optimizer's curse predicts | latent regret rose in 28 of 48 cases (34 needed) and exceeded the null in 20 (32 needed); the chosen plan's task cost fell from 0.192 to 0.163 |

**In this setting, "better model" depended on the use.** Planner data helped the planner that chose
it, and random data helped closed-loop control. Held-out loss registered neither.

## Setup at a glance

| component | choice |
| --- | --- |
| task | **PushT**: push a T-shaped block to a goal pose, with 18,685 expert episodes. **TwoRoom** navigation is used for the audit only |
| world model | the released LeWM checkpoint: a ViT-tiny image encoder, a 192-d latent and a 6-layer transformer predictor. The image encoder stays frozen; only the dynamics are fine-tuned |
| planners | CEM and random shooting over plans of 5 action blocks (25 simulator steps), scored by the predicted latent distance to the goal |
| scale | 49 evaluated models × 48 held-out cases = 2,352 result rows. Stage 1 executed 48 × 4,096 candidate plans in the simulator |
| statistics | **repair study:** the seed is the unit (n = 12), with exact sign-flip permutation tests, 95% t intervals and effect sizes. **Stage 1:** the case is the unit (n = 48), with Monte-Carlo sign-flip tests |
| compute | about 4 GPU-hours on one RTX 3090 for the confirmatory pipeline |

## How the study works

1. **Audit, the "Atlas".**
   - CEM and random shooting run with the released models at four search pressures, P0 to P3:
     128×1, 128×2, 256×4 and 512×6 samples × iterations.
   - Random shooting draws the same total number of plans in one round, so the two planners
     coincide at P0.
   - The plan each planner chooses is compared with 32 plans drawn from its initial proposal
     distribution q0.
   - This part is descriptive, on PushT (50 cases) and TwoRoom (100 cases).
2. **Acquisition.**
   - In each of 12 seeds, the three strategies each pick one plan for each of 1,600 shared training
     contexts and execute it. That is 1,600 × 25 = 40,000 transitions per strategy.
   - Selection sees only the model's predictions, never outcomes.
3. **Repair.**
   - Each strategy's data fine-tunes the released model's dynamics for 1,000 steps. Half of each
     batch comes from the original data and half from the new data.
   - These repaired models are called branches. There are four per seed, including a **continued**
     control that takes the same steps on the original data only.
4. **Evaluation.**
   - On 48 held-out cases, each model ranks two fixed candidate pools (defined below).
   - Each model also runs one closed-loop MPC episode per case, replanning after every action block.

**Terms.**
- **Latent cost** of a plan: the squared distance between the predicted final latent and the goal
  latent, summed over the 192 latent dimensions. The **realized** latent cost uses the latent of the
  frame the simulator actually reaches.
- **Optimism**: realized minus predicted latent cost. A positive value means the model thought the
  plan ended closer to the goal than it did.
- **Rollout error**: the mean squared error between predicted and realized latents over the 5
  action blocks. It is a mean over dimensions, so its scale is much smaller than a cost's.
- **Top-choice regret** of a pool: the true task cost of the candidate the model ranks first, minus
  the best true task cost in the pool. Lower is better.
- **q0 pool**: 32 plans per case drawn from q0, before any search.
- **Planner-region pool**: 32 plans per case, sampled from the final proposal distribution of the
  released model's CEM at pressure P2 (256 samples × 4 iterations).
- **Task cost**: block position error / 512 + angle error / π.
- **Semantic success**: the T-block ends within 20 px and π/9 of its goal pose.

## Results

### Search selects optimistic errors

![Selection amplification of CEM and random shooting against search pressure, on PushT and TwoRoom](docs/figures/atlas_amplification.png)

The plan CEM chooses is more optimistic than a typical proposal, and increasingly so under more
search pressure. *Selection amplification* is the chosen plan's optimism minus the mean optimism of
32 proposals. PushT, 50 cases, mean ± SE:

| | P0 128×1 | P1 128×2 | P2 256×4 | P3 512×6 |
| --- | ---: | ---: | ---: | ---: |
| selection amplification, CEM | 10.6 ± 4.7 | 24.1 ± 7.9 | 32.1 ± 9.5 | 51.1 ± 13.3 |
| selection amplification, random shooting | 10.6 ± 4.7 | 11.7 ± 7.7 | 22.4 ± 5.1 | 19.3 ± 7.5 |
| rollout error, CEM's chosen plan | 0.056 | 0.087 | 0.099 | 0.128 |
| rollout error, a typical proposal | 0.042 | 0.044 | 0.045 | 0.041 |

- **Average-case error understates the error on the plans that actually get executed.**
- **TwoRoom.** Amplification is positive (13 to 26) but shows no trend with pressure.
- **Status.** This audit is descriptive.
- **The later test.** A pre-registered study ([below](#later-studies-both-closed)) covered random
  shooting over a fixed proposal. It compared how decision regret grows with search against an
  exchangeable-error null, in which the same prediction errors are shuffled across candidates. It
  found no excess over that null. CEM was not tested.

### What each strategy collects

| 12 seeds × 1,600 plans each | random | uncertainty | planner |
| --- | ---: | ---: | ---: |
| signed optimism | +8.8 | +18.9 | +48.4 |
| absolute optimism | 20.3 | 51.3 | 50.8 |
| rollout error | 0.0468 | 0.1299 | 0.1227 |
| true task cost | 0.269 | 0.245 | 0.105 |
| open-loop semantic success | 0.08 | 0.17 | 0.67 |

- **Both targeted strategies find high-error regions.** Uncertainty has 2.8× random's rollout
  error, and the planner 2.6×.
- **The planner's errors are almost all optimistic,** and its plans mostly succeed.
- **Every planner-vs-random ordering holds in all 12 seeds**
  ([figure](docs/figures/acquisition.png)).

### Confirmatory repair experiment

The [pre-registration](docs/protocols/confirmatory-protocol.md) fixed the hypothesis, metric, seeds,
analysis and stopping rule before any confirmatory model was trained. One
[amendment](docs/protocols/confirmatory-amendment-1.md) before launch replaced the evaluation
manifest and the seeds.

- **Hypothesis.** At 40,000 transitions, planner-selected data gives lower planner-region
  top-choice regret than random data after identical repair compute.
- **Design.**
  - 12 seeds (4 to 15), each an independent acquisition stream and repair run.
  - Four branches per seed: continued, random, uncertainty and planner.
  - 48 held-out evaluation cases ([manifest](docs/protocols/confirmatory-manifest.json)).
- **Test.** The seed is the unit. The pre-registered test is the exact two-sided sign-flip
  permutation test over all 2¹² sign assignments.

![Per-seed primary contrast, and within-seed differences from the continued control in regret and MPC success](docs/figures/overview.png)

*The figure has three panels. Error bars are 95% t intervals over 12 seeds.*
- *(a) Planner − random regret per seed, with the mean and its 95% CI. This is the pre-registered
  contrast.*
- *(b) Within-seed difference from the continued control in planner-region regret. Planner data
  improves the ranking of the planner's own candidates.*
- *(c) The same difference in closed-loop MPC success. Random data improves it; this metric is
  secondary.*

| model (mean over 12 seeds, 95% CI) | planner-region regret | q0 regret | MPC semantic success | held-out loss |
| --- | ---: | ---: | ---: | ---: |
| released model (reference) | 0.0559 | 0.0394 | 0.625 | 0.00989 |
| continued (no new data) | 0.0591 [0.0557, 0.0625] | 0.0450 | 0.644 [0.613, 0.675] | 0.00675 |
| random | 0.0572 [0.0529, 0.0614] | 0.0451 | 0.743 [0.715, 0.771] | 0.00726 |
| uncertainty | 0.0517 [0.0466, 0.0568] | 0.0423 | 0.670 [0.639, 0.701] | 0.00750 |
| planner | **0.0489** [0.0450, 0.0529] | 0.0427 | 0.655 [0.613, 0.696] | 0.00740 |

**Primary result (confirmed).** Planner − random planner-region regret:

| statistic | value |
| --- | --- |
| mean | **−0.00826**, a 14% reduction of the random branch's regret |
| 95% CI | [−0.01176, −0.00476] |
| seeds with the predicted sign | 11 of 12 |
| exact sign-flip p (pre-registered) | **0.0015** |
| Cohen's d_z | −1.50 |
| paired t-test p (supplementary) | 0.0003 |
| exact Wilcoxon p (supplementary) | 0.0015 |

The effect is about half the superseded 3-seed exploratory estimate (−0.0147). That estimate ran
under a since-corrected protocol, and the shrinkage is consistent with the winner's curse.

**Scope.** The repair data and the evaluation pool come from the same planner configuration, the
released model's P2 CEM. The confirmed claim is therefore:

> Training on trajectories selected by a given planner improves candidate ranking on trajectories
> generated by that same planner, relative to random acquisition.

Two wider checks are unresolved:
- on the q0 pool, planner − random regret is −0.0024 [−0.0053, +0.0006], p = 0.11;
- on each repaired model's own higher-pressure plan (P3), planner − random task cost is −0.0039
  [−0.0109, +0.0031], p = 0.25.

Generalization beyond the matched planner is therefore not established.

### Two different kinds of "better"

Within each seed, against the equal-compute continued control:
- **Planner data** improves planner-region regret by −0.0102 [−0.0138, −0.0065] (11/12 seeds). It
  leaves closed-loop MPC success unchanged: +0.010 [−0.050, +0.071].
- **Random data** leaves planner-region regret unchanged: −0.0019 [−0.0065, +0.0027]. It improves
  MPC success by +0.099 [+0.056, +0.142] (11/12 seeds).

MPC was a secondary, descriptive metric. The MPC gain is therefore not a separately confirmed
hypothesis, and it is not evidence that planner repair hurts closed-loop control.

**Held-out loss.**
- Every repair branch has worse held-out loss than the continued control, in 12 of 12 seeds.
- That is partly by design. The continued control trains only on the original data distribution,
  which is what held-out loss measures, while the repair branches spend half of each batch on new
  data.
- Yet the best ranking and the best closed-loop control come from two different branches, and
  neither of them has the best held-out loss ([figure](docs/figures/held_out_vs_downstream.png)).

### Exploratory: replay weight

> This analysis ran after the confirmatory result was inspected, against the pre-registered
> stopping rule, so it is exploratory. Its [design note](docs/protocols/replay-weight-protocol.md)
> was recorded before its models were trained.

![Planner-region regret, MPC success and held-out loss against the weight on acquired data](docs/figures/replay_weight.png)

The weight on acquired data is its share of each training batch. The values were 0 (the continued
control), 0.125, 0.25 and 0.5 (the confirmatory setting).
- **More planner data** steadily improves planner-region regret, from 0.0591 to 0.0489. MPC success
  stays within ±0.02 of the continued control.
- **More random data** leaves ranking roughly flat and raises MPC success from 0.644 to 0.743.
- **Held-out loss** rises with the weight for both strategies.
- **Forgetting is not supported** as the explanation for the MPC gap between random and planner
  data. Lowering the planner-data weight recovers held-out loss, but planner MPC stays level.

### Later studies (both closed)

Two later studies asked whether a stronger, more general claim could be made.
- Each had pre-declared go/no-go gates.
- Neither retrained or re-analysed a repaired model, though Stage 1 reused the 48 evaluation cases.
- Whether they count as follow-ups under the confirmatory stopping rule is a judgment call; see
  [write-up §9.1](docs/writeup.md#91-protocol). Both are reported in full.

**Planner-conditioned reliability (E1 in the write-up): closed at its validity gates, before any
outcome.**
- *Plan.* Repair on each planner's data, CEM or random shooting at equal compute, and test every
  repaired model on both planners' candidates.
- *Gates.* First, the two planners' candidate sets had to be shown to differ. Both pre-declared
  gates failed:
  - median per-dimension Bhattacharyya coefficient 0.909, where ≤ 0.88 was needed;
  - median normalized energy distance 0.0995, where ≥ 0.10 was needed.

  By these measures, the candidate sets overlap too much for the comparison to be informative.
- *Record.* The gate values were computed interactively and not saved to disk.

**Optimization depth (Stage 1): pre-registered, NO-GO.**
- *Design.* For each of the 48 cases, 4,096 candidate plans were drawn once and executed in the
  simulator. Search at depth N picks the model's favourite among the first N.
- *Protocol.* Pre-registered in `ebdfa8a`, amended before implementation in `8b47d24`, and
  executed at `65d3f00` with no deviations.

![Stage 1: latent regret against the exchangeable null, task cost, and the per-case slopes tested by gate G1](docs/figures/stage1_depth.png)

| gate: per-case slope per doubling of N, N = 256 to 4,096 (n = 48) | mean [95% CI] | cases > 0 | d_z | result |
| --- | --- | --- | --- | --- |
| G1, latent regret | +4.02 [−0.04, +8.08] | 28 (34 needed) | 0.29 (0.5 needed) | **FAIL: NO-GO** |
| G2, excess over the exchangeable-error null | −2.21 [−6.34, +1.93] | 20 (32 needed) | −0.16 | FAIL |
| G3, task regret | +0.0049 [−0.0012, +0.0110] | 28 (29 needed) | 0.23 (0.3 needed) | FAIL |

- **Latent regret** rose on average, but inconsistently across cases (Monte-Carlo sign-flip
  p = 0.051), and no faster than the null.
- **Task regret** also rose on average, from 0.055 to 0.077, without a consistent case-level trend.
- **The chosen plan's task cost still fell,** from 0.192 to 0.163, because larger pools contain
  better plans: the best plan's cost fell from 0.137 to 0.086.

Larger latent-model error under optimization was not the same thing as worse decisions.

## Limitations

- **Matched planner.** The confirmed gain is measured on candidates from the same planner
  configuration that selected the repair data.
- **Scale.**
  - One environment, one budget (40,000 transitions), one data mixture and one model family.
  - CEM-style planners and 48 evaluation cases.
  - Smaller budgets exist only in a superseded 3-seed grid, where no effect appeared: +0.0052 at
    2,500 transitions and −0.0013 at 10,000.
- **Planner vs uncertainty is unresolved** on the primary metric: −0.0027 [−0.0102, +0.0047].
  Uncertainty-selected data also beats the continued control (−0.0074, exploratory).
- **The MPC results are secondary,** and the replay-weight analysis is exploratory.
- **Why random data helps closed-loop control is untested.** Random data drifts further from the
  expert data than planner data does (nearest-base latent distance 2.84 against 2.64). That it
  covers off-nominal recovery states is a hypothesis, not a result.
- **Goal rendering.** The confirmatory runs encoded goals from recorded dataset frames, which sit
  0.69 latent L2 from the live renders used elsewhere. This changes a case's start-to-goal cost by
  1.3% on average. All models share it. It is fixed for future runs, but nothing was rerun.

## Research practices

- **Pre-registration.** The hypothesis, primary metric, seeds, analysis and stopping rule were fixed
  before any confirmatory model was trained. The one amendment before launch is published with its
  reason.
- **Equal budgets and a control.** Every strategy gets the same 40,000 transitions and 1,000
  training steps. An equal-compute continued-training control separates "more training" from
  "which data".
- **Shared outcomes.** Candidate pools are executed once and shared, so every model ranks identical
  plans against identical outcomes.
- **Inference.** In the repair study the seed is the unit of inference; in Stage 1 it is the case.
  Every contrast comes with a permutation test, an interval and an effect size. Secondary and
  exploratory results are labelled as such.
- **Provenance.** Every result row records the commit, the protocol digest, the manifest and the
  checkpoint.
- **Disclosed deviations and negative results.**
  - The analysis that broke the stopping rule is labelled exploratory.
  - The superseded exploratory estimate is reported.
  - Two later studies were closed at their gates, without rescue.
- **Audits.** Separate review passes re-derived the reported numbers from the raw artifacts and
  checked the documents against them. The corrections are listed in
  [the write-up](docs/writeup.md#15-corrections-audits-and-known-gaps).

## Documentation

| document | contents |
| --- | --- |
| [docs/writeup.md](docs/writeup.md) | the complete project, phase by phase, with every table |
| [docs/report.md](docs/report.md) | paper-style report of the confirmatory repair experiment |
| [docs/results/scientific_ledger.md](docs/results/scientific_ledger.md) | every claim with its evidence and status |
| [docs/results/optimization_depth_stage1.md](docs/results/optimization_depth_stage1.md) | the complete Stage-1 result |
| [docs/protocols/](docs/protocols/) | pre-registrations, amendments, the evaluation manifest and the replay-weight design note, verbatim |
| [docs/results/summary.json](docs/results/summary.json) | every repair-study number, computed by [scripts/summarize_results.py](scripts/summarize_results.py) with the digests of its inputs |
| [docs/results/optimization_depth_stage1_summary.json](docs/results/optimization_depth_stage1_summary.json) | the Stage-1 analysis summary |

## Reproduce

**Requirements.** Python 3.10, [uv](https://docs.astral.sh/uv/) and, for the experiments, a CUDA
GPU. The environment, data, latent cache and run outputs live on local disk outside the repository.

```bash
export UV_PROJECT_ENVIRONMENT=/local/disk/venvs/planner-atlas   # outside the repository
uv sync --locked
uv run pytest   # the test suite runs on CPU and needs no data or checkpoints
```

**Inputs.**
- *Dataset:* `pusht_expert_train.h5.zst` from `huggingface.co/datasets/quentinll/lewm-pusht` at
  revision `655cd446b9929369d7d406001da85c15d1457850` (archive sha256 `7cfbd6d9…`). Decompress it
  with `zstd -d pusht_expert_train.h5.zst`.
- *Checkpoint:* `weights.pt` from `huggingface.co/quentinll/lewm-pusht` (sha256 `48938400…`,
  recorded in every result row).

<details>
<summary><b>Full confirmatory pipeline</b> (about 4 GPU-hours on one RTX 3090, plus a one-time 1.8 GB latent cache)</summary>

```bash
DATA=/local/disk                # pusht_expert_train.h5 and weights.pt
RUNS=$DATA/runs/confirm         # outputs
mkdir -p $RUNS
cp docs/protocols/confirmatory-manifest.json $RUNS/manifest.json   # or let seed 307 regenerate it
cp docs/protocols/confirmatory-amendment-1.md $RUNS/protocol-amendment.md
git checkout results-confirm2   # the exact code that produced the confirmatory results

# 1. live-render latent cache (built on the first run) and the 5-member uncertainty ensemble
for m in 0 1 2 3 4; do
  uv run python scripts/train_dynamics.py --env pusht --dataset $DATA/pusht_expert_train.h5 \
    --checkpoint $DATA/weights.pt --latent-cache $DATA/pusht-live-latents.h5 \
    --output $DATA/member$m.pt --member $m --seed 0 --epochs 1
done

# 2. acquisition: 3 strategies x 12 seeds, 40,000 transitions each
for seed in $(seq 4 15); do for s in random uncertainty planner; do
  uv run python scripts/acquire.py --strategy $s --budget 40000 --stream-seed $seed \
    --dataset $DATA/pusht_expert_train.h5 --checkpoint $DATA/weights.pt \
    --members $DATA/member{0,1,2,3,4}.pt --output $RUNS/acquisition/$s-seed$seed.h5
done; done

# 3. repair and evaluation: 49 models x 48 cases
uv run python scripts/repair_experiment.py --dataset $DATA/pusht_expert_train.h5 \
  --checkpoint $DATA/weights.pt --latent-cache $DATA/pusht-live-latents.h5 \
  --acquisition-dir $RUNS/acquisition --manifest $RUNS/manifest.json --manifest-seed 307 \
  --protocol $RUNS/protocol-amendment.md --budgets 40000 --seeds $(seq 4 15) \
  --output $RUNS/grid.jsonl --dynamics-dir $RUNS/branches

# 4. summary and figures (back on main; these scripts only read the outputs)
git checkout main
uv run python scripts/summarize_results.py --confirm-dir $RUNS --output $RUNS/summary.json
uv run --group figures python scripts/make_figures.py \
  --summary $RUNS/summary.json --output $RUNS/figures
```

- **Determinism.** Training and evaluation are seeded. Rerunning a branch reproduced its 48 result
  rows bit for bit on the same machine.
- **Replay-weight branches.** Rerun step 3 with `--repair-fraction 0.125` or `0.25` and
  `--branches random:40000:<seed> planner:40000:<seed>`.
- **The committed summary.** `summarize_results.py` also accepts `--replay-rows`, `--atlas` and
  `--coverage`. The committed `docs/results/summary.json` used all three; the exact command is in
  the [report](docs/report.md#10-reproducibility).
- **Running on `main`.** On `main`, steps 1 to 3 render goals live (protocol `frozen_rep_v3`) and
  refuse the older `frozen_rep_v2` checkpoints and streams. Their outputs will therefore differ from
  the confirmatory ones.

</details>

<details>
<summary><b>Stage 1</b> (the build took 1 h 41 min on one RTX 3090)</summary>

From the tag `optimization-depth-stage1-frozen`, run `scripts/optimization_depth_stage1.py` with
`build`, `verify-determinism`, `analyze` and `summarize`, in that order. The arguments are in the
script's docstring.

</details>

**Figures.** Every figure is drawn from the committed summaries:

```bash
uv run --group figures python scripts/make_figures.py --summary docs/results/summary.json \
  --output docs/figures --stage1-summary docs/results/optimization_depth_stage1_summary.json
```

**Lint.** `uv run ruff check`, `uv run ruff format --check` and `uv run pre-commit run --all-files`.

## Repository layout

```text
src/planner_atlas/
  models/reference_lewm.py  released LeWM; rollout and cost bit-identical to upstream (encoder in TF32)
  envs.py data.py           environment boundary and the action-block contract
  planning.py atlas.py      CEM, random shooting, search pressures, Atlas metrics
  pusht.py evaluation.py    PushT replay reconstruction, task metrics, MPC; TwoRoom cases
  training.py repair.py     frozen-representation dynamics training and mixed-batch repair
  acquisition.py            random, uncertainty and planner selection with transition accounting
  uncertainty.py            bootstrap ensemble and cost-variance uncertainty
  coverage.py               latent coverage of acquired data against the base distribution
  depth.py                  Stage 1: candidate banks, outcome caches, regrets, null, gates
  statistics.py             paired inference: exact and Monte-Carlo sign-flip, t, Wilcoxon, sign test
scripts/
  tworoom_atlas.py pusht_atlas.py        the Atlas
  train_dynamics.py acquire.py           ensemble members and acquisition streams
  repair_experiment.py                   repair and evaluation
  optimization_depth_stage1.py           Stage 1: build, verify-determinism, analyze, summarize
  summarize_results.py make_figures.py   every reported number and figure
docs/                                    write-up, report, protocols, results, figures
tests/                                   unit and integration tests
```

## Provenance

| results | produced by |
| --- | --- |
| confirmatory experiment and replay-weight analysis | commit `2d6364d` (tag `results-confirm2`) |
| Stage 1 | commit `65d3f00` (tag `optimization-depth-stage1-frozen`) |
| Atlas, PushT and TwoRoom | commits `7a7f75a` and `7d62e8c` |
| superseded exploratory grid | commit `2fcb9e7` |
| planner-conditioning gates (E1) | no commit: computed interactively and not saved |

No other commit has produced a result.

## Status, license and citation

The project is complete, and the confirmatory result is final as reported. The two later studies
are closed, and no experiment follows from them. The code is released under the
[MIT license](LICENSE).

```bibtex
@misc{bhardwaj2026planneratlas,
  author = {Yash Bhardwaj},
  title  = {Planner Atlas: planner-selected errors and planner-targeted repair
            in latent world models},
  year   = {2026},
  url    = {https://github.com/sudo-YashBhardwaj/planner-atlas},
  note   = {Confirmatory results produced by commit 2d6364d (tag results-confirm2)}
}
```

This work builds on:
- the released LeWM checkpoints and datasets (`quentinll/lewm-pusht`, `quentinll/lewm-tworooms`);
- the [LeWM reference implementation](https://github.com/lucas-maes/le-wm), read at `8edfeb3`;
- the `stable-worldmodel` environments.

Please cite those works as well.
