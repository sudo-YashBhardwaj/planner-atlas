# Planner Atlas

Is a planner's own experience the best data for repairing the world model it plans with? This is a
pre-registered study on the PushT manipulation task, using the released
[LeWM](https://github.com/lucas-maes/le-wm) latent world model.

![The study in three panels: planning inside the world model, the optimism of the plans search selects, and repair with 40,000 transitions per strategy](docs/figures/schematic.png)

## Results

When a planner searches through a learned world model, it favours plans whose cost the model
underestimates. We gave three kinds of repair data the same environment budget (40,000
transitions) and the same training compute:
- random plans;
- the plans a model ensemble disagrees on most;
- the planner's own plans.

### 1. Planner data repairs that planner's ranking (pre-registered, confirmed)

Compared with random data, planner data lowered the model's top-choice regret on the planner's
candidate plans by 14%.
- The effect was −0.0083, 95% CI [−0.0118, −0.0048], in 11 of 12 seeds.
- The pre-registered exact sign-flip test gave p = 0.0015.
- The effect was not distinguishable from that of uncertainty-selected data.
- It was not shown to transfer to other planners.

![Per-seed primary contrast, and within-seed changes against the continued control in regret and closed-loop MPC success](docs/figures/overview.png)

*The figure has three panels.*
- *(a) Planner − random regret per seed, with the mean and 95% CI. This is the pre-registered
  test.*
- *(b) Change in regret against an equal-compute continued-training control. Planner data improves
  ranking.*
- *(c) Change in closed-loop MPC success against the same control. Random data improves it; this
  metric is secondary.*

### 2. Each metric favours a different model

| model (mean of 12 seeds) | regret on the planner's candidates ↓ | closed-loop MPC success ↑ | held-out prediction loss ↓ |
| --- | ---: | ---: | ---: |
| released model | 0.0559 | 0.625 | 0.00989 |
| continued training, no new data | 0.0591 | 0.644 | **0.00675** |
| \+ random data | 0.0572 | **0.743** | 0.00726 |
| \+ uncertainty data | 0.0517 | 0.670 | 0.00750 |
| \+ planner data | **0.0489** | 0.655 | 0.00740 |

Each metric is best for a different model:
- planner data gives the best ranking;
- random data gives the best closed-loop control (+0.099 over continued training, a secondary
  metric);
- continued training gives the best held-out loss, and the worst ranking.

Held-out loss alone would pick the wrong model for either use.

### 3. Search selects optimistic errors

![Selection amplification of CEM and random shooting against search pressure, on PushT and TwoRoom](docs/figures/atlas_amplification.png)

- **The pattern.** On PushT, the plan CEM chooses is more over-optimistic than a typical
  proposal: its realized cost exceeds the model's prediction by more. The gap grows with search
  pressure.
- **In the acquisition data.** Planner-chosen plans had optimism +48.4, against +8.8 for random
  plans.
- **TwoRoom.** Amplification is positive but shows no trend with pressure.

### 4. No consistent degradation from deeper search (pre-registered, NO-GO)

![Stage 1: latent regret against the exchangeable null, task cost, and the per-case slopes tested by gate G1](docs/figures/stage1_depth.png)

A later pre-registered study tested random shooting over a fixed bank of 4,096 plans per case.
- **Latent regret** rose with search depth about as an exchangeable-error null predicts, and not
  consistently across cases: 28 of 48 positive, where 34 were needed.
- **Task cost** of the chosen plan still fell, from 0.192 to 0.163.
- **Verdict.** The hypothesis that deeper search degrades the model's choices beyond the ordinary
  optimizer's curse was a NO-GO.

In this setting, "a better world model" depended on what it was used for.

## Method

**Setup.**
- *Model:* the released LeWM PushT checkpoint, with a ViT-tiny encoder, a 192-d latent and a
  transformer predictor. The encoder is frozen and only the dynamics are fine-tuned.
- *Planning:* CEM and random shooting over 25-step plans, scored by predicted latent distance to
  the goal.
- *Evaluation:* 48 held-out cases.

**Rigor.**
- The hypothesis, metric, seeds, analysis and stopping rule were pre-registered.
- There are 12 independent seeds, and the seed is the unit of inference.
- Every strategy has equal budgets, and an equal-compute continued-training control is included.
- Candidate pools are executed once and shared by every model.
- Every result row is stamped with its commit and protocol digest.
- Exploratory analyses and negative results are reported as such.

## Documentation

- [Report](docs/report.md): every experiment, result, closed study and correction, plus the full
  reproduction pipeline.
- [Claims ledger](docs/results/scientific_ledger.md): each claim with its evidence and status.
- [Protocols](docs/protocols/): the pre-registrations and amendments, verbatim.

## Repository

```text
src/planner_atlas/   world model, planners, environments, training, acquisition, statistics
scripts/             experiments, result summaries and figures
docs/                report, protocols, results and figures
tests/               unit and integration tests
```

## Quickstart

```bash
export UV_PROJECT_ENVIRONMENT=/path/outside/the/repo   # keep the environment out of the repo
uv sync --locked
uv run pytest                                          # CPU only, no data needed
uv run --group figures python scripts/make_figures.py --summary docs/results/summary.json \
  --output docs/figures --stage1-summary docs/results/optimization_depth_stage1_summary.json
```

The figures are redrawn from the committed result summaries. Rerunning the experiments needs the
LeWM PushT dataset and checkpoint and a CUDA GPU; the confirmatory experiment takes about 4
GPU-hours on an RTX 3090. The [report](docs/report.md#16-reproducibility) gives the full pipeline,
the pinned inputs and the tagged commit behind each result.

## Citation

```bibtex
@misc{bhardwaj2026planneratlas,
  author = {Yash Bhardwaj},
  title  = {Planner Atlas: planner-selected errors and planner-targeted repair
            in latent world models},
  year   = {2026},
  url    = {https://github.com/sudo-YashBhardwaj/planner-atlas}
}
```

This work builds on the released [LeWM](https://github.com/lucas-maes/le-wm) checkpoints, datasets
and reference implementation, and on the `stable-worldmodel` environments. The code is released
under the [MIT license](LICENSE).
