"""Repair the PushT dynamics on each acquired dataset and score every branch on the same starts.

    uv run python scripts/repair_experiment.py --dataset pusht_expert_train.h5 \
        --checkpoint weights.pt --latent-cache pusht_latents.h5 \
        --repair-data random=random.h5 uncertainty=uncertainty.h5 planner=planner.h5 \
        --output repair.jsonl --steps 300

Every branch starts from the same base checkpoint and shares optimizer, learning rate, batch size,
step count, normalization protocol, mixture and seed; only the acquired transitions differ. The
unrepaired base model is evaluated as the zero-budget branch.

Each evaluation case draws one candidate set, which does not depend on any model, so all branches
are scored on identical plans and identical executed outcomes.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from planner_atlas.acquisition import load_acquisitions
from planner_atlas.atlas import (
    HORIZON,
    PRESSURES,
    make_planner,
    pairwise_rank_agreement,
    planner_settings,
)
from planner_atlas.data import action_stats
from planner_atlas.envs import make_env
from planner_atlas.evaluation import encode_frame, evaluate_candidates
from planner_atlas.models.reference_lewm import ReferenceLeWM
from planner_atlas.planning import initial_proposal, model_action_bounds, sample_proposal
from planner_atlas.pusht import (
    reconstruct_pusht_start,
    rollout_pusht,
    run_pusht_mpc,
    sample_pusht_cases,
)
from planner_atlas.repair import repair_dynamics, repair_windows
from planner_atlas.training import (
    LatentWindows,
    TrainableDynamics,
    file_digest,
    latent_windows,
    load_latent_cache,
    save_dynamics,
    split_episodes,
    validation_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="base LeWM weights.pt")
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--repair-data", nargs="+", required=True, help="strategy=path.h5")
    parser.add_argument("--output", type=Path, required=True, help="JSONL of evaluation rows")
    parser.add_argument("--dynamics-dir", type=Path, help="where repaired dynamics are written")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repair-fraction", type=float, default=0.5, help="new data per batch")
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case-seed", type=int, default=11, help="must match acquisition")
    parser.add_argument("--holdout-cases", type=int, default=24)
    parser.add_argument("--evaluation-cases", type=int, default=0, help="0 uses every holdout")
    parser.add_argument("--candidates", type=int, default=16, help="scored plans per case")
    parser.add_argument("--validation-windows", type=int, default=20_000)
    parser.add_argument("--pressure", default="P2", choices=list(PRESSURES), help="MPC planner")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def evaluate(model, env, case, *, stats, bounds, settings, candidates, seed, device) -> dict:
    """Score one model on one case: a fixed candidate set, then closed-loop control.

    The candidate set depends only on the case and the seed, so every branch is scored on the same
    plans, and their executions are deterministic repeats of the same simulator trajectories.
    """
    reconstruct_pusht_start(env, case)
    latent = encode_frame(model, env.render(), device)
    goal = encode_frame(model, case.goal_frame, device)
    generator = torch.Generator(bounds[0].device).manual_seed(seed)
    plans = sample_proposal(
        initial_proposal(HORIZON), num_samples=candidates, bounds=bounds, generator=generator
    )
    scored = evaluate_candidates(
        model, latent, goal, partial(rollout_pusht, env, case, stats), plans
    )
    chosen = int(scored.predicted_cost.argmin())
    mpc = run_pusht_mpc(
        env, model, case, stats, goal, make_planner("cem", model, bounds, *settings, seed)
    )
    return {
        "top_choice_regret": float(scored.task_cost[chosen] - scored.task_cost.min()),
        "chosen_task_cost": float(scored.task_cost[chosen]),
        "best_task_cost": float(scored.task_cost.min()),
        "absolute_optimism": float(np.abs(scored.optimism).mean()),
        "signed_optimism": float(scored.optimism.mean()),
        "rollout_error": float(scored.rollout_error.mean()),
        "terminal_error": float(scored.terminal_error.mean()),
        "rank_dynamics": pairwise_rank_agreement(scored.predicted_cost, scored.realized_cost),
        "rank_objective": pairwise_rank_agreement(scored.realized_cost, scored.task_cost),
        "rank_end_to_end": pairwise_rank_agreement(scored.predicted_cost, scored.task_cost),
        "mpc_raw_steps": mpc.raw_steps,
        **{f"mpc_{key}": value for key, value in mpc.task.items()},
    }


def main() -> None:
    args = parse_args()
    branches = dict(entry.split("=", 1) for entry in args.repair_data)
    stats = action_stats(args.dataset)
    bounds = model_action_bounds(stats, device=args.device)
    settings = planner_settings(*PRESSURES[args.pressure])["cem"]
    cache = load_latent_cache(args.latent_cache, args.dataset, args.checkpoint)
    validation_episodes = split_episodes(len(cache.lengths), validation_fraction=0.1, seed=0)
    base_windows = latent_windows(cache, args.dataset, stats, episodes=~validation_episodes)
    held_out = latent_windows(cache, args.dataset, stats, episodes=validation_episodes)
    held_out = LatentWindows(
        held_out.latents, held_out.actions, held_out.starts[: args.validation_windows]
    )
    print(f"base windows {len(base_windows):,}, held out {len(held_out):,}")

    models = {"base": ReferenceLeWM.from_checkpoint(args.checkpoint, device=args.device)}
    training = {"base": {"transitions": 0}}
    for strategy, path in branches.items():
        latents, blocks, transitions = load_acquisitions(Path(path))
        windows = repair_windows(latents, blocks)
        model = TrainableDynamics.from_checkpoint(args.checkpoint, device=args.device)
        print(f"\n== repair {strategy}: {transitions} transitions, {len(windows)} windows")
        history = repair_dynamics(
            model,
            base_windows,
            windows,
            steps=args.steps,
            batch_size=args.batch_size,
            repair_fraction=args.repair_fraction,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            validation=held_out,
        )
        training[strategy] = {
            "transitions": transitions,
            "trajectories": len(latents),
            "windows": len(windows),
            "history": history,
            "held_out_loss": history[-1]["validation_loss"],
        }
        if args.dynamics_dir:
            args.dynamics_dir.mkdir(parents=True, exist_ok=True)
            save_dynamics(
                args.dynamics_dir / f"{strategy}.pt",
                model,
                {
                    "strategy": strategy,
                    "base_checkpoint": file_digest(args.checkpoint),
                    **training[strategy],
                },
            )
        models[strategy] = model.model.eval()

    base_model = TrainableDynamics(models["base"])
    training["base"]["held_out_loss"] = validation_loss(
        base_model, held_out, batch_size=args.batch_size
    )
    losses = {name: round(record["held_out_loss"], 6) for name, record in training.items()}
    print(f"\nheld-out prediction loss: {losses}")

    pool = sample_pusht_cases(args.dataset, num_cases=args.holdout_cases, seed=args.case_seed)
    cases = pool[: args.evaluation_cases or args.holdout_cases]
    env = make_env("pusht")
    rows = []
    with args.output.open("w") as out:
        for index, case in enumerate(cases):
            for name, model in models.items():
                row = {
                    "strategy": name,
                    "case": index,
                    "episode": case.episode,
                    "transitions": training[name]["transitions"],
                    "held_out_loss": training[name]["held_out_loss"],
                    **evaluate(
                        model,
                        env,
                        case,
                        stats=stats,
                        bounds=bounds,
                        settings=settings,
                        candidates=args.candidates,
                        seed=1000 + index,
                        device=args.device,
                    ),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                rows.append(row)
            print(f"case {index + 1}/{len(cases)} scored", flush=True)
    summarize(rows)


def summarize(rows: list[dict]) -> None:
    keys = (
        "top_choice_regret",
        "absolute_optimism",
        "rollout_error",
        "rank_end_to_end",
        "mpc_semantic_reached_success",
        "mpc_task_cost",
    )
    print(f"\n{'branch':12}{'held out':>10}" + "".join(f"{key:>28}" for key in keys))
    for strategy in dict.fromkeys(row["strategy"] for row in rows):
        group = [row for row in rows if row["strategy"] == strategy]
        cells = []
        for key in keys:
            values = np.array([row[key] for row in group], dtype=float)
            values = values[~np.isnan(values)]
            sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
            cells.append(f"{values.mean():.4f} ± {sem:.4f}")
        print(
            f"{strategy:12}{group[0]['held_out_loss']:>10.6f}" + "".join(f"{c:>28}" for c in cells)
        )


if __name__ == "__main__":
    main()
