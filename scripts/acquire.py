"""Collect a PushT repair dataset with one acquisition strategy, under a transition budget.

    uv run python scripts/acquire.py --strategy planner --budget 500 \
        --dataset pusht_expert_train.h5 --checkpoint weights.pt --output planner.h5

Every strategy pays the same budget, counted in true simulator transitions: one plan of 5 blocks
costs 25 of them, and one plan is executed per case start. The case pool is drawn once and split
so that the evaluation starts of the repair experiment are never acquired from; pass the same
--case-seed and --holdout-cases there.

Writes the training tensors to --output and one metadata record per acquisition beside it.
"""

import argparse
from functools import partial
from pathlib import Path

import torch

from planner_atlas.acquisition import (
    STRATEGIES,
    acquire,
    planner_selector,
    plans_for_budget,
    random_selector,
    save_acquisitions,
    uncertainty_selector,
)
from planner_atlas.atlas import HORIZON, PRESSURES, make_planner, planner_settings
from planner_atlas.data import action_stats
from planner_atlas.envs import make_env
from planner_atlas.evaluation import encode_frame
from planner_atlas.models.reference_lewm import ReferenceLeWM
from planner_atlas.planning import model_action_bounds
from planner_atlas.pusht import reconstruct_pusht_start, rollout_pusht, sample_pusht_cases
from planner_atlas.training import file_digest
from planner_atlas.uncertainty import load_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--budget", type=int, required=True, help="true simulator transitions")
    parser.add_argument("--dataset", type=Path, required=True, help="pusht_expert_train.h5")
    parser.add_argument("--checkpoint", type=Path, required=True, help="base LeWM weights.pt")
    parser.add_argument("--output", type=Path, required=True, help="repair dataset (.h5)")
    parser.add_argument("--members", type=Path, nargs="*", default=(), help="ensemble dynamics")
    parser.add_argument("--case-seed", type=int, default=11, help="the shared case pool")
    parser.add_argument("--holdout-cases", type=int, default=24, help="reserved for evaluation")
    parser.add_argument("--candidates", type=int, default=128, help="uncertainty candidate set")
    parser.add_argument("--pressure", default="P2", choices=list(PRESSURES), help="planner budget")
    parser.add_argument("--seed", type=int, default=0, help="proposal and planner seed")
    parser.add_argument("--round", type=int, default=0, help="acquisition round")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plans = plans_for_budget(args.budget, horizon=HORIZON)
    stats = action_stats(args.dataset)
    bounds = model_action_bounds(stats, device=args.device)
    model = ReferenceLeWM.from_checkpoint(args.checkpoint, device=args.device)
    env = make_env("pusht")

    pool = sample_pusht_cases(
        args.dataset, num_cases=args.holdout_cases + plans, seed=args.case_seed
    )
    cases = pool[args.holdout_cases :]
    print(f"{args.strategy}: {plans} plans x {HORIZON * 5} transitions = {args.budget}")

    generator = torch.Generator(bounds[0].device).manual_seed(args.seed)
    if args.strategy == "random":
        select = random_selector(horizon=HORIZON, bounds=bounds, generator=generator)
    elif args.strategy == "uncertainty":
        ensemble = load_ensemble(args.checkpoint, args.members, device=args.device)
        select = uncertainty_selector(
            horizon=HORIZON,
            bounds=bounds,
            generator=generator,
            ensemble=ensemble,
            num_candidates=args.candidates,
        )
    else:
        settings = planner_settings(*PRESSURES[args.pressure])["cem"]
        select = planner_selector(make_planner("cem", model, bounds, *settings, args.seed))

    acquisitions = []
    for index, case in enumerate(cases):
        reconstruct_pusht_start(env, case)
        latent = encode_frame(model, env.render(), args.device)
        goal = encode_frame(model, case.goal_frame, args.device)
        acquisitions.append(
            acquire(
                args.strategy,
                select,
                partial(rollout_pusht, env, case, stats),
                model=model,
                latent=latent,
                goal=goal,
                case=index,
                episode=case.episode,
                start_step=case.start_step,
            )
        )
        if (index + 1) % 10 == 0:
            print(f"   {index + 1}/{len(cases)} plans executed", flush=True)

    run = {
        "acquisition_round": args.round,
        "environment": "pusht",
        "base_checkpoint": file_digest(args.checkpoint),
        "case_seed": args.case_seed,
        "holdout_cases": args.holdout_cases,
        "seed": args.seed,
        "pressure": args.pressure if args.strategy == "planner" else None,
        "candidates": args.candidates if args.strategy == "uncertainty" else None,
    }
    save_acquisitions(args.output, acquisitions, run)
    transitions = sum(one.transitions for one in acquisitions)
    assert transitions == args.budget, f"spent {transitions}, budget {args.budget}"
    print(f"saved {args.output}: {len(acquisitions)} trajectories, {transitions} transitions")


if __name__ == "__main__":
    main()
