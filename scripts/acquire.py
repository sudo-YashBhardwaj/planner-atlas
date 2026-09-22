"""Collect a PushT repair stream with one acquisition strategy, under a transition budget.

    uv run python scripts/acquire.py --strategy planner --budget 40000 \
        --dataset pusht_expert_train.h5 --checkpoint weights.pt --output planner-seed0.h5

Contexts come from one deterministic ordered stream over training-split episodes, shared by every
strategy at the same --stream-seed, so for a given plan index all strategies face the same start,
goal and history and only the chosen continuation differs. Evaluation episodes are the validation
side of the same episode split and are never acquired from.

Budgets are nested: a seed's stream always walks the same fixed pool of 1600 contexts in the same
permuted order, so the first n contexts are the same whatever budget is asked for and a 40000
stream contains the 10000 and 2500 budgets as exact prefixes. One plan of 5 blocks costs 25 true transitions;
replaying a recorded episode to reconstruct a start is diagnostic and is not counted.

Random and uncertainty seed their proposal generator per context from the same number, so their
candidate innovations are paired; CEM is left alone.
"""

import argparse
import subprocess
from functools import partial
from pathlib import Path

import h5py
import torch

from planner_atlas.acquisition import (
    STRATEGIES,
    STREAM_PLANS,
    acquire,
    planner_selector,
    plans_for_budget,
    random_selector,
    save_acquisitions,
    stream_order,
    uncertainty_selector,
)
from planner_atlas.atlas import HORIZON, PRESSURES, make_planner, planner_settings
from planner_atlas.data import action_stats
from planner_atlas.envs import make_env
from planner_atlas.evaluation import encode_frame
from planner_atlas.models.reference_lewm import ReferenceLeWM
from planner_atlas.planning import model_action_bounds
from planner_atlas.pusht import (
    reconstruct_pusht_start,
    render_pusht_goal,
    rollout_pusht,
    sample_pusht_cases,
)
from planner_atlas.training import (
    DYNAMICS_PROTOCOL,
    cache_identity,
    file_digest,
    split_episodes,
)
from planner_atlas.uncertainty import load_ensemble

VALIDATION_FRACTION = 0.1  # the episode split repair training uses
SPLIT_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--budget", type=int, required=True, help="true simulator transitions")
    parser.add_argument("--dataset", type=Path, required=True, help="pusht_expert_train.h5")
    parser.add_argument("--checkpoint", type=Path, required=True, help="base LeWM weights.pt")
    parser.add_argument("--output", type=Path, required=True, help="repair stream (.h5)")
    parser.add_argument("--members", type=Path, nargs="*", default=(), help="ensemble dynamics")
    parser.add_argument("--stream-seed", type=int, default=0, help="acquisition seed")
    parser.add_argument("--candidates", type=int, default=128, help="uncertainty candidate set")
    parser.add_argument("--pressure", default="P2", choices=list(PRESSURES), help="planner budget")
    parser.add_argument("--round", type=int, default=0, help="acquisition round")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_commit() -> str:
    """The working tree's commit, recorded so a run can be traced back to its code."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def main() -> None:
    args = parse_args()
    plans = plans_for_budget(args.budget, horizon=HORIZON)
    stats = action_stats(args.dataset)
    bounds = model_action_bounds(stats, device=args.device)
    model = ReferenceLeWM.from_checkpoint(args.checkpoint, device=args.device)
    env = make_env("pusht")

    with h5py.File(args.dataset, "r") as file:
        num_episodes = len(file["ep_len"])
    validation = split_episodes(
        num_episodes, validation_fraction=VALIDATION_FRACTION, seed=SPLIT_SEED
    )
    if plans > STREAM_PLANS:
        raise ValueError(f"a stream holds {STREAM_PLANS} contexts, {plans} were asked for")
    pool = sample_pusht_cases(
        args.dataset, num_cases=STREAM_PLANS, seed=args.stream_seed, episodes=~validation
    )
    order = stream_order(len(pool), seed=args.stream_seed)
    contexts = [pool[i] for i in order[:plans]]  # budgets are prefixes of this fixed order
    print(
        f"{args.strategy} seed {args.stream_seed}: {plans} plans x {HORIZON * 5} transitions "
        f"= {args.budget}, from {(~validation).sum()} training-split episodes"
    )

    ensemble = None
    if args.strategy == "uncertainty":
        ensemble = load_ensemble(
            args.checkpoint,
            args.members,
            device=args.device,
            expect={"cache_identity": cache_identity(args.dataset, args.checkpoint)},
        )
    settings = planner_settings(*PRESSURES[args.pressure])["cem"]

    acquisitions = []
    for index, case in enumerate(contexts):
        # random and uncertainty draw their proposal from the same per-context seed
        seed = args.stream_seed * 1_000_000 + index
        generator = torch.Generator(bounds[0].device).manual_seed(seed)
        if args.strategy == "random":
            select = random_selector(horizon=HORIZON, bounds=bounds, generator=generator)
        elif args.strategy == "uncertainty":
            select = uncertainty_selector(
                horizon=HORIZON,
                bounds=bounds,
                generator=generator,
                ensemble=ensemble,
                num_candidates=args.candidates,
            )
        else:
            select = planner_selector(make_planner("cem", model, bounds, *settings, seed))

        goal = encode_frame(model, render_pusht_goal(env, case), args.device)
        reconstruct_pusht_start(env, case)
        latent = encode_frame(model, env.render(), args.device)
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
        if (index + 1) % 100 == 0:
            print(f"   {index + 1}/{len(contexts)} plans executed", flush=True)

    run = {
        "acquisition_round": args.round,
        "environment": "pusht",
        "git_commit": git_commit(),
        "dynamics_protocol": DYNAMICS_PROTOCOL,
        "base_checkpoint": file_digest(args.checkpoint),
        "dataset": cache_identity(args.dataset, args.checkpoint),
        "stream_seed": args.stream_seed,
        "split_seed": SPLIT_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "pressure": args.pressure if args.strategy == "planner" else None,
        "candidates": args.candidates if args.strategy == "uncertainty" else None,
        "members": [file_digest(path) for path in args.members],
    }
    save_acquisitions(args.output, acquisitions, run)
    transitions = sum(one.transitions for one in acquisitions)
    if transitions != args.budget:
        raise AssertionError(f"spent {transitions} transitions, budget was {args.budget}")
    print(
        f"saved {args.output}: {len(acquisitions)} trajectories, {transitions} transitions, "
        f"digest {file_digest(args.output)[:12]}"
    )


if __name__ == "__main__":
    main()
