"""The equal-interaction, equal-compute repair grid on PushT.

    uv run python scripts/repair_experiment.py --dataset pusht_expert_train.h5 \
        --checkpoint weights.pt --latent-cache pusht_latents.h5 \
        --acquisition-dir runs/acquisition --manifest runs/manifest.json \
        --output runs/grid.jsonl

Branches are the untouched released checkpoint, an equal-compute continued-training control that
sees only base data, and each acquisition strategy at each budget. Every trained branch starts
from the same released checkpoint and shares optimizer, learning rate, batch size, step count,
mixture, normalization protocol and training seed; only which transitions exist differs. Budgets
are prefixes of one acquired stream, so the budget curve is nested.

Each held-out case carries two candidate pools that no branch influences: one from the planners'
initial proposal, one from the final proposal of the frozen base model's P2 CEM, which is the
region optimization actually visits. Both are executed once in the simulator and their true task
costs cached, so every branch ranks identical plans with identical outcomes. Each branch then also
plans for itself with P3 CEM, which is a separate question: not how well it ranks given plans, but
how well it controls when it chooses.
"""

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from planner_atlas.acquisition import load_acquisitions
from planner_atlas.atlas import (
    ELITE_FRACTION,
    HORIZON,
    PRESSURES,
    make_planner,
    pairwise_rank_agreement,
    planner_settings,
)
from planner_atlas.data import ActionStats, action_stats
from planner_atlas.envs import make_env
from planner_atlas.evaluation import (
    TaskOutcome,
    compare_with_model,
    encode_frame,
    frames_to_tensor,
)
from planner_atlas.models.reference_lewm import ReferenceLeWM
from planner_atlas.planning import cem, initial_proposal, model_action_bounds, sample_proposal
from planner_atlas.pusht import (
    PushTCase,
    reconstruct_pusht_start,
    render_pusht_goal,
    rollout_pusht,
    run_pusht_mpc,
    sample_pusht_cases,
)
from planner_atlas.repair import repair_dynamics, repair_windows
from planner_atlas.training import (
    DYNAMICS_PROTOCOL,
    TrainableDynamics,
    cache_identity,
    file_digest,
    latent_windows,
    load_latent_cache,
    save_dynamics,
    split_episodes,
    subsample_windows,
    validation_loss,
    windows_episodes,
)

VALIDATION_FRACTION = 0.1
SPLIT_SEED = 0
TRANSITIONS_PER_PLAN = HORIZON * 5


@dataclass(frozen=True)
class Pool:
    """Candidate plans executed once, with the true outcomes every branch is scored against."""

    plans: torch.Tensor  # [S, H, 10]
    realized: torch.Tensor  # [S, H, 192]
    outcomes: tuple[TaskOutcome, ...]
    task_cost: np.ndarray  # [S] true semantic task cost


@dataclass(frozen=True)
class SharedCase:
    """A held-out case and the branch-independent pools it is scored on."""

    case: PushTCase
    latent: torch.Tensor
    goal: torch.Tensor
    pools: dict[str, Pool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="base LeWM weights.pt")
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--acquisition-dir", type=Path, required=True, help="{strategy}-seed{n}.h5")
    parser.add_argument("--manifest", type=Path, required=True, help="frozen evaluation cases")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamics-dir", type=Path)
    parser.add_argument("--protocol", type=Path, help="pre-registered protocol, hashed into rows")
    parser.add_argument("--branches", nargs="*", help="strategy[:budget[:seed]], else the grid")
    parser.add_argument("--strategies", nargs="+", default=["random", "uncertainty", "planner"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2500, 10000, 40000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--manifest-cases", type=int, default=48)
    parser.add_argument("--manifest-seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repair-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--candidates", type=int, default=32, help="plans per shared pool")
    parser.add_argument("--evaluation-seed", type=int, default=7000)
    parser.add_argument("--validation-windows", type=int, default=20_000)
    parser.add_argument("--held-out-seed", type=int, default=909, help="held-out subsample")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def load_manifest(
    path: Path, dataset: Path, *, num_cases: int, seed: int, validation: np.ndarray
) -> list[PushTCase]:
    """The frozen evaluation cases, drawn only from validation-split episodes.

    The manifest is written the first time and verified afterwards, so the cases cannot drift
    once repair results have been seen.
    """
    cases = sample_pusht_cases(dataset, num_cases=num_cases, seed=seed, episodes=validation)
    identity = [
        {"episode": case.episode, "start_step": case.start_step, "goal_step": case.goal_step}
        for case in cases
    ]
    if path.exists():
        stored = json.loads(path.read_text())
        if stored["cases"] != identity:
            raise ValueError(f"{path} does not match the cases this configuration draws")
    else:
        path.write_text(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "num_cases": num_cases,
                    "seed": seed,
                    "split_seed": SPLIT_SEED,
                    "validation_fraction": VALIDATION_FRACTION,
                    "cases": identity,
                },
                indent=1,
            )
        )
    return cases


def build_pools(
    cases: list[PushTCase],
    base: ReferenceLeWM,
    env: gym.Env,
    *,
    stats: ActionStats,
    bounds: tuple[torch.Tensor, torch.Tensor],
    candidates: int,
    seed: int,
    device: str,
) -> list[SharedCase]:
    """Execute both candidate pools of every case once; all branches reuse these outcomes."""
    settings = planner_settings(*PRESSURES["P2"])["cem"]
    shared = []
    for index, case in enumerate(cases):
        goal = encode_frame(base, render_pusht_goal(env, case), device)
        reconstruct_pusht_start(env, case)
        latent = encode_frame(base, env.render(), device)
        region = cem(
            base,
            latent,
            goal,
            horizon=HORIZON,
            num_samples=settings[0],
            iterations=settings[1],
            elite_fraction=ELITE_FRACTION,
            bounds=bounds,
            generator=torch.Generator(bounds[0].device).manual_seed(seed + index),
        ).final
        pools = {}
        for name, proposal in (("q0", initial_proposal(HORIZON)), ("planner_region", region)):
            generator = torch.Generator(bounds[0].device).manual_seed(seed + index)
            plans = sample_proposal(
                proposal, num_samples=candidates, bounds=bounds, generator=generator
            )
            frames, outcomes = zip(
                *(rollout_pusht(env, case, stats, plan) for plan in plans), strict=True
            )
            realized = base.encode(frames_to_tensor(np.stack(frames), device))
            pools[name] = Pool(
                plans,
                realized,
                tuple(outcomes),
                np.array([outcome["task_cost"] for outcome in outcomes]),
            )
        shared.append(SharedCase(case, latent, goal, pools))
        if (index + 1) % 8 == 0:
            print(f"   pools for {index + 1}/{len(cases)} cases", flush=True)
    return shared


def score(
    model: ReferenceLeWM,
    shared: SharedCase,
    env: gym.Env,
    *,
    stats: ActionStats,
    bounds: tuple[torch.Tensor, torch.Tensor],
    seed: int,
    device: str,
) -> dict:
    """One branch on one case: ranking on the shared pools, then its own planning."""
    row = {}
    for name, pool in shared.pools.items():
        ranked = compare_with_model(
            model, shared.latent, shared.goal, pool.plans, pool.realized, pool.outcomes
        )
        chosen = int(ranked.predicted_cost.argmin())
        row |= {
            f"{name}_regret": float(pool.task_cost[chosen] - pool.task_cost.min()),
            f"{name}_chosen_task_cost": float(pool.task_cost[chosen]),
            f"{name}_best_task_cost": float(pool.task_cost.min()),
            f"{name}_absolute_optimism": float(np.abs(ranked.optimism).mean()),
            f"{name}_rollout_error": float(ranked.rollout_error.mean()),
            f"{name}_terminal_error": float(ranked.terminal_error.mean()),
            f"{name}_rank_dynamics": pairwise_rank_agreement(
                ranked.predicted_cost, ranked.realized_cost
            ),
            f"{name}_rank_objective": pairwise_rank_agreement(ranked.realized_cost, pool.task_cost),
            f"{name}_rank_end_to_end": pairwise_rank_agreement(
                ranked.predicted_cost, pool.task_cost
            ),
        }

    settings = planner_settings(*PRESSURES["P3"])["cem"]
    # the open-loop and closed-loop measurements are separate estimates, so they plan separately
    plan = make_planner("cem", model, bounds, *settings, seed)(shared.latent, shared.goal)
    frames, outcome = rollout_pusht(env, shared.case, stats, plan.actions)
    realized = model.encode(frames_to_tensor(frames[None], device))
    selected = compare_with_model(
        model, shared.latent, shared.goal, plan.actions[None], realized, [outcome]
    )
    row |= {
        "selected_predicted_cost": float(selected.predicted_cost[0]),
        "selected_optimism": float(selected.optimism[0]),
        "selected_absolute_optimism": float(np.abs(selected.optimism[0])),
        "selected_rollout_error": float(selected.rollout_error[0]),
        "selected_terminal_error": float(selected.terminal_error[0]),
        "selected_task_cost": float(selected.task_cost[0]),
        "selected_semantic_success": bool(selected.task["semantic_final_success"][0]),
    }
    mpc = run_pusht_mpc(
        env,
        model,
        shared.case,
        stats,
        shared.goal,
        make_planner("cem", model, bounds, *settings, seed + 500_000),
    )
    row |= {"mpc_raw_steps": mpc.raw_steps, **{f"mpc_{k}": v for k, v in mpc.task.items()}}
    return row


def grid(args) -> list[dict]:
    """Every branch to train and evaluate, as (strategy, budget, seed)."""
    if args.branches:
        specs = []
        for entry in args.branches:
            parts = entry.split(":")
            strategy = parts[0]
            budget = int(parts[1]) if len(parts) > 1 else 0
            seed = int(parts[2]) if len(parts) > 2 else None
            specs.append({"strategy": strategy, "budget": budget, "seed": seed})
        return specs
    return [
        {"strategy": "original_base", "budget": 0, "seed": None},
        *({"strategy": "continued", "budget": 0, "seed": seed} for seed in args.seeds),
        *(
            {"strategy": strategy, "budget": budget, "seed": seed}
            for budget in args.budgets
            for strategy in args.strategies
            for seed in args.seeds
        ),
    ]


def main() -> None:
    args = parse_args()
    stats = action_stats(args.dataset)
    bounds = model_action_bounds(stats, device=args.device)
    base = ReferenceLeWM.from_checkpoint(args.checkpoint, device=args.device)
    env = make_env("pusht")

    with h5py.File(args.dataset, "r") as file:
        num_episodes = len(file["ep_len"])
    validation = split_episodes(
        num_episodes, validation_fraction=VALIDATION_FRACTION, seed=SPLIT_SEED
    )
    cache = load_latent_cache(args.latent_cache, args.dataset, args.checkpoint)
    base_windows = latent_windows(cache, args.dataset, stats, episodes=~validation)
    held_out_windows = subsample_windows(
        latent_windows(cache, args.dataset, stats, episodes=validation),
        args.validation_windows,
        seed=args.held_out_seed,
    )
    cases = load_manifest(
        args.manifest,
        args.dataset,
        num_cases=args.manifest_cases,
        seed=args.manifest_seed,
        validation=validation,
    )
    print(
        f"{len(cases)} held-out cases from {validation.sum()} validation episodes | "
        f"base windows {len(base_windows):,} | manifest {file_digest(args.manifest)[:12]}"
    )
    print(
        f"held-out loss on {len(held_out_windows):,} windows from "
        f"{windows_episodes(held_out_windows, cache.offsets)} episodes (seed {args.held_out_seed})"
    )
    shared = build_pools(
        cases,
        base,
        env,
        stats=stats,
        bounds=bounds,
        candidates=args.candidates,
        seed=args.evaluation_seed,
        device=args.device,
    )

    identity = {
        "git_commit": git_commit(),
        "dynamics_protocol": DYNAMICS_PROTOCOL,
        "base_checkpoint": file_digest(args.checkpoint),
        "dataset": cache_identity(args.dataset, args.checkpoint),
        "manifest": file_digest(args.manifest),
        "protocol": file_digest(args.protocol) if args.protocol else None,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "repair_fraction": args.repair_fraction,
        "candidates": args.candidates,
        "held_out_seed": args.held_out_seed,
        "held_out_windows": len(held_out_windows),
        "held_out_episodes": windows_episodes(held_out_windows, cache.offsets),
    }
    rows = []
    with args.output.open("w") as out:
        for spec in grid(args):
            strategy, budget, seed = spec["strategy"], spec["budget"], spec["seed"]
            name = (
                f"{strategy}"
                + (f"-B{budget}" if budget else "")
                + (f"-s{seed}" if seed is not None else "")
            )
            model, training = train_branch(args, spec, base_windows, held_out_windows, name=name)
            for index, case in enumerate(shared):
                row = {
                    "branch": name,
                    "strategy": strategy,
                    "budget": budget,
                    "seed": seed,
                    "case": index,
                    "episode": case.case.episode,
                    **training,
                    **identity,
                    **score(
                        model,
                        case,
                        env,
                        stats=stats,
                        bounds=bounds,
                        seed=args.evaluation_seed + index,
                        device=args.device,
                    ),
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                rows.append(row)
            print(f"-- {name}: {len(shared)} cases scored", flush=True)
    summarize(rows)


def train_branch(args, spec, base_windows, held_out, *, name):
    """Train one branch from the released checkpoint, or return it untouched."""
    strategy, budget, seed = spec["strategy"], spec["budget"], spec["seed"]
    if strategy == "original_base":
        model = TrainableDynamics.from_checkpoint(args.checkpoint, device=args.device)
        loss = validation_loss(model, held_out, batch_size=args.batch_size)
        return model.model.eval(), {"held_out_loss": loss, "transitions": 0, "windows": 0}

    if seed is None:
        raise ValueError(f"branch {name} trains and needs an explicit seed")
    model = TrainableDynamics.from_checkpoint(args.checkpoint, device=args.device)
    acquired, windows, digest = None, 0, None
    if budget:
        path = args.acquisition_dir / f"{strategy}-seed{seed}.h5"
        latents, blocks, total = load_acquisitions(
            path,
            horizon=HORIZON,
            expect={
                "environment": "pusht",
                "base_checkpoint": file_digest(args.checkpoint),
                "dynamics_protocol": DYNAMICS_PROTOCOL,  # v2 streams were chosen for recorded goals
            },
        )
        plans = budget // TRANSITIONS_PER_PLAN
        if plans > len(latents):
            raise ValueError(f"{path} holds {total} transitions, budget needs {budget}")
        acquired = repair_windows(latents[:plans], blocks[:plans])  # budgets are nested prefixes
        windows, digest = len(acquired), file_digest(path)
    print(f"\n== {name}: {budget} transitions, {windows} new windows", flush=True)
    history = repair_dynamics(
        model,
        base_windows,
        acquired,
        steps=args.steps,
        batch_size=args.batch_size,
        repair_fraction=args.repair_fraction if budget else 0.0,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=seed,
        validation=held_out,
        log_every=max(1, args.steps // 2),
    )
    training = {
        "held_out_loss": history[-1]["validation_loss"],
        "transitions": budget,
        "windows": windows,
        "training_seed": seed,
        "window_resampling": (
            args.steps * args.batch_size * args.repair_fraction / windows if windows else 0.0
        ),
        "acquisition_digest": digest,
    }
    if args.dynamics_dir:
        args.dynamics_dir.mkdir(parents=True, exist_ok=True)
        save_dynamics(
            args.dynamics_dir / f"{name}.pt",
            model,
            {"branch": name, "strategy": strategy, "seed": seed, **training},
        )
    return model.model.eval(), training


SUMMARY_KEYS = (
    "planner_region_regret",
    "q0_regret",
    "selected_absolute_optimism",
    "selected_rollout_error",
    "mpc_semantic_reached_success",  # the only MPC success flag; see run_pusht_mpc
    "held_out_loss",
)


def summarize(rows: list[dict]) -> None:
    """Per-branch case means, then seed-level inference for the strategy contrasts.

    The 48 cases are repeated measurements of one trained model, not independent replicates of an
    acquisition method, so anything comparing methods is aggregated per seed first and the seeds
    are the unit of inference. Case-level spread is printed as a descriptive only.
    """
    groups = dict.fromkeys((row["strategy"], row["budget"]) for row in rows)
    print(
        f"\n{'branch':24}{'seeds':>7}"
        + "".join(f"{key.replace('_', ' '):>26}" for key in SUMMARY_KEYS)
    )
    for strategy, budget in groups:
        per_seed = seed_means(rows, strategy, budget)
        cells = []
        for key in SUMMARY_KEYS:
            values = np.array([means[key] for means in per_seed.values()])
            spread = (
                f" ± {values.std(ddof=1) / np.sqrt(len(values)):.4f}" if len(values) > 1 else ""
            )
            cells.append(f"{values.mean():.4f}{spread}")
        label = f"{strategy}" + (f"-B{budget}" if budget else "")
        print(f"{label:24}{len(per_seed):>7}" + "".join(f"{cell:>26}" for cell in cells))

    contrasts = [("random", "continued"), ("uncertainty", "random"), ("planner", "random")]
    budgets = sorted({budget for _, budget in groups if budget})
    for key in SUMMARY_KEYS[:-1]:
        print(f"\n-- {key}: paired per-seed differences (negative favours the first)")
        for budget in budgets:
            for strategy, reference in contrasts:
                report_contrast(rows, strategy, budget, reference, key)


def seed_means(rows: list[dict], strategy: str, budget: int) -> dict:
    """Mean over the held-out cases, per training seed."""
    means = {}
    for seed in sorted(
        {row["seed"] for row in rows if row["strategy"] == strategy and row["budget"] == budget},
        key=lambda value: (value is None, value),
    ):
        group = [
            row
            for row in rows
            if row["strategy"] == strategy and row["budget"] == budget and row["seed"] == seed
        ]
        means[seed] = {key: float(np.nanmean([row[key] for row in group])) for key in SUMMARY_KEYS}
    return means


def report_contrast(rows, strategy: str, budget: int, reference: str, key: str) -> None:
    """One strategy against a reference: per-seed differences, then the seed-level estimate."""
    treatment = seed_means(rows, strategy, budget)
    control = seed_means(rows, reference, budget if reference != "continued" else 0)
    seeds = sorted(set(treatment) & set(control))
    if not seeds:
        return
    differences = np.array([treatment[seed][key] - control[seed][key] for seed in seeds])
    line = f"   B={budget:<6} {strategy:11} - {reference:11} {differences.mean():+10.5f}"
    if len(differences) > 1:
        sd = differences.std(ddof=1)
        line += f" sd {sd:.5f} se {sd / np.sqrt(len(differences)):.5f}"
    line += f"  n={len(seeds)} seeds, {(differences < 0).sum()} negative"
    print(line + "   per seed " + " ".join(f"{value:+.5f}" for value in differences))


if __name__ == "__main__":
    main()
