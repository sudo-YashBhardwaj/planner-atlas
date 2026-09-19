"""TwoRoom Planner Atlas: matched-budget random shooting vs CEM, audited in the simulator.

    uv run python scripts/tworoom_atlas.py --dataset tworoom.h5 --checkpoint weights.pt \
        --output tworoom_atlas.jsonl

Writes one JSON row per case x pressure x planner (Atlas quantities, then mpc_* fields) and
prints mean ± standard error of the main quantities per pressure and planner. Random shooting and
CEM get separate generators with the same seed (paired random innovations) and share one
initial-proposal audit per case and pressure.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from planner_atlas.atlas import atlas_metrics, audit_samples
from planner_atlas.data import action_stats
from planner_atlas.envs import make_env
from planner_atlas.evaluation import (
    encode_frame,
    evaluate_candidates,
    run_mpc,
    sample_tworoom_cases,
)
from planner_atlas.models.reference_lewm import ReferenceLeWM
from planner_atlas.planning import cem, model_action_bounds, random_shooting

PRESSURES = {"P0": (128, 1), "P1": (128, 2), "P2": (256, 4), "P3": (512, 6)}  # CEM N and I
PLANNERS = ("random_shooting", "cem")
HORIZON, ELITE_FRACTION = 5, 0.1
SEED_STRIDE = 1000  # seed = base + SEED_STRIDE * case + pressure index
SUMMARY = {
    "selected_optimism": "optimism",
    "selection_amplification": "amplification",
    "selected_task_cost": "task cost",
    "selected_success": "open-loop success",
    "mpc_success": "MPC success",
    "adaptive_optimism_shift": "adapt. optimism",
    "adaptive_rollout_error_shift": "adapt. rollout err",
    "q0_rank_end_to_end": "q0 rank e2e",
}


def make_planner(name, model, bounds, num_samples, iterations, seed):
    """Planning function of (latent, goal) with its own generator, seeded with seed."""
    common = {"horizon": HORIZON, "num_samples": num_samples, "bounds": bounds}
    generator = torch.Generator(bounds[0].device).manual_seed(seed)
    if name == "random_shooting":
        return partial(random_shooting, model, **common, generator=generator)
    return partial(
        cem,
        model,
        **common,
        iterations=iterations,
        elite_fraction=ELITE_FRACTION,
        generator=generator,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="official tworoom.h5")
    parser.add_argument("--checkpoint", type=Path, required=True, help="lewm-tworooms weights.pt")
    parser.add_argument("--output", type=Path, required=True, help="JSONL results file")
    parser.add_argument("--num-cases", type=int, default=100)
    parser.add_argument("--case-seed", type=int, default=42)
    parser.add_argument("--planner-seed", type=int, default=0)
    parser.add_argument("--audit-seed", type=int, default=1_000_000)
    parser.add_argument("--audit-samples", type=int, default=32, help="per audited proposal")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = action_stats(args.dataset)
    bounds = model_action_bounds(stats, device=args.device)
    model = ReferenceLeWM.from_checkpoint(args.checkpoint, device=args.device)
    env = make_env("tworoom")
    cases = sample_tworoom_cases(args.dataset, num_cases=args.num_cases, seed=args.case_seed)
    rows = []
    with args.output.open("w") as out:
        for c, case in enumerate(cases):
            latent = encode_frame(model, case.start_frame, args.device)
            goal = encode_frame(model, case.goal_frame, args.device)
            evaluate = partial(evaluate_candidates, model, env, case, stats, latent, goal)
            for p, (pressure, (num_samples, iterations)) in enumerate(PRESSURES.items()):
                planner_seed = args.planner_seed + SEED_STRIDE * c + p
                audit_seed = args.audit_seed + SEED_STRIDE * c + p
                audit = partial(
                    audit_samples, num_samples=args.audit_samples, bounds=bounds, seed=audit_seed
                )
                # random shooting spends the whole CEM budget in a single iteration
                settings = {
                    "random_shooting": (num_samples * iterations, 1),
                    "cem": (num_samples, iterations),
                }
                results = {}
                for name in PLANNERS:
                    plan = make_planner(name, model, bounds, *settings[name], planner_seed)
                    results[name] = plan(latent, goal)
                initial = evaluate(audit(results["cem"].initial))  # one q0 audit for both planners
                for name, result in results.items():
                    final = evaluate(audit(result.final)) if result.iterations else None
                    planner = make_planner(name, model, bounds, *settings[name], planner_seed)
                    mpc = run_mpc(env, model, case, stats, goal, planner)
                    row = {
                        "case": c,
                        "episode": case.episode,
                        "start_step": case.start_step,
                        "goal_step": case.goal_step,
                        "initial_distance": case.initial_distance,
                        "pressure": pressure,
                        "planner": name,
                        "num_samples": settings[name][0],
                        "iterations": settings[name][1],
                        "model_evaluations": result.model_evaluations,
                        "planner_seed": planner_seed,
                        "audit_seed": audit_seed,
                        **atlas_metrics(evaluate(result.actions[None]), initial, final),
                        "mpc_success": mpc.success,
                        "mpc_raw_steps": mpc.raw_steps,
                        "mpc_final_distance": mpc.final_distance,
                    }
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    rows.append(row)
    print_summary(rows)


def print_summary(rows: list[dict]) -> None:
    print(f"{'':3} {'planner':16}" + "".join(f"{label:>20}" for label in SUMMARY.values()))
    for pressure in PRESSURES:
        for name in PLANNERS:
            group = [row for row in rows if row["pressure"] == pressure and row["planner"] == name]
            cells = []
            for key in SUMMARY:
                values = np.array([row[key] for row in group if key in row], dtype=float)
                sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
                cells.append(f"{values.mean():.3f} ± {sem:.3f}" if len(values) else "-")
            print(f"{pressure:3} {name:16}" + "".join(f"{cell:>20}" for cell in cells))


if __name__ == "__main__":
    main()
