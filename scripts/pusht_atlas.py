"""PushT Planner Atlas: matched-budget random shooting vs CEM, audited in the simulator.

    uv run python scripts/pusht_atlas.py --dataset pusht_expert_train.h5 --checkpoint weights.pt \
        --output pusht_atlas.jsonl

The TwoRoom protocol on PushT: one JSON row per case x pressure x planner, paired planners, one
shared initial-proposal audit per case and pressure. PushT differs in how a case start is reached
(replay from the episode start, see planner_atlas.pusht), in what the model observes (live renders
of the replayed start and goal, never recorded frames) and in the task: the semantic block pose
error is the task cost, and PushT's official success is reported next to it.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import gymnasium as gym
import torch

from planner_atlas.atlas import (
    PLANNERS,
    PRESSURES,
    SEED_STRIDE,
    atlas_metrics,
    audit_samples,
    make_planner,
    planner_settings,
    print_summary,
)
from planner_atlas.data import action_stats
from planner_atlas.envs import env_state, make_env
from planner_atlas.evaluation import encode_frame, evaluate_candidates
from planner_atlas.models.reference_lewm import ReferenceLeWM, terminal_cost
from planner_atlas.planning import bound_fraction, model_action_bounds
from planner_atlas.pusht import (
    BLOCK_POSE,
    PushTCase,
    block_pose_errors,
    reconstruct_pusht_start,
    render_pusht_goal,
    rollout_pusht,
    run_pusht_mpc,
    sample_pusht_cases,
)

SUMMARY = {
    "selected_optimism": "optimism",
    "selection_amplification": "amplification",
    "selected_task_cost": "task cost",
    "selected_semantic_final_success": "open-loop success",
    "mpc_semantic_reached_success": "MPC semantic",
    "mpc_official_reached_before_stop": "official pre-stop",
    "adaptive_optimism_shift": "adapt. optimism",
    "adaptive_rollout_error_shift": "adapt. rollout err",
    "q0_rank_end_to_end": "q0 rank e2e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="pusht_expert_train.h5")
    parser.add_argument("--checkpoint", type=Path, required=True, help="lewm-pusht weights.pt")
    parser.add_argument("--output", type=Path, required=True, help="JSONL results file")
    parser.add_argument("--num-cases", type=int, default=50)
    parser.add_argument("--case-seed", type=int, default=42)
    parser.add_argument("--planner-seed", type=int, default=0)
    parser.add_argument("--audit-seed", type=int, default=1_000_000)
    parser.add_argument("--audit-samples", type=int, default=32, help="per audited proposal")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def start_diagnostics(
    model: ReferenceLeWM,
    env: gym.Env,
    case: PushTCase,
    latent: torch.Tensor,
    device: torch.device | str,
) -> dict[str, float]:
    """How faithfully the reconstructed start reproduces the recording, and how far its live
    render sits from the recorded frame of the same state."""
    position_error, angle_error = block_pose_errors(
        env_state(env)[BLOCK_POSE], case.start_state[BLOCK_POSE]
    )
    recorded = encode_frame(model, case.start_frame, device)
    return {
        "start_reconstruction_position_error": float(position_error),
        "start_reconstruction_angle_error": float(angle_error),
        "start_frame_latent_gap": float(terminal_cost(recorded[None], latent)[0, 0]),
    }


def main() -> None:
    args = parse_args()
    stats = action_stats(args.dataset)
    bounds = model_action_bounds(stats, device=args.device)
    model = ReferenceLeWM.from_checkpoint(args.checkpoint, device=args.device)
    env = make_env("pusht")
    cases = sample_pusht_cases(args.dataset, num_cases=args.num_cases, seed=args.case_seed)
    rows = []
    with args.output.open("w") as out:
        for c, case in enumerate(cases):
            goal = encode_frame(model, render_pusht_goal(env, case), args.device)
            reconstruct_pusht_start(env, case)
            latent = encode_frame(model, env.render(), args.device)  # live, not the recorded frame
            diagnostics = start_diagnostics(model, env, case, latent, args.device)
            execute = partial(rollout_pusht, env, case, stats)
            evaluate = partial(evaluate_candidates, model, latent, goal, execute)
            for p, (pressure, (num_samples, iterations)) in enumerate(PRESSURES.items()):
                planner_seed = args.planner_seed + SEED_STRIDE * c + p
                audit_seed = args.audit_seed + SEED_STRIDE * c + p
                audit = partial(
                    audit_samples, num_samples=args.audit_samples, bounds=bounds, seed=audit_seed
                )
                settings = planner_settings(num_samples, iterations)
                results = {}
                for name in PLANNERS:
                    plan = make_planner(name, model, bounds, *settings[name], planner_seed)
                    results[name] = plan(latent, goal)
                q0 = audit(results["cem"].initial)  # one q0 audit for both planners
                initial = evaluate(q0)
                for name, result in results.items():
                    qfinal = audit(result.final) if result.iterations else None
                    final = None if qfinal is None else evaluate(qfinal)
                    planner = make_planner(name, model, bounds, *settings[name], planner_seed)
                    mpc = run_pusht_mpc(env, model, case, stats, goal, planner)
                    row = {
                        "case": c,
                        "episode": case.episode,
                        "start_step": case.start_step,
                        "goal_step": case.goal_step,
                        **case.initial_task,
                        **diagnostics,
                        "pressure": pressure,
                        "planner": name,
                        "num_samples": settings[name][0],
                        "iterations": settings[name][1],
                        "model_evaluations": result.model_evaluations,
                        "planner_seed": planner_seed,
                        "audit_seed": audit_seed,
                        **atlas_metrics(evaluate(result.actions[None]), initial, final),
                        "selected_bound_fraction": bound_fraction(result.actions, stats),
                        "q0_bound_fraction": bound_fraction(q0, stats),
                        "mpc_raw_steps": mpc.raw_steps,
                        **{f"mpc_{key}": value for key, value in mpc.task.items()},
                    }
                    if qfinal is not None:
                        row["qfinal_bound_fraction"] = bound_fraction(qfinal, stats)
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    rows.append(row)
    print_summary(rows, SUMMARY)


if __name__ == "__main__":
    main()
