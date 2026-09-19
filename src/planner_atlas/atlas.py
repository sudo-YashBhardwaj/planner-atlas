"""Planner Atlas quantities: selection amplification, adaptive exploitation and ranking quality.

A selected plan is compared with audit plans sampled from bounded proposals, each drawn with a fresh
generator so auditing never changes planning. Random shooting and CEM at the same case and pressure
share one initial-proposal (q0) audit. CEM's final-proposal (qfinal) audit uses the same audit seed:
paired random innovations, the same noise applied to different proposal parameters.

Latent optimism O = realized latent cost - predicted latent cost (see CandidateEvaluation).
Ranking agreement is decomposed into dynamics (predicted vs realized latent cost), objective
(realized latent cost vs task cost) and end to end (predicted latent cost vs task cost).
"""

import numpy as np
import torch

from planner_atlas.evaluation import ARENA_SIZE, SUCCESS_RADIUS, CandidateEvaluation
from planner_atlas.planning import Proposal, sample_proposal


def audit_samples(
    proposal: Proposal, *, num_samples: int, bounds: tuple[torch.Tensor, torch.Tensor], seed: int
) -> torch.Tensor:
    """Bounded audit plans from proposal, drawn with a fresh generator seeded with seed."""
    generator = torch.Generator(bounds[0].device).manual_seed(seed)
    return sample_proposal(proposal, num_samples=num_samples, bounds=bounds, generator=generator)


def atlas_metrics(
    selected: CandidateEvaluation,
    initial: CandidateEvaluation,
    final: CandidateEvaluation | None = None,
) -> dict[str, float | bool]:
    """Quantities of a selected plan (one candidate) against the q0 audit and, for CEM, qfinal."""
    metrics = {
        "selected_predicted_cost": float(selected.predicted_cost[0]),
        "selected_realized_latent_cost": float(selected.realized_cost[0]),
        "selected_optimism": float(selected.optimism[0]),
        "selected_rollout_error": float(selected.rollout_error[0]),
        "selected_terminal_error": float(selected.terminal_error[0]),
        "selected_task_distance": float(selected.distance[0]),
        "selected_task_cost": float(selected.distance[0] / ARENA_SIZE),
        "selected_success": bool(selected.distance[0] < SUCCESS_RADIUS),
        "selected_reached_goal": bool(selected.reached[0]),
        **_audit_metrics("q0", initial),
        "selection_amplification": float(selected.optimism[0] - initial.optimism.mean()),
    }
    if final is not None:
        metrics |= _audit_metrics("qfinal", final)
        metrics["adaptive_optimism_shift"] = (
            metrics["qfinal_mean_optimism"] - metrics["q0_mean_optimism"]
        )
        metrics["adaptive_rollout_error_shift"] = (
            metrics["qfinal_mean_rollout_error"] - metrics["q0_mean_rollout_error"]
        )
    return metrics


def pairwise_rank_agreement(predicted: np.ndarray, true: np.ndarray) -> float:
    """(concordant - discordant) / pairs untied in both rankings; NaN without such pairs.

    Pairs tied in either ranking are excluded, which makes this Goodman and Kruskal's gamma.
    """
    i, j = np.triu_indices(len(predicted), k=1)
    agreement = np.sign(predicted[i] - predicted[j]) * np.sign(true[i] - true[j])
    comparable = agreement != 0
    return float(agreement[comparable].mean()) if comparable.any() else float("nan")


def _audit_metrics(name: str, audit: CandidateEvaluation) -> dict[str, float]:
    predicted, realized, task = audit.predicted_cost, audit.realized_cost, audit.distance
    return {
        f"{name}_mean_optimism": float(audit.optimism.mean()),
        f"{name}_mean_rollout_error": float(audit.rollout_error.mean()),
        f"{name}_rank_dynamics": pairwise_rank_agreement(predicted, realized),
        f"{name}_rank_objective": pairwise_rank_agreement(realized, task),
        f"{name}_rank_end_to_end": pairwise_rank_agreement(predicted, task),
    }
