"""Planner Atlas quantities: selection amplification, adaptive exploitation and ranking quality.

A selected plan is compared with audit plans sampled from bounded proposals, each drawn with a fresh
generator so auditing never changes planning. Random shooting and CEM at the same case and pressure
share one initial-proposal (q0) audit. CEM's final-proposal (qfinal) audit uses the same audit seed:
paired random innovations, the same noise applied to different proposal parameters.

Latent optimism O = realized latent cost - predicted latent cost (see CandidateEvaluation).
Ranking agreement is decomposed into dynamics (predicted vs realized latent cost), objective
(realized latent cost vs task cost) and end to end (predicted latent cost vs task cost).

The pressures, the matched budgets and the seeding below are the experiment protocol shared by
every environment; nothing in this module is specific to one.
"""

from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from planner_atlas.evaluation import CandidateEvaluation
from planner_atlas.models.reference_lewm import ReferenceLeWM
from planner_atlas.planning import PlanResult, Proposal, cem, random_shooting, sample_proposal

PRESSURES = {"P0": (128, 1), "P1": (128, 2), "P2": (256, 4), "P3": (512, 6)}  # CEM N and I
PLANNERS = ("random_shooting", "cem")
HORIZON, ELITE_FRACTION = 5, 0.1
SEED_STRIDE = 1000  # seed = base + SEED_STRIDE * case + pressure index

Planner = Callable[[torch.Tensor, torch.Tensor], PlanResult]


def planner_settings(num_samples: int, iterations: int) -> dict[str, tuple[int, int]]:
    """Samples and iterations per planner at one pressure: random shooting spends the whole CEM
    budget in a single iteration."""
    return {"random_shooting": (num_samples * iterations, 1), "cem": (num_samples, iterations)}


def make_planner(
    name: str,
    model: ReferenceLeWM,
    bounds: tuple[torch.Tensor, torch.Tensor],
    num_samples: int,
    iterations: int,
    seed: int,
) -> Planner:
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
    """Quantities of a selected plan (one candidate) against the q0 audit and, for CEM, qfinal.

    The selected plan's task outcome is reported as the environment named it; the audits enter
    only through their latent quantities and their ranking against the task cost.
    """
    metrics = {
        "selected_predicted_cost": float(selected.predicted_cost[0]),
        "selected_realized_latent_cost": float(selected.realized_cost[0]),
        "selected_optimism": float(selected.optimism[0]),
        "selected_rollout_error": float(selected.rollout_error[0]),
        "selected_terminal_error": float(selected.terminal_error[0]),
        **{f"selected_{name}": values[0].item() for name, values in selected.task.items()},
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


def print_summary(rows: list[dict], labels: dict[str, str]) -> None:
    """Mean ± standard error per pressure and planner of the labelled quantities present in rows.

    A rank is NaN on cases where no audited pair is comparable; those cases are left out of the
    column rather than turning the whole column into NaN.
    """
    labels = {key: label for key, label in labels.items() if any(key in row for row in rows)}
    print(f"{'':3} {'planner':16}" + "".join(f"{label:>20}" for label in labels.values()))
    for pressure in PRESSURES:
        for name in PLANNERS:
            group = [row for row in rows if row["pressure"] == pressure and row["planner"] == name]
            cells = []
            for key in labels:
                values = np.array([row[key] for row in group if key in row], dtype=float)
                values = values[~np.isnan(values)]
                sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
                cells.append(f"{values.mean():.3f} ± {sem:.3f}" if len(values) else "-")
            print(f"{pressure:3} {name:16}" + "".join(f"{cell:>20}" for cell in cells))


def _audit_metrics(name: str, audit: CandidateEvaluation) -> dict[str, float]:
    predicted, realized, task = audit.predicted_cost, audit.realized_cost, audit.task_cost
    return {
        f"{name}_mean_optimism": float(audit.optimism.mean()),
        f"{name}_mean_rollout_error": float(audit.rollout_error.mean()),
        f"{name}_rank_dynamics": pairwise_rank_agreement(predicted, realized),
        f"{name}_rank_objective": pairwise_rank_agreement(realized, task),
        f"{name}_rank_end_to_end": pairwise_rank_agreement(predicted, task),
    }
