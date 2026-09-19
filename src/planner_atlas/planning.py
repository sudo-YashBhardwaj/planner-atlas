"""Latent planners for the reference LeWM: random shooting and bounded CEM.

A plan is [H, 10]: H action blocks in the model's normalized action space. Both planners
sample plans from a diagonal Gaussian and clamp them to the model-space image of the raw action
bounds [-1, 1], so every plan they evaluate or return is executable. They only roll out cached
latents and never encode images.

A CEM run with N samples and I iterations evaluates N * I plans (PlanResult.model_evaluations);
random shooting with that many samples from the same initial proposal is its matched control.
With identically seeded generators, its first N plans are exactly CEM's first iteration.
"""

from dataclasses import dataclass

import torch

from planner_atlas.data import (
    BLOCK_STEPS,
    ENV_ACTION_DIM,
    ActionStats,
    normalize_actions,
    pack_action_blocks,
)
from planner_atlas.models.reference_lewm import ACTION_DIM, ReferenceLeWM


@dataclass(frozen=True)
class Proposal:
    """Diagonal Gaussian over plans: mean and std [H, 10], on the CPU. Samples are clamped."""

    mean: torch.Tensor
    std: torch.Tensor


@dataclass(frozen=True)
class CEMIteration:
    proposal: Proposal  # the distribution this iteration sampled from
    best_cost: float
    elite_mean_cost: float


@dataclass(frozen=True)
class PlanResult:
    actions: torch.Tensor  # [H, 10] lowest-cost plan evaluated, on the planning device
    predicted_cost: float
    model_evaluations: int
    initial: Proposal
    final: Proposal  # after the last CEM update; the initial proposal for random shooting
    iterations: tuple[CEMIteration, ...] = ()


def model_action_bounds(
    stats: ActionStats, *, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounds [10] of a normalized action block whose raw actions all lie in [-1, 1]."""
    raw = torch.tensor([-1.0, 1.0]).view(2, 1, 1).expand(2, BLOCK_STEPS, ENV_ACTION_DIM)
    low, high = pack_action_blocks(normalize_actions(raw, stats))
    return low.to(device), high.to(device)


@torch.no_grad()
def random_shooting(
    model: ReferenceLeWM,
    latent: torch.Tensor,
    goal: torch.Tensor,
    *,
    horizon: int,
    num_samples: int,
    bounds: tuple[torch.Tensor, torch.Tensor],
    generator: torch.Generator,
    score_batch_size: int = 1024,
) -> PlanResult:
    """Lowest-cost plan among num_samples independent samples of the initial proposal."""
    if horizon < 1 or num_samples < 1:
        raise ValueError("expected positive horizon and num_samples")
    mean = torch.zeros(horizon, ACTION_DIM, device=latent.device)
    std = torch.ones_like(mean)
    plans = _sample(mean, std, num_samples, bounds, generator)
    costs = _score(model, latent, goal, plans, score_batch_size)
    best = costs.argmin()
    initial = Proposal(mean.cpu(), std.cpu())
    return PlanResult(plans[best], costs[best].item(), num_samples, initial, initial)


@torch.no_grad()
def cem(
    model: ReferenceLeWM,
    latent: torch.Tensor,
    goal: torch.Tensor,
    *,
    horizon: int,
    num_samples: int,
    iterations: int,
    elite_fraction: float,
    bounds: tuple[torch.Tensor, torch.Tensor],
    generator: torch.Generator,
    min_std: float = 0.05,
    score_batch_size: int = 1024,
) -> PlanResult:
    """Cross-entropy method returning the lowest-cost plan evaluated in any iteration.

    Each iteration samples num_samples plans from the current proposal, scores them, and refits
    the proposal to the lowest-cost elites: their mean and population std, the std floored at
    min_std (in normalized action units). The initial proposal is N(0, 1).
    """
    if horizon < 1 or num_samples < 1 or iterations < 1 or not 0 < elite_fraction <= 1:
        raise ValueError(
            "expected positive horizon, num_samples, iterations; elite_fraction in (0, 1]"
        )
    num_elites = max(1, int(num_samples * elite_fraction))
    mean = torch.zeros(horizon, ACTION_DIM, device=latent.device)
    std = torch.ones_like(mean)
    initial = Proposal(mean.cpu(), std.cpu())
    history, best_plans = [], []
    for _ in range(iterations):
        plans = _sample(mean, std, num_samples, bounds, generator)
        costs = _score(model, latent, goal, plans, score_batch_size)
        elite_costs, elite_indices = costs.topk(num_elites, largest=False)
        proposal = Proposal(mean.cpu(), std.cpu())
        history.append(CEMIteration(proposal, elite_costs[0].item(), elite_costs.mean().item()))
        best_plans.append(plans[elite_indices[0]])
        elites = plans[elite_indices]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0, correction=0).clamp(min=min_std)
    best = min(range(iterations), key=lambda i: history[i].best_cost)
    final = Proposal(mean.cpu(), std.cpu())
    return PlanResult(
        best_plans[best],
        history[best].best_cost,
        num_samples * iterations,
        initial,
        final,
        tuple(history),
    )


def _sample(
    mean: torch.Tensor,
    std: torch.Tensor,
    num_samples: int,
    bounds: tuple[torch.Tensor, torch.Tensor],
    generator: torch.Generator,
) -> torch.Tensor:
    """Plans [num_samples, H, 10] from N(mean, std), clamped to bounds."""
    noise = torch.randn((num_samples, *mean.shape), generator=generator, device=mean.device)
    low, high = bounds
    return (mean + std * noise).clamp(low, high)


def _score(
    model: ReferenceLeWM,
    latent: torch.Tensor,
    goal: torch.Tensor,
    plans: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Costs [S] of plans [S, H, 10] from one state, scored batch_size plans at a time."""
    return torch.cat(
        [model.cost(latent, goal, chunk[None])[0] for chunk in plans.split(batch_size)]
    )
