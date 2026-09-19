"""Latent planners for the reference LeWM: random shooting and bounded CEM.

A plan is [H, 10]: H action blocks in the model's normalized action space. Both planners draw
bounded proposal samples: a diagonal Gaussian clamped to the model-space image of the raw action
bounds [-1, 1], so every plan they evaluate or return is executable. They only roll out cached
latents and never encode images.

A CEM run with N samples and I iterations evaluates N * I plans (PlanResult.model_evaluations);
random shooting with that many samples from the same initial proposal is its matched control.
Paired random innovations: separate generators with the same seed give both planners the same
underlying noise, so random shooting's first N samples are exactly CEM's first iteration.
"""

from dataclasses import dataclass

import torch

from planner_atlas.data import (
    BLOCK_STEPS,
    ENV_ACTION_DIM,
    ActionStats,
    denormalize_actions,
    normalize_actions,
    pack_action_blocks,
    unpack_action_blocks,
)
from planner_atlas.models.reference_lewm import ACTION_DIM, ReferenceLeWM


@dataclass(frozen=True)
class Proposal:
    """Pre-clamp diagonal Gaussian over plans: mean and std [H, 10], on the CPU."""

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


def bound_fraction(blocks: torch.Tensor, stats: ActionStats, *, tolerance: float = 1e-5) -> float:
    """Fraction of the raw action coordinates of plans [..., 10] that sit at a bound of [-1, 1].

    Plans made of clamped samples are the proposal pressing against the action constraint rather
    than shaping a trajectory inside it.
    """
    actions = denormalize_actions(unpack_action_blocks(blocks), stats)
    return float((actions.abs() >= 1.0 - tolerance).to(torch.float64).mean())


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
    initial = _initial_proposal(horizon)
    plans = sample_proposal(initial, num_samples=num_samples, bounds=bounds, generator=generator)
    costs = _score(model, latent, goal, plans, score_batch_size)
    best = costs.argmin()
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
    initial = proposal = _initial_proposal(horizon)
    history, best_plans = [], []
    for _ in range(iterations):
        plans = sample_proposal(
            proposal, num_samples=num_samples, bounds=bounds, generator=generator
        )
        costs = _score(model, latent, goal, plans, score_batch_size)
        elite_costs, elite_indices = costs.topk(num_elites, largest=False)
        history.append(CEMIteration(proposal, elite_costs[0].item(), elite_costs.mean().item()))
        best_plans.append(plans[elite_indices[0]])
        elites = plans[elite_indices]
        std = elites.std(dim=0, correction=0).clamp(min=min_std)
        proposal = Proposal(elites.mean(dim=0).cpu(), std.cpu())
    best = min(range(iterations), key=lambda i: history[i].best_cost)
    return PlanResult(
        best_plans[best],
        history[best].best_cost,
        num_samples * iterations,
        initial,
        proposal,
        tuple(history),
    )


def sample_proposal(
    proposal: Proposal,
    *,
    num_samples: int,
    bounds: tuple[torch.Tensor, torch.Tensor],
    generator: torch.Generator,
) -> torch.Tensor:
    """Bounded proposal samples [num_samples, H, 10]: mean + std * epsilon, clamped to bounds.

    Samples are drawn on the bounds' device, which generator must share.
    """
    low, high = bounds
    mean, std = proposal.mean.to(low.device), proposal.std.to(low.device)
    noise = torch.randn((num_samples, *mean.shape), generator=generator, device=low.device)
    return (mean + std * noise).clamp(low, high)


def _initial_proposal(horizon: int) -> Proposal:
    return Proposal(torch.zeros(horizon, ACTION_DIM), torch.ones(horizon, ACTION_DIM))


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
