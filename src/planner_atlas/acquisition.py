"""Acquisition: which simulator interactions to pay for, and what they buy.

Three policies choose one plan to execute from a case start, each charged the same budget of true
simulator transitions:

- random: one plan from the proposal the planners themselves start from, using no model
  information at all, so it is extra interaction without targeting;
- uncertainty: the most uncertain plan of a candidate set drawn from that same proposal, scored by
  the ensemble's canonical uncertainty (planner_atlas.uncertainty);
- planner: the plan CEM returns, the lowest predicted cost among everything it evaluated. This is
  planner-selected, not exploitation-selected: whether the plan exploited the model is only known
  after it runs.

The scientific constraint is that a selector may read only what exists before the transition is
paid for - proposals, model predictions, ensemble disagreement - so selectors take (latent, goal)
and never touch the environment. Realized latent cost, optimism, rollout error and task outcome
are computed after execution and stored beside the trajectory, where they cannot reach a selector.

Reconstructing a case start replays recorded actions, which re-creates recorded data rather than
buying new interaction, so only the executed plan counts against a budget: one plan of H blocks
costs H * 5 transitions.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from planner_atlas.data import BLOCK_STEPS
from planner_atlas.evaluation import TaskOutcome, compare_with_model, frames_to_tensor
from planner_atlas.models.reference_lewm import ReferenceLeWM
from planner_atlas.planning import PlanResult, initial_proposal, sample_proposal
from planner_atlas.uncertainty import DynamicsEnsemble

STRATEGIES = ("random", "uncertainty", "planner")
STREAM_PLANS = 1600  # the fixed context pool a seed's stream is drawn from; budgets are prefixes
PROVENANCE = ("environment", "base_checkpoint", "dataset", "dynamics_protocol")


@dataclass(frozen=True)
class Selection:
    """A chosen plan and the evidence that was available before paying for it."""

    plan: torch.Tensor  # [H, 10] normalized action blocks
    candidate: int  # index in the set the strategy scored; 0 when it drew a single plan
    diagnostics: dict[str, float]
    model_evaluations: int


Selector = Callable[[torch.Tensor, torch.Tensor], Selection]


@dataclass(frozen=True)
class Acquisition:
    """One executed acquisition: the plan, what it cost, and what came back."""

    strategy: str
    case: int
    episode: int
    start_step: int
    candidate: int
    transitions: int  # true simulator transitions paid for this trajectory
    latents: np.ndarray  # [H + 1, 192]: the start latent, then one per executed block
    blocks: np.ndarray  # [H, 10] normalized action blocks executed
    selection: dict[str, float]  # what the strategy knew when it chose
    outcome: dict[str, float]  # what execution revealed, recorded but never selected on

    @property
    def record(self) -> dict:
        """This acquisition's scientific metadata, without the tensors."""
        return {
            "strategy": self.strategy,
            "case": self.case,
            "episode": self.episode,
            "start_step": self.start_step,
            "candidate": self.candidate,
            "transitions": self.transitions,
            **{f"selection_{key}": value for key, value in self.selection.items()},
            **{f"outcome_{key}": value for key, value in self.outcome.items()},
        }


def random_selector(
    *, horizon: int, bounds: tuple[torch.Tensor, torch.Tensor], generator: torch.Generator
) -> Selector:
    """One plan from the planners' own initial proposal: N(0, 1) over normalized action blocks,
    clamped to the model-space image of the raw bounds [-1, 1]."""

    def select(latent: torch.Tensor, goal: torch.Tensor) -> Selection:
        plans = sample_proposal(
            initial_proposal(horizon), num_samples=1, bounds=bounds, generator=generator
        )
        return Selection(plans[0], candidate=0, diagnostics={}, model_evaluations=0)

    return select


def uncertainty_selector(
    *,
    horizon: int,
    bounds: tuple[torch.Tensor, torch.Tensor],
    generator: torch.Generator,
    ensemble: DynamicsEnsemble,
    num_candidates: int,
) -> Selector:
    """The most uncertain candidate of a set drawn from the same proposal random draws from."""

    def select(latent: torch.Tensor, goal: torch.Tensor) -> Selection:
        plans = sample_proposal(
            initial_proposal(horizon),
            num_samples=num_candidates,
            bounds=bounds,
            generator=generator,
        )
        uncertainty = ensemble.uncertainty(latent, goal, plans[None])[0]  # [S]
        chosen = int(uncertainty.argmax())
        return Selection(
            plans[chosen],
            candidate=chosen,
            diagnostics={
                "uncertainty": float(uncertainty[chosen]),
                "candidate_mean_uncertainty": float(uncertainty.mean()),
            },
            model_evaluations=num_candidates * len(ensemble),
        )

    return select


def planner_selector(plan: Callable[[torch.Tensor, torch.Tensor], PlanResult]) -> Selector:
    """What the planner would do anyway: the lowest predicted cost it found."""

    def select(latent: torch.Tensor, goal: torch.Tensor) -> Selection:
        result = plan(latent, goal)
        return Selection(
            result.actions,
            candidate=0,
            diagnostics={"predicted_cost": result.predicted_cost},
            model_evaluations=result.model_evaluations,
        )

    return select


def stream_order(count: int, *, seed: int) -> np.ndarray:
    """The order a seed's acquisition stream visits its context pool.

    A permutation of the whole pool, so the first n of it are the same contexts whatever budget a
    run asks for: budget prefixes are nested by construction rather than by how rows happen to be
    sorted for storage.
    """
    return np.random.default_rng([seed, 0xACC]).permutation(count)


def plans_for_budget(budget: int, *, horizon: int) -> int:
    """How many plans a budget of true transitions buys; the budget must divide exactly, so that
    every strategy spends the same number of transitions and none is truncated mid-plan."""
    cost = horizon * BLOCK_STEPS
    if budget <= 0 or budget % cost:
        raise ValueError(
            f"a budget of {budget} transitions is not a whole number of {cost}-transition plans"
        )
    return budget // cost


def acquire(
    strategy: str,
    select: Selector,
    execute: Callable[[torch.Tensor], tuple[np.ndarray, TaskOutcome]],
    *,
    model: ReferenceLeWM,
    latent: torch.Tensor,
    goal: torch.Tensor,
    case: int,
    episode: int,
    start_step: int,
) -> Acquisition:
    """Select a plan, execute it once, and record both sides of the transaction.

    Selection happens strictly before execution and sees none of its results.
    """
    selection = select(latent, goal)
    frames, task = execute(selection.plan)
    realized = model.encode(frames_to_tensor(frames[None], latent.device))  # [1, H, 192]
    evaluation = compare_with_model(model, latent, goal, selection.plan[None], realized, [task])
    return Acquisition(
        strategy=strategy,
        case=case,
        episode=episode,
        start_step=start_step,
        candidate=selection.candidate,
        transitions=len(selection.plan) * BLOCK_STEPS,
        latents=torch.cat([latent, realized[0]]).cpu().numpy(),
        blocks=selection.plan.cpu().numpy(),
        selection={**selection.diagnostics, "model_evaluations": selection.model_evaluations},
        outcome={
            "predicted_cost": float(evaluation.predicted_cost[0]),
            "realized_cost": float(evaluation.realized_cost[0]),
            "optimism": float(evaluation.optimism[0]),
            "rollout_error": float(evaluation.rollout_error[0]),
            "terminal_error": float(evaluation.terminal_error[0]),
            **{key: value[0].item() for key, value in evaluation.task.items()},
        },
    )


def save_acquisitions(path: Path, acquisitions: Sequence[Acquisition], run: dict) -> None:
    """Training tensors in path, one metadata record per acquisition in its .jsonl sibling.

    The tensors stay narrow because training reads them every step; everything a later analysis
    needs about where a transition came from lives in the records.
    """
    if not acquisitions:
        raise ValueError("nothing to save")
    horizon = acquisitions[0].blocks.shape[0]
    with h5py.File(path, "w") as file:
        file["latents"] = np.stack([one.latents for one in acquisitions])
        file["blocks"] = np.stack([one.blocks for one in acquisitions])
        file.attrs["transitions"] = sum(one.transitions for one in acquisitions)
        file.attrs["trajectories"] = len(acquisitions)
        file.attrs["horizon"] = horizon
        file.attrs["transitions_per_plan"] = horizon * BLOCK_STEPS
        file.attrs["strategy"] = acquisitions[0].strategy
        for field in PROVENANCE:
            file.attrs[field] = json.dumps(run.get(field))
    with path.with_suffix(".jsonl").open("w") as out:
        for one in acquisitions:
            out.write(json.dumps({**run, **one.record}) + "\n")


def load_acquisitions(
    path: Path, *, horizon: int | None = None, expect: dict | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Latents [K, H + 1, 192], action blocks [K, H, 10], and the transitions they cost.

    A stream is refused unless its horizon and provenance match what the caller expects: a
    horizon mismatch would silently make budget // transitions_per_plan the wrong count.
    """
    with h5py.File(path, "r") as file:
        if "horizon" not in file.attrs:
            raise ValueError(f"{path} predates the acquisition provenance protocol; recollect it")
        stored = int(file.attrs["horizon"])
        if horizon is not None and stored != horizon:
            raise ValueError(f"{path} holds {stored}-block plans, this run uses {horizon}")
        for field, value in (expect or {}).items():
            if json.loads(file.attrs[field]) != value:
                raise ValueError(f"{path} was collected with a different {field}")
        if int(file.attrs["transitions_per_plan"]) != stored * BLOCK_STEPS:
            raise ValueError(f"{path} disagrees with itself about transition accounting")
        return file["latents"][:], file["blocks"][:], int(file.attrs["transitions"])
