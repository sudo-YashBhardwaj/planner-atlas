import pytest
import torch

from planner_atlas.models.reference_lewm import (
    LATENT_DIM,
    ReferenceLeWM,
    _normalize,
    terminal_cost,
)


@pytest.fixture(scope="module")
def model() -> ReferenceLeWM:
    torch.manual_seed(0)
    return ReferenceLeWM().eval()


def test_architecture_matches_released_checkpoint(model: ReferenceLeWM) -> None:
    # Tensor count and total size of the official quentinll/lewm-* weights.pt state dicts.
    state = model.state_dict()
    assert len(state) == 303
    assert sum(tensor.numel() for tensor in state.values()) == 18_042_672


def test_normalize_applies_imagenet_statistics_per_channel() -> None:
    observations = torch.tensor([0.0, 0.5, 1.0]).view(1, 1, 3, 1, 1).expand(1, 1, 3, 224, 224)
    expected = torch.tensor([-0.485 / 0.229, (0.5 - 0.456) / 0.224, (1.0 - 0.406) / 0.225])
    torch.testing.assert_close(_normalize(observations)[0, 0, :, 0, 0], expected)


@pytest.mark.parametrize(
    "observations",
    [
        torch.zeros(1, 1, 3, 224, 224, dtype=torch.float64),
        torch.zeros(1, 3, 224, 224),
        torch.zeros(1, 1, 1, 224, 224),
        torch.zeros(1, 1, 3, 64, 64),
        torch.full((1, 1, 3, 224, 224), 255.0),
    ],
    ids=["dtype", "rank", "channels", "size", "range"],
)
def test_encode_rejects_invalid_observations(
    model: ReferenceLeWM, observations: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        model.encode(observations)


def test_encode_returns_one_latent_per_frame(model: ReferenceLeWM) -> None:
    assert model.encode(torch.rand(2, 3, 3, 224, 224)).shape == (2, 3, LATENT_DIM)


def test_rollout_starts_from_given_latent(model: ReferenceLeWM) -> None:
    latent = torch.randn(2, LATENT_DIM)
    latents = model.rollout(latent, torch.randn(2, 3, 5, 10))
    assert latents.shape == (2, 3, 6, LATENT_DIM)
    torch.testing.assert_close(latents[:, :, 0], latent[:, None].expand(2, 3, -1))


def test_rollout_rejects_mismatched_batch(model: ReferenceLeWM) -> None:
    with pytest.raises(ValueError):
        model.rollout(torch.randn(1, LATENT_DIM), torch.randn(2, 3, 5, 10))


def test_terminal_cost_sums_squared_error_against_own_goal() -> None:
    predicted = torch.zeros(2, 3, LATENT_DIM)  # B=2 environments, S=3 candidates
    goal = torch.stack([torch.zeros(LATENT_DIM), torch.ones(LATENT_DIM)])
    expected = torch.tensor([[0.0] * 3, [float(LATENT_DIM)] * 3])
    torch.testing.assert_close(terminal_cost(predicted, goal), expected)


def test_batched_cost_matches_per_environment_cost(model: ReferenceLeWM) -> None:
    latent, goal = torch.randn(2, LATENT_DIM), torch.randn(2, LATENT_DIM)
    action_blocks = torch.randn(2, 3, 5, 10)
    per_environment = torch.cat(
        [model.cost(latent[[i]], goal[[i]], action_blocks[[i]]) for i in range(2)]
    )
    torch.testing.assert_close(model.cost(latent, goal, action_blocks), per_environment)
