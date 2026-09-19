import h5py
import numpy as np
import pytest
import torch

from planner_atlas.data import (
    BLOCK_STEPS,
    ENV_ACTION_DIM,
    ActionStats,
    action_stats,
    denormalize_actions,
    normalize_actions,
    pack_action_blocks,
    unpack_action_blocks,
)
from planner_atlas.models.reference_lewm import ACTION_DIM

STATS = ActionStats(mean=torch.tensor([1.0, -2.0]), std=torch.tensor([2.0, 4.0]))


def write_actions(path, actions) -> None:
    with h5py.File(path, "w") as file:
        file["action"] = np.array(actions, dtype=np.float32)


def test_blocks_match_the_model_action_dimension() -> None:
    assert BLOCK_STEPS * ENV_ACTION_DIM == ACTION_DIM


def test_pack_flattens_actions_in_time_order() -> None:
    actions = torch.arange(10.0).view(5, 2)  # a_t = (2t, 2t + 1)
    assert torch.equal(pack_action_blocks(actions), torch.arange(10.0))


@pytest.mark.parametrize("shape", [(5, 2), (4, 5, 2), (2, 3, 4, 5, 2)])
def test_unpack_inverts_pack(shape: tuple[int, ...]) -> None:
    actions = torch.randn(shape)
    assert torch.equal(unpack_action_blocks(pack_action_blocks(actions)), actions)


def test_normalization_is_an_unclipped_affine_map() -> None:
    actions = torch.tensor([[3.0, 2.0], [5.0, 10.0]])
    normalized = normalize_actions(actions, STATS)
    assert torch.equal(normalized, torch.tensor([[1.0, 1.0], [2.0, 3.0]]))
    assert torch.equal(denormalize_actions(normalized, STATS), actions)


def test_normalization_round_trip() -> None:
    actions = torch.randn(4, 3, 5, 2)
    round_trip = denormalize_actions(normalize_actions(actions, STATS), STATS)
    torch.testing.assert_close(round_trip, actions)


@pytest.mark.parametrize(
    ("transform", "shape"),
    [
        (pack_action_blocks, (4, 2)),
        (pack_action_blocks, (5, 3)),
        (unpack_action_blocks, (9,)),
        (lambda actions: normalize_actions(actions, STATS), (10,)),
        (lambda actions: denormalize_actions(actions, STATS), (5, 3)),
    ],
)
def test_rejects_malformed_trailing_dimensions(transform, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        transform(torch.zeros(shape))


def test_action_stats_excludes_nan_rows_and_uses_population_std(tmp_path) -> None:
    write_actions(tmp_path / "actions.h5", [[1, 2], [3, 6], [np.nan, np.nan]])
    stats = action_stats(tmp_path / "actions.h5")
    assert torch.equal(stats.mean, torch.tensor([2.0, 4.0]))
    assert torch.equal(stats.std, torch.tensor([1.0, 2.0]))


def test_action_stats_rejects_zero_std(tmp_path) -> None:
    write_actions(tmp_path / "actions.h5", [[1, 2], [3, 2]])
    with pytest.raises(ValueError):
        action_stats(tmp_path / "actions.h5")
