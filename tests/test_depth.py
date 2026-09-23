"""The Stage-1 optimization-depth pipeline on synthetic data only.

Nothing here builds a real candidate bank, loads the consumed manifest for execution, or runs the
simulator: every array is synthetic, and the fixtures are the ones frozen in protocol section 15.
"""

import inspect
import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from planner_atlas import depth
from planner_atlas.data import ActionStats
from planner_atlas.evaluation import CandidateEvaluation
from planner_atlas.planning import initial_proposal, model_action_bounds, sample_proposal
from planner_atlas.statistics import case_summary, sign_matrix

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "optimization_depth_stage1.py"
M = depth.N_MAX
FIXTURE_SEED = 20260923  # frozen fixture constants (protocol section 15)


@pytest.fixture(scope="module")
def signs() -> np.ndarray:
    return sign_matrix(depth.SIGN_DRAWS, depth.NUM_CASES, seed=depth.SIGN_SEED)


def synthetic_fields(true_latent, predicted, *, task=None, seed=0, kind="rs") -> dict:
    """Every cache field the build writes, around hand-set latent costs."""
    n = len(true_latent)
    rng = np.random.default_rng(seed)
    task = rng.random(n) if task is None else np.asarray(task, dtype=float)
    actions = rng.normal(size=(n, 5, 10)).astype(np.float32)
    residual = np.asarray(true_latent, dtype=float) - np.asarray(predicted, dtype=float)
    flat = actions.reshape(n, -1).astype(np.float64)
    fields = {
        "candidate_index": np.arange(n),
        "actions": actions,
        "predicted_latent_cost": np.asarray(predicted, dtype=float),
        "true_latent_cost": np.asarray(true_latent, dtype=float),
        "signed_optimism": residual,
        "absolute_optimism": np.abs(residual),
        "true_task_cost": task,
        "position_error": task * 512,
        "angle_error": np.zeros(n),
        "rollout_error": rng.random(n),
        "terminal_error": rng.random(n),
        "semantic_success": task < 0.05,
        "semantic_reached_success": task < 0.05,
        "terminal_true_latent": rng.normal(size=(n, 192)).astype(np.float32),
        "action_norm": np.linalg.norm(flat, axis=1),
        "q0_negative_log_density": 0.5 * np.square(flat).sum(axis=1),
    }
    if kind == "cem":
        fields["iteration"] = np.repeat(np.arange(depth.CEM_ITERATIONS), depth.CEM_POPULATION)
    return fields


def random_case(seed=0, kind="rs") -> dict:
    count = depth.candidate_count(kind)
    rng = np.random.default_rng(seed)
    true_latent = rng.gamma(4.0, 12.0, count)
    return synthetic_fields(
        true_latent, true_latent - rng.normal(0, 10, count), seed=seed, kind=kind
    )


def metadata(case_index=0, kind="rs", **overrides) -> dict:
    return (
        depth.expected_metadata(case_index, kind)
        | {
            "git_commit": "0" * 40,
            "episode": 7,
            "start_step": 11,
            "action_bounds_identity": {},
            "objective_identity": depth.OBJECTIVE_IDENTITY,
            "sampler_identity": depth.SAMPLER_IDENTITY,
            "planner_identity": {},
            "simulator_cache_identity": {},
        }
        | overrides
    )


def fixture_costs() -> np.ndarray:
    return np.random.default_rng(FIXTURE_SEED).gamma(4.0, 12.0, M)


def excess(fields, case_index=0, draws=depth.NULL_DRAWS) -> tuple[float, dict]:
    null = depth.exchangeable_null(fields, case_index, draws=draws)
    observed = float(depth.ols_slope(depth.prefix_curves(fields)["r_latent"]))
    return observed - null["mean_beta"], null | {"observed": observed}


# ---- the bank ----------------------------------------------------------------------------------


def test_every_rs_n_is_a_literal_prefix_of_one_bank() -> None:
    assert list(inspect.signature(depth.draw_bank).parameters) == ["bounds", "case_index"]
    stats = ActionStats(mean=torch.tensor([0.0, 0.0]), std=torch.tensor([0.21, 0.21]))
    bounds = model_action_bounds(stats, device="cpu")
    bank = depth.draw_bank(bounds, 5)
    assert bank.shape == (M, 5, 10)
    assert torch.equal(bank, depth.draw_bank(bounds, 5))  # the same case, the same bank
    assert not torch.equal(bank, depth.draw_bank(bounds, 6))
    for n in depth.PREFIXES:  # an RS-N run with the bank's seed draws exactly its prefix
        generator = torch.Generator().manual_seed(depth.BANK_SEED + 5)
        alone = sample_proposal(
            initial_proposal(5), num_samples=n, bounds=bounds, generator=generator
        )
        assert torch.equal(alone, bank[:n])


# ---- selection and regret -----------------------------------------------------------------------


def test_selection_is_the_argmin_over_each_prefix_and_never_gets_worse() -> None:
    fields = random_case(1)
    curves = depth.prefix_curves(fields)
    predicted = fields["predicted_latent_cost"]
    for column, n in enumerate(depth.PREFIXES):
        assert curves["selected_index"][column] == np.argmin(predicted[:n])
    assert np.all(np.diff(curves["predicted_latent_cost"]) <= 0)


def test_regrets_are_the_selected_minus_the_oracle_and_never_negative() -> None:
    fields = random_case(2)
    curves = depth.prefix_curves(fields)
    for column, n in enumerate(depth.PREFIXES):
        chosen = int(np.argmin(fields["predicted_latent_cost"][:n]))
        latent, task = fields["true_latent_cost"], fields["true_task_cost"]
        assert curves["r_latent"][column] == latent[chosen] - latent[:n].min()
        assert curves["r_task"][column] == task[chosen] - task[:n].min()
        assert curves["oracle_task_cost"][column] == task[:n].min()
    assert np.all(curves["r_latent"] >= 0) and np.all(curves["r_task"] >= 0)


def test_hand_computed_regret_on_a_tiny_ladder() -> None:
    true = np.array([5.0, 1.0, 3.0, 0.5])
    predicted = np.array([4.0, 2.0, 1.0, 3.0])  # picks 0, then 2 (true 3.0) among 4
    curves = depth.prefix_curves(synthetic_fields(true, predicted), prefixes=(1, 2, 4))
    assert curves["selected_index"].tolist() == [0, 1, 2]
    assert curves["r_latent"].tolist() == [0.0, 0.0, 2.5]


@pytest.mark.parametrize(("slope", "offset"), [(2.0, 1.0), (-3.0, 40.0), (0.0, 7.0)])
def test_ols_slope_against_log2_n(slope, offset) -> None:
    curve = slope * np.log2(np.array(depth.PREFIXES, dtype=float)) + offset
    assert depth.ols_slope(curve) == pytest.approx(slope, abs=1e-12)
    batch = np.stack([curve, 2 * curve])
    assert depth.ols_slope(batch) == pytest.approx([slope, 2 * slope], abs=1e-12)


# ---- the exchangeable-error null ----------------------------------------------------------------


def test_null_permutes_residuals_as_preregistered() -> None:
    fields = random_case(3)
    residual = fields["true_latent_cost"] - fields["predicted_latent_cost"]
    permutation = np.random.default_rng([depth.NULL_SEED, 3]).permutation(M)
    permuted = residual[permutation]
    assert np.array_equal(np.sort(permuted), np.sort(residual))  # the same multiset
    assert np.mean(permuted != residual) > 0.99  # candidates carry other residuals
    # replicate 0 of the null is exactly this permutation, on the same bank and ladder
    null = depth.exchangeable_null(fields, 3, draws=2)
    replicate = synthetic_fields(fields["true_latent_cost"], fields["true_latent_cost"] - permuted)
    expected = depth.ols_slope(depth.prefix_curves(replicate)["r_latent"])
    assert null["beta"][0] == pytest.approx(float(expected), rel=1e-12, abs=1e-12)
    assert len(null["mean_r_latent"]) == len(depth.PREFIXES)
    assert inspect.signature(depth.exchangeable_null).parameters["draws"].default == 10_000


def test_fixture_a_exchangeable_error_is_consistent_with_its_null() -> None:
    true = fixture_costs()
    predicted = true - np.random.default_rng(FIXTURE_SEED + 1).normal(0.0, 10.0, M)
    value, null = excess(synthetic_fields(true, predicted))
    quantiles = null["beta_quantiles"]
    assert quantiles["2.5"] <= null["observed"] <= quantiles["97.5"]
    assert abs(value) < 2 * null["sd_beta"]


def test_exchangeable_error_leaves_no_excess_on_average_across_cases() -> None:
    """What G2 relies on: under exchangeability the observation is itself one null replicate."""
    values = []
    for case in range(24):
        rng = np.random.default_rng([FIXTURE_SEED, case])
        true = rng.gamma(4.0, 12.0, M)
        values.append(excess(synthetic_fields(true, true - rng.normal(0, 10, M)), case, 2000)[0])
    values = np.array(values)
    assert abs(values.mean()) < 3 * values.std(ddof=1) / np.sqrt(len(values))


def test_fixture_b_trap_error_yields_clearly_positive_excess() -> None:
    true = fixture_costs()
    residual = np.zeros(M)
    residual[np.argsort(true)[-int(0.01 * M) :]] = 200.0  # the 1% worst look far better
    value, null = excess(synthetic_fields(true, true - residual))
    assert value > 3.0  # protocol section 15 records +3.62
    assert null["mean_beta"] < 0 < null["observed"]  # the null falls; the observation does not


def test_fixture_c_perfect_predictor_has_no_regret_and_no_excess() -> None:
    true = fixture_costs()
    fields = synthetic_fields(true, true.copy())
    assert np.all(depth.prefix_curves(fields)["r_latent"] == 0)
    value, null = excess(fields)
    assert np.all(null["beta"] == 0) and value == 0 and null["observed"] == 0
    assert np.isnan(null["corr_true_residual"])  # undefined when residuals do not vary


def test_fixture_d1_is_search_useful_while_ranking_efficiency_degrades() -> None:
    selected = np.array([0.40, 0.34, 0.30, 0.27, 0.25, 0.24, 0.235, 0.233])
    oracle = np.array([0.30, 0.22, 0.16, 0.11, 0.07, 0.05, 0.035, 0.025])
    selected_slope = float(depth.ols_slope(selected, depth.LADDER))
    regret_slope = float(depth.ols_slope(selected - oracle, depth.LADDER))
    assert selected_slope < 0 < regret_slope
    reading = depth.classify_task_curve(
        selected_slope,
        {"mean": regret_slope, "ci_low": regret_slope / 2, "ci_high": 2 * regret_slope},
    )
    assert reading.startswith("B: search useful, ranking efficiency degrades")


# ---- CEM, compute-matching, rank bins ------------------------------------------------------------


def test_cem_depth_reads_prefixes_of_one_trace() -> None:
    trace = random_case(4, kind="cem")
    curves = depth.cem_depth(trace)
    assert curves["prefixes"].tolist() == [256 * i for i in range(1, 7)]
    for column, depth_ in enumerate(range(1, 7)):
        best = np.argmin(trace["predicted_latent_cost"][: 256 * depth_])
        assert curves["selected_index"][column] == best
    shuffled = trace | {"iteration": np.random.default_rng(0).permutation(trace["iteration"])}
    with pytest.raises(ValueError, match="iteration order"):
        depth.cem_depth(shuffled)


def test_compute_matched_rs_uses_exactly_the_same_evaluation_count() -> None:
    rs, cem = random_case(5), random_case(6, kind="cem")
    matched = depth.compute_matched(rs, cem)
    assert matched["rs"]["prefixes"] == list(depth.MATCHED) == matched["cem"]["prefixes"]
    for column, n in enumerate(depth.MATCHED):
        assert matched["rs"]["selected_index"][column] == np.argmin(rs["predicted_latent_cost"][:n])


def test_rank_bins_are_exact_disjoint_and_exhaustive() -> None:
    slices = list(depth.RANK_BINS.values())
    assert [high - low for low, high in slices] == [41, 164, 205, 614, 1024, 2048]
    assert slices[0][0] == 0 and slices[-1][1] == M
    assert all(a[1] == b[0] for a, b in pairwise(slices))
    fields = random_case(7)
    bins = depth.rank_conditioned(fields)
    assert [cell["size"] for cell in bins.values()] == [41, 164, 205, 614, 1024, 2048]
    best = np.sort(fields["predicted_latent_cost"])[:41]
    assert bins["0-1%"]["predicted_latent_cost"] == pytest.approx(best.mean())


# ---- cache schema --------------------------------------------------------------------------------


def test_signed_optimism_is_realized_minus_predicted() -> None:
    evaluation = CandidateEvaluation(
        predicted_cost=np.array([1.0, 5.0]),
        realized_cost=np.array([3.0, 4.0]),
        rollout_error=np.zeros(2),
        terminal_error=np.zeros(2),
        task={
            "task_cost": np.zeros(2),
            "position_error": np.zeros(2),
            "angle_error": np.zeros(2),
            "semantic_final_success": np.array([False, True]),
            "semantic_reached_success": np.array([False, True]),
        },
    )
    fields = depth.candidate_fields(evaluation, np.zeros((2, 192)), np.zeros((2, 5, 10)))
    assert fields["signed_optimism"].tolist() == [2.0, -1.0]  # positive = over-optimistic
    assert np.array_equal(fields["signed_optimism"], evaluation.optimism)
    assert fields["absolute_optimism"].tolist() == [2.0, 1.0]


def test_cache_round_trips_bitwise_and_is_written_once(tmp_path) -> None:
    path = depth.case_path(tmp_path, "rs", 0)
    fields = random_case(8)
    depth.save_case(path, fields, metadata())
    first, meta = depth.load_case(path, expect=depth.expected_metadata(0, "rs"))
    second, _ = depth.load_case(path, expect=depth.expected_metadata(0, "rs"))
    for name, values in fields.items():
        assert first[name].tobytes() == np.asarray(values).tobytes() == second[name].tobytes()
    assert meta["episode"] == 7
    with pytest.raises(FileExistsError):
        depth.save_case(path, fields, metadata())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("protocol_commit", "ebdfa8a"),
        ("checkpoint_digest", "0" * 64),
        ("manifest_digest", "86ed349f"),
        ("n_max", 2048),
        ("metric_definition_version", "stage1-v0"),
        ("dynamics_protocol", "frozen_rep_v2"),
        ("case_index", 1),
    ],
)
def test_cache_with_foreign_provenance_is_refused(tmp_path, key, value) -> None:
    path = depth.case_path(tmp_path, "rs", 0)
    depth.save_case(path, random_case(9), metadata(**{key: value}))
    with pytest.raises(ValueError, match=key):
        depth.load_case(path, expect=depth.expected_metadata(0, "rs"))


def test_truncated_cache_is_refused(tmp_path) -> None:
    path = depth.case_path(tmp_path, "rs", 0)
    with pytest.raises(ValueError, match="candidate counts"):
        depth.save_case(path, {k: v[:100] for k, v in random_case(10).items()}, metadata())
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as file:  # written by hand, bypassing save_case
        for name, values in random_case(10).items():
            file[name] = values[:100]
        file.attrs["metadata"] = json.dumps(metadata())
    with pytest.raises(ValueError, match="incomplete"):
        depth.load_case(path, expect=depth.expected_metadata(0, "rs"))


def test_determinism_comparison_is_bitwise(tmp_path) -> None:
    first, second = (depth.case_path(tmp_path / name, "rs", 0) for name in ("a", "b"))
    fields = random_case(11)
    depth.save_case(first, fields, metadata())
    depth.save_case(second, fields, metadata())
    assert depth.compare_case_files(first, second) == []
    third = depth.case_path(tmp_path / "c", "rs", 0)
    nudged = fields | {"true_latent_cost": np.nextafter(fields["true_latent_cost"], np.inf)}
    depth.save_case(third, nudged, metadata(git_commit="1" * 40))
    assert depth.compare_case_files(first, third) == ["true_latent_cost", "metadata.git_commit"]


def test_no_manifest_is_ever_made(tmp_path) -> None:
    missing = tmp_path / "manifest.json"
    with pytest.raises(FileNotFoundError, match="never makes one"):
        depth.load_frozen_manifest(missing)
    assert not missing.exists()
    missing.write_text("{}")
    with pytest.raises(ValueError, match="consumed confirmatory manifest"):
        depth.load_frozen_manifest(missing)


def test_the_protocol_file_is_the_frozen_one() -> None:
    protocol = REPO / "docs" / "protocols" / "optimization_depth_stage1.md"
    assert depth.verify_protocol(protocol)["protocol_digest"] == depth.PROTOCOL_DIGEST


# ---- pipeline guards -------------------------------------------------------------------------------


def test_analysis_refuses_missing_or_half_written_caches(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=depth.INCOMPLETE):
        depth.require_complete(tmp_path)
    for kind in ("rs", "cem"):
        for case in range(depth.NUM_CASES):
            path = depth.case_path(tmp_path, kind, case)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    depth.require_complete(tmp_path)
    depth.case_path(tmp_path, "cem", 47).unlink()
    with pytest.raises(FileNotFoundError, match="1 caches missing"):
        depth.require_complete(tmp_path)
    depth.case_path(tmp_path, "cem", 47).touch()
    (tmp_path / "rs" / "case-003.h5.partial").touch()
    with pytest.raises(FileNotFoundError, match="1 half-written"):
        depth.require_complete(tmp_path)


def test_analyze_command_fails_rather_than_build(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "analyze", "--output", str(tmp_path), "--dataset", "x.h5"]
        + ["--checkpoint", "x.pt", "--latent-cache", "x.h5", "--device", "cpu"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and depth.INCOMPLETE in result.stderr
    assert list(tmp_path.iterdir()) == []  # nothing was created


def test_stage1_has_no_uncertainty_path() -> None:
    probe = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('stage1', {str(SCRIPT)!r}); "
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "print(sorted(m for m in sys.modules if m in "
        "('planner_atlas.uncertainty', 'planner_atlas.acquisition')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
    for source in (SCRIPT, REPO / "src" / "planner_atlas" / "depth.py"):
        text = source.read_text()
        assert "uncertainty" not in text and "ensemble" not in text.lower()


# ---- across cases ----------------------------------------------------------------------------------


def gate_summaries(signs, beta_latent, excess_beta, beta_task) -> dict:
    return {
        "beta_latent": case_summary(np.asarray(beta_latent, dtype=float), signs),
        "excess_beta": case_summary(np.asarray(excess_beta, dtype=float), signs),
        "beta_task": case_summary(np.asarray(beta_task, dtype=float), signs),
    }


def vector(positive: int, *, mean_shift: float = 1.0, size: int = 48) -> np.ndarray:
    """size values, exactly `positive` of them > 0, with a controllable mean."""
    rng = np.random.default_rng(positive)
    values = np.r_[rng.uniform(0.5, 1.5, positive), -rng.uniform(0.05, 0.2, size - positive)]
    return values * mean_shift


@pytest.mark.parametrize(
    ("latent", "excess", "task", "case", "verdict"),
    [
        (vector(40), vector(40), vector(40), 1, "STRONG GO"),
        (vector(40), vector(40), vector(20), 2, "default NO-GO"),
        (vector(40), vector(20), vector(40), 3, "NO-GO"),
        (vector(20), vector(40), vector(40), 4, "NO-GO"),
        (-vector(40), vector(40), vector(40), 4, "NO-GO"),
    ],
)
def test_gates_reproduce_the_frozen_decision_matrix(signs, latent, excess, task, case, verdict):
    decision = depth.evaluate_stage1_gates(gate_summaries(signs, latent, excess, task))
    assert (decision["case"], decision["verdict"]) == (case, verdict)


def test_gate_thresholds_are_the_frozen_integer_counts(signs) -> None:
    passes = depth.evaluate_stage1_gates(gate_summaries(signs, vector(34), vector(32), vector(29)))
    assert all(gate["checks"]["positive_count"] for gate in passes["gates"].values())
    fails = depth.evaluate_stage1_gates(gate_summaries(signs, vector(33), vector(31), vector(28)))
    assert not any(gate["checks"]["positive_count"] for gate in fails["gates"].values())
    assert "mc_sign_flip_p" not in passes["gates"]["G3"]["checks"]  # G3 has no p-value gate
    summaries = gate_summaries(signs, vector(40), vector(40), vector(40))
    summaries["beta_latent"]["mc_sign_flip_p"] = 0.05  # strict inequality
    assert depth.evaluate_stage1_gates(summaries)["verdict"] == "NO-GO"


def test_fewer_than_48_cases_is_an_incomplete_execution(signs) -> None:
    short = {name: {"n": 47} for name in ("beta_latent", "excess_beta", "beta_task")}
    assert depth.evaluate_stage1_gates(short)["verdict"] == depth.INCOMPLETE
    assert depth.summarize_cases([{"case_index": 0}], signs)["verdict"] == depth.INCOMPLETE


def test_pure_analysis_is_repeatable_and_summarizes_48_cases(signs) -> None:
    rs, cem = random_case(12), random_case(13, kind="cem")
    first, beta = depth.analyze_case(rs, cem, metadata())
    second, again = depth.analyze_case(rs, cem, metadata())
    assert json.dumps(first) == json.dumps(second) and np.array_equal(beta, again)
    rows = [first | {"case_index": case} for case in range(depth.NUM_CASES)]
    summary = depth.summarize_cases(rows, signs)
    assert summary["verdict"] in {"STRONG GO", "default NO-GO", "NO-GO"}
    assert summary["summaries"]["beta_latent"]["n"] == depth.NUM_CASES
