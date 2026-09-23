import numpy as np
import pytest

from planner_atlas.statistics import (
    binomial_sign_test,
    case_summary,
    incomplete_beta,
    monte_carlo_sign_flip,
    paired_summary,
    sign_flip_p,
    sign_matrix,
    t_critical,
    t_two_sided,
    wilcoxon_p,
    wilcoxon_signed_rank,
)

SIGNS = sign_matrix(100_000, 48, seed=20260923)  # the Stage-1 frozen matrix (protocol section 22)


def test_t_distribution_matches_published_values() -> None:
    # two-sided critical values from standard t tables
    assert t_critical(11) == pytest.approx(2.201, abs=5e-4)
    assert t_critical(1) == pytest.approx(12.706, abs=5e-3)
    assert t_critical(30) == pytest.approx(2.042, abs=5e-4)
    assert t_critical(11, 0.99) == pytest.approx(3.106, abs=5e-4)
    assert t_two_sided(2.201, 11) == pytest.approx(0.05, abs=2e-4)
    assert t_two_sided(0.0, 11) == pytest.approx(1.0)
    # df = 1 is Cauchy: P(|T| > t) = 1 - 2 atan(t) / pi
    assert t_two_sided(3.0, 1) == pytest.approx(1 - 2 * np.arctan(3.0) / np.pi, rel=1e-9)
    assert incomplete_beta(2.0, 3.0, 0.4) == pytest.approx(0.5248, abs=1e-4)


def test_sign_flip_counts_every_assignment_at_least_as_extreme() -> None:
    # all negative: only the observed signs and their mirror image reach the observed sum
    assert sign_flip_p(-np.arange(1.0, 7.0)) == pytest.approx(2 / 64)
    # a small positive and a near-zero difference among twelve: flipping neither, the one, or
    # the other keeps the sum as large, so 3 assignments and their mirrors of 4096
    differences = -np.linspace(1.0, 2.0, 12)
    differences[0], differences[1] = 0.3, -0.1
    assert sign_flip_p(differences) == pytest.approx(6 / 4096)
    assert sign_flip_p(np.array([1.0, -1.0])) == 1.0


def test_wilcoxon_exact_on_small_samples() -> None:
    assert wilcoxon_p(-np.arange(1.0, 7.0)) == pytest.approx(2 / 64)
    # the single positive difference has rank 2: W+ = 2, reached by 3 of 4096 subsets per tail
    differences = -np.arange(1.0, 13.0)
    differences[1] = 2.0
    assert wilcoxon_p(differences) == pytest.approx(6 / 4096)
    assert wilcoxon_p(np.array([0.0, -1.0, -2.0])) == pytest.approx(wilcoxon_p(-np.array([1, 2])))


def test_paired_summary_reports_the_protocol_quantities() -> None:
    differences = np.array([-3.0, -1.0, -2.0, 0.5])
    summary = paired_summary(differences)
    assert summary.n == 4 and summary.negative == 3
    assert summary.mean == pytest.approx(-1.375)
    assert summary.sd == pytest.approx(np.std(differences, ddof=1))
    assert summary.se == pytest.approx(summary.sd / 2)
    assert summary.d_z == pytest.approx(summary.mean / summary.sd)
    half = t_critical(3) * summary.se
    assert (summary.ci_low, summary.ci_high) == pytest.approx(
        (summary.mean - half, summary.mean + half)
    )
    assert summary.t_p == pytest.approx(t_two_sided(summary.t, 3))


def test_sign_matrix_is_the_frozen_stage1_draw() -> None:
    expected = 2 * np.random.default_rng(20260923).integers(0, 2, size=(100_000, 48)) - 1
    assert np.array_equal(SIGNS, expected)
    assert set(np.unique(SIGNS)) == {-1, 1}


def test_monte_carlo_sign_flip_uses_the_plus_one_correction_and_is_never_zero() -> None:
    strong = np.linspace(1.0, 2.0, 48)  # no sign flip reaches the observed mean
    p, se = monte_carlo_sign_flip(strong, SIGNS)
    assert p == 1 / 100_001 > 0
    assert se == pytest.approx(np.sqrt(p * (1 - p) / 100_000))
    assert monte_carlo_sign_flip(np.zeros(48), SIGNS)[0] == 1.0  # every draw ties
    symmetric = np.tile([1.0, -1.0], 24)  # observed mean exactly zero
    assert monte_carlo_sign_flip(symmetric, SIGNS)[0] == 1.0
    assert monte_carlo_sign_flip(strong, SIGNS) == monte_carlo_sign_flip(strong, SIGNS)
    noisy = np.random.default_rng(0).normal(0.0, 1.0, 48)
    assert 0.05 < monte_carlo_sign_flip(noisy, SIGNS)[0] <= 1.0
    with pytest.raises(ValueError):
        monte_carlo_sign_flip(np.ones(12), SIGNS)


def test_binomial_sign_test_is_exact_and_discards_zeros() -> None:
    assert binomial_sign_test(np.arange(1.0, 7.0)) == pytest.approx(2 / 64)
    assert binomial_sign_test(np.r_[np.arange(1.0, 7.0), 0.0, 0.0]) == pytest.approx(2 / 64)
    assert binomial_sign_test(np.array([1.0, -1.0, 2.0, -2.0])) == 1.0
    assert binomial_sign_test(np.zeros(5)) == 1.0


def test_wilcoxon_by_dynamic_programming_matches_full_enumeration() -> None:
    rng = np.random.default_rng(3)
    for n in (5, 8, 11):
        values = rng.normal(0.3, 1.0, n)
        p, exact = wilcoxon_signed_rank(values)
        assert exact and p == pytest.approx(wilcoxon_p(values), abs=1e-12)
    p, exact = wilcoxon_signed_rank(rng.normal(0.2, 1.0, 48))  # 2^48 subsets, counted by DP
    assert exact and 0 < p <= 1


def test_wilcoxon_with_tied_magnitudes_is_labelled_approximate() -> None:
    p, exact = wilcoxon_signed_rank(np.array([1.0, -1.0, 2.0, 2.0, 3.0]))
    assert not exact and 0 < p <= 1


def test_case_summary_follows_the_preregistered_definitions() -> None:
    values = np.r_[np.linspace(0.5, 2.0, 40), -np.linspace(0.1, 0.4, 6), 0.0, 0.0]
    summary = case_summary(values, SIGNS)
    sd = values.std(ddof=1)
    assert summary["n"] == 48 and summary["positive"] == 40 and summary["nonzero"] == 46
    assert summary["sign_proportion"] == pytest.approx(40 / 46)  # zeros excluded
    assert summary["d_z"] == pytest.approx(values.mean() / sd)
    half = t_critical(47) * sd / np.sqrt(48)
    assert (summary["ci_low"], summary["ci_high"]) == pytest.approx(
        (values.mean() - half, values.mean() + half)
    )
    assert summary["binomial_sign_p"] == pytest.approx(binomial_sign_test(values))
    assert summary["mc_sign_flip_p"] == monte_carlo_sign_flip(values, SIGNS)[0]
