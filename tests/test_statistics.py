import numpy as np
import pytest

from planner_atlas.statistics import (
    incomplete_beta,
    paired_summary,
    sign_flip_p,
    t_critical,
    t_two_sided,
    wilcoxon_p,
)


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
