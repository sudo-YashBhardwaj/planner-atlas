"""Seed-level paired inference, as the confirmatory protocol's analysis plan specifies it.

Seeds are the unit of replication: each is one independent acquisition stream and training run,
and the held-out cases are repeated measurements of the model it produced. Every contrast is
therefore a vector of per-seed differences of case means, never a pool of case-level values.

The pre-registered test is the exact two-sided sign-flip permutation test over all 2^n sign
assignments of those differences; the paired t-test and the exact Wilcoxon signed-rank test are
reported beside it as checks. Student's t comes from the regularized incomplete beta function, so
nothing beyond numpy is needed.

Exact enumeration stops being possible well before n = 48. The Stage-1 optimization-depth study
(docs/protocols/optimization_depth_stage1.md, section 22) therefore uses a Monte-Carlo sign-flip
test on one frozen sign matrix, never called exact, beside an exact binomial sign test and a
Wilcoxon signed-rank test that is exact by dynamic programming whenever it can be.
"""

from dataclasses import dataclass
from itertools import product
from math import comb, erf, exp, lgamma, log, sqrt

import numpy as np

_TINY = 1e-300


def _beta_fraction(a: float, b: float, x: float, iterations: int = 300) -> float:
    """Continued fraction of the incomplete beta function, by the modified Lentz method."""
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > _TINY else _TINY)
    h = d
    for m in range(1, iterations):
        for numerator in (
            m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
            -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1)),
        ):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > _TINY else _TINY)
            c = 1.0 + numerator / c
            c = c if abs(c) > _TINY else _TINY
            h *= c * d
    return h


def incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_fraction(a, b, x) / a
    return 1.0 - front * _beta_fraction(b, a, 1.0 - x) / b


def t_two_sided(t: float, df: int) -> float:
    """Two-sided p-value of Student's t with df degrees of freedom."""
    return incomplete_beta(df / 2.0, 0.5, df / (df + t * t))


def t_critical(df: int, level: float = 0.95) -> float:
    """The two-sided critical value, by bisection on the same distribution the test uses."""
    low, high = 0.0, 1e3
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if t_two_sided(middle, df) > 1 - level else (low, middle)
    return (low + high) / 2


def sign_flip_p(differences: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p-value of the mean paired difference.

    Under the null each difference is as likely to carry either sign, so the reference
    distribution is the sum over all 2^n sign assignments of the absolute differences.
    """
    values = np.abs(np.asarray(differences, dtype=float))
    signs = np.array(list(product((1.0, -1.0), repeat=len(values))))
    sums = np.abs(signs @ values)
    observed = abs(float(np.sum(differences)))
    return float(np.mean(sums >= observed * (1 - 1e-12)))


def wilcoxon_p(differences: np.ndarray) -> float:
    """Exact two-sided Wilcoxon signed-rank p-value; zeros dropped, ties given average ranks."""
    values = np.asarray(differences, dtype=float)
    values = values[values != 0]
    magnitude = np.abs(values)
    ranks = np.empty(len(values))
    ranks[np.argsort(magnitude, kind="stable")] = np.arange(1, len(values) + 1)
    for value in np.unique(magnitude):
        ranks[magnitude == value] = ranks[magnitude == value].mean()
    positive = ranks[values > 0].sum()
    centre = ranks.sum() / 2
    totals = np.array(list(product((0.0, 1.0), repeat=len(values)))) @ ranks
    return float(np.mean(np.abs(totals - centre) >= abs(positive - centre) - 1e-9))


@dataclass(frozen=True)
class PairedSummary:
    """The protocol's report of one paired contrast over seeds."""

    n: int
    mean: float
    sd: float
    se: float
    ci_low: float
    ci_high: float
    t: float
    t_p: float
    sign_flip_p: float  # the pre-registered test
    wilcoxon_p: float
    d_z: float
    negative: int  # seeds whose difference is below zero


def paired_summary(differences: np.ndarray, *, level: float = 0.95) -> PairedSummary:
    """Mean, spread, t interval and the three tests of per-seed paired differences."""
    values = np.asarray(differences, dtype=float)
    n = len(values)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = sd / sqrt(n)
    half = t_critical(n - 1, level) * se
    t = mean / se if se > 0 else float("nan")
    return PairedSummary(
        n=n,
        mean=mean,
        sd=sd,
        se=se,
        ci_low=mean - half,
        ci_high=mean + half,
        t=t,
        t_p=t_two_sided(t, n - 1) if se > 0 else float("nan"),
        sign_flip_p=sign_flip_p(values),
        wilcoxon_p=wilcoxon_p(values),
        d_z=mean / sd if sd > 0 else float("nan"),
        negative=int((values < 0).sum()),
    )


def sign_matrix(draws: int, n: int, *, seed: int) -> np.ndarray:
    """The frozen ±1 matrix [draws, n] every Monte-Carlo sign-flip test of one study shares."""
    return 2 * np.random.default_rng(seed).integers(0, 2, size=(draws, n)) - 1


def _flip_statistic(signs: np.ndarray, values: np.ndarray) -> np.ndarray:
    """|mean(s * d)| for each sign vector s in signs [B, n]."""
    return np.abs(np.mean(signs * values[None, :], axis=1))


def monte_carlo_sign_flip(values: np.ndarray, signs: np.ndarray) -> tuple[float, float]:
    """Two-sided Monte-Carlo sign-flip p-value, with the +1 correction, and its Monte-Carlo SE.

    p = (1 + #{b : T_b >= T_obs}) / (B + 1), T = |mean(s * d)|, with T_obs computed by the same
    routine on the all-(+1) sign vector. It is a randomization test, not an exact one, and it is
    never zero.
    """
    values = np.asarray(values, dtype=float)
    if signs.shape[1] != len(values):
        raise ValueError(f"the sign matrix is for n = {signs.shape[1]}, got {len(values)} values")
    observed = _flip_statistic(np.ones((1, len(values))), values)[0]
    draws = len(signs)
    p = (1 + int(np.sum(_flip_statistic(signs, values) >= observed))) / (draws + 1)
    return p, sqrt(p * (1 - p) / draws)


def binomial_sign_test(values: np.ndarray) -> float:
    """Exact two-sided sign test: zeros discarded, X ~ Bin(#nonzero, 1/2), doubled smaller tail."""
    values = np.asarray(values, dtype=float)
    n, k = int(np.sum(values != 0)), int(np.sum(values > 0))
    if n == 0:
        return 1.0
    lower = sum(comb(n, j) for j in range(k + 1))
    upper = sum(comb(n, j) for j in range(k, n + 1))
    return min(1.0, 2 * min(lower, upper) / 2**n)


def wilcoxon_signed_rank(values: np.ndarray) -> tuple[float, bool]:
    """Two-sided Wilcoxon signed-rank p-value, and whether it is exact.

    Zeros are discarded. Without ties in |d| the ranks are the integers 1..n and the null
    distribution of the positive rank sum is counted exactly by dynamic programming over subset
    sums, so any n works. With ties the ranks are averaged, no integer DP applies, and the normal
    approximation with tie correction is returned, labelled not exact.
    """
    values = np.asarray(values, dtype=float)
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return 1.0, True
    magnitude = np.abs(values)
    if len(np.unique(magnitude)) == n:
        ranks = np.empty(n, dtype=int)
        ranks[np.argsort(magnitude)] = np.arange(1, n + 1)
        positive = int(ranks[values > 0].sum())
        counts = [1] + [0] * (n * (n + 1) // 2)  # subsets of {1..n} by their sum
        for rank in range(1, n + 1):
            for total in range(len(counts) - 1, rank - 1, -1):
                counts[total] += counts[total - rank]
        centre = n * (n + 1) / 4
        extreme = sum(
            c for total, c in enumerate(counts) if abs(total - centre) >= abs(positive - centre)
        )
        return extreme / 2**n, True
    ranks = np.empty(n)
    ranks[np.argsort(magnitude, kind="stable")] = np.arange(1, n + 1)
    for value in np.unique(magnitude):
        ranks[magnitude == value] = ranks[magnitude == value].mean()
    positive = ranks[values > 0].sum()
    _, ties = np.unique(magnitude, return_counts=True)
    variance = n * (n + 1) * (2 * n + 1) / 24 - float(np.sum(ties**3 - ties)) / 48
    z = (positive - n * (n + 1) / 4) / sqrt(variance)
    return min(1.0, 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))), False


def case_summary(values: np.ndarray, signs: np.ndarray, *, level: float = 0.95) -> dict:
    """Everything the Stage-1 protocol reports for one case-level vector d (section 22, A2-A3)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = sd / sqrt(n)
    half = t_critical(n - 1, level) * se
    p, mc_se = monte_carlo_sign_flip(values, signs)
    wilcoxon, exact = wilcoxon_signed_rank(values)
    nonzero = int(np.sum(values != 0))
    positive = int(np.sum(values > 0))
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(values)),
        "sd": sd,
        "se": se,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "d_z": mean / sd if sd > 0 else float("nan"),
        "positive": positive,
        "nonzero": nonzero,
        "sign_proportion": positive / nonzero if nonzero else float("nan"),
        "mc_sign_flip_p": p,
        "mc_sign_flip_se": mc_se,
        "binomial_sign_p": binomial_sign_test(values),
        "wilcoxon_p": wilcoxon,
        "wilcoxon_exact": exact,
    }
