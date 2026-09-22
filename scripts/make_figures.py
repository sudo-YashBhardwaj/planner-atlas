"""The write-up's figures, drawn from the results summary alone.

    uv run --group figures python scripts/make_figures.py \
        --summary docs/results/summary.json --output docs/figures

Every interval is over seeds, the unit of replication: a 95% t interval of per-seed values, each
of which is already a mean over the 48 held-out cases. The Atlas figure is the one exception: it
audits one fixed released model, so its error bars are standard errors over cases.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {  # Okabe-Ito, safe for colour-blind readers
    "original_base": "#000000",
    "continued": "#7a7a7a",
    "random": "#E69F00",
    "uncertainty": "#009E73",
    "planner": "#0072B2",
}
LABELS = {
    "original_base": "released",
    "continued": "continued\n(B = 0)",
    "random": "random",
    "uncertainty": "uncertainty",
    "planner": "planner",
}
STYLE = {
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "svg.hashsalt": "planner-atlas",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def save(figure, output: Path, name: str) -> None:
    figure.savefig(output / f"{name}.png", dpi=180, metadata={"Software": None})
    figure.savefig(output / f"{name}.pdf", metadata={"CreationDate": None, "Creator": None})
    plt.close(figure)


def jitter(count: int, width: float = 0.09) -> np.ndarray:
    return np.linspace(-width, width, count)


def seeds_and_mean(ax, x: float, cell: dict, color: str, *, marker: str = "o") -> None:
    """Per-seed points beside the seed mean and its 95% t interval."""
    values = np.asarray(cell["per_seed"])
    ax.scatter(x + jitter(len(values)) - 0.12, values, s=9, color=color, alpha=0.55, lw=0)
    low, high = cell["ci95"] if "ci95" in cell else (cell["ci_low"], cell["ci_high"])
    ax.errorbar(
        x + 0.12,
        cell["mean"],
        yerr=[[cell["mean"] - low], [high - cell["mean"]]],
        fmt=marker,
        color=color,
        ms=4.5,
        capsize=2.5,
        lw=1.2,
    )


def acquisition_figure(summary: dict, output: Path) -> None:
    strategies = ("random", "uncertainty", "planner")
    data = summary["acquisition"]["strategies"]
    panels = (
        ("signed_optimism", "signed optimism\n(realized − predicted latent cost)"),
        ("absolute_optimism", "absolute optimism\n|realized − predicted|"),
        ("rollout_error", "rollout error\n(latent MSE over the 5 blocks)"),
        ("terminal_error", "terminal error\n(latent MSE at the last block)"),
        ("task_cost", "true task cost\n(block pose error at the end)"),
        ("semantic_success", "semantic success\n(block at goal pose, open loop)"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
    for ax, (key, label) in zip(axes.flat, panels, strict=True):
        for x, strategy in enumerate(strategies):
            seeds_and_mean(ax, x, data[strategy][key], COLORS[strategy])
        ax.set_xticks(range(3), ["random", "uncert.", "planner"])
        ax.set_title(label, loc="left", fontsize=8)
        ax.set_xlim(-0.5, 2.5)
        if key == "signed_optimism":
            ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax.set_ylim(bottom=0)
    figure.suptitle(
        "What each strategy acquires (descriptive): 12 seeds × 1,600 plans = 40,000 transitions each",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save(figure, output, "acquisition")


def primary_figure(summary: dict, output: Path) -> None:
    primary = summary["confirmatory"]["primary"]
    seeds = summary["confirmatory"]["provenance"]["seeds"]
    figure, ax = plt.subplots(figsize=(4.2, 2.9))
    draw_primary(ax, primary, seeds)
    save(figure, output, "primary_effect")


def draw_primary(ax, primary: dict, seeds: list[int]) -> None:
    values = np.asarray(primary["per_seed"])
    ax.axhspan(primary["ci_low"], primary["ci_high"], color=COLORS["planner"], alpha=0.12, lw=0)
    ax.axhline(primary["mean"], color=COLORS["planner"], lw=1.2)
    ax.axhline(0, color="0.3", lw=0.7, ls="--")
    ax.scatter(
        range(len(values)),
        values,
        s=22,
        color=[COLORS["planner"] if value < 0 else COLORS["random"] for value in values],
        zorder=3,
    )
    spread = max(values.max(), 0.0) - values.min()
    ax.set_ylim(values.min() - 0.42 * spread, max(values.max(), 0.0) + 0.12 * spread)
    ax.set_xticks(range(len(seeds)), seeds)
    ax.set_xlabel("seed (independent acquisition stream and repair run)")
    ax.set_ylabel("Δ planner-region regret\n(planner − random)")
    ax.set_title("Confirmatory primary contrast (pre-registered)", loc="left")
    ax.text(
        0.02,
        0.03,
        f"mean {primary['mean']:+.5f}, 95% CI [{primary['ci_low']:+.5f}, {primary['ci_high']:+.5f}]\n"
        f"{primary['negative']}/{primary['n']} seeds negative; exact sign-flip p = "
        f"{primary['sign_flip_p']:.4f}; d_z = {primary['d_z']:.2f}",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )


def strategy_figure(summary: dict, output: Path) -> None:
    confirm = summary["confirmatory"]
    strategies = ("continued", "random", "uncertainty", "planner")
    panels = (
        ("planner_region_regret", "planner-region regret (primary metric)"),
        ("q0_regret", "q0 regret (secondary)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ax, (key, title) in zip(axes, panels, strict=True):
        cells = [confirm["branches"][strategy][key] for strategy in strategies]
        paths = np.array([cell["per_seed"] for cell in cells])  # [strategy, seed]
        for path in paths.T:
            ax.plot(np.arange(4) - 0.12, path, color="0.8", lw=0.5, zorder=1)
        for x, (strategy, cell) in enumerate(zip(strategies, cells, strict=True)):
            seeds_and_mean(ax, x, cell, COLORS[strategy])
        ax.axhline(confirm["original_base"][key], color="k", lw=0.8, ls="--")
        ax.text(3.45, confirm["original_base"][key], "released\nmodel", fontsize=6.5, va="center")
        ax.set_xticks(range(4), [LABELS[s] for s in strategies])
        ax.set_xlim(-0.5, 3.9)
        ax.set_title(title, loc="left")
        ax.set_ylabel("top-choice regret (task cost)")
    figure.suptitle(
        "Candidate ranking after repair, B = 40,000, 12 seeds (grey lines join one seed)",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save(figure, output, "strategy_regret")


def draw_versus_continued(ax, summary: dict, key: str, title: str, ylabel: str) -> None:
    contrasts = summary["confirmatory"]["contrasts"][key]
    strategies = ("random", "uncertainty", "planner")
    for x, strategy in enumerate(strategies):
        seeds_and_mean(ax, x, contrasts[f"{strategy}-continued"], COLORS[strategy])
    ax.axhline(0, color="0.3", lw=0.7, ls="--")
    ax.set_xticks(range(3), [LABELS[s] for s in strategies])
    ax.set_xlim(-0.5, 2.5)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)


def dissociation_figure(summary: dict, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    draw_versus_continued(
        axes[0],
        summary,
        "planner_region_regret",
        "Ranking in the planner's region",
        "Δ planner-region regret vs continued\n(lower is better)",
    )
    draw_versus_continued(
        axes[1],
        summary,
        "mpc_semantic_reached_success",
        "Closed-loop control (secondary)",
        "Δ MPC semantic success vs continued\n(higher is better)",
    )
    figure.suptitle(
        "Within-seed differences from the equal-compute continued control, B = 40,000",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save(figure, output, "dissociation")


def overview_figure(summary: dict, output: Path) -> None:
    figure, axes = plt.subplots(
        1, 3, figsize=(10.2, 3.0), gridspec_kw={"width_ratios": [1.45, 1, 1]}
    )
    draw_primary(
        axes[0], summary["confirmatory"]["primary"], summary["confirmatory"]["provenance"]["seeds"]
    )
    axes[0].set_title("a  Confirmatory: planner − random (pre-registered)", loc="left")
    draw_versus_continued(
        axes[1],
        summary,
        "planner_region_regret",
        "b  Matched-planner ranking",
        "Δ planner-region regret vs continued",
    )
    draw_versus_continued(
        axes[2],
        summary,
        "mpc_semantic_reached_success",
        "c  Closed-loop MPC (secondary)",
        "Δ MPC semantic success vs continued",
    )
    figure.tight_layout()
    save(figure, output, "overview")


def replay_figure(summary: dict, output: Path) -> None:
    replay = summary["replay"]
    alphas = replay["alphas"]
    panels = (
        ("planner_region_regret", "planner-region regret", "lower is better"),
        ("mpc_semantic_reached_success", "MPC semantic success", "higher is better"),
        ("held_out_loss", "held-out prediction loss", "lower is better"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, (key, label, direction) in zip(axes, panels, strict=True):
        for offset, strategy in ((-0.006, "random"), (0.006, "planner")):
            cells = [replay["strategies"][strategy][key][str(alpha)] for alpha in alphas]
            means = np.array([cell["mean"] for cell in cells])
            low = means - np.array([cell["ci95"][0] for cell in cells])
            high = np.array([cell["ci95"][1] for cell in cells]) - means
            ax.errorbar(
                np.array(alphas) + offset,
                means,
                yerr=[low, high],
                fmt="-o",
                color=COLORS[strategy],
                ms=3.5,
                capsize=2,
                lw=1.1,
                label=strategy,
            )
        ax.set_xticks(alphas, ["0\n(continued)", ".125", ".25", ".5"])
        ax.set_xlabel("weight α on acquired data")
        ax.set_title(f"{label}\n({direction})", loc="left")
    axes[2].legend(frameon=False, loc="upper left")
    figure.suptitle(
        "EXPLORATORY post-confirmation replay-weight analysis (not pre-registered; run after the "
        "confirmatory result was inspected)",
        fontsize=8.5,
        color="#8a1c1c",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save(figure, output, "replay_weight")


def held_out_figure(summary: dict, output: Path) -> None:
    confirm = summary["confirmatory"]
    panels = (
        ("planner_region_regret", "planner-region regret"),
        ("mpc_semantic_reached_success", "MPC semantic success"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, (key, label) in zip(axes, panels, strict=True):
        for strategy in ("continued", "random", "uncertainty", "planner"):
            branch = confirm["branches"][strategy]
            ax.scatter(
                np.asarray(branch["held_out_loss"]["per_seed"]) * 1e3,
                branch[key]["per_seed"],
                s=12,
                color=COLORS[strategy],
                alpha=0.8,
                lw=0,
                label=LABELS[strategy].replace("\n", " "),
            )
        ax.set_xlabel("held-out loss (×10⁻³, lower is better)")
        ax.set_ylabel(label)
    axes[1].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    figure.suptitle(
        "Held-out loss against the two downstream metrics (one point per trained branch)",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save(figure, output, "held_out_vs_downstream")


def atlas_figure(summary: dict, output: Path) -> None:
    atlas = summary["atlas"]
    names = {"pusht": "PushT (50 cases)", "tworoom": "TwoRoom (100 cases)"}
    figure, axes = plt.subplots(1, len(atlas), figsize=(3.3 * len(atlas), 2.7), sharey=False)
    for ax, (env, data) in zip(np.atleast_1d(axes), atlas.items(), strict=True):
        pressures = list(data["pressures"])
        for planner, color, label in (
            ("cem", COLORS["planner"], "CEM"),
            ("random_shooting", COLORS["random"], "random shooting"),
        ):
            cells = [data["pressures"][p][f"{planner}.selection_amplification"] for p in pressures]
            ax.errorbar(
                np.arange(len(pressures)),
                [cell["mean"] for cell in cells],
                yerr=[cell["se"] for cell in cells],
                fmt="-o",
                color=color,
                ms=3.5,
                capsize=2,
                lw=1.1,
                label=label,
            )
        ax.axhline(0, color="0.3", lw=0.7, ls="--")
        budgets = [data["pressures"][p]["samples"] for p in pressures]
        ax.set_xticks(
            range(len(pressures)),
            [
                f"{p}\n{b['cem_samples']}×{b['cem_iterations']}"
                for p, b in zip(pressures, budgets, strict=True)
            ],
        )
        ax.set_xlabel("pressure (CEM samples × iterations)")
        ax.set_ylabel("selection amplification\n(optimism of chosen − mean of q0)")
        ax.set_title(names.get(env, env), loc="left")
    np.atleast_1d(axes)[0].legend(frameon=False)
    figure.suptitle(
        "Released models: chosen plans are more optimistic than typical proposals (± SE over cases)",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save(figure, output, "atlas_amplification")


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    overview_figure(summary, args.output)
    acquisition_figure(summary, args.output)
    primary_figure(summary, args.output)
    strategy_figure(summary, args.output)
    dissociation_figure(summary, args.output)
    held_out_figure(summary, args.output)
    if "replay" in summary:
        replay_figure(summary, args.output)
    if "atlas" in summary:
        atlas_figure(summary, args.output)
    for path in sorted(args.output.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
