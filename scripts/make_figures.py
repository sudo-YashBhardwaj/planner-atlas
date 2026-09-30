"""The write-up's figures, drawn from the results summaries alone.

    uv run --group figures python scripts/make_figures.py \
        --summary docs/results/summary.json --output docs/figures \
        --stage1-summary docs/results/optimization_depth_stage1_summary.json

Every interval is over seeds, the unit of replication: a 95% t interval of per-seed values, each
of which is already a mean over the 48 held-out cases. The Atlas figure is the one exception: it
audits one fixed released model, so its error bars are standard errors over cases. The Stage-1
figure shows means over its 48 cases and the per-case slopes its gate tested.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

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


INK = "#1f2328"
MUTED = "#57606a"
PANEL = "#f4f6f8"
FRAME = "#eaeef2"
LATENT_SHADES = ("#d8e1ea", "#aebfd0", "#8ea6bd", "#c3d0dd", "#9fb3c7", "#e4eaf0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage1-summary", type=Path, help="Stage-1 analysis summary; adds its figure when given"
    )
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


def rounded(ax, x, y, width, height, *, face, edge="none", lw=0.0, radius=0.9, zorder=1):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            fc=face,
            ec=edge,
            lw=lw,
            zorder=zorder,
        )
    )


def link(ax, start, end, *, style="arc3,rad=0", lw=1.1) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            color=MUTED,
            lw=lw,
            connectionstyle=style,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )


def pusht_frame(ax, x, y, size, *, angle, shift, agent, label) -> None:
    """A stylized PushT frame: the T block (a 4 x 1 bar over a 1 x 3 stem) and the agent disk."""
    rounded(ax, x, y, size, size, face="white", edge="#c9d1d9", lw=0.8, radius=0.6, zorder=2)
    tee = np.array([(-2, 4), (2, 4), (2, 3), (0.5, 3), (0.5, 0), (-0.5, 0), (-0.5, 3), (-2, 3)])
    tee = (tee - (0.0, 2.64)) * size / 11  # about the T's centre of mass
    turn = np.deg2rad(angle)
    rotation = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
    centre = np.array([x + size / 2 + shift[0], y + size / 2 + shift[1]])
    ax.add_patch(Polygon(tee @ rotation.T + centre, closed=True, fc="#8796a8", ec="none", zorder=3))
    ax.add_patch(Circle((x + agent[0] * size, y + agent[1] * size), 0.06 * size, fc=INK, zorder=3))
    ax.text(x + size / 2, y - 0.9, label, ha="center", va="top", fontsize=8.5, color=MUTED)


def encoder_icon(ax, x, middle) -> None:
    corners = [
        (x, middle - 3.0),
        (x + 3.0, middle - 1.7),
        (x + 3.0, middle + 1.7),
        (x, middle + 3.0),
    ]
    ax.add_patch(Polygon(corners, closed=True, fc="#dde3ea", ec="#aab4bf", lw=0.8, zorder=3))


def latent_icon(ax, x, bottom, label) -> None:
    for i, shade in enumerate(LATENT_SHADES):
        ax.add_patch(
            Rectangle((x, bottom + 1.05 * i), 1.3, 1.05, fc=shade, ec="white", lw=0.8, zorder=3)
        )
    ax.text(x + 0.65, bottom + 6.9, label, ha="center", va="bottom", fontsize=10.5, color=INK)


def panel_header(ax, x, top, number: int, title: str, subtitle: str) -> None:
    ax.add_patch(Circle((x + 1.2, top), 1.2, fc=INK, zorder=3))
    ax.text(
        x + 1.2,
        top - 0.05,
        str(number),
        ha="center",
        va="center",
        fontsize=9.5,
        color="white",
        fontweight="bold",
        zorder=4,
    )
    ax.text(x + 3.2, top, title, ha="left", va="center", fontsize=12, color=INK, fontweight="bold")
    ax.text(x, top - 2.9, subtitle, ha="left", va="center", fontsize=9, color=MUTED)


def signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}".replace("-", "\u2212")


def schematic_figure(summary: dict, output: Path) -> None:
    """The README's one-picture account: plan in the model, what search selects, repair."""
    figure = plt.figure(figsize=(12.4, 4.5))
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 45)
    ax.axis("off")
    for x, width in ((0.5, 44.0), (45.5, 34.0), (80.5, 43.0)):
        rounded(ax, x, 0.5, width, 44.0, face=PANEL, radius=1.6, zorder=0)

    # 1. Planning inside the world model.
    panel_header(
        ax,
        2.0,
        41.6,
        1,
        "Plan inside the world model",
        "Released LeWM on PushT; CEM minimizes the predicted cost.",
    )
    pusht_frame(ax, 2.3, 24.0, 9.0, angle=28, shift=(0.6, 0.2), agent=(0.2, 0.25), label="start")
    pusht_frame(
        ax,
        2.3,
        4.0,
        9.0,
        angle=-12,
        shift=(-0.4, 0.4),
        agent=(0.78, 0.3),
        label="goal: 25 steps later",
    )
    for middle in (28.5, 8.5):
        link(ax, (11.3, middle), (12.9, middle))
        encoder_icon(ax, 13.0, middle)
        link(ax, (16.0, middle), (17.7, middle))
    ax.text(14.5, 32.6, "encoder\n(frozen)", ha="center", va="bottom", fontsize=8.5, color=MUTED)
    latent_icon(ax, 17.8, 25.35, "$z_0$")
    latent_icon(ax, 17.8, 5.35, r"$z_{\mathrm{goal}}$")
    rounded(ax, 21.5, 24.3, 9.6, 8.4, face="white", edge="#aab4bf", lw=0.8, radius=0.8, zorder=2)
    ax.text(
        26.3, 30.3, "predictor", ha="center", va="center", fontsize=10, color=INK, fontweight="bold"
    )
    ax.text(
        26.3,
        27.2,
        "rolls the plan\nforward 5 blocks",
        ha="center",
        va="center",
        fontsize=8,
        color=MUTED,
        linespacing=1.3,
    )
    link(ax, (19.1, 28.5), (21.5, 28.5))
    link(ax, (31.1, 28.5), (33.6, 28.5))
    latent_icon(ax, 33.7, 25.35, r"$\hat{z}_5$")
    for i in range(5):
        rounded(ax, 21.5 + 1.85 * i, 16.6, 1.6, 2.6, face="#c3d0dd", radius=0.35, zorder=2)
    ax.text(
        20.9,
        17.9,
        "plan: 5 action\nblocks, 25 steps",
        ha="right",
        va="center",
        fontsize=8.5,
        color=MUTED,
        linespacing=1.3,
    )
    link(ax, (26.3, 19.3), (26.3, 24.3))
    rounded(ax, 32.0, 12.8, 12.0, 7.2, face="white", edge="#aab4bf", lw=0.8, radius=0.8, zorder=2)
    ax.text(38.0, 18.2, "predicted cost", ha="center", va="center", fontsize=8.5, color=MUTED)
    ax.text(
        38.0,
        15.2,
        r"$\hat{C} = \Vert\hat{z}_5 - z_{\mathrm{goal}}\Vert^2$",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    link(ax, (34.35, 25.3), (34.35, 20.0))
    link(ax, (19.1, 8.5), (40.5, 12.8), style="angle,angleA=0,angleB=90,rad=0")
    link(ax, (33.6, 12.8), (23.3, 16.5), style="arc3,rad=-0.3", lw=1.3)
    ax.text(
        28.3,
        11.0,
        "CEM search",
        ha="center",
        va="center",
        fontsize=8.5,
        color=INK,
        fontweight="bold",
    )

    # 2. What the search selects: the acquisition streams of the confirmatory run.
    panel_header(
        ax,
        47.0,
        41.6,
        2,
        "Search picks optimistic plans",
        "optimism = realized \u2212 predicted latent cost",
    )
    strategies = ("random", "uncertainty", "planner")
    names = ("random plan", "most uncertain\nof 128 plans", "planner's plan\n(CEM 256 \u00d7 4)")
    acquisition = summary["acquisition"]["strategies"]
    to_figure = figure.transFigure.inverted()
    left, bottom = to_figure.transform(ax.transData.transform((58.0, 14.5)))
    right, top = to_figure.transform(ax.transData.transform((76.5, 33.0)))
    bars = figure.add_axes((left, bottom, right - left, top - bottom))
    for row, strategy in enumerate(strategies):
        cell = acquisition[strategy]["signed_optimism"]
        y = len(strategies) - 1 - row
        bars.barh(y, cell["mean"], height=0.56, color=COLORS[strategy], lw=0)
        seeds = np.asarray(cell["per_seed"])
        bars.scatter(
            seeds,
            y + np.linspace(-0.16, 0.16, len(seeds)),
            s=6,
            color=INK,
            alpha=0.55,
            lw=0,
            zorder=3,
        )
        bars.text(
            seeds.max() + 1.5, y, signed(cell["mean"], 1), va="center", fontsize=9.5, color=INK
        )
    bars.set_yticks(range(len(strategies)), names[::-1], fontsize=8.5, color=INK)
    bars.tick_params(axis="y", length=0)
    bars.set_xlim(0, 60)
    bars.set_xticks([0, 20, 40, 60])
    bars.tick_params(axis="x", labelsize=8, colors=MUTED)
    bars.set_xlabel("signed optimism", fontsize=8.5, color=MUTED)
    bars.set_facecolor(PANEL)
    for side in ("top", "right", "left"):
        bars.spines[side].set_visible(False)
    bars.spines["bottom"].set_color("#aab4bf")
    success = ", ".join(f"{acquisition[s]['semantic_success']['mean']:.2f}" for s in strategies)
    ax.text(
        47.0,
        7.0,
        f"Open-loop task success of the same plans:\n{success}; dots are the 12 seeds.",
        ha="left",
        va="center",
        fontsize=8.5,
        color=MUTED,
        linespacing=1.35,
    )
    ax.text(
        47.0,
        3.0,
        "Picking the best-looking plan favours plans\nwhose cost the model underestimates.",
        ha="left",
        va="center",
        fontsize=8.5,
        color=INK,
        linespacing=1.35,
    )

    # 3. Repair with 40,000 transitions and what it changed.
    panel_header(
        ax,
        82.0,
        41.6,
        3,
        "Repair with 40,000 new transitions",
        "Fine-tune on each strategy's data, 12 seeds each, then test.",
    )
    chips = (
        ("random", "random plans"),
        ("uncertainty", "most uncertain plans"),
        ("planner", "the planner's plans"),
        ("continued", "continued: no new data"),
    )
    for i, (strategy, text) in enumerate(chips):
        x, y = 82.0 + 20.4 * (i % 2), 32.0 - 3.3 * (i // 2)
        rounded(ax, x, y, 19.4, 2.7, face="white", edge="#d0d7de", lw=0.6, radius=0.5, zorder=2)
        rounded(ax, x + 0.7, y + 0.6, 1.5, 1.5, face=COLORS[strategy], radius=0.3, zorder=3)
        ax.text(x + 2.9, y + 1.35, text, ha="left", va="center", fontsize=8.8, color=INK)
    link(ax, (101.9, 28.6), (101.9, 26.4))
    rounded(ax, 82.0, 21.4, 39.8, 5.0, face="white", edge="#aab4bf", lw=0.8, radius=0.8, zorder=2)
    ax.text(
        101.9,
        24.9,
        "fine-tune the dynamics, image encoder frozen",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        101.9,
        22.6,
        "1,000 steps; half base, half new data (continued: base only)",
        ha="center",
        va="center",
        fontsize=8.5,
        color=MUTED,
    )
    primary = summary["confirmatory"]["primary"]
    mpc = summary["confirmatory"]["contrasts"]["mpc_semantic_reached_success"]
    gain = mpc["random-continued"]
    ranking = (
        f"{signed(primary['mean'], 4)}  "
        f"[{signed(primary['ci_low'], 4)}, {signed(primary['ci_high'], 4)}]"
    )
    control = (
        f"{signed(gain['mean'], 3)}  [{signed(gain['ci_low'], 3)}, {signed(gain['ci_high'], 3)}]"
    )
    cards = (
        (
            82.0,
            "planner",
            "Planner-region ranking",
            [
                ("regret: planner \u2212 random", MUTED),
                (ranking, INK),
                (
                    f"{primary['negative']} of {primary['n']} seeds, p = {primary['sign_flip_p']:.4f}",
                    INK,
                ),
                ("pre-registered: confirmed", INK),
            ],
        ),
        (
            102.4,
            "random",
            "Closed-loop MPC",
            [
                ("success: random \u2212 continued", MUTED),
                (control, INK),
                (f"planner \u2212 continued: {signed(mpc['planner-continued']['mean'], 3)}", INK),
                ("secondary, descriptive", INK),
            ],
        ),
    )
    for x, strategy, title, lines in cards:
        link(ax, (x + 9.7, 21.4), (x + 9.7, 19.6))
        rounded(ax, x, 5.2, 19.4, 14.4, face="white", edge="#d0d7de", lw=0.6, radius=0.8, zorder=2)
        ax.add_patch(Rectangle((x + 0.9, 17.9), 17.6, 0.5, fc=COLORS[strategy], lw=0, zorder=3))
        ax.text(
            x + 0.9, 16.0, title, ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold"
        )
        for j, (text, color) in enumerate(lines):
            ax.text(
                x + 0.9, 13.4 - 2.2 * j, text, ha="left", va="center", fontsize=8.4, color=color
            )
    ax.text(
        82.0,
        2.7,
        "Held-out loss ranked the branches differently from both.",
        ha="left",
        va="center",
        fontsize=8.5,
        color=MUTED,
    )
    save(figure, output, "schematic")


def depth_figure(stage1: dict, output: Path) -> None:
    """Stage 1: fixed-proposal random shooting at N = 32 ... 4,096 candidates, 48 cases."""
    curves, null = stage1["mean_curves"], stage1["mean_null_curves"]
    ladder = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    index = [curves["prefixes"].index(float(n)) for n in ladder]
    x = np.log2(ladder)
    ticks = ["32", "64", "128", "256", "512", "1k", "2k", "4k"]
    figure, axes = plt.subplots(
        1, 3, figsize=(7.6, 2.75), gridspec_kw={"width_ratios": [1, 1, 1.1]}
    )
    panels = (
        (
            axes[0],
            "a  Latent regret of the chosen plan",
            "latent regret",
            (
                ("r_latent", curves, "observed", COLORS["planner"]),
                ("mean_r_latent", null, "exchangeable null", "#8c8c8c"),
            ),
        ),
        (
            axes[1],
            "b  True task cost (lower is better)",
            "task cost",
            (
                ("selected_task_cost", curves, "chosen plan", COLORS["planner"]),
                ("oracle_task_cost", curves, "best plan in the pool", "#8c8c8c"),
            ),
        ),
    )
    for (ax, title, ylabel, series), corner in zip(
        panels, ("upper left", "lower left"), strict=True
    ):
        ax.axvspan(8, 12, color="0.94", lw=0, zorder=0)
        for key, source, label, color in series:
            ax.plot(
                x, [source[key][i] for i in index], "-o", color=color, ms=3.2, lw=1.3, label=label
            )
        ax.set_xticks(x, ticks)
        ax.set_xlabel("candidates N (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc=corner, fontsize=7)
    axes[0].text(
        10,
        axes[0].get_ylim()[0],
        "gate window",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=MUTED,
    )
    ax = axes[2]
    slopes = np.asarray(stage1["case_vectors"]["beta_latent"])
    cell = stage1["summaries"]["beta_latent"]
    ax.plot([0, 0], [-0.45, 0.4], color="0.3", lw=0.7, ls="--")
    rows = np.random.default_rng(0).uniform(-0.28, 0.28, len(slopes))
    ax.scatter(
        slopes,
        rows,
        s=10,
        color=[COLORS["planner"] if v > 0 else "#8c8c8c" for v in slopes],
        lw=0,
        alpha=0.8,
    )
    ax.errorbar(
        cell["mean"],
        0.55,
        xerr=[[cell["mean"] - cell["ci_low"]], [cell["ci_high"] - cell["mean"]]],
        fmt="o",
        color=INK,
        ms=4,
        capsize=2.5,
        lw=1.2,
    )
    ax.text(cell["mean"], 0.72, "mean, 95% CI", ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_ylim(-0.5, 1.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("per-case slope of latent regret\nper doubling of N (256 to 4,096)")
    ax.set_title("c  Pre-registered gate G1", loc="left")
    ax.text(
        0.02,
        0.97,
        f"{cell['positive']}/{cell['n']} cases > 0 (needed 34)\n"
        f"d_z = {cell['d_z']:.2f} (needed 0.5)\n"
        f"sign-flip p = {cell['mc_sign_flip_p']:.3f} (needed < 0.05)\n"
        f"verdict: {stage1['verdict']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=INK,
    )
    figure.suptitle(
        f"Stage 1 (pre-registered, {stage1['verdict']}): regret grows about as the null predicts, "
        "and task cost falls overall",
        fontsize=8.5,
    )
    figure.tight_layout()
    save(figure, output, "stage1_depth")


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
    schematic_figure(summary, args.output)
    if args.stage1_summary is not None:
        depth_figure(json.loads(args.stage1_summary.read_text()), args.output)
    for path in sorted(args.output.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
