"""Every number of the write-up, computed from the experiment artifacts into one JSON summary.

    uv run python scripts/summarize_results.py --confirm-dir runs/confirm2 \
        --replay-rows runs/replay/alpha0125-merged.jsonl runs/replay/alpha025.jsonl \
        --replay-protocol runs/replay/protocol.md \
        --atlas pusht=runs/pusht-atlas/atlas50.jsonl tworoom=runs/tworoom-atlas/dev100.jsonl \
        --output docs/results/summary.json

This only reads: nothing is trained, planned or executed. The confirmatory directory holds the
grid, the manifest and the acquisition streams exactly as the protocol produced them. Replay rows
are the exploratory post-confirmation replay-weight branches, grouped by the mixture each row
records. --coverage also reads the acquired latents and the latent cache, which needs the dataset,
the checkpoint and a GPU for a few minutes.

Seeds are the unit of inference throughout (see planner_atlas.statistics). The summary records
input files by name and digest, never by path, so it can live in the repository.
"""

import argparse
import json
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from planner_atlas.statistics import paired_summary
from planner_atlas.training import file_digest

PRIMARY = ("planner", "random", "planner_region_regret")
STRATEGIES = ("continued", "random", "uncertainty", "planner")
GRID_METRICS = (
    "planner_region_regret",
    "q0_regret",
    "mpc_semantic_reached_success",
    "held_out_loss",
    "selected_task_cost",
    "selected_semantic_success",
    "selected_optimism",
    "selected_absolute_optimism",
    "selected_rollout_error",
    "selected_terminal_error",
    "planner_region_absolute_optimism",
    "planner_region_rollout_error",
    "planner_region_rank_end_to_end",
    "q0_rank_end_to_end",
)
CONTRASTS = (
    ("planner", "random"),
    ("planner", "continued"),
    ("random", "continued"),
    ("uncertainty", "random"),
    ("uncertainty", "continued"),
    ("planner", "uncertainty"),
)
ACQUISITION_METRICS = {
    "signed_optimism": lambda r: r["outcome_optimism"],
    "absolute_optimism": lambda r: abs(r["outcome_optimism"]),
    "rollout_error": lambda r: r["outcome_rollout_error"],
    "terminal_error": lambda r: r["outcome_terminal_error"],
    "task_cost": lambda r: r["outcome_task_cost"],
    "semantic_success": lambda r: float(r["outcome_semantic_final_success"]),
}
REPLAY_METRICS = (
    "planner_region_regret",
    "q0_regret",
    "mpc_semantic_reached_success",
    "held_out_loss",
    "selected_task_cost",
    "selected_absolute_optimism",
    "selected_rollout_error",
)
ATLAS_METRICS = {  # summary name: (PushT key, TwoRoom key)
    "selected_optimism": ("selected_optimism", "selected_optimism"),
    "q0_mean_optimism": ("q0_mean_optimism", "q0_mean_optimism"),
    "selection_amplification": ("selection_amplification", "selection_amplification"),
    "selected_rollout_error": ("selected_rollout_error", "selected_rollout_error"),
    "q0_mean_rollout_error": ("q0_mean_rollout_error", "q0_mean_rollout_error"),
    "selected_task_cost": ("selected_task_cost", "selected_task_cost"),
    "open_loop_success": ("selected_semantic_final_success", "selected_success"),
    "mpc_success": ("mpc_semantic_reached_success", "mpc_success"),
    "q0_rank_end_to_end": ("q0_rank_end_to_end", "q0_rank_end_to_end"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirm-dir", type=Path, required=True)
    parser.add_argument("--replay-rows", type=Path, nargs="*", default=())
    parser.add_argument("--replay-protocol", type=Path, help="recorded before the replay branches")
    parser.add_argument("--atlas", nargs="*", default=(), help="env=atlas.jsonl")
    parser.add_argument("--coverage", action="store_true", help="latent coverage of the streams")
    parser.add_argument("--dataset", type=Path, help="for --coverage")
    parser.add_argument("--checkpoint", type=Path, help="for --coverage")
    parser.add_argument("--latent-cache", type=Path, help="for --coverage")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file]


def one(values, what: str):
    """The single value every row agrees on, or an error naming the disagreement."""
    distinct = {json.dumps(value, sort_keys=True) for value in values}
    if len(distinct) != 1:
        raise ValueError(f"rows disagree on {what}: {sorted(distinct)[:3]}")
    return json.loads(distinct.pop())


def provenance(path: Path) -> dict:
    return {"file": path.name, "sha256": file_digest(path)}


def seed_means(rows: list[dict], metrics) -> tuple[dict, list[int]]:
    """{(strategy, seed): {metric: mean over cases}}; every branch must cover the same cases."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["strategy"], row["seed"]].append(row)
    cases = {key: sorted(row["case"] for row in group) for key, group in groups.items()}
    reference = one(cases.values(), "the evaluated cases")
    means = {
        key: {metric: float(np.nanmean([row[metric] for row in group])) for metric in metrics}
        for key, group in groups.items()
    }
    return means, reference


def seed_vector(means: dict, strategy: str, seeds: list[int], metric: str) -> np.ndarray:
    return np.array([means[strategy, seed][metric] for seed in seeds])


def describe(values: np.ndarray) -> dict:
    """Seed-level mean, spread and 95% t interval of one branch's per-seed values."""
    summary = paired_summary(values)
    return {
        "per_seed": values.tolist(),
        "mean": summary.mean,
        "sd": summary.sd,
        "se": summary.se,
        "ci95": [summary.ci_low, summary.ci_high],
    }


def contrast(treatment: np.ndarray, control: np.ndarray) -> dict:
    return {
        "per_seed": (treatment - control).tolist(),
        **asdict(paired_summary(treatment - control)),
    }


def confirmatory(directory: Path) -> dict:
    rows = read_rows(directory / "grid.jsonl")
    trained = [row for row in rows if row["strategy"] != "original_base"]
    means, cases = seed_means(trained, GRID_METRICS)
    seeds = sorted({seed for _, seed in means})
    for strategy in STRATEGIES:
        missing = [seed for seed in seeds if (strategy, seed) not in means]
        if missing:
            raise ValueError(f"{strategy} lacks seeds {missing}")
    base = [row for row in rows if row["strategy"] == "original_base"]
    summary = {
        "provenance": {
            "grid": provenance(directory / "grid.jsonl"),
            "manifest": provenance(directory / "manifest.json"),
            "rows": len(rows),
            "branches": len({row["branch"] for row in rows}),
            "cases": len(cases),
            "seeds": seeds,
            "budget": one([row["budget"] for row in trained if row["budget"]], "budget"),
            "repair_fraction": one([row["repair_fraction"] for row in rows], "repair fraction"),
            "git_commit": one([row["git_commit"] for row in rows], "git commit"),
            "dynamics_protocol": one([row["dynamics_protocol"] for row in rows], "protocol"),
            "manifest_digest": one([row["manifest"] for row in rows], "manifest"),
            "protocol_digest": one([row["protocol"] for row in rows], "protocol digest"),
            "held_out_windows": one([row["held_out_windows"] for row in rows], "held-out"),
            "held_out_episodes": one([row["held_out_episodes"] for row in rows], "held-out"),
        },
        "original_base": {
            metric: float(np.nanmean([row[metric] for row in base])) for metric in GRID_METRICS
        },
        "branches": {
            strategy: {
                metric: describe(seed_vector(means, strategy, seeds, metric))
                for metric in GRID_METRICS
            }
            for strategy in STRATEGIES
        },
        "contrasts": {
            metric: {
                f"{a}-{b}": contrast(
                    seed_vector(means, a, seeds, metric), seed_vector(means, b, seeds, metric)
                )
                for a, b in CONTRASTS
            }
            for metric in GRID_METRICS
        },
    }
    if summary["provenance"]["manifest_digest"] != summary["provenance"]["manifest"]["sha256"]:
        raise ValueError("the grid rows were not scored on this manifest")
    a, b, metric = PRIMARY
    summary["primary"] = {"treatment": a, "control": b, "metric": metric}
    summary["primary"] |= summary["contrasts"][metric][f"{a}-{b}"]
    return summary


def acquisition(directory: Path, seeds: list[int]) -> dict:
    """What each strategy acquired: per-seed means over every executed plan of its stream."""
    per_seed = defaultdict(lambda: defaultdict(list))
    contexts, commits = {}, set()
    for strategy in ("random", "uncertainty", "planner"):
        for seed in seeds:
            records = read_rows(directory / f"{strategy}-seed{seed}.jsonl")
            one([record["strategy"] for record in records], "strategy")
            if one([record["stream_seed"] for record in records], "seed") != seed:
                raise ValueError(f"{strategy}-seed{seed} holds another seed's stream")
            commits.add(one([record["git_commit"] for record in records], "commit"))
            contexts[strategy, seed] = [(r["episode"], r["start_step"]) for r in records]
            for name, value in ACQUISITION_METRICS.items():
                per_seed[strategy][name].append(float(np.mean([value(r) for r in records])))
            per_seed[strategy]["transitions"].append(sum(r["transitions"] for r in records))
            per_seed[strategy]["plans"].append(len(records))
    shared = all(
        contexts["random", seed] == contexts[other, seed]
        for seed in seeds
        for other in ("uncertainty", "planner")
    )
    if not shared:
        raise ValueError("strategies did not face identical contexts")
    return {
        "provenance": {"git_commit": sorted(commits), "identical_contexts": shared},
        "strategies": {
            strategy: {
                name: describe(np.array(values)) if name not in ("transitions", "plans") else values
                for name, values in metrics.items()
            }
            for strategy, metrics in per_seed.items()
        },
    }


def replay(confirm_rows: list[dict], files: list[Path], protocol: Path | None) -> dict:
    """The exploratory post-confirmation replay-weight analysis, on seed means per alpha."""
    confirm_means, cases = seed_means(
        [row for row in confirm_rows if row["strategy"] != "original_base"], REPLAY_METRICS
    )
    pools = {
        row["case"]: (row["q0_best_task_cost"], row["planner_region_best_task_cost"])
        for row in confirm_rows
    }
    by_alpha, sources = defaultdict(list), []
    for path in files:
        rows = read_rows(path)
        sources.append(provenance(path))
        for row in rows:
            if (row["q0_best_task_cost"], row["planner_region_best_task_cost"]) != pools[
                row["case"]
            ]:
                raise ValueError(f"{path.name} was scored on other candidate pools")
            by_alpha[row["repair_fraction"]].append(row)
    confirm_commit = one([row["git_commit"] for row in confirm_rows], "commit")
    replay_rows = [row for rows in by_alpha.values() for row in rows]
    if one([row["git_commit"] for row in replay_rows], "commit") != confirm_commit:
        raise ValueError("the replay branches ran on other code than the confirmatory grid")
    seeds = sorted({seed for _, seed in confirm_means})
    alphas = [0.0, *sorted(by_alpha), 0.5]
    means = {}
    for alpha in alphas:
        for strategy in ("random", "planner"):
            if alpha == 0.0:  # repair fraction 0 is the continued control, shared by both
                source, name = confirm_means, "continued"
            elif alpha == 0.5:
                source, name = confirm_means, strategy
            else:
                source, _ = seed_means(by_alpha[alpha], REPLAY_METRICS)
                name = strategy
            for seed in seeds:
                means[alpha, strategy, seed] = source[name, seed]

    def vector(alpha, strategy, metric):
        return np.array([means[alpha, strategy, seed][metric] for seed in seeds])

    rises = {}  # seeds whose held-out loss rises at each step of alpha, and at every step
    for strategy in ("random", "planner"):
        steps = np.diff([vector(a, strategy, "held_out_loss") for a in alphas], axis=0) > 0
        rises[strategy] = {
            f"{low}->{high}": int(step.sum())
            for low, high, step in zip(alphas[:-1], alphas[1:], steps, strict=True)
        } | {"every_step": int(steps.all(axis=0).sum())}
    return {
        "status": "exploratory: run after the confirmatory result was inspected, against the "
        "protocol's stopping rule",
        "provenance": {
            "rows": sources,
            "protocol": provenance(protocol) if protocol else None,
            "protocol_digest": one([row["protocol"] for row in replay_rows], "protocol digest"),
            "git_commit": confirm_commit,
            "alpha_0_and_0.5_from": "confirmatory grid",
            "branches_per_alpha": {
                str(alpha): len({row["branch"] for row in rows}) for alpha, rows in by_alpha.items()
            },
            "cases": len(cases),
        },
        "alphas": alphas,
        "seeds": seeds,
        "strategies": {
            strategy: {
                metric: {str(a): describe(vector(a, strategy, metric)) for a in alphas}
                for metric in REPLAY_METRICS
            }
            for strategy in ("random", "planner")
        },
        "versus_continued": {
            strategy: {
                metric: {
                    str(a): contrast(vector(a, strategy, metric), vector(0.0, strategy, metric))
                    for a in alphas[1:]
                }
                for metric in REPLAY_METRICS
            }
            for strategy in ("random", "planner")
        },
        "planner_minus_random": {
            metric: {
                str(a): contrast(vector(a, "planner", metric), vector(a, "random", metric))
                for a in alphas[1:]
            }
            for metric in REPLAY_METRICS
        },
        "held_out_loss_rises_in_seeds": rises,
    }


def atlas(entries: list[str]) -> dict:
    """The released models' Atlas, per environment, pressure and planner; cases are the unit here,
    since one fixed model is audited on independently drawn cases."""
    summary = {}
    for entry in entries:
        env, path = entry.split("=", 1)
        rows = read_rows(Path(path))
        column = 0 if env == "pusht" else 1
        cells = defaultdict(dict)
        for pressure in sorted({row["pressure"] for row in rows}):
            for planner in ("random_shooting", "cem"):
                group = [r for r in rows if r["pressure"] == pressure and r["planner"] == planner]
                for name, keys in ATLAS_METRICS.items():
                    values = np.array([row[keys[column]] for row in group], dtype=float)
                    values = values[~np.isnan(values)]
                    cells[pressure][f"{planner}.{name}"] = {
                        "mean": float(values.mean()),
                        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
                        "n": len(values),
                    }
        summary[env] = {
            "provenance": provenance(Path(path)) | {"cases": len({r["case"] for r in rows})},
            "pressures": {
                pressure: {"samples": planner_budget(rows, pressure), **cells[pressure]}
                for pressure in cells
            },
        }
    return summary


def planner_budget(rows: list[dict], pressure: str) -> dict:
    row = next(r for r in rows if r["pressure"] == pressure and r["planner"] == "cem")
    return {"cem_samples": row["num_samples"], "cem_iterations": row["iterations"]}


def coverage(args, seeds: list[int]) -> dict:
    """How the random and planner streams sit relative to the base training latents."""
    import h5py
    import torch

    from planner_atlas.training import load_latent_cache, split_episodes

    cache = load_latent_cache(args.latent_cache, args.dataset, args.checkpoint)
    validation = split_episodes(len(cache.lengths), validation_fraction=0.1, seed=0)
    episode_of_row = np.repeat(np.arange(len(cache.lengths)), cache.lengths)
    train_rows = np.flatnonzero(~validation[episode_of_row])
    chosen = np.sort(np.random.default_rng(0).choice(train_rows, 100_000, replace=False))
    reference = torch.from_numpy(cache.latents[chosen]).to(args.device)
    centre = reference.mean(0)
    precision = torch.linalg.inv(
        torch.cov(reference.T) + 1e-6 * torch.eye(reference.shape[1], device=args.device)
    )

    def nearest(z):
        return torch.cat([torch.cdist(part, reference).min(1).values for part in z.split(2048)])

    def mahalanobis(z):
        offset = z - centre
        return torch.sqrt(((offset @ precision) * offset).sum(1))

    def participation(z):
        eigen = torch.linalg.eigvalsh(torch.cov(z.T))
        return float(eigen.sum() ** 2 / (eigen**2).sum())

    values = defaultdict(lambda: defaultdict(list))
    stream = args.confirm_dir / "acquisition"
    for strategy in ("random", "planner"):
        for seed in seeds:
            with h5py.File(stream / f"{strategy}-seed{seed}.h5", "r") as file:
                latents = torch.from_numpy(file["latents"][:]).to(args.device)  # [K, H + 1, D]
            flat = latents.reshape(-1, latents.shape[-1])
            by_block = nearest(flat).reshape(len(latents), -1)
            distance = mahalanobis(flat).reshape(len(latents), -1)
            record = values[strategy]
            record["latent_total_variance"].append(float(torch.cov(flat.T).trace()))
            record["participation_ratio"].append(participation(flat))
            record["nearest_base_distance"].append(float(by_block[:, 1:].mean()))
            record["mahalanobis_to_base"].append(float(distance[:, 1:].mean()))
            record["nearest_base_by_block"].append(by_block.mean(0).cpu().numpy().tolist())
    summary = {
        "status": "descriptive",
        "reference": "100,000 training-split latents of the live cache (rng seed 0); executed "
        "blocks 1..H, the shared start excluded",
        "strategies": {},
    }
    for strategy, record in values.items():
        summary["strategies"][strategy] = {
            name: describe(np.array(series))
            for name, series in record.items()
            if name != "nearest_base_by_block"
        } | {"nearest_base_by_block": np.mean(record["nearest_base_by_block"], axis=0).tolist()}
    return summary


def generation() -> dict:
    """The code that generated this summary: the repository HEAD when it ran.

    A summary is committed after it is generated, so the commit that stores it is always a later
    one; the results themselves come from the commit their rows record.
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    changes = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return {
        "source_commit": commit.stdout.strip() or "unknown",
        "source_tree_clean": not changes.stdout.strip(),
        "meaning": "source_commit is the code that generated this file, not the commit that "
        "stores it; the results were produced by confirmatory.provenance.git_commit",
    }


def main() -> None:
    args = parse_args()
    confirm = confirmatory(args.confirm_dir)
    seeds = confirm["provenance"]["seeds"]
    protocols = [provenance(path) for path in sorted(args.confirm_dir.glob("protocol*.md"))]
    if confirm["provenance"]["protocol_digest"] not in {file["sha256"] for file in protocols}:
        raise ValueError("the grid rows were not produced under a protocol in --confirm-dir")
    confirm["provenance"]["protocol_files"] = protocols
    summary = {
        "generation": generation(),
        "confirmatory": confirm,
        "acquisition": acquisition(args.confirm_dir / "acquisition", seeds),
    }
    if args.replay_rows:
        rows = read_rows(args.confirm_dir / "grid.jsonl")
        summary["replay"] = replay(rows, list(args.replay_rows), args.replay_protocol)
    if args.atlas:
        summary["atlas"] = atlas(list(args.atlas))
    if args.coverage:
        summary["coverage"] = coverage(args, seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1) + "\n")

    primary = confirm["primary"]
    print(
        f"primary {primary['treatment']} - {primary['control']} {primary['metric']}: "
        f"{primary['mean']:+.6f} [{primary['ci_low']:+.6f}, {primary['ci_high']:+.6f}], "
        f"{primary['negative']}/{primary['n']} negative, sign-flip p {primary['sign_flip_p']:.4f}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
