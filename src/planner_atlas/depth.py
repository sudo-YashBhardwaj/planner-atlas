"""Stage 1 of the optimization-depth study: its per-case cache and every analysis step.

The study is defined by docs/protocols/optimization_depth_stage1.md, frozen at commit 8b47d24 and
amended before implementation in its section 22. For each of the 48 consumed confirmatory cases the
build step (scripts/optimization_depth_stage1.py) executes, once, an immutable bank of 4096 q0 plans
and one six-iteration CEM trace, and caches what the model predicted and what the simulator did.
Everything here then works on those cached arrays alone: nothing in this module steps an
environment or runs a model, so no analysis can change the data it analyses.

Random shooting over N candidates is a prefix of the one bank, never a fresh draw, so the only thing
that changes along the search ladder is how many candidates the model gets to choose from.
"""

import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import torch

from planner_atlas.atlas import HORIZON, pairwise_rank_agreement
from planner_atlas.coverage import BaseReference, mahalanobis_to_base, nearest_base_distance
from planner_atlas.planning import initial_proposal, sample_proposal
from planner_atlas.statistics import case_summary
from planner_atlas.training import file_digest

PROTOCOL_COMMIT = "8b47d24c3af277af451962949d6de051b95c0889"
PROTOCOL_DIGEST = "c87b34acce37c9e5e77d63038b01ecdf1250880bf4b0ba99579055c585302dda"
MANIFEST_DIGEST = "5e9458839dccb124ccd94394853a8592b79b27c94a11c4819bbe040dd8b5d746"
CHECKPOINT_DIGEST = "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
STUDY_DYNAMICS_PROTOCOL = "frozen_rep_v3"
MANIFEST_SEED, NUM_CASES, ENVIRONMENT = 307, 48, "pusht"
N_MAX = 4096
LADDER = (32, 64, 128, 256, 512, 1024, 2048, 4096)
PRIMARY = (256, 512, 1024, 2048, 4096)
MATCHED = (256, 512, 768, 1024, 1280, 1536)
PREFIXES = tuple(sorted(set(LADDER) | set(MATCHED)))
BANK_SEED = NULL_SEED = SIGN_SEED = 20260923
NULL_DRAWS, SIGN_DRAWS = 10_000, 100_000
CEM_POPULATION, CEM_ITERATIONS = 256, 6
DETERMINISM_CASE = 0
METRIC_VERSION = "stage1-v1"
OBJECTIVE_IDENTITY = "terminal latent cost: sum over 192 latent dims of (z_5 - z_goal)^2"
SAMPLER_IDENTITY = (
    "q0 = initial_proposal(5): N(0, I) over [5, 10] normalized blocks, clamped to "
    "model_action_bounds; sample_proposal with torch.Generator(device).manual_seed(20260923 + case)"
)
SIMULATOR_IDENTITY = (
    "rollout_pusht: reconstruct the case start by replaying its recorded episode, then execute 5 "
    "blocks of 5 raw actions; goal = live render of the replayed goal step (render_pusht_goal)"
)
RANK_BINS = {  # half-open slices of the candidates sorted by predicted cost, best first
    "0-1%": (0, 41),
    "1-5%": (41, 205),
    "5-10%": (205, 410),
    "10-25%": (410, 1024),
    "25-50%": (1024, 2048),
    "50-100%": (2048, 4096),
}
CANDIDATE_FIELDS = (
    "candidate_index",
    "actions",
    "predicted_latent_cost",
    "true_latent_cost",
    "signed_optimism",
    "absolute_optimism",
    "true_task_cost",
    "position_error",
    "angle_error",
    "rollout_error",
    "terminal_error",
    "semantic_success",
    "semantic_reached_success",
    "terminal_true_latent",
    "action_norm",
    "q0_negative_log_density",
)
METADATA_FIELDS = (
    "kind",
    "git_commit",
    "protocol_commit",
    "protocol_digest",
    "environment",
    "manifest_digest",
    "case_index",
    "episode",
    "start_step",
    "checkpoint_digest",
    "dynamics_protocol",
    "bank_seed",
    "n_max",
    "horizon",
    "action_bounds_identity",
    "objective_identity",
    "sampler_identity",
    "planner_identity",
    "metric_definition_version",
    "simulator_cache_identity",
)
GATES = {  # section 7, 9 and 10, as integer case counts out of all 48 (section 22, A3)
    "G1": {"statistic": "beta_latent", "positive": 34, "d_z": 0.5, "p_below": 0.05},
    "G2": {"statistic": "excess_beta", "positive": 32, "d_z": 0.5, "p_below": 0.05},
    "G3": {"statistic": "beta_task", "positive": 29, "d_z": 0.3, "p_below": None},
}
INCOMPLETE = "protocol execution incomplete"


# ---- provenance and cache -------------------------------------------------------------------


def verify_protocol(path: Path) -> dict:
    """The frozen protocol's identity, refusing any file that is not exactly it."""
    digest = file_digest(path)
    if digest != PROTOCOL_DIGEST:
        raise ValueError(f"{path} is not the frozen Stage-1 protocol ({digest[:12]})")
    return {"protocol_commit": PROTOCOL_COMMIT, "protocol_digest": PROTOCOL_DIGEST}


def load_frozen_manifest(path: Path) -> dict:
    """The consumed confirmatory manifest, verified; Stage 1 never writes a manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"{path}: Stage 1 reuses the consumed manifest and never makes one")
    if file_digest(path) != MANIFEST_DIGEST:
        raise ValueError(f"{path} is not the consumed confirmatory manifest {MANIFEST_DIGEST[:12]}")
    return json.loads(path.read_text())


def draw_bank(bounds: tuple[torch.Tensor, torch.Tensor], case_index: int) -> torch.Tensor:
    """The case's immutable q0 bank [N_MAX, 5, 10]; RS-N is always a prefix of it."""
    generator = torch.Generator(bounds[0].device).manual_seed(BANK_SEED + case_index)
    return sample_proposal(
        initial_proposal(HORIZON), num_samples=N_MAX, bounds=bounds, generator=generator
    )


def bounds_identity(bounds: tuple[torch.Tensor, torch.Tensor]) -> dict:
    low, high = (bound.cpu().numpy().astype(np.float64) for bound in bounds)
    return {
        "low": low.tolist(),
        "high": high.tolist(),
        "sha256": sha256(low.tobytes() + high.tobytes()).hexdigest(),
    }


def candidate_fields(evaluation, terminal_latents: np.ndarray, plans: np.ndarray) -> dict:
    """Cache fields of executed plans [K, 5, 10] from their CandidateEvaluation.

    Signed optimism is the evaluation's own realized − predicted latent cost, the convention of every
    earlier result row: positive means the model was over-optimistic.
    """
    flat = plans.reshape(len(plans), -1).astype(np.float64)
    return {
        "actions": plans,
        "predicted_latent_cost": evaluation.predicted_cost,
        "true_latent_cost": evaluation.realized_cost,
        "signed_optimism": evaluation.optimism,
        "absolute_optimism": np.abs(evaluation.optimism),
        "true_task_cost": evaluation.task["task_cost"],
        "position_error": evaluation.task["position_error"],
        "angle_error": evaluation.task["angle_error"],
        "rollout_error": evaluation.rollout_error,
        "terminal_error": evaluation.terminal_error,
        "semantic_success": evaluation.task["semantic_final_success"],
        "semantic_reached_success": evaluation.task["semantic_reached_success"],
        "terminal_true_latent": terminal_latents,
        "action_norm": np.linalg.norm(flat, axis=1),
        "q0_negative_log_density": 0.5 * np.square(flat).sum(axis=1),  # up to a constant
    }


def case_path(output: Path, kind: str, case_index: int) -> Path:
    return output / kind / f"case-{case_index:03d}.h5"


def expected_metadata(case_index: int, kind: str) -> dict:
    """What a cache must state to be accepted as one of this study's cases."""
    return {
        "kind": kind,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_digest": PROTOCOL_DIGEST,
        "environment": ENVIRONMENT,
        "manifest_digest": MANIFEST_DIGEST,
        "case_index": case_index,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "dynamics_protocol": STUDY_DYNAMICS_PROTOCOL,
        "bank_seed": BANK_SEED + case_index,
        "n_max": N_MAX,
        "horizon": HORIZON,
        "metric_definition_version": METRIC_VERSION,
    }


def candidate_count(kind: str) -> int:
    return N_MAX if kind == "rs" else CEM_POPULATION * CEM_ITERATIONS


def save_case(path: Path, fields: dict, metadata: dict, extra: dict | None = None) -> None:
    """Write one case cache, atomically and once: an existing cache is never overwritten."""
    if path.exists():
        raise FileExistsError(f"{path} exists; Stage-1 caches are immutable")
    missing = [name for name in CANDIDATE_FIELDS if name not in fields]
    missing += [name for name in METADATA_FIELDS if name not in metadata]
    if missing:
        raise ValueError(f"refusing to write {path}: missing {missing}")
    lengths = {len(fields[name]) for name in fields}
    if lengths != {candidate_count(metadata["kind"])}:
        raise ValueError(f"refusing to write {path}: candidate counts {sorted(lengths)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with h5py.File(partial, "w") as file:
        for name, values in (fields | (extra or {})).items():
            file[name] = np.asarray(values)
        file.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
    partial.replace(path)


def load_case(path: Path, *, expect: dict) -> tuple[dict[str, np.ndarray], dict]:
    """A complete case cache, refused unless it states exactly the expected study identity."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: Stage 1 is {INCOMPLETE!r} until it is built")
    with h5py.File(path, "r") as file:
        metadata = json.loads(file.attrs["metadata"])
        fields = {name: file[name][:] for name in file}
    missing = [name for name in METADATA_FIELDS if name not in metadata]
    if missing:
        raise ValueError(f"{path} lacks provenance {missing}")
    for key, value in expect.items():
        if metadata.get(key) != value:
            raise ValueError(f"{path} has {key}={metadata.get(key)!r}, expected {value!r}")
    count = candidate_count(metadata["kind"])
    lengths = {name: len(fields[name]) for name in CANDIDATE_FIELDS if name in fields}
    if set(CANDIDATE_FIELDS) - set(fields) or set(lengths.values()) != {count}:
        raise ValueError(f"{path} is incomplete: expected {count} candidates in every field")
    return fields, metadata


def require_complete(output: Path) -> None:
    """Refuse to analyse unless every preregistered case has both caches, and nothing half-written."""
    missing = [
        str(case_path(output, kind, case).relative_to(output))
        for kind in ("rs", "cem")
        for case in range(NUM_CASES)
        if not case_path(output, kind, case).is_file()
    ]
    partial = sorted(str(path.relative_to(output)) for path in output.glob("*/*.partial"))
    if missing or partial:
        raise FileNotFoundError(
            f"Stage 1 is {INCOMPLETE!r}: {len(missing)} caches missing {missing[:4]}, "
            f"{len(partial)} half-written {partial[:4]}; analysis never builds caches"
        )


def compare_case_files(first: Path, second: Path) -> list[str]:
    """Names of the datasets and metadata keys that differ bitwise between two case caches."""
    with h5py.File(first, "r") as a, h5py.File(second, "r") as b:
        names = sorted(set(a) | set(b))
        differ = [
            name
            for name in names
            if name not in a
            or name not in b
            or a[name].dtype != b[name].dtype
            or a[name].shape != b[name].shape
            or a[name][()].tobytes() != b[name][()].tobytes()
        ]
        meta_a, meta_b = json.loads(a.attrs["metadata"]), json.loads(b.attrs["metadata"])
    differ += [
        f"metadata.{key}"
        for key in sorted(set(meta_a) | set(meta_b))
        if meta_a.get(key) != meta_b.get(key)
    ]
    return differ


# ---- per-case analysis ----------------------------------------------------------------------


def selected_indices(predicted: np.ndarray, prefixes: Sequence[int] = PREFIXES) -> np.ndarray:
    """The model's choice among the first N candidates, for each prefix N (first on ties)."""
    return np.array([int(np.argmin(predicted[:n])) for n in prefixes])


def prefix_curves(fields: dict, prefixes: Sequence[int] = PREFIXES) -> dict[str, np.ndarray]:
    """What the model selects at every prefix of the bank, and what that choice was worth."""
    predicted = np.asarray(fields["predicted_latent_cost"], dtype=np.float64)
    latent = np.asarray(fields["true_latent_cost"], dtype=np.float64)
    task = np.asarray(fields["true_task_cost"], dtype=np.float64)
    index = selected_indices(predicted, prefixes)
    oracle_latent = np.array([latent[:n].min() for n in prefixes])
    oracle_task = np.array([task[:n].min() for n in prefixes])
    curves = {
        "prefixes": np.array(prefixes),
        "selected_index": index,
        "predicted_latent_cost": predicted[index],
        "true_latent_cost": latent[index],
        "oracle_latent_cost": oracle_latent,
        "r_latent": latent[index] - oracle_latent,
        "selected_task_cost": task[index],
        "oracle_task_cost": oracle_task,
        "r_task": task[index] - oracle_task,
    }
    for name in (
        "signed_optimism",
        "absolute_optimism",
        "rollout_error",
        "terminal_error",
        "semantic_success",
        "action_norm",
        "q0_negative_log_density",
    ):
        curves[name] = np.asarray(fields[name], dtype=np.float64)[index]
    return curves


def ols_slope(
    curve: np.ndarray, prefixes: Sequence[int] = PREFIXES, at: Sequence[int] = PRIMARY
) -> np.ndarray:
    """Least-squares slope of curve [..., len(prefixes)] against log2 N over the prefixes in at."""
    columns = [list(prefixes).index(n) for n in at]
    y = np.asarray(curve, dtype=np.float64)[..., columns]
    x = np.log2(np.asarray(at, dtype=np.float64))
    x = x - x.mean()
    return (y - y.mean(axis=-1, keepdims=True)) @ x / (x @ x)


def exchangeable_null(
    fields: dict,
    case_index: int,
    *,
    draws: int = NULL_DRAWS,
    prefixes: Sequence[int] = PREFIXES,
    chunk: int = 250,
) -> dict:
    """The preregistered residual-permutation null of one case (section 8).

    Replicate b applies the b-th successive rng.permutation(K) of np.random.default_rng([20260923,
    case]) to the residuals o = C − Ĉ, sets Ĉ_b = C − o[π_b], and lets the same selection rule choose
    on every prefix. True costs, the residual marginal and the ladder are exactly the observed ones;
    only which candidate carries which residual is exchanged. Chunked, so memory stays at a few
    [chunk, K] arrays whatever the number of draws.
    """
    latent = np.asarray(fields["true_latent_cost"], dtype=np.float64)
    task = np.asarray(fields["true_task_cost"], dtype=np.float64)
    residual = latent - np.asarray(fields["predicted_latent_cost"], dtype=np.float64)
    oracle_latent = np.array([latent[:n].min() for n in prefixes])
    oracle_task = np.array([task[:n].min() for n in prefixes])
    rng = np.random.default_rng([NULL_SEED, case_index])
    r_latent = np.empty((draws, len(prefixes)))
    r_task = np.empty((draws, len(prefixes)))
    optimism = np.empty((draws, len(prefixes)))
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        permuted = residual[np.stack([rng.permutation(len(residual)) for _ in range(size)])]
        predicted = latent[None, :] - permuted
        rows = np.arange(size)
        for column, n in enumerate(prefixes):
            index = np.argmin(predicted[:, :n], axis=1)
            r_latent[start : start + size, column] = latent[index] - oracle_latent[column]
            r_task[start : start + size, column] = task[index] - oracle_task[column]
            optimism[start : start + size, column] = permuted[rows, index]
    beta = ols_slope(r_latent, prefixes)
    return {
        "beta": beta,
        "mean_beta": float(beta.mean()),
        "sd_beta": float(beta.std(ddof=1)),
        "beta_quantiles": dict(
            zip(
                ("2.5", "5", "50", "95", "97.5"),
                np.quantile(beta, [0.025, 0.05, 0.5, 0.95, 0.975]).tolist(),
                strict=True,
            )
        ),
        "mean_r_latent": r_latent.mean(axis=0),
        "mean_r_task": r_task.mean(axis=0),
        "mean_selected_optimism": optimism.mean(axis=0),
        # undefined, not zero, when the residuals do not vary (a perfect predictor)
        "corr_true_residual": float(np.corrcoef(latent, residual)[0, 1])
        if residual.std() > 0
        else float("nan"),
    }


def rank_conditioned(fields: dict) -> dict[str, dict]:
    """Section 12, S4: the bank split into disjoint predicted-rank bins, best predicted first."""
    predicted = np.asarray(fields["predicted_latent_cost"], dtype=np.float64)
    order = np.argsort(predicted, kind="stable")
    bins = {}
    for label, (low, high) in RANK_BINS.items():
        index = order[low:high]
        cell = {
            name: float(np.mean(np.asarray(fields[name], dtype=np.float64)[index]))
            for name in (
                "signed_optimism",
                "absolute_optimism",
                "true_latent_cost",
                "predicted_latent_cost",
                "true_task_cost",
                "rollout_error",
                "terminal_error",
                "semantic_success",
            )
        }
        cell["size"] = len(index)
        cell["rank_agreement_latent"] = pairwise_rank_agreement(
            predicted[index], np.asarray(fields["true_latent_cost"], dtype=np.float64)[index]
        )
        cell["rank_agreement_task"] = pairwise_rank_agreement(
            predicted[index], np.asarray(fields["true_task_cost"], dtype=np.float64)[index]
        )
        bins[label] = cell
    return bins


def cem_depth(fields: dict) -> dict[str, np.ndarray]:
    """Best candidate CEM has evaluated after each iteration, read from one trace (section 14)."""
    iteration = np.asarray(fields["iteration"])
    expected = np.repeat(np.arange(CEM_ITERATIONS), CEM_POPULATION)
    if not np.array_equal(iteration, expected):
        raise ValueError("a CEM trace must hold 256 candidates per iteration, in iteration order")
    return prefix_curves(fields, [CEM_POPULATION * depth for depth in range(1, CEM_ITERATIONS + 1)])


def compute_matched(rs: dict, cem: dict) -> dict[str, dict]:
    """CEM after I iterations against random shooting over the same 256·I model evaluations."""
    matched = {"cem": cem_depth(cem), "rs": prefix_curves(rs, MATCHED)}
    keys = ("predicted_latent_cost", "r_latent", "r_task", "signed_optimism", "selected_task_cost")
    return {
        side: {key: curves[key].tolist() for key in ("prefixes", "selected_index", *keys)}
        for side, curves in matched.items()
    }


def latent_geometry(
    terminal_latents: np.ndarray, reference: BaseReference, device: torch.device | str
) -> dict[str, list[float]]:
    """Nearest-base and Mahalanobis distance of executed terminal latents [K, D] (descriptive)."""
    latents = torch.from_numpy(np.asarray(terminal_latents, dtype=np.float32)).to(device)
    return {
        "nearest_base_distance": nearest_base_distance(latents, reference).cpu().tolist(),
        "mahalanobis_to_base": mahalanobis_to_base(latents, reference).cpu().tolist(),
    }


def analyze_case(
    rs: dict,
    cem: dict,
    metadata: dict,
    *,
    reference: BaseReference | None = None,
    device: torch.device | str = "cpu",
) -> tuple[dict, np.ndarray]:
    """Every per-case quantity the protocol reports, and the case's null slope distribution."""
    curves = prefix_curves(rs)
    column = {n: PREFIXES.index(n) for n in PREFIXES}
    null = exchangeable_null(rs, metadata["case_index"])
    beta_latent = float(ols_slope(curves["r_latent"]))
    order = np.argsort(np.asarray(rs["predicted_latent_cost"], dtype=np.float64), kind="stable")
    optimism = np.asarray(rs["signed_optimism"], dtype=np.float64)
    top, bottom = RANK_BINS["0-1%"], RANK_BINS["50-100%"]
    row = {
        "case_index": metadata["case_index"],
        "episode": metadata["episode"],
        "start_step": metadata["start_step"],
        "curves": {name: values.tolist() for name, values in curves.items()},
        "beta_latent": beta_latent,
        "beta_task": float(ols_slope(curves["r_task"])),
        "selected_task_slope": float(ols_slope(curves["selected_task_cost"])),
        "delta_latent": float(curves["r_latent"][column[4096]] - curves["r_latent"][column[256]]),
        "delta_task": float(curves["r_task"][column[4096]] - curves["r_task"][column[256]]),
        "excess_beta": beta_latent - null["mean_beta"],
        "null": {
            name: (value.tolist() if isinstance(value, np.ndarray) else value)
            for name, value in null.items()
            if name != "beta"
        },
        "sanity": {
            "beta_signed_optimism": float(ols_slope(curves["signed_optimism"])),
            "beta_absolute_optimism": float(ols_slope(curves["absolute_optimism"])),
            "top1_minus_bottom50_optimism": float(
                optimism[order[slice(*top)]].mean() - optimism[order[slice(*bottom)]].mean()
            ),
        },
        "rank_bins": rank_conditioned(rs),
        "compute_matched": compute_matched(rs, cem),
        "geometry": {
            "beta_action_norm": float(ols_slope(curves["action_norm"])),
            "beta_q0_negative_log_density": float(ols_slope(curves["q0_negative_log_density"])),
            "bank_sd_action_norm": float(np.std(rs["action_norm"], ddof=1)),
            "bank_sd_q0_negative_log_density": float(np.std(rs["q0_negative_log_density"], ddof=1)),
        },
    }
    if reference is not None:
        selected = np.asarray(rs["terminal_true_latent"])[curves["selected_index"]]
        distances = latent_geometry(selected, reference, device)
        row["geometry"] |= {name: values for name, values in distances.items()}
        row["geometry"] |= {
            f"beta_{name}": float(ols_slope(np.asarray(values)))
            for name, values in distances.items()
        }
    return row, null["beta"]


# ---- across cases ---------------------------------------------------------------------------


def evaluate_stage1_gates(summaries: dict[str, dict]) -> dict:
    """The frozen G1/G2/G3 rules and decision matrix, applied literally (sections 7-11)."""
    if any(summaries[rule["statistic"]]["n"] != NUM_CASES for rule in GATES.values()):
        return {"verdict": INCOMPLETE, "case": None, "gates": {}}
    gates = {}
    for gate, rule in GATES.items():
        summary = summaries[rule["statistic"]]
        checks = {
            "mean_positive": summary["mean"] > 0,
            "positive_count": summary["positive"] >= rule["positive"],
            "d_z": summary["d_z"] >= rule["d_z"],
        }
        if rule["p_below"] is not None:
            checks["mc_sign_flip_p"] = summary["mc_sign_flip_p"] < rule["p_below"]
        gates[gate] = {"pass": all(checks.values()), "checks": checks}
    if not gates["G1"]["pass"]:
        case, verdict = 4, "NO-GO"
    elif not gates["G2"]["pass"]:
        case, verdict = 3, "NO-GO"
    elif not gates["G3"]["pass"]:
        case, verdict = 2, "default NO-GO"
    else:
        case, verdict = 1, "STRONG GO"
    return {"verdict": verdict, "case": case, "gates": gates}


def classify_task_curve(selected_task_slope: float, task_regret: dict) -> str:
    """Descriptive reading A-D of section 10; it never changes a gate or the verdict.

    'Flat' is a presentation rule only: the preregistered 95% t interval of mean beta_task
    contains zero. No new threshold is introduced.
    """
    if task_regret["ci_low"] <= 0 <= task_regret["ci_high"]:
        return "D: task regret flat; latent degradation not clearly task-relevant"
    if task_regret["mean"] > 0:
        if selected_task_slope > 0:
            return "A: selected task cost and task regret both worsen (outright decision harm)"
        return "B: search useful, ranking efficiency degrades"
    if selected_task_slope <= 0:
        return "C: deeper search improves model-mediated decisions"
    return "other: selected task cost worsens while task regret falls"


def summarize_cases(rows: list[dict], signs: np.ndarray) -> dict:
    """Case-level statistics, the gates and the descriptive readings, from the 48 per-case rows."""
    if sorted(row["case_index"] for row in rows) != list(range(NUM_CASES)):
        return {"verdict": INCOMPLETE, "cases": len(rows)}
    rows = sorted(rows, key=lambda row: row["case_index"])
    vectors = {
        name: np.array([row[name] for row in rows])
        for name in (
            "beta_latent",
            "excess_beta",
            "beta_task",
            "delta_latent",
            "delta_task",
            "selected_task_slope",
        )
    }
    for name in ("beta_signed_optimism", "beta_absolute_optimism", "top1_minus_bottom50_optimism"):
        vectors[name] = np.array([row["sanity"][name] for row in rows])
    summaries = {name: case_summary(values, signs) for name, values in vectors.items()}
    decision = evaluate_stage1_gates(summaries)
    task_reading = classify_task_curve(
        summaries["selected_task_slope"]["mean"], summaries["beta_task"]
    )
    both_fall = summaries["beta_latent"]["mean"] < 0 and summaries["beta_task"]["mean"] < 0
    return {
        "verdict": decision["verdict"],
        "decision_case": decision["case"],
        "gates": decision["gates"],
        "task_reading": task_reading,
        "case_6_both_regrets_fall": both_fall,
        "case_5_search_still_useful": decision["case"] == 1
        and summaries["selected_task_slope"]["mean"] <= 0,
        "summaries": summaries,
        "case_vectors": {name: values.tolist() for name, values in vectors.items()},
        "mean_curves": {
            name: np.mean([row["curves"][name] for row in rows], axis=0).tolist()
            for name in rows[0]["curves"]
            if name != "selected_index"
        },
        "mean_null_curves": {
            name: np.mean([row["null"][name] for row in rows], axis=0).tolist()
            for name in ("mean_r_latent", "mean_r_task", "mean_selected_optimism")
        },
        "corr_true_residual": [row["null"]["corr_true_residual"] for row in rows],
    }
