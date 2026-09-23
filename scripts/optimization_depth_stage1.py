"""Stage 1 of the optimization-depth study: build the immutable per-case caches, then analyse them.

    uv run python scripts/optimization_depth_stage1.py build --dataset pusht_expert_train.h5 \
        --checkpoint weights.pt --manifest manifest.json \
        --protocol docs/protocols/optimization_depth_stage1.md --output runs/optimization_depth_stage1
    uv run python scripts/optimization_depth_stage1.py verify-determinism <the build arguments>
    uv run python scripts/optimization_depth_stage1.py analyze --output runs/optimization_depth_stage1 \
        --dataset pusht_expert_train.h5 --checkpoint weights.pt --latent-cache pusht_latents.h5
    uv run python scripts/optimization_depth_stage1.py summarize --output runs/optimization_depth_stage1

The study is docs/protocols/optimization_depth_stage1.md, frozen at commit 8b47d24. build is the
only command that runs the simulator or the model: it writes every case once and never overwrites
one, so rerunning it finishes an interrupted build. analyze and summarize only read, and analyze
fails rather than build a missing case. verify-determinism rebuilds case 0 in a scratch directory and
compares it with the stored cache bit for bit.
"""

import argparse
import json
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import torch

from planner_atlas import depth
from planner_atlas.atlas import ELITE_FRACTION, HORIZON
from planner_atlas.coverage import base_reference
from planner_atlas.data import ActionStats, action_stats
from planner_atlas.envs import make_env
from planner_atlas.evaluation import compare_with_model, encode_frame, frames_to_tensor
from planner_atlas.models.reference_lewm import ReferenceLeWM
from planner_atlas.planning import cem, model_action_bounds, random_shooting
from planner_atlas.pusht import (
    PushTCase,
    reconstruct_pusht_start,
    render_pusht_goal,
    rollout_pusht,
    sample_pusht_cases,
)
from planner_atlas.statistics import sign_matrix
from planner_atlas.training import (
    DYNAMICS_PROTOCOL,
    file_digest,
    load_latent_cache,
    split_episodes,
)

RS_SCORE_CHUNK = 1024  # random_shooting's own scoring batch, so cached costs are RS-4096's costs
ENCODE_CHUNK = 128  # executed plans encoded at once: 640 frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "verify-determinism"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--manifest", type=Path, required=True, help="the consumed manifest")
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        if name == "build":
            command.add_argument("--cases", type=int, nargs="*", help="a subset, to shard")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--dataset", type=Path, required=True)
    analyze.add_argument("--checkpoint", type=Path, required=True)
    analyze.add_argument("--latent-cache", type=Path, required=True, help="geometry reference")
    analyze.add_argument("--device", default="cuda")
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def git(*arguments: str) -> str:
    result = subprocess.run(["git", *arguments], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def write_once(path: Path, content: bytes) -> None:
    """Write a study record, or confirm an existing one is byte-identical."""
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"{path} exists with different content")
        return
    path.write_bytes(content)


def study_identity(args) -> tuple[dict, list[PushTCase]]:
    """Verify every frozen input and re-derive the 48 consumed cases."""
    if DYNAMICS_PROTOCOL != depth.STUDY_DYNAMICS_PROTOCOL:
        raise ValueError(f"Stage 1 was preregistered under {depth.STUDY_DYNAMICS_PROTOCOL}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("build from a clean tree: the recorded commit must be the code that ran")
    identity = depth.verify_protocol(args.protocol)
    if file_digest(args.checkpoint) != depth.CHECKPOINT_DIGEST:
        raise ValueError(f"{args.checkpoint} is not the base checkpoint")
    manifest = depth.load_frozen_manifest(args.manifest)
    with h5py.File(args.dataset, "r") as file:
        num_episodes = len(file["ep_len"])
    validation = split_episodes(num_episodes, validation_fraction=0.1, seed=0)
    cases = sample_pusht_cases(
        args.dataset, num_cases=depth.NUM_CASES, seed=depth.MANIFEST_SEED, episodes=validation
    )
    drawn = [
        {"episode": case.episode, "start_step": case.start_step, "goal_step": case.goal_step}
        for case in cases
    ]
    if drawn != manifest["cases"]:
        raise ValueError("the cases drawn with seed 307 do not match the consumed manifest")
    identity |= {
        "git_commit": git("rev-parse", "HEAD"),
        "environment": depth.ENVIRONMENT,
        "manifest_digest": depth.MANIFEST_DIGEST,
        "checkpoint_digest": depth.CHECKPOINT_DIGEST,
        "dynamics_protocol": DYNAMICS_PROTOCOL,
        "n_max": depth.N_MAX,
        "horizon": HORIZON,
        "objective_identity": depth.OBJECTIVE_IDENTITY,
        "sampler_identity": depth.SAMPLER_IDENTITY,
        "metric_definition_version": depth.METRIC_VERSION,
        "simulator_cache_identity": {
            "environment_id": "swm/PushT-v1",
            "stable_worldmodel": version("stable-worldmodel"),
            "execution": depth.SIMULATOR_IDENTITY,
        },
    }
    return identity, cases


@torch.no_grad()
def evaluate_plans(
    model, env, case: PushTCase, stats: ActionStats, latent, goal, plans, *, score_chunk, device
) -> dict[str, np.ndarray]:
    """Execute every plan once and cache what the model predicted against what happened.

    Plans are scored score_chunk at a time, the batch their planner scored them in, so the cached
    predicted costs are the planner's own; execution replays the case start for every plan.
    """
    parts = []
    for start in range(0, len(plans), score_chunk):
        chunk = plans[start : start + score_chunk]
        realized, outcomes = [], []
        for sub in chunk.split(ENCODE_CHUNK):
            frames, executed = zip(
                *(rollout_pusht(env, case, stats, plan) for plan in sub), strict=True
            )
            realized.append(model.encode(frames_to_tensor(np.stack(frames), device)))
            outcomes.extend(executed)
        realized = torch.cat(realized)
        evaluation = compare_with_model(model, latent, goal, chunk, realized, outcomes)
        parts.append(
            depth.candidate_fields(evaluation, realized[:, -1].cpu().numpy(), chunk.cpu().numpy())
        )
    fields = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    fields["candidate_index"] = np.arange(len(plans))
    return fields


def build_case(model, env, stats, bounds, case, index: int, identity: dict, output: Path, device):
    """Write the case's RS bank and CEM trace caches, whichever is not there yet."""
    goal = encode_frame(model, render_pusht_goal(env, case), device)
    reconstruct_pusht_start(env, case)
    latent = encode_frame(model, env.render(), device)
    common = identity | {
        "case_index": index,
        "episode": case.episode,
        "start_step": case.start_step,
        "bank_seed": depth.BANK_SEED + index,
        "action_bounds_identity": depth.bounds_identity(bounds),
    }

    def seeded() -> torch.Generator:
        return torch.Generator(bounds[0].device).manual_seed(depth.BANK_SEED + index)

    path = depth.case_path(output, "rs", index)
    if not path.exists():
        bank = depth.draw_bank(bounds, index)
        fields = evaluate_plans(
            model, env, case, stats, latent, goal, bank, score_chunk=RS_SCORE_CHUNK, device=device
        )
        shipped = random_shooting(
            model,
            latent,
            goal,
            horizon=HORIZON,
            num_samples=depth.N_MAX,
            bounds=bounds,
            generator=seeded(),
            keep_candidates=True,
        ).candidates
        if not torch.equal(shipped.plans, bank) or not np.array_equal(
            shipped.costs.cpu().numpy(), fields["predicted_latent_cost"]
        ):
            raise RuntimeError(f"case {index}: the bank is not random_shooting's own candidate set")
        planner = {"algorithm": "random_shooting", "num_samples": depth.N_MAX, "prefixes": True}
        depth.save_case(path, fields, common | {"kind": "rs", "planner_identity": planner})

    path = depth.case_path(output, "cem", index)
    if not path.exists():
        result = cem(
            model,
            latent,
            goal,
            horizon=HORIZON,
            num_samples=depth.CEM_POPULATION,
            iterations=depth.CEM_ITERATIONS,
            elite_fraction=ELITE_FRACTION,
            bounds=bounds,
            generator=seeded(),  # the bank's seed: CEM's first iteration is exactly RS-256
            keep_candidates=True,
        )
        trace = result.candidates
        fields = evaluate_plans(
            model,
            env,
            case,
            stats,
            latent,
            goal,
            trace.plans,
            score_chunk=depth.CEM_POPULATION,
            device=device,
        )
        if not np.array_equal(fields["predicted_latent_cost"], trace.costs.cpu().numpy()):
            raise RuntimeError(f"case {index}: cached CEM costs differ from the planner's own")
        if not torch.equal(trace.plans[int(trace.costs.argmin())], result.actions):
            raise RuntimeError(f"case {index}: the trace's best plan is not the CEM result")
        fields["iteration"] = trace.iteration.numpy()
        planner = {
            "algorithm": "cem",
            "population": depth.CEM_POPULATION,
            "iterations": depth.CEM_ITERATIONS,
            "elite_fraction": ELITE_FRACTION,
            "min_std": 0.05,
            "seed": depth.BANK_SEED + index,
        }
        extra = {
            "proposal_mean": np.stack([step.proposal.mean.numpy() for step in result.iterations]),
            "proposal_std": np.stack([step.proposal.std.numpy() for step in result.iterations]),
        }
        depth.save_case(path, fields, common | {"kind": "cem", "planner_identity": planner}, extra)


def build(args, output: Path, cases_to_build: list[int] | None, *, record: bool = True) -> None:
    identity, cases = study_identity(args)
    if record:
        output.mkdir(parents=True, exist_ok=True)
        write_once(output / "protocol.json", json.dumps(identity, indent=1).encode())
        write_once(output / "manifest.json", args.manifest.read_bytes())
    stats = action_stats(args.dataset)
    bounds = model_action_bounds(stats, device=args.device)
    model = ReferenceLeWM.from_checkpoint(args.checkpoint, device=args.device)
    env = make_env("pusht")
    for index in cases_to_build if cases_to_build is not None else range(depth.NUM_CASES):
        build_case(model, env, stats, bounds, cases[index], index, identity, output, args.device)
        print(f"case {index:2d} cached", flush=True)


def verify_determinism(args) -> None:
    stored = {
        kind: depth.case_path(args.output, kind, depth.DETERMINISM_CASE) for kind in ("rs", "cem")
    }
    if not all(path.is_file() for path in stored.values()):
        raise FileNotFoundError(f"case {depth.DETERMINISM_CASE} is not built yet")
    with tempfile.TemporaryDirectory(dir=args.output, prefix="determinism-") as scratch:
        build(args, Path(scratch), [depth.DETERMINISM_CASE], record=False)
        differ = {
            kind: depth.compare_case_files(
                path, depth.case_path(Path(scratch), kind, depth.DETERMINISM_CASE)
            )
            for kind, path in stored.items()
        }
    for kind, names in differ.items():
        print(
            f"{kind} case {depth.DETERMINISM_CASE}: "
            + ("bitwise identical" if not names else f"DIFFERS in {names}")
        )
    if any(differ.values()):
        raise SystemExit("determinism check failed: stop and report before weakening it")


def analyze(args) -> None:
    depth.require_complete(args.output)
    target = args.output / "analysis"
    for name in ("per_case.jsonl", "null_beta.npy"):
        if (target / name).exists():
            raise FileExistsError(f"{target / name} exists; analysis outputs are written once")
    reference = base_reference(
        load_latent_cache(args.latent_cache, args.dataset, args.checkpoint), device=args.device
    )
    rows, nulls = [], []
    for index in range(depth.NUM_CASES):
        rs, metadata = depth.load_case(
            depth.case_path(args.output, "rs", index), expect=depth.expected_metadata(index, "rs")
        )
        trace, _ = depth.load_case(
            depth.case_path(args.output, "cem", index), expect=depth.expected_metadata(index, "cem")
        )
        row, beta = depth.analyze_case(rs, trace, metadata, reference=reference, device=args.device)
        rows.append(row)
        nulls.append(beta)
        print(f"case {index:2d} analysed", flush=True)
    target.mkdir(exist_ok=True)
    with (target / "per_case.jsonl").open("x") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    np.save(target / "null_beta.npy", np.stack(nulls))


def summarize(args) -> None:
    source = args.output / "analysis" / "per_case.jsonl"
    rows = [json.loads(line) for line in source.open()]
    signs = sign_matrix(depth.SIGN_DRAWS, depth.NUM_CASES, seed=depth.SIGN_SEED)
    summary = depth.summarize_cases(rows, signs)
    summary["provenance"] = {
        "per_case": {"file": source.name, "sha256": file_digest(source)},
        "protocol_commit": depth.PROTOCOL_COMMIT,
        "protocol_digest": depth.PROTOCOL_DIGEST,
        "source_commit": git("rev-parse", "HEAD"),
    }
    target = args.output / "analysis" / "summary.json"
    with target.open("x") as out:
        out.write(json.dumps(summary, indent=1) + "\n")
    print(f"verdict: {summary['verdict']}")


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build(args, args.output, args.cases)
    elif args.command == "verify-determinism":
        verify_determinism(args)
    elif args.command == "analyze":
        analyze(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
