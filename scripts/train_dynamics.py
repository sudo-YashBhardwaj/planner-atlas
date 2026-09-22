"""Train one bootstrap member of the LeWM latent dynamics, with the representation frozen.

    uv run python scripts/train_dynamics.py --dataset pusht_expert_train.h5 \
        --checkpoint weights.pt --latent-cache pusht_latents.h5 \
        --output member0.pt --member 0 --seed 0 --epochs 2

The encoder and projector stay as released; only the dynamics train (see planner_atlas.training).
Frames are encoded once into the latent cache, which is built on the first run and reused after,
and every member draws its own bootstrap of the training windows. Members are independent runs:
train them one at a time, here or on different machines, then load them as an ensemble.
"""

import argparse
import json
from pathlib import Path

import torch

from planner_atlas.data import action_stats
from planner_atlas.envs import make_env
from planner_atlas.evaluation import tworoom_frames
from planner_atlas.pusht import replay_frames
from planner_atlas.training import (
    TrainableDynamics,
    bootstrap_indices,
    build_latent_cache,
    cache_identity,
    file_digest,
    latent_windows,
    load_latent_cache,
    save_dynamics,
    split_episodes,
    subsample_windows,
    train_dynamics,
    validation_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", choices=("pusht", "tworoom"), required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="official tworoom/pusht h5")
    parser.add_argument("--checkpoint", type=Path, required=True, help="released LeWM weights.pt")
    parser.add_argument("--latent-cache", type=Path, required=True, help="built if missing")
    parser.add_argument("--output", type=Path, required=True, help="trained dynamics of a member")
    parser.add_argument("--member", type=int, default=0, help="ensemble member index")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap and shuffling seed")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=0, help="episode split, shared by all")
    parser.add_argument("--train-windows", type=int, default=0, help="0 uses every window")
    parser.add_argument("--validation-windows", type=int, default=20_000)
    parser.add_argument("--cache-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.latent_cache.exists():
        print(f"building latent cache {args.latent_cache} from live renders")
        source = replay_frames if args.env == "pusht" else tworoom_frames
        build_latent_cache(
            args.dataset,
            args.checkpoint,
            args.latent_cache,
            source(make_env(args.env), args.dataset),
            device=args.device,
            batch_size=args.cache_batch_size,
        )
    cache = load_latent_cache(args.latent_cache, args.dataset, args.checkpoint)
    stats = action_stats(args.dataset)

    validation_episodes = split_episodes(
        len(cache.lengths), validation_fraction=args.validation_fraction, seed=args.split_seed
    )
    train = latent_windows(cache, args.dataset, stats, episodes=~validation_episodes)
    validation = latent_windows(cache, args.dataset, stats, episodes=validation_episodes)
    train = subsample_windows(train, args.train_windows, seed=args.split_seed)
    validation = subsample_windows(validation, args.validation_windows, seed=args.split_seed)
    print(f"windows: {len(train):,} train, {len(validation):,} validation")

    model = TrainableDynamics.from_checkpoint(args.checkpoint, device=args.device)
    initial = validation_loss(model, validation, batch_size=args.batch_size)
    indices = bootstrap_indices(len(train), member=args.member, seed=args.seed)
    print(f"member {args.member}: initial validation loss {initial:.6f}")

    history = train_dynamics(
        model,
        train,
        indices=indices,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed + args.member,
        validation=validation,
    )

    metadata = {
        "member": args.member,
        "seed": args.seed,
        "base_checkpoint": file_digest(args.checkpoint),
        "cache_identity": cache_identity(args.dataset, args.checkpoint),
        "train_windows": len(train),
        "validation_windows": len(validation),
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "initial_validation_loss": initial,
        "history": history,
    }
    save_dynamics(args.output, model, metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "cache_identity"}, indent=1))
    if torch.cuda.is_available():
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
