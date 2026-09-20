"""Inference with the released LeWM architecture and weights.

Reproduces the model behind the official quentinll/lewm-tworooms and quentinll/lewm-pusht
checkpoints, built from stable-worldmodel 0.1.1 modules and a Hugging Face ViT: the encoder's
CLS token is projected to a latent, and a causal predictor advances latents by action blocks.
"""

from pathlib import Path

import torch
from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor
from torch import nn
from transformers import ViTConfig, ViTModel

IMAGE_SIZE = 224
LATENT_DIM = 192
ACTION_DIM = 10  # one action block: 5 consecutive normalized 2-d environment actions
CONTEXT_FRAMES = 3  # latents the predictor attends to

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class ReferenceLeWM(nn.Module):
    """The released LeWM, for inference.

    Observations are float32 RGB frames in [0, 1] shaped [B, T, 3, 224, 224]; ImageNet
    normalization happens here. Rollouts and costs work on latents, so a planner encodes the
    observation and goal once rather than per candidate batch. Action blocks are shaped
    [B, S, T, 10] for S candidates of T blocks each, already in the normalized action space.
    """

    def __init__(self) -> None:
        super().__init__()
        vit = ViTConfig(
            hidden_size=LATENT_DIM,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
            image_size=IMAGE_SIZE,
            patch_size=14,
        )
        self.encoder = ViTModel(vit, add_pooling_layer=False, use_mask_token=False)
        self.projector = MLP(LATENT_DIM, 2048, LATENT_DIM, norm_fn=nn.BatchNorm1d)
        self.action_encoder = Embedder(input_dim=ACTION_DIM, emb_dim=LATENT_DIM)
        self.predictor = Predictor(
            num_frames=CONTEXT_FRAMES,
            depth=6,
            heads=16,
            dim_head=64,
            mlp_dim=2048,
            input_dim=LATENT_DIM,
            hidden_dim=LATENT_DIM,
            output_dim=LATENT_DIM,
            dropout=0.1,
        )
        self.pred_proj = MLP(LATENT_DIM, 2048, LATENT_DIM, norm_fn=nn.BatchNorm1d)

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: torch.device | str) -> "ReferenceLeWM":
        """Strictly load released weights (a weights.pt state dict), in eval mode on device."""
        model = cls()
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()

    @torch.no_grad()
    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Latents [B, T, 192] of observations [B, T, 3, 224, 224]."""
        pixels = _normalize(observations).flatten(0, 1)
        cls_tokens = self.encoder(pixels).last_hidden_state[:, 0]
        return self.projector(cls_tokens).unflatten(0, observations.shape[:2])

    def predict(self, latents: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        """Next latents [B, T, 192] predicted at every position of a causal window.

        Position t predicts the latent reached by applying its action block, attending to the
        positions up to t. The rollout iterates this one step at a time; training supervises all
        positions of a window at once, which is the step the released model learned.
        """
        predicted = self.predictor(latents, action_embeddings)
        return self.pred_proj(predicted.flatten(0, 1)).unflatten(0, predicted.shape[:2])

    @torch.no_grad()
    def rollout(self, latent: torch.Tensor, action_blocks: torch.Tensor) -> torch.Tensor:
        """Latents [z_0, z_1, ..., z_T] shaped [B, S, T + 1, 192], where z_0 = latent [B, 192].

        Each later latent is predicted from the latest (at most 3) latents and the action blocks
        applied at them.
        """
        if (
            action_blocks.ndim != 4
            or action_blocks.shape[-1] != ACTION_DIM
            or latent.shape != (action_blocks.shape[0], LATENT_DIM)
        ):
            raise ValueError(
                f"expected latent [B, 192] and action blocks [B, S, T, 10], "
                f"got {list(latent.shape)} and {list(action_blocks.shape)}"
            )
        b, s, t, _ = action_blocks.shape
        latents = [latent[:, None].expand(b, s, -1).flatten(0, 1)]
        actions = self.action_encoder(action_blocks.flatten(0, 1))
        for step in range(t):
            start = max(0, step + 1 - CONTEXT_FRAMES)
            window = torch.stack(latents[start:], dim=1)
            # Like the reference, predict the whole window, then keep the newest latent.
            latents.append(self.predict(window, actions[:, start : step + 1])[:, -1])
        return torch.stack(latents, dim=1).unflatten(0, (b, s))

    @torch.no_grad()
    def cost(
        self, latent: torch.Tensor, goal: torch.Tensor, action_blocks: torch.Tensor
    ) -> torch.Tensor:
        """Terminal goal cost [B, S] of each candidate, from latent and goal latent [B, 192]."""
        return terminal_cost(self.rollout(latent, action_blocks)[:, :, -1], goal)


def terminal_cost(predicted: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Squared error summed over latent dims: predicted [B, S, D], goal [B, D] -> [B, S]."""
    if predicted.ndim != 3 or goal.shape != (predicted.shape[0], predicted.shape[2]):
        raise ValueError(
            f"expected predictions [B, S, D] and goals [B, D], "
            f"got {list(predicted.shape)} and {list(goal.shape)}"
        )
    return (predicted - goal[:, None]).square().sum(dim=-1)


def _normalize(observations: torch.Tensor) -> torch.Tensor:
    if observations.dtype != torch.float32 or observations.shape[2:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"expected float32 observations [B, T, 3, 224, 224], "
            f"got {observations.dtype} {list(observations.shape)}"
        )
    if observations.min() < 0 or observations.max() > 1:
        raise ValueError("expected observation values in [0, 1]")
    device = observations.device
    return (observations - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)
