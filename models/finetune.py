"""
DownstreamModel — encoder + task-specific head for fine-tuning.

Loads pre-trained encoder weights and attaches a new head.
The encoder can be frozen or fine-tuned jointly.
"""

import torch
import torch.nn as nn
from typing import Optional

from .transformer.encoder import PlayerEncoder
from .heads import build_head


class DownstreamModel(nn.Module):
    """Generic downstream model: pre-trained encoder + task head.

    Usage:
        1. Construct with same encoder config as pre-training.
        2. Call load_pretrained_encoder() to load weights from MPP checkpoint.
        3. Optionally freeze the encoder.
        4. Train only the head (or both).

    Args:
        encoder_config: dict with PlayerEncoder constructor kwargs.
        head_config: dict with head type and params (passed to build_head).
        freeze_encoder: if True, set encoder params to requires_grad=False.
    """

    def __init__(
        self,
        encoder_config: dict,
        head_config: dict,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        ...

    def load_pretrained_encoder(
        self,
        checkpoint_path: str,
        strict: bool = False,
    ) -> None:
        """Load encoder weights from a pre-trained MaskedPlayerModel checkpoint.

        Handles the key prefix mismatch: checkpoint has "encoder.xxx" keys,
        we load into self.encoder which expects "xxx" keys.

        Also handles the case where form_embeddings input dimension changed
        (e.g. 39 → 234 for NMSP): re-initializes that layer.

        Args:
            checkpoint_path: path to .safetensors or .pt file.
            strict: if False, ignore missing/unexpected keys.
        """
        ...

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        ...

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        position_id: torch.Tensor,
        team_id: torch.Tensor,
        form_stats: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        target_stats: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: encoder → head → loss (if targets provided).

        Args:
            input_ids, position_id, team_id, form_stats, attention_mask:
                standard encoder inputs.
            labels: target for classification tasks (optional).
            target_stats: target for regression/NMSP tasks (optional).

        Returns:
            Dict with keys:
                "logits" or "predictions": head output.
                "loss": computed loss (if targets provided).
                "hidden_states": encoder output.
                "attentions": list of attention matrices.
        """
        ...
