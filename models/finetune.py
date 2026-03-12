"""
DownstreamModel — encoder + task-specific head for fine-tuning.

Loads pre-trained encoder weights and attaches a new head.
The encoder can be frozen or fine-tuned jointly.
"""

import torch
import torch.nn as nn
from typing import Optional

from .encoder import PlayerEncoder
from .heads import build_head ###


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
        self.encoder = PlayerEncoder(**encoder_config)
        self.head_config = dict(head_config)

        self.head = build_head(
            self.head_config,
            embed_size=encoder_config["embed_size"],
        )

        if freeze_encoder:
            self.freeze_encoder()

    def load_pretrained_encoder(
        self,
        checkpoint_path: str,
        strict: bool = False,
    ) -> None:
        """Load encoder weights from a pre-trained MaskedPlayerModel checkpoint.

        Handles the key prefix mismatch: checkpoint has "encoder.xxx" keys,
        we load into self.encoder which expects "xxx" keys.

        Also handles the case where form_embeddings input dimension changed
        (e.g. 39 → 234 for NMSP): incompatible form_embeddings weights are skipped.

        Args:
            checkpoint_path: path to .pt/.bin checkpoint.
            strict: if False, ignore missing/unexpected keys.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint format is not supported")

        encoder_state = {}
        current_state = self.encoder.state_dict()

        for key, value in checkpoint.items():
            if not key.startswith("encoder."):
                continue

            new_key = key[len("encoder.") :]

            # Skip weights that do not exist in current encoder
            if new_key not in current_state:
                continue

            # Skip incompatible shapes (important for form_embeddings 39 -> 234)
            if current_state[new_key].shape != value.shape:
                continue

            encoder_state[new_key] = value

        self.encoder.load_state_dict(encoder_state, strict=strict)

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

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
        hidden_states, attentions = self.encoder(
            player_id=input_ids,
            position_id=position_id,
            team_id=team_id,
            form_stats=form_stats,
            attention_mask=attention_mask,
        )

        head_type = self.head_config.get("type")

        if head_type == "nmsp":
            predictions = self.head(hidden_states)
            loss = None
            if target_stats is not None:
                loss_fct = nn.MSELoss()
                loss = loss_fct(predictions, target_stats)

            return {
                "predictions": predictions,
                "loss": loss,
                "hidden_states": hidden_states,
                "attentions": attentions,
            }

        elif head_type == "classification":
            logits = self.head(hidden_states, attention_mask=attention_mask)
            loss = None

            if labels is not None:
                if logits.dim() == 3:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(
                        logits.reshape(-1, logits.size(-1)),
                        labels.reshape(-1),
                    )
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits, labels)

            return {
                "logits": logits,
                "loss": loss,
                "hidden_states": hidden_states,
                "attentions": attentions,
            }

        elif head_type == "regression":
            predictions = self.head(hidden_states, attention_mask=attention_mask)
            loss = None

            if target_stats is not None:
                loss_fct = nn.MSELoss()
                loss = loss_fct(predictions, target_stats)

            return {
                "predictions": predictions,
                "loss": loss,
                "hidden_states": hidden_states,
                "attentions": attentions,
            }

        else:
            raise ValueError(f"Unsupported head type: {head_type}")
