"""
MaskedPlayerModel — full model for MPP pre-training.

Combines PlayerEncoder + MPPHead + CrossEntropyLoss.
Returns HuggingFace-compatible MaskedLMOutput for use with HF Trainer.
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput

from .encoder import PlayerEncoder
from ..heads import MPPHead


class MaskedPlayerModel(nn.Module):
    """RisingBALLER pre-training model for Masked Player Prediction.

    Architecture:
        PlayerEncoder → MPPHead (Linear → logits over player vocab)
        Loss: CrossEntropyLoss(ignore_index=-100)

    The model outputs MaskedLMOutput so it integrates with HuggingFace Trainer
    (automatic loss handling, logging, checkpointing).

    Args:
        embed_size: embedding dimension.
        num_layers: number of transformer blocks.
        heads: number of attention heads.
        forward_expansion: FFN expansion factor.
        dropout: dropout rate.
        form_stats_size: input stat vector dimension (39).
        players_vocab_size: total unique players (decoder output size).
        teams_vocab_size: total unique teams.
        positions_vocab_size: total unique positions (25).
        use_teams_embeddings: include team affiliation embeddings.
    """

    def __init__(
        self,
        embed_size: int = 128,
        num_layers: int = 1,
        heads: int = 2,
        forward_expansion: int = 4,
        dropout: float = 0.05,
        form_stats_size: int = 39,
        players_vocab_size: int = 5107,
        teams_vocab_size: int = 141,
        positions_vocab_size: int = 25,
        use_teams_embeddings: bool = False,
        position_enc_type: str = "learned",
    ):
        super().__init__()
        self.encoder = PlayerEncoder(
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            form_stats_size=form_stats_size,
            players_vocab_size=players_vocab_size,
            teams_vocab_size=teams_vocab_size,
            positions_vocab_size=positions_vocab_size,
            use_teams_embeddings=use_teams_embeddings,
            position_enc_type=position_enc_type,
        )
        # MPP head: predict among real players only (labels 0 .. n_players-1). Encoder pad_idx=players_vocab_size → n_players = players_vocab_size - 1.
        num_player_classes = players_vocab_size - 1
        self.head = MPPHead(embed_size=embed_size, players_vocab_size=num_player_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        position_id: torch.Tensor,
        team_id: torch.Tensor,
        form_stats: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> MaskedLMOutput:
        """Forward pass for MPP.

        Args:
            input_ids:      (batch, seq_len)             — player IDs with [MASK] tokens.
            labels:         (batch, seq_len)             — target player IDs; -100 for non-masked.
            position_id:    (batch, seq_len)
            team_id:        (batch, seq_len)
            form_stats:     (batch, seq_len, stats_dim)
            attention_mask: (batch, seq_len)

        Returns:
            MaskedLMOutput with:
                loss:           scalar CrossEntropy loss (only over masked positions).
                logits:         (batch, seq_len, players_vocab_size)
                hidden_states:  (batch, seq_len, embed_size) — encoder output.
                attentions:     list of attention matrices per layer.

        Notes:
            - If tensors have an extra leading dim of 1 (from PreCollatedDataset),
              squeeze(0) is applied first.
        """
        if input_ids.dim() == 3:
            input_ids = input_ids.squeeze(0)
            labels = labels.squeeze(0)
            position_id = position_id.squeeze(0)
            team_id = team_id.squeeze(0)
            form_stats = form_stats.squeeze(0)
            attention_mask = attention_mask.squeeze(0)

        hidden_states, attentions = self.encoder(
            player_id=input_ids,
            position_id=position_id,
            team_id=team_id,
            form_stats=form_stats,
            attention_mask=attention_mask,
        )
        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def get_encoder(self) -> PlayerEncoder:
        """Return the encoder submodule (for weight transfer to downstream)."""
        return self.encoder
