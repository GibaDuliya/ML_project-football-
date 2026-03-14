"""
gMLPMaskedPlayerModel — full model for MPP pre-training with gMLP encoder.

Combines gMLPEncoder + MPPHead + CrossEntropyLoss.
Returns HuggingFace-compatible MaskedLMOutput.
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput

from .encoder import gMLPEncoder
from ..heads import MPPHead


class gMLPMaskedPlayerModel(nn.Module):
    """gMLP pre-training model for Masked Player Prediction.

    Architecture:
        gMLPEncoder → MPPHead (Linear → logits over player vocab)
        Loss: CrossEntropyLoss(ignore_index=-100)

    Args:
        embed_size: embedding dimension.
        num_layers: number of gMLP blocks.
        d_ffn: hidden FFN dimension (None = 4 * embed_size).
        dropout: dropout rate.
        form_stats_size: input stat vector dimension (39).
        players_vocab_size: total unique players.
        teams_vocab_size: total unique teams.
        use_teams_embeddings: include team affiliation embeddings.
        seq_len: fixed sequence length for gMLP (default 50).
    """

    def __init__(
        self,
        embed_size: int = 128,
        num_layers: int = 2,
        d_ffn: int | None = None,
        dropout: float = 0.05,
        form_stats_size: int = 39,
        players_vocab_size: int = 5107,
        teams_vocab_size: int = 141,
        use_teams_embeddings: bool = False,
        seq_len: int = 50,
    ):
        super().__init__()
        self.encoder = gMLPEncoder(
            embed_size=embed_size,
            num_layers=num_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            form_stats_size=form_stats_size,
            players_vocab_size=players_vocab_size,
            teams_vocab_size=teams_vocab_size,
            use_teams_embeddings=use_teams_embeddings,
            seq_len=seq_len,
        )
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
            input_ids:      (batch, seq_len)
            labels:         (batch, seq_len) — target player IDs; -100 for non-masked.
            position_id:    (batch, seq_len)
            team_id:        (batch, seq_len)
            form_stats:     (batch, seq_len, stats_dim)
            attention_mask: (batch, seq_len)

        Returns:
            MaskedLMOutput with loss, logits, hidden_states, attentions.
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

    def get_encoder(self) -> gMLPEncoder:
        return self.encoder
