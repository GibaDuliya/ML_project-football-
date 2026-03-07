"""
PlayerEncoder — the core RisingBALLER encoder.

Combines four embeddings per player token:
    PE  = players_embeddings(player_id)        — who this player is
    TPE = form_embeddings(form_stats)          — match statistics (temporal)
    SPE = positions_embeddings(position_id)    — field position (spatial)
    TE  = teams_embeddings(team_id)            — team affiliation (optional)

    token_repr = ReLU(PE + TPE + SPE [+ TE]) → Dropout

Then passes through N × PlayerTransformerBlock.

The encoder is the reusable backbone: it is shared between pre-training (MPP)
and all downstream tasks. Different tasks attach different heads on top.

Position encoding strategies (controlled via `position_enc_type`):
    "learned"    — trainable nn.Embedding, as in the paper (default).
    "sinusoidal" — fixed sin/cos encoding (BERT-style), not trainable.
"""

import math

import torch
import torch.nn as nn

from .attention import PlayerTransformerBlock


class SinusoidalEncoding(nn.Module):
    """Fixed sinusoidal position encoding for discrete integer IDs (BERT-style).

    Precomputes a lookup table of shape (max_id + 1, embed_size) at init time
    and registers it as a non-trainable buffer.  Row max_id is reserved as the
    padding vector (all zeros), matching the nn.Embedding padding_idx convention.

    Args:
        embed_size: dimensionality of output vectors.
        max_id: largest valid ID; pad index = max_id.
    """

    def __init__(self, embed_size: int, max_id: int):
        super().__init__()
        pe = torch.zeros(max_id + 1, embed_size)
        position = torch.arange(0, max_id, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size)
        )
        pe[:max_id, 0::2] = torch.sin(position * div_term)
        pe[:max_id, 1::2] = torch.cos(position * div_term[: embed_size // 2])
        # Row max_id stays zeros — padding vector
        self.register_buffer("pe", pe)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return sinusoidal vectors for the given integer IDs.

        Args:
            ids: integer tensor of shape (batch, seq_len).

        Returns:
            Float tensor of shape (batch, seq_len, embed_size).
        """
        return self.pe[ids]


class PlayerEncoder(nn.Module):
    """RisingBALLER encoder backbone.

    Args:
        embed_size: dimension D of all embeddings.
        num_layers: number of stacked transformer blocks.
        heads: number of attention heads per block.
        forward_expansion: FFN multiplier in transformer blocks.
        dropout: dropout rate.
        form_stats_size: dimensionality of raw stat vector (39 for MPP, 234 for NMSP).
        players_vocab_size: total unique players (excl. pad; pad_idx = this value).
        teams_vocab_size: total unique teams (excl. pad).
        positions_vocab_size: total unique positions (25 for StatsBomb).
        use_teams_embeddings: whether to include team affiliation embeddings.
        position_enc_type: encoding strategy for player field positions.
            "learned"    — trainable nn.Embedding (paper default).
            "sinusoidal" — fixed SinusoidalEncoding (BERT-style, not trainable).
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
        self.use_teams_embeddings = use_teams_embeddings
        self.position_enc_type = position_enc_type

        # PE — player identity (pad_idx = players_vocab_size)
        self.players_embeddings = nn.Embedding(
            players_vocab_size + 1, embed_size, padding_idx=players_vocab_size
        )

        # TPE — match statistics linear projection
        self.form_embeddings = nn.Linear(form_stats_size, embed_size)

        # SPE — field position encoding
        if position_enc_type == "learned":
            self.positions_embeddings = nn.Embedding(
                positions_vocab_size + 1, embed_size, padding_idx=positions_vocab_size
            )
        else:
            self.positions_embeddings = SinusoidalEncoding(embed_size, positions_vocab_size)

        # TE — team affiliation (optional)
        if use_teams_embeddings:
            self.teams_embeddings = nn.Embedding(
                teams_vocab_size + 1, embed_size, padding_idx=teams_vocab_size
            )

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                PlayerTransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        player_id: torch.Tensor,
        position_id: torch.Tensor,
        team_id: torch.Tensor,
        form_stats: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode a batch of matches.

        Args:
            player_id:      (batch, seq_len)             — player IDs (may include mask token).
            position_id:    (batch, seq_len)             — position IDs.
            team_id:        (batch, seq_len)             — team IDs.
            form_stats:     (batch, seq_len, stats_dim)  — per-player statistics.
            attention_mask: (batch, seq_len)             — 1=real, 0=pad.

        Returns:
            player_representations: (batch, seq_len, embed_size)
            attention_matrices: list of (batch, heads, seq_len, seq_len), one per layer.
        """
        token_repr = (
            self.players_embeddings(player_id)
            + self.form_embeddings(form_stats)
            + self.positions_embeddings(position_id)
        )

        if self.use_teams_embeddings:
            token_repr = token_repr + self.teams_embeddings(team_id)

        token_repr = self.dropout(torch.relu(token_repr))

        # Build transformer mask: (batch, seq_len) → (batch, 1, 1, seq_len)
        mask = attention_mask.unsqueeze(1).unsqueeze(2)

        attention_matrices = []
        x = token_repr
        for layer in self.layers:
            x, attn = layer(x, x, x, mask)
            attention_matrices.append(attn)

        return x, attention_matrices

    def get_player_embeddings_weight(self) -> torch.Tensor:
        """Return the player embedding lookup table (for analysis/extraction).

        Returns:
            Tensor of shape (players_vocab_size + 1, embed_size).
        """
        return self.players_embeddings.weight

    def get_position_embeddings_weight(self) -> torch.Tensor:
        """Return the position embedding table regardless of encoding strategy.

        For "learned":    returns the trainable nn.Embedding weight matrix.
        For "sinusoidal": returns the precomputed non-trainable buffer.

        Returns:
            Tensor of shape (positions_vocab_size + 1, embed_size).
        """
        if self.position_enc_type == "learned":
            return self.positions_embeddings.weight
        else:
            return self.positions_embeddings.pe
