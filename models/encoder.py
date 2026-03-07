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
        ...

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Return sinusoidal vectors for the given integer IDs.

        Args:
            ids: integer tensor of shape (batch, seq_len).

        Returns:
            Float tensor of shape (batch, seq_len, embed_size).
        """
        ...


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
        ...

    # ------------------------------------------------------------------
    # Per-component encoding helpers
    # ------------------------------------------------------------------

    def _encode_player(self, player_id: torch.Tensor) -> torch.Tensor:
        """Compute PE: learned player embedding.

        Args:
            player_id: (batch, seq_len) — player IDs.

        Returns:
            (batch, seq_len, embed_size)
        """
        ...

    def _encode_stats(self, form_stats: torch.Tensor) -> torch.Tensor:
        """Compute TPE: linear projection of match statistics.

        Args:
            form_stats: (batch, seq_len, form_stats_size)

        Returns:
            (batch, seq_len, embed_size)
        """
        ...

    def _encode_position(self, position_id: torch.Tensor) -> torch.Tensor:
        """Compute SPE using the strategy selected by `position_enc_type`.

        Dispatches to the learned embedding or sinusoidal encoding depending
        on self.position_enc_type set at construction time.

        Args:
            position_id: (batch, seq_len) — position IDs.

        Returns:
            (batch, seq_len, embed_size)
        """
        ...

    def _encode_team(self, team_id: torch.Tensor) -> torch.Tensor:
        """Compute TE: learned team embedding (only when use_teams_embeddings=True).

        Args:
            team_id: (batch, seq_len) — team IDs.

        Returns:
            (batch, seq_len, embed_size)
        """
        ...

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

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

        Steps:
            1. Compute PE via _encode_player, TPE via _encode_stats,
               SPE via _encode_position (strategy-aware), TE via _encode_team.
            2. Sum → ReLU → Dropout.
            3. Pass through each transformer block.
            4. Collect attention matrices from each layer.
        """
        ...

    def get_player_embeddings_weight(self) -> torch.Tensor:
        """Return the player embedding lookup table (for analysis/extraction).

        Returns:
            Tensor of shape (players_vocab_size + 1, embed_size).
        """
        ...

    def get_position_embeddings_weight(self) -> torch.Tensor:
        """Return the position embedding table regardless of encoding strategy.

        For "learned":    returns the trainable nn.Embedding weight matrix.
        For "sinusoidal": returns the precomputed non-trainable buffer.

        Returns:
            Tensor of shape (positions_vocab_size + 1, embed_size).
        """
        ...
