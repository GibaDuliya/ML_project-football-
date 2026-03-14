"""
gMLPEncoder — gMLP-based encoder for RisingBALLER.

Replaces the transformer encoder with a stack of gMLP blocks.
Designed for position-indexed sparse sequences (seq_len = 50 = 2 * 25 positions).

Each player occupies slot = team_idx * 25 + (position_id - 1).
Empty slots are masked (attention_mask = 0) and zeroed out after each block.

Position encoding (SPE) is NOT used here — the player's field position is
implicitly encoded by the slot index, and the static spatial projection
matrix W in SGU learns slot-to-slot interactions directly.

Embeddings used:
    PE  = players_embeddings(player_id)        — who this player is
    TPE = form_embeddings(form_stats)          — match statistics (temporal)
    TE  = teams_embeddings(team_id)            — team identity (optional)
"""

import torch
import torch.nn as nn

from .gmlp_block import gMLPBlock


# Number of StatsBomb positions (1-25)
NUM_POSITIONS = 25
# gMLP sequence length: 2 teams × 25 position slots
GMLP_SEQ_LEN = 2 * NUM_POSITIONS


class gMLPEncoder(nn.Module):
    """gMLP-based encoder backbone.

    Args:
        embed_size: dimension D of all embeddings.
        num_layers: number of stacked gMLP blocks.
        d_ffn: hidden FFN dimension in gMLP blocks.
            If None, defaults to 4 * embed_size.
        dropout: dropout rate.
        form_stats_size: dimensionality of raw stat vector (39 for MPP).
        players_vocab_size: total unique players (excl. pad; pad_idx = this value).
        teams_vocab_size: total unique teams (excl. pad).
        use_teams_embeddings: whether to include team affiliation embeddings.
        seq_len: fixed sequence length for spatial projection (default 50).
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
        seq_len: int = GMLP_SEQ_LEN,
    ):
        super().__init__()
        self.use_teams_embeddings = use_teams_embeddings
        self.seq_len = seq_len

        if d_ffn is None:
            d_ffn = 4 * embed_size

        # PE — player identity
        self.players_embeddings = nn.Embedding(
            players_vocab_size + 1, embed_size, padding_idx=players_vocab_size
        )

        # TPE — match statistics
        self.form_embeddings = nn.Linear(form_stats_size, embed_size)

        # TE — team identity (optional; encodes which specific team, not just 0/1)
        if use_teams_embeddings:
            self.teams_embeddings = nn.Embedding(
                teams_vocab_size + 1, embed_size, padding_idx=teams_vocab_size
            )

        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                gMLPBlock(embed_size, d_ffn, seq_len, dropout)
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
    ) -> tuple[torch.Tensor, list]:
        """Encode a batch of matches.

        Args:
            player_id:      (batch, seq_len)
            position_id:    (batch, seq_len)             — unused (position encoded by slot)
            team_id:        (batch, seq_len)
            form_stats:     (batch, seq_len, stats_dim)
            attention_mask: (batch, seq_len) — 1=real player, 0=empty slot

        Returns:
            player_representations: (batch, seq_len, embed_size)
            empty list (no attention matrices in gMLP)
        """
        token_repr = (
            self.players_embeddings(player_id)
            + self.form_embeddings(form_stats)
        )

        if self.use_teams_embeddings:
            token_repr = token_repr + self.teams_embeddings(team_id)

        token_repr = self.dropout(torch.relu(token_repr))

        # Mask: zero out empty position slots
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        x = token_repr * mask

        for layer in self.layers:
            x = layer(x) * mask

        return x, []

    def get_player_embeddings_weight(self) -> torch.Tensor:
        return self.players_embeddings.weight
