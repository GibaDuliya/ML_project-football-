"""
MLP Baseline for Masked Player Prediction (MPP).

Replaces the Transformer encoder with a stack of per-token MLP blocks.
Each token is processed independently — there is NO inter-token interaction
(no attention). This serves as a baseline to measure the contribution of
self-attention in the main RisingBALLER model.

Architecture:
    Same embeddings as PlayerEncoder:
        PE  = players_embeddings(player_id)
        TPE = form_embeddings(form_stats)
        SPE = positions_embeddings(position_id)
        TE  = teams_embeddings(team_id)  (optional)
        token_repr = ReLU(PE + TPE + SPE [+ TE]) → Dropout

    Then N × MLPBlock (per-token, no cross-token interaction):
        x → Linear(D, D*expansion) → ReLU → Dropout → Linear(D*expansion, D) → Add&Norm

    Then MPPHead on top for classification.

To match the parameter count of the Transformer model, adjust num_layers
and forward_expansion. The Transformer block with embed_size=128, heads=2,
forward_expansion=4 has ~161K params. Each MLP block with forward_expansion=4
has ~132K params. So 1 MLP block with expansion=5 (~165K) or 1 block with
expansion=4 + a small extra layer are reasonable matches.
"""

import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput

from models.encoder import SinusoidalEncoding
from models.heads import MPPHead


class MLPBlock(nn.Module):
    """Single MLP block with residual connection and LayerNorm.

    Architecture: x → Linear → ReLU → Dropout → Linear → Dropout → Add(x) → LayerNorm
    """

    def __init__(self, embed_size: int, forward_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_size * forward_expansion
        self.net = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_size),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_size)
        Returns:
            (batch, seq_len, embed_size)
        """
        return self.norm(x + self.net(x))


class MLPEncoder(nn.Module):
    """MLP-based encoder baseline (no attention, per-token processing).

    Has the same embedding structure as PlayerEncoder and the same
    forward() signature, so it's a drop-in replacement.

    Args:
        embed_size: dimension D of all embeddings.
        num_layers: number of MLP blocks.
        forward_expansion: hidden dim = embed_size * forward_expansion.
        dropout: dropout rate.
        form_stats_size: input stat vector dimension (39).
        players_vocab_size: total unique players (excl. pad; pad_idx = this value).
        teams_vocab_size: total unique teams (excl. pad).
        positions_vocab_size: total unique positions (25).
        use_teams_embeddings: whether to include team affiliation embeddings.
        position_enc_type: "learned" or "sinusoidal".
    """

    def __init__(
        self,
        embed_size: int = 128,
        num_layers: int = 1,
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

        # MLP blocks (per-token, no attention)
        self.layers = nn.ModuleList([
            MLPBlock(embed_size, forward_expansion, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        player_id: torch.Tensor,
        position_id: torch.Tensor,
        team_id: torch.Tensor,
        form_stats: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode a batch of matches (per-token MLP, no cross-token interaction).

        Args:
            player_id:      (batch, seq_len)
            position_id:    (batch, seq_len)
            team_id:        (batch, seq_len)
            form_stats:     (batch, seq_len, stats_dim)
            attention_mask: (batch, seq_len) — 1=real, 0=pad.

        Returns:
            player_representations: (batch, seq_len, embed_size)
            attention_matrices: empty list (no attention in MLP baseline)
        """
        token_repr = (
            self.players_embeddings(player_id)
            + self.form_embeddings(form_stats)
            + self.positions_embeddings(position_id)
        )

        if self.use_teams_embeddings:
            token_repr = token_repr + self.teams_embeddings(team_id)

        token_repr = self.dropout(torch.relu(token_repr))

        # Zero out padding positions so they don't affect norms
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        x = token_repr * mask

        for layer in self.layers:
            x = layer(x) * mask

        return x, []

    def get_player_embeddings_weight(self) -> torch.Tensor:
        return self.players_embeddings.weight

    def get_position_embeddings_weight(self) -> torch.Tensor:
        if self.position_enc_type == "learned":
            return self.positions_embeddings.weight
        else:
            return self.positions_embeddings.pe


class MLPMaskedPlayerModel(nn.Module):
    """MLP baseline model for MPP pre-training.

    Architecture:
        MLPEncoder → MPPHead (Linear → logits over player vocab)
        Loss: CrossEntropyLoss(ignore_index=-100)

    Same interface as MaskedPlayerModel — drop-in replacement for training.
    """

    def __init__(
        self,
        embed_size: int = 128,
        num_layers: int = 1,
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
        self.encoder = MLPEncoder(
            embed_size=embed_size,
            num_layers=num_layers,
            forward_expansion=forward_expansion,
            dropout=dropout,
            form_stats_size=form_stats_size,
            players_vocab_size=players_vocab_size,
            teams_vocab_size=teams_vocab_size,
            positions_vocab_size=positions_vocab_size,
            use_teams_embeddings=use_teams_embeddings,
            position_enc_type=position_enc_type,
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
        """Forward pass for MPP (same signature as MaskedPlayerModel)."""
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

    def get_encoder(self) -> MLPEncoder:
        return self.encoder
