"""
Task-specific heads that attach on top of PlayerEncoder.

Each head takes encoder output (batch, seq_len, embed_size) and produces
task-specific predictions.

MPPHead             — Masked Player Prediction: per-token classification over player vocab.
NMSPHead            — Next Match Stats Prediction: flatten all tokens → MLP → 2*N_stats.
ClassificationHead  — Generic per-token or per-sequence classification.
RegressionHead      — Generic regression head.

build_head()        — factory function: config dict → Head instance.
"""

from typing import Optional

import torch
import torch.nn as nn


class MPPHead(nn.Module):
    """Head for Masked Player Prediction.

    Projects each token embedding to logits over the player vocabulary.
    Loss uses only masked positions (labels != -100); logits over real players only
    (indices 0 .. players_vocab_size - 1), not mask/pad.

    Args:
        embed_size: encoder embedding dimension.
        players_vocab_size: number of classes (unique players, excl. mask and pad).
    """

    def __init__(self, embed_size: int, players_vocab_size: int):
        super().__init__()
        self.projection = nn.Linear(embed_size, players_vocab_size)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, embed_size)

        Returns:
            logits: (batch, seq_len, players_vocab_size)
        """
        return self.projection(encoder_output)


class StatsPredictionHead(nn.Module):
    """Head for predicting 39 player statistics from encoder embeddings.

    Input: embeddings after the transformer block (batch, seq_len, embed_size).
    Output: per-token regression to 39 stats (batch, seq_len, num_stats).
    Loss: MSE (or L1) only on non-padded positions (attention_mask).

    Args:
        embed_size: encoder embedding dimension.
        num_stats: number of stat targets (default 39).
        hidden_dim: optional hidden size for MLP; if 0 or None, use single Linear.
    """

    def __init__(
        self,
        embed_size: int,
        num_stats: int = 39,
        hidden_dim: Optional[int] = 256,
    ):
        super().__init__()
        self.num_stats = num_stats
        if hidden_dim and hidden_dim > 0:
            self.mlp = nn.Sequential(
                nn.Linear(embed_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_stats),
            )
        else:
            self.mlp = nn.Linear(embed_size, num_stats)

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, embed_size)

        Returns:
            predictions: (batch, seq_len, num_stats)
        """
        return self.mlp(encoder_output)


class NMSPHead(nn.Module):
    """Head for Next Match Statistics Prediction.

    Flattens all player representations and projects to team-level stats.
    Output: 2 * num_target_stats (predictions for both teams).

    Args:
        embed_size: encoder embedding dimension.
        max_seq_length: sequence length (for flattening).
        num_target_stats: number of stats to predict per team.
        hidden_dim: MLP hidden layer size.
    """

    def __init__(
        self,
        embed_size: int,
        max_seq_length: int = 36,
        num_target_stats: int = 18,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.num_target_stats = num_target_stats
        flat_size = max_seq_length * embed_size
        self.mlp = nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2 * num_target_stats),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, embed_size)

        Returns:
            predictions: (batch, 2 * num_target_stats)
        """
        b, s, e = encoder_output.shape
        x = encoder_output.reshape(b, s * e)
        return self.mlp(x)


class ClassificationHead(nn.Module):
    """Generic classification head.

    Supports two modes:
        - "per_token":   predict a class for each token → (batch, seq_len, num_classes)
        - "per_sequence": pool tokens then classify    → (batch, num_classes)

    Args:
        embed_size: input embedding dimension.
        num_classes: number of output classes.
        hidden_dim: MLP hidden size.
        pool: "per_token" or "per_sequence" (mean-pool over non-padded tokens).
    """

    def __init__(
        self,
        embed_size: int,
        num_classes: int,
        hidden_dim: int = 128,
        pool: str = "per_token",
    ):
        super().__init__()
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, embed_size)
            attention_mask: (batch, seq_len) — needed for per_sequence pooling.

        Returns:
            logits: (batch, [seq_len,] num_classes)
        """
        if self.pool == "per_sequence":
            if attention_mask is None:
                x = encoder_output.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).float()
                x = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return self.mlp(x)
        return self.mlp(encoder_output)


class RegressionHead(nn.Module):
    """Generic regression head.

    Supports per_token or per_sequence regression.
    For rating prediction: output_dim=1, pool="per_sequence" (one scalar per sample).

    Args:
        embed_size: input dimension.
        output_dim: number of regression targets (e.g. 1 for overall rating).
        hidden_dim: MLP hidden size.
        pool: "per_token" or "per_sequence".
    """

    def __init__(
        self,
        embed_size: int,
        output_dim: int = 1,
        hidden_dim: int = 128,
        pool: str = "per_sequence",
    ):
        super().__init__()
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, embed_size)
            attention_mask: (batch, seq_len) — for per_sequence: mask padding (0) before mean.

        Returns:
            (batch, output_dim) if per_sequence else (batch, seq_len, output_dim)
        """
        if self.pool == "per_sequence":
            if attention_mask is None:
                x = encoder_output.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).float()
                x = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return self.mlp(x)
        else:
            return self.mlp(encoder_output)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

HEAD_REGISTRY = {
    "mpp": MPPHead,
    "nmsp": NMSPHead,
    "stats_prediction": StatsPredictionHead,
    "classification": ClassificationHead,
    "regression": RegressionHead,
}


def build_head(head_config: dict, **kwargs) -> nn.Module:
    """Instantiate a head from a config dict.

    Args:
        head_config: dict with at least {"type": "<name>"} plus head-specific params.
        **kwargs: additional keyword args forwarded to the head constructor
                  (e.g. embed_size, players_vocab_size).

    Returns:
        nn.Module — the constructed head.

    Raises:
        ValueError: if head_config["type"] not in HEAD_REGISTRY.
    """
    config = dict(head_config)
    head_type = config.pop("type", None)
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type: {head_type}. Known: {list(HEAD_REGISTRY)}")
    return HEAD_REGISTRY[head_type](**{**config, **kwargs})
