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

import torch
import torch.nn as nn


class MPPHead(nn.Module):
    """Head for Masked Player Prediction.

    Projects each token embedding to logits over the player vocabulary.

    Args:
        embed_size: encoder embedding dimension.
        players_vocab_size: number of classes (unique players, excl. special tokens).
    """

    def __init__(self, embed_size: int, players_vocab_size: int):
        super().__init__()
        ...

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, embed_size)

        Returns:
            logits: (batch, seq_len, players_vocab_size)
        """
        ...


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
        ...

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, embed_size)

        Returns:
            predictions: (batch, 2 * num_target_stats)
        """
        ...


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
        ...

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
        ...


class RegressionHead(nn.Module):
    """Generic regression head.

    Supports per_token or per_sequence regression.

    Args:
        embed_size: input dimension.
        output_dim: number of regression targets.
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
        ...

    def forward(
        self,
        encoder_output: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

HEAD_REGISTRY = {
    "mpp": MPPHead,
    "nmsp": NMSPHead,
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
    ...
