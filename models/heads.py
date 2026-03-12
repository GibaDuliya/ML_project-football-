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

        input_dim = max_seq_length * embed_size
        output_dim = 2 * num_target_stats

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, embed_size)

        Returns:
            predictions: (batch, 2 * num_target_stats)
        """
        batch_size = encoder_output.shape[0]
        x = encoder_output.reshape(batch_size, -1)
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
    """Instantiate a head from a config dict."""
    head_type = head_config.get("type")
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head type: {head_type}")

    head_cls = HEAD_REGISTRY[head_type]

    config = {k: v for k, v in head_config.items() if k != "type"}
    config.update(kwargs)

    return head_cls(**config)
