"""
Transformer building blocks for RisingBALLER.

PlayerSelfAttention   — multi-head self-attention (manual implementation).
PlayerTransformerBlock — one transformer layer: Attn → Add&Norm → FFN → Add&Norm.
"""

import torch
import torch.nn as nn


class PlayerSelfAttention(nn.Module):
    """Multi-head self-attention implemented from scratch.

    Args:
        embed_size: total embedding dimension D.
        heads: number of attention heads.

    Shapes:
        Input:  values, keys, queries — each (batch, seq_len, embed_size)
                mask — (batch, 1, 1, seq_len) or None
        Output: (batch, seq_len, embed_size), attention_matrix (batch, heads, seq_len, seq_len)
    """

    def __init__(self, embed_size: int, heads: int):
        super().__init__()
        ...

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        queries: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head scaled dot-product attention.

        Steps:
            1. Linear projections Q, K, V per head.
            2. energy = Q @ K^T / sqrt(head_dim).
            3. Apply mask (fill -1e20 where mask==0).
            4. attention = softmax(energy).
            5. out = attention @ V → reshape → fc_out.

        Returns:
            (output, attention_matrix)
        """
        ...


class PlayerTransformerBlock(nn.Module):
    """Single transformer encoder block.

    Architecture:
        x → MultiHeadAttention → Add & LayerNorm → FFN → Add & LayerNorm

    Args:
        embed_size: embedding dimension.
        heads: number of attention heads.
        dropout: dropout rate.
        forward_expansion: FFN hidden dim = embed_size * forward_expansion.
    """

    def __init__(
        self,
        embed_size: int,
        heads: int,
        dropout: float = 0.1,
        forward_expansion: int = 4,
    ):
        super().__init__()
        ...

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through one transformer block.

        Returns:
            (output, attention_matrix)
        """
        ...
