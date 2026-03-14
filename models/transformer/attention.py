"""
Transformer building blocks for RisingBALLER.

PlayerSelfAttention   — multi-head self-attention (manual implementation).
PlayerTransformerBlock — one transformer layer: Attn → Add&Norm → FFN → Add&Norm.
"""

import torch
import torch.nn as nn


class PlayerSelfAttention(nn.Module):
    """Multi-head self-attention (как в risingBALLER: split → per-head Linear).

    Сначала split по головам, затем Q/K/V — Linear(head_dim, head_dim, bias=False).
    """

    def __init__(self, embed_size: int, heads: int):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "embed_size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, embed_size)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        queries: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = queries.shape[0]
        value_len = values.shape[1]
        key_len = keys.shape[1]
        query_len = queries.shape[1]

        # Split into heads first (как в risingBALLER)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        energy = energy / (self.head_dim ** 0.5)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out, attention


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
        self.attention = PlayerSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

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
        attn_out, attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attn_out + query))
        ff_out = self.feed_forward(x)
        out = self.dropout(self.norm2(ff_out + x))
        return out, attention
