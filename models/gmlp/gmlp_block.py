"""
gMLP building blocks for RisingBALLER.

SpatialGatingUnit — cross-token interaction via static spatial projection + gating.
gMLPBlock          — one gMLP layer: Norm → Channel Up → GELU → SGU → Channel Down → Residual.

Reference: Liu et al., "Pay Attention to MLPs" (2021), arXiv:2105.08050
"""

import torch
import torch.nn as nn


class SpatialGatingUnit(nn.Module):
    """Spatial Gating Unit (SGU) from gMLP.

    Splits input along channel dim into two halves (Z1, Z2).
    Applies a static linear spatial projection W to Z2, then gates: Z1 * f(Z2).

    The spatial projection W ∈ R^{seq_len × seq_len} is initialized near zero
    with bias = 1, so that f(Z2) ≈ 1 at the start of training (identity gate).

    Args:
        d_ffn: channel dimension of input (will be split in half).
        seq_len: fixed sequence length (determines spatial projection size).
    """

    def __init__(self, d_ffn: int, seq_len: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn // 2)
        self.spatial_proj = nn.Linear(seq_len, seq_len)
        # Critical initialization: W ≈ 0, b = 1 → gate starts as identity
        nn.init.uniform_(self.spatial_proj.weight, -1e-3, 1e-3)
        nn.init.ones_(self.spatial_proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, seq_len, d_ffn)

        Returns:
            (batch, seq_len, d_ffn // 2)
        """
        z1, z2 = z.chunk(2, dim=-1)           # each (batch, seq_len, d_ffn // 2)
        z2 = self.norm(z2)
        z2 = z2.transpose(-1, -2)             # (batch, d_ffn // 2, seq_len)
        z2 = self.spatial_proj(z2)            # W @ z2 + b
        z2 = z2.transpose(-1, -2)             # (batch, seq_len, d_ffn // 2)
        return z1 * z2


class gMLPBlock(nn.Module):
    """Single gMLP block.

    Architecture:
        x → LayerNorm → Linear(D → d_ffn) → GELU → SGU → Linear(d_ffn//2 → D) → + residual

    Args:
        d_model: embedding dimension (input and output).
        d_ffn: hidden dimension after channel expansion.
        seq_len: fixed sequence length for spatial projection.
        dropout: dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        seq_len: int,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj_up = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.channel_proj_down = nn.Linear(d_ffn // 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        shortcut = x
        x = self.norm(x)
        x = self.channel_proj_up(x)
        x = self.activation(x)
        x = self.sgu(x)
        x = self.channel_proj_down(x)
        x = self.dropout(x)
        return x + shortcut
