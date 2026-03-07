"""
Data utilities: padding, encoding helpers, stat aggregation for NMSP.
"""

import numpy as np
import pandas as pd
import torch


def pad_sequence_1d(
    seq: list | np.ndarray,
    max_length: int,
    pad_value: int = 0,
) -> np.ndarray:
    """Pad or truncate a 1-D sequence to max_length.

    Args:
        seq: input sequence of ints/floats.
        max_length: desired length.
        pad_value: value used for padding.

    Returns:
        np.ndarray of shape (max_length,).
    """
    arr = np.asarray(seq)
    if len(arr) >= max_length:
        return arr[:max_length].copy()
    out = np.full(max_length, pad_value, dtype=arr.dtype)
    out[: len(arr)] = arr
    return out


def pad_sequence_2d(
    seq: np.ndarray,
    max_length: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Pad or truncate a 2-D array along axis 0 to max_length.

    Args:
        seq: input array of shape (n, d).
        max_length: desired length along axis 0.
        pad_value: fill value for padding rows.

    Returns:
        np.ndarray of shape (max_length, d).
    """
    arr = np.asarray(seq)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    n, d = arr.shape
    if n >= max_length:
        return arr[:max_length].astype(np.float32)
    out = np.full((max_length, d), pad_value, dtype=np.float32)
    out[:n] = arr.astype(np.float32)
    return out


def build_attention_mask(real_length: int, max_length: int) -> np.ndarray:
    """Create attention mask: 1 for first `real_length` positions, 0 for rest.

    Returns:
        np.ndarray of shape (max_length,), dtype int.
    """
    mask = np.zeros(max_length, dtype=np.int64)
    mask[: min(real_length, max_length)] = 1
    return mask


def aggregate_player_stats_for_nmsp(
    df: pd.DataFrame,
    stat_columns: list[str],
    match_id_col: str = "match_id",
    player_name_col: str = "player_name",
    team_name_col: str = "team_name",
    season_col: str = "season_name",
    window_size: int = 5,
) -> pd.DataFrame:
    """Compute NMSP temporal positional encodings.

    For each player at each match, compute:
        - season-to-date: sum, mean, std of each stat
        - last `window_size` matches: sum, mean, std of each stat
    Total features = 39 stats × 3 aggregations × 2 windows = 234

    Args:
        df: raw DataFrame sorted by kickoff date within each player.
        stat_columns: list of 39 stat column names.
        match_id_col, player_name_col, team_name_col, season_col: column names.
        window_size: rolling window size (default 5).

    Returns:
        DataFrame with same index as df but stat columns replaced by 234 aggregated features.
    """
    ...


def custom_collate_fn(batch: list) -> dict:
    """Filter None items from batch and apply default_collate.

    Used when MatchDataset may return None for corrupt matches.
    """
    ...
