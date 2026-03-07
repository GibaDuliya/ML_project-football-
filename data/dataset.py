"""
PyTorch Dataset classes for RisingBALLER.

MatchDatasetMPP   — one item = one match; used for Masked Player Prediction.
MatchDatasetNMSP  — one item = one match; used for Next Match Stats Prediction.
PreCollatedDataset — wrapper that stores pre-batched tensors (for HF Trainer trick).
"""

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class MatchDatasetMPP(Dataset):
    """Dataset for Masked Player Prediction.

    Each item corresponds to a single match and returns a dict of tensors:
        input_ids      : LongTensor  (max_seq_length,)     — player IDs (mask applied later by collator)
        position_id    : LongTensor  (max_seq_length,)
        team_id        : LongTensor  (max_seq_length,)
        form_stats     : FloatTensor (max_seq_length, form_stats_size)
        attention_mask : LongTensor  (max_seq_length,)      — 1 for real players, 0 for padding

    Masking is NOT applied here — it's done in DataCollatorMPP so that
    different masks are generated each epoch (data augmentation).

    Args:
        df: preprocessed DataFrame (one row per player-match).
        player_name2id: mapping player_name → int id.
        team_name2id: mapping team_name → int id.
        max_seq_length: pad/truncate to this length (default 36).
        player_pad_token_id: id used for padding player positions.
        team_pad_token_id: id used for padding team positions.
        position_pad_token_id: id used for padding position positions.
        id_columns: list of metadata column names to drop for stats.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        player_name2id: dict,
        team_name2id: dict,
        max_seq_length: int = 36,
        player_pad_token_id: int = 0,
        team_pad_token_id: int = 0,
        position_pad_token_id: int = 25,
        id_columns: Optional[list[str]] = None,
    ):
        ...

    def __len__(self) -> int:
        """Return number of unique matches."""
        ...

    def __getitem__(self, idx: int) -> Optional[dict[str, torch.Tensor]]:
        """Build padded tensor dict for match at index `idx`.

        Steps:
            1. Filter rows for match_id at this index.
            2. Sort players: team 0 first, then team 1.
            3. Encode player_name → id, pad to max_seq_length.
            4. Encode team_name → id, position_id, pad.
            5. Extract form_stats (drop id_columns), pad.
            6. Build attention_mask.
            7. Return dict of tensors.

        Returns None if the match has != 2 teams (corrupt data).
        """
        ...


class MatchDatasetNMSP(Dataset):
    """Dataset for Next Match Statistics Prediction.

    Each item corresponds to a single match and returns:
        input tensors  — same as MatchDatasetMPP (but with aggregated TPE)
        target_stats   — FloatTensor (2 * num_target_stats,) — targets for both teams

    For NMSP, the temporal positional encoding (form_stats) uses 234
    aggregated variables instead of 39 raw counts.

    Args:
        df: preprocessed DataFrame with aggregated stats.
        target_columns: list of stat column names to predict.
        (... same args as MatchDatasetMPP ...)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_columns: list[str],
        player_name2id: dict,
        team_name2id: dict,
        max_seq_length: int = 36,
        player_pad_token_id: int = 0,
        team_pad_token_id: int = 0,
        position_pad_token_id: int = 25,
        id_columns: Optional[list[str]] = None,
    ):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Optional[dict[str, torch.Tensor]]:
        """Return input tensors + target_stats for match idx."""
        ...


class PreCollatedDataset(Dataset):
    """Thin wrapper: stores a list of pre-collated batches as individual items.

    Used with the HF Trainer batch_size=1 trick: each "item" is already a
    full batch (e.g. 256 matches), and the model does squeeze(0) internally.

    Args:
        batches: list of dicts, each dict maps str → Tensor.
    """

    def __init__(self, batches: list[dict[str, torch.Tensor]]):
        ...

    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ...
