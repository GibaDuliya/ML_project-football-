"""
PyTorch Dataset classes for RisingBALLER.

MatchDatasetMPP      — one item = one match; sequential layout for transformer encoder.
MatchDatasetGMLP_MPP — one item = one match; position-indexed sparse layout for gMLP encoder.
MatchDatasetNMSP     — one item = one match; used for Next Match Stats Prediction.
PreCollatedDataset   — wrapper that stores pre-batched tensors (for HF Trainer trick).
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import REQUIRED_ID_COLUMNS
from .utils import build_attention_mask, pad_sequence_1d, pad_sequence_2d


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
        self.df = df.reset_index(drop=True)
        self.player_name2id = player_name2id
        self.team_name2id = team_name2id
        self.max_seq_length = max_seq_length
        self.player_pad_token_id = player_pad_token_id
        self.team_pad_token_id = team_pad_token_id
        self.position_pad_token_id = position_pad_token_id
        self.id_columns = id_columns if id_columns is not None else REQUIRED_ID_COLUMNS
        self.stat_columns = [c for c in self.df.columns if c not in self.id_columns]
        self.form_stats_size = len(self.stat_columns)
        # Unique match_id in order of first appearance
        self._match_ids = self.df["match_id"].drop_duplicates().tolist()

    def __len__(self) -> int:
        """Return number of unique matches."""
        return len(self._match_ids)

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
        match_id = self._match_ids[idx]
        rows = self.df[self.df["match_id"] == match_id]

        teams = rows["team_name"].unique()
        if len(teams) != 2:
            return None

        # Sort teams (stable order), then concatenate players: team0, then team1
        team_a, team_b = sorted(teams)
        rows_a = rows[rows["team_name"] == team_a]
        rows_b = rows[rows["team_name"] == team_b]
        rows_ordered = pd.concat([rows_a, rows_b], ignore_index=True)

        n_players = len(rows_ordered)
        player_ids = []
        team_ids = []
        position_ids = []
        for _, row in rows_ordered.iterrows():
            player_ids.append(self.player_name2id.get(row["player_name"], self.player_pad_token_id))
            team_ids.append(self.team_name2id.get(row["team_name"], self.team_pad_token_id))
            # StatsBomb position_id 1-25 → 0-24; pad = position_pad_token_id (25)
            pos_id = int(row["position_id"])
            position_ids.append(pos_id - 1 if 1 <= pos_id <= 25 else self.position_pad_token_id)

        form_stats = rows_ordered[self.stat_columns].values.astype(np.float32)

        # Pad or truncate to max_seq_length
        player_ids = pad_sequence_1d(
            player_ids, self.max_seq_length, pad_value=self.player_pad_token_id
        )
        team_ids = pad_sequence_1d(team_ids, self.max_seq_length, pad_value=self.team_pad_token_id)
        position_ids = pad_sequence_1d(
            position_ids, self.max_seq_length, pad_value=self.position_pad_token_id
        )
        form_stats = pad_sequence_2d(form_stats, self.max_seq_length, pad_value=0.0)
        attention_mask = build_attention_mask(n_players, self.max_seq_length)

        return {
            "input_ids": torch.from_numpy(player_ids).long(),
            "position_id": torch.from_numpy(position_ids).long(),
            "team_id": torch.from_numpy(team_ids).long(),
            "form_stats": torch.from_numpy(form_stats),
            "attention_mask": torch.from_numpy(attention_mask).long(),
        }


# Number of StatsBomb positions
_NUM_POSITIONS = 25
# gMLP sequence length: 2 teams × 25 position slots
_GMLP_SEQ_LEN = 2 * _NUM_POSITIONS


class MatchDatasetGMLP_MPP(Dataset):
    """Dataset for Masked Player Prediction with position-indexed sparse layout.

    Designed for gMLP encoder where each player is placed at a fixed slot
    determined by their team and position:
        slot = team_idx * 25 + (position_id - 1)

    This gives a fixed sequence of length 50 (2 × 25 positions) where
    only ~22 slots are occupied per match and the rest are masked.

    Each item returns a dict of tensors with the same keys as MatchDatasetMPP,
    so DataCollatorMPP works without modification.

    Args:
        df: preprocessed DataFrame (one row per player-match).
        player_name2id: mapping player_name → int id.
        team_name2id: mapping team_name → int id.
        player_pad_token_id: id used for padding player positions.
        team_pad_token_id: id used for padding team positions.
        position_pad_token_id: id used for padding position positions (25).
        id_columns: list of metadata column names to drop for stats.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        player_name2id: dict,
        team_name2id: dict,
        player_pad_token_id: int = 0,
        team_pad_token_id: int = 0,
        position_pad_token_id: int = 25,
        id_columns: Optional[list[str]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.player_name2id = player_name2id
        self.team_name2id = team_name2id
        self.max_seq_length = _GMLP_SEQ_LEN
        self.player_pad_token_id = player_pad_token_id
        self.team_pad_token_id = team_pad_token_id
        self.position_pad_token_id = position_pad_token_id
        self.id_columns = id_columns if id_columns is not None else REQUIRED_ID_COLUMNS
        self.stat_columns = [c for c in self.df.columns if c not in self.id_columns]
        self.form_stats_size = len(self.stat_columns)
        self._match_ids = self.df["match_id"].drop_duplicates().tolist()

    def __len__(self) -> int:
        return len(self._match_ids)

    def __getitem__(self, idx: int) -> Optional[dict[str, torch.Tensor]]:
        """Build position-indexed sparse tensor dict for match at index `idx`.

        Each player is placed at slot = team_idx * 25 + (position_id - 1).
        Empty slots have pad values and attention_mask = 0.

        Returns None if the match has != 2 teams.
        """
        match_id = self._match_ids[idx]
        rows = self.df[self.df["match_id"] == match_id]

        teams = rows["team_name"].unique()
        if len(teams) != 2:
            return None

        S = self.max_seq_length  # 50

        # Initialize with pad values
        player_ids = np.full(S, self.player_pad_token_id, dtype=np.int64)
        team_ids = np.full(S, self.team_pad_token_id, dtype=np.int64)
        position_ids = np.full(S, self.position_pad_token_id, dtype=np.int64)
        form_stats = np.zeros((S, self.form_stats_size), dtype=np.float32)
        attention_mask = np.zeros(S, dtype=np.int64)

        # Assign team indices: team_a = 0, team_b = 1 (alphabetical)
        team_a, team_b = sorted(teams)
        team_to_idx = {team_a: 0, team_b: 1}

        for _, row in rows.iterrows():
            pos_id = int(row["position_id"])
            if not (1 <= pos_id <= _NUM_POSITIONS):
                continue

            team_idx = team_to_idx[row["team_name"]]
            slot = team_idx * _NUM_POSITIONS + (pos_id - 1)

            player_ids[slot] = self.player_name2id.get(
                row["player_name"], self.player_pad_token_id
            )
            team_ids[slot] = self.team_name2id.get(
                row["team_name"], self.team_pad_token_id
            )
            position_ids[slot] = pos_id - 1  # 0-indexed
            form_stats[slot] = row[self.stat_columns].values.astype(np.float32)
            attention_mask[slot] = 1

        return {
            "input_ids": torch.from_numpy(player_ids).long(),
            "position_id": torch.from_numpy(position_ids).long(),
            "team_id": torch.from_numpy(team_ids).long(),
            "form_stats": torch.from_numpy(form_stats),
            "attention_mask": torch.from_numpy(attention_mask).long(),
        }


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
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.batches[idx]
