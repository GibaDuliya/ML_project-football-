"""
PyTorch Dataset classes for RisingBALLER.

MatchDatasetMPP   — one item = one match; used for Masked Player Prediction.
MatchDatasetNMSP  — one item = one match; used for Next Match Stats Prediction.
PreCollatedDataset — wrapper that stores pre-batched tensors (for HF Trainer trick).
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


class MatchDatasetNMSP(Dataset):
    """Dataset for Next Match Statistics Prediction.

    Each item corresponds to a single match and returns:
        input tensors  — same as MatchDatasetMPP (but with aggregated TPE)
        target_stats   — FloatTensor (2 * num_target_stats,) — targets for both teams

    For NMSP, the temporal positional encoding (form_stats) uses aggregated
    variables instead of raw per-match counts.
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
        self.df = df.reset_index(drop=True)
        self.target_columns = target_columns
        self.player_name2id = player_name2id
        self.team_name2id = team_name2id
        self.max_seq_length = max_seq_length
        self.player_pad_token_id = player_pad_token_id
        self.team_pad_token_id = team_pad_token_id
        self.position_pad_token_id = position_pad_token_id
        self.id_columns = id_columns if id_columns is not None else REQUIRED_ID_COLUMNS

        self.stat_columns = [
            c for c in self.df.columns
            if c not in self.id_columns and c not in self.target_columns
        ]
        self.form_stats_size = len(self.stat_columns)
        self._match_ids = self.df["match_id"].drop_duplicates().tolist()

    def __len__(self) -> int:
        return len(self._match_ids)

    def __getitem__(self, idx: int) -> Optional[dict[str, torch.Tensor]]:
        match_id = self._match_ids[idx]
        rows = self.df[self.df["match_id"] == match_id]

        teams = rows["team_name"].unique()
        if len(teams) != 2:
            return None

        team_a, team_b = sorted(teams)
        rows_a = rows[rows["team_name"] == team_a]
        rows_b = rows[rows["team_name"] == team_b]
        rows_ordered = pd.concat([rows_a, rows_b], ignore_index=True)

        n_players = len(rows_ordered)

        player_ids = []
        team_ids = []
        position_ids = []

        for _, row in rows_ordered.iterrows():
            player_ids.append(
                self.player_name2id.get(row["player_name"], self.player_pad_token_id)
            )
            team_ids.append(
                self.team_name2id.get(row["team_name"], self.team_pad_token_id)
            )
            pos_id = int(row["position_id"])
            position_ids.append(
                pos_id - 1 if 1 <= pos_id <= 25 else self.position_pad_token_id
            )

        form_stats = rows_ordered[self.stat_columns].values.astype(np.float32)

        team_a_target = rows_a[self.target_columns].sum().values.astype(np.float32)
        team_b_target = rows_b[self.target_columns].sum().values.astype(np.float32)
        target_stats = np.concatenate([team_a_target, team_b_target], axis=0)

        player_ids = pad_sequence_1d(
            player_ids, self.max_seq_length, pad_value=self.player_pad_token_id
        )
        team_ids = pad_sequence_1d(
            team_ids, self.max_seq_length, pad_value=self.team_pad_token_id
        )
        position_ids = pad_sequence_1d(
            position_ids, self.max_seq_length, pad_value=self.position_pad_token_id
        )
        form_stats = pad_sequence_2d(
            form_stats, self.max_seq_length, pad_value=0.0
        )
        attention_mask = build_attention_mask(n_players, self.max_seq_length)

        return {
            "input_ids": torch.from_numpy(player_ids).long(),
            "position_id": torch.from_numpy(position_ids).long(),
            "team_id": torch.from_numpy(team_ids).long(),
            "form_stats": torch.from_numpy(form_stats).float(),
            "attention_mask": torch.from_numpy(attention_mask).long(),
            "target_stats": torch.from_numpy(target_stats).float(),
        }


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
