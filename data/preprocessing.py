"""
Preprocessing pipeline: raw CSV → clean DataFrame + vocabulary mappings.

The raw CSV has one row per (player, match) with columns:
    player_name, team_name, competition_name, season_name, match_id,
    match_date, is_aligned, position_id, position_name, <39 stat columns>

Preprocessing steps:
    1. Load and validate raw CSV
    2. Build vocabulary mappings (player_name ↔ id, team_name ↔ id)
    3. Assign special token IDs (mask, pad)
    4. Optionally filter by season/competition
    5. Save processed data + metadata pickles
"""

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd


# Expected schema (configs/data.yaml id_columns + match_date + 39 stats)
REQUIRED_ID_COLUMNS = [
    "player_name",
    "team_name",
    "competition_name",
    "season_name",
    "match_id",
    "match_date",
    "is_aligned",
    "position_id",
    "position_name",
]
FORM_STATS_SIZE = 39
PROCESSED_CSV_NAME = "processed.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_raw_csv(
    csv_path: str,
    output_dir: str,
    seasons: Optional[list[str]] = None,
    competitions: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load raw CSV, validate columns, optionally filter, and save processed df.

    Args:
        csv_path: path to the raw CSV file.
        output_dir: directory to write processed CSV and metadata pickles.
        seasons: if provided, keep only these season_name values.
        competitions: if provided, keep only these competition_name values.

    Returns:
        Cleaned DataFrame (same schema, potentially filtered).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    df = pd.read_csv(path)
    _validate_columns(df) 

    # Optional filters
    if seasons is not None:
        df = df[df["season_name"].isin(seasons)].copy()
    if competitions is not None:
        df = df[df["competition_name"].isin(competitions)].copy()

    # Stat columns: fill NaN with 0 (paper: non-participating players have 0 stats)
    stat_columns = [c for c in df.columns if c not in REQUIRED_ID_COLUMNS]
    df[stat_columns] = df[stat_columns].fillna(0)

    # Ensure position_id is integer (StatsBomb 1–25)
    df["position_id"] = df["position_id"].astype(int)

    # Parse match_date (required for NMSP sorting by kickoff date)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_csv = out_path / PROCESSED_CSV_NAME
    df.to_csv(output_csv, index=False)

    return df


def build_vocab_mappings(
    df: pd.DataFrame,
    output_dir: str,
) -> dict:
    """Build and persist name↔id mappings for players and teams.

    Creates four pickle files in output_dir/metadata/:
        player_name2id.pickle, id2player_name.pickle,
        team_name2id.pickle,   id2team_name.pickle

    Also determines special token IDs:
        player_mask_token_id = len(unique_players)
        player_pad_token_id  = len(unique_players) + 1
        team_pad_token_id    = len(unique_teams)

    Args:
        df: processed DataFrame (output of preprocess_raw_csv).
        output_dir: directory containing metadata/ subfolder.

    Returns:
        Dict with keys: player_name2id, id2player_name, team_name2id,
        id2team_name, player_mask_token_id, player_pad_token_id,
        team_pad_token_id, players_vocab_size, teams_vocab_size.
    """
    unique_players = df["player_name"].dropna().unique()
    unique_teams = df["team_name"].dropna().unique()

    player_name2id = {name: i for i, name in enumerate(sorted(unique_players))}
    id2player_name = {i: name for name, i in player_name2id.items()}

    team_name2id = {name: i for i, name in enumerate(sorted(unique_teams))}
    id2team_name = {i: name for name, i in team_name2id.items()}

    n_players = len(player_name2id)
    n_teams = len(team_name2id)
    player_mask_token_id = n_players
    player_pad_token_id = n_players + 1
    team_pad_token_id = n_teams
    players_vocab_size = n_players + 2  # players + mask + pad
    teams_vocab_size = n_teams + 1     # teams + pad

    metadata_dir = Path(output_dir) / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    _save_pickle(player_name2id, str(metadata_dir / "player_name2id.pickle"))
    _save_pickle(id2player_name, str(metadata_dir / "id2player_name.pickle"))
    _save_pickle(team_name2id, str(metadata_dir / "team_name2id.pickle"))
    _save_pickle(id2team_name, str(metadata_dir / "id2team_name.pickle"))

    return {
        "player_name2id": player_name2id,
        "id2player_name": id2player_name,
        "team_name2id": team_name2id,
        "id2team_name": id2team_name,
        "player_mask_token_id": player_mask_token_id,
        "player_pad_token_id": player_pad_token_id,
        "team_pad_token_id": team_pad_token_id,
        "players_vocab_size": players_vocab_size,
        "teams_vocab_size": teams_vocab_size,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame) -> None:
    """Check that all expected columns are present in the DataFrame."""
    missing = [c for c in REQUIRED_ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    stat_cols = [c for c in df.columns if c not in REQUIRED_ID_COLUMNS]
    if len(stat_cols) != FORM_STATS_SIZE:
        raise ValueError(
            f"Expected {FORM_STATS_SIZE} stat columns, got {len(stat_cols)}. "
            f"Stat columns: {stat_cols[:5]}... ({len(stat_cols)} total)"
        )


def _save_pickle(obj, path: str) -> None:
    """Serialize object to pickle file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: str):
    """Deserialize object from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)
