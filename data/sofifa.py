"""
Обработка данных SoFIFA для предсказания рейтинга (overall).

- load_sofifa_csv()     — загрузка и базовая очистка CSV.
- normalize_player_name() — нормализация имени для сопоставления с нашим словарём.
- build_rating_splits() — сопоставление с player_name2id, разбиение на train/val.
- SofifaRatingDataset  — PyTorch Dataset (player_id, overall) для обучения головы.
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Колонки SoFIFA (dataset/sofifa_players.csv)
SOFIFA_COLUMNS = ["name", "url", "positions", "country", "age", "overall", "potential", "value", "wage", "total_stats"]
RATING_COLUMN = "overall"
NAME_COLUMN = "name"


def load_sofifa_csv(csv_path: str | Path) -> pd.DataFrame:
    """Загружает CSV SoFIFA, оставляет нужные колонки, отбрасывает строки без рейтинга.

    Args:
        csv_path: путь к dataset/sofifa_players.csv.

    Returns:
        DataFrame с колонками name, overall (и опционально age, positions, ...).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"SoFIFA CSV not found: {path}")

    df = pd.read_csv(path)
    if NAME_COLUMN not in df.columns or RATING_COLUMN not in df.columns:
        raise ValueError(f"Expected columns {NAME_COLUMN}, {RATING_COLUMN}. Got: {list(df.columns)}")

    df = df.dropna(subset=[NAME_COLUMN, RATING_COLUMN])
    df[RATING_COLUMN] = pd.to_numeric(df[RATING_COLUMN], errors="coerce")
    df = df.dropna(subset=[RATING_COLUMN])
    df[RATING_COLUMN] = df[RATING_COLUMN].astype(float)
    return df


def normalize_player_name(name: str) -> str:
    """Нормализация имени для сопоставления с нашим словарём игроков.

    - lowercase, strip
    - убираем лишние пробелы и тире
    - опционально: убираем диакритику (можно добавить unidecode)
    """
    if not isinstance(name, str) or pd.isna(name):
        return ""
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[-–—]", " ", s).strip()
    return s


def build_rating_splits(
    sofifa_df: pd.DataFrame,
    player_name2id: dict[str, int],
    *,
    dev_ratio: float = 0.15,
    seed: int = 42,
    min_id: int = 0,
    max_id: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Сопоставляет игроков SoFIFA со словарём и разбивает на train/val.

    Игроки, которых нет в player_name2id, отбрасываются. Не учитываются
    специальные id (mask, pad) — задаются через max_id (например, player_pad_token_id - 1).

    Args:
        sofifa_df: DataFrame из load_sofifa_csv (колонки name, overall).
        player_name2id: маппинг имя → id из нашего словаря.
        dev_ratio: доля выборки для валидации.
        seed: seed для разбиения.
        min_id: минимальный player_id для учёта (обычно 0).
        max_id: максимальный player_id (например, только «реальные» игроки без mask/pad).
                 Если None, допускаются все id из словаря.

    Returns:
        (train_df, eval_df) с колонками player_id, overall; индексы сброшены.
    """
    normalized2id = {}
    for name, pid in player_name2id.items():
        n = normalize_player_name(name)
        if n and pid not in (None,):
            normalized2id[n] = pid

    rows = []
    for _, row in sofifa_df.iterrows():
        n = normalize_player_name(row[NAME_COLUMN])
        if not n:
            continue
        pid = normalized2id.get(n)
        if pid is None:
            continue
        if max_id is not None and pid > max_id:
            continue
        if pid < min_id:
            continue
        rows.append({"player_id": pid, RATING_COLUMN: row[RATING_COLUMN]})

    if not rows:
        return pd.DataFrame(columns=["player_id", RATING_COLUMN]), pd.DataFrame(columns=["player_id", RATING_COLUMN])

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["player_id"], keep="first")

    n = len(df)
    n_dev = max(1, int(n * dev_ratio))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    dev_idx = idx[:n_dev]
    train_idx = idx[n_dev:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    eval_df = df.iloc[dev_idx].reset_index(drop=True)
    return train_df, eval_df


class SofifaRatingDataset(Dataset):
    """Dataset для предсказания рейтинга по player_id.

    Возвращает (player_id, overall). Эмбеддинг берётся из encoder.players_embeddings(player_id)
    при forward; здесь только пары для обучения головы.

    Args:
        df: DataFrame с колонками player_id, overall (из build_rating_splits).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        return {
            "player_id": torch.tensor(row["player_id"], dtype=torch.long),
            "overall": torch.tensor(row["overall"], dtype=torch.float32),
        }
