"""
Обработка данных SoFIFA для предсказания рейтинга (overall).

- load_sofifa_csv() — загрузка dataset/sofifa_players.csv.
- normalize_player_name() — нормализация имени для сопоставления со словарём.
- build_player_id_to_overall() — player_id → overall из SoFIFA.
- build_aggregated_embeddings() — по матчам энкодер, усреднение по (player_id, season).
- SofifaAggregatedDataset — (mean_embedding, overall) для обучения головы.
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


def build_player_id_to_overall(
    sofifa_df: pd.DataFrame,
    player_name2id: dict[str, int],
    *,
    max_id: Optional[int] = None,
) -> dict[int, float]:
    """Строит маппинг player_id → overall для игроков, присутствующих в SoFIFA и в словаре.

    Args:
        sofifa_df: DataFrame из load_sofifa_csv (name, overall).
        player_name2id: словарь из метаданных MPP.
        max_id: если задан, игнорировать player_id > max_id (mask/pad).

    Returns:
        dict[player_id, overall].
    """
    normalized2id = {}
    for name, pid in player_name2id.items():
        n = normalize_player_name(name)
        if n and pid is not None:
            normalized2id[n] = pid
    out = {}
    for _, row in sofifa_df.iterrows():
        n = normalize_player_name(row[NAME_COLUMN])
        if not n:
            continue
        pid = normalized2id.get(n)
        if pid is None:
            continue
        if max_id is not None and pid > max_id:
            continue
        out[pid] = float(row[RATING_COLUMN])
    return out


def build_aggregated_embeddings(
    encoder: torch.nn.Module,
    match_dataset: Dataset,
    match_df: pd.DataFrame,
    player_id_to_overall: dict[int, float],
    device: torch.device,
    *,
    batch_size: int = 1,
) -> tuple[np.ndarray, pd.DataFrame]:
    """По каждому матчу прогоняет энкодер, собирает эмбеддинги по (player_id, season_name), усредняет по сезону.

    Ожидает match_dataset с __getitem__ → dict с input_ids, position_id, team_id, form_stats, attention_mask
    (как MatchDatasetMPP, без маскирования). match_df должен содержать match_id и season_name.

    Returns:
        embeddings: (N, embed_size) float32 — усреднённые эмбеддинги по (player_id, season).
        meta: DataFrame с колонками player_id, season_name, overall (N строк, порядок как в embeddings).
        Только игроки из player_id_to_overall; без overall отбрасываются.
    """
    encoder.eval()
    encoder.to(device)
    match_ids = getattr(match_dataset, "_match_ids", None)
    if match_ids is None:
        raise ValueError("match_dataset must have _match_ids (e.g. MatchDatasetMPP)")

    # Собираем (player_id, season_name, embedding) по всем матчам
    list_player_season_emb: list[tuple[int, str, np.ndarray]] = []
    embed_size = None

    for idx in range(len(match_dataset)):
        batch = match_dataset[idx]
        if batch is None:
            continue
        match_id = match_ids[idx]
        seasons = match_df.loc[match_df["match_id"] == match_id, "season_name"]
        if len(seasons) == 0:
            continue
        season_name = str(seasons.iloc[0])

        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        position_id = batch["position_id"].unsqueeze(0).to(device)
        team_id = batch["team_id"].unsqueeze(0).to(device)
        form_stats = batch["form_stats"].unsqueeze(0).to(device)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            enc_out, _ = encoder(input_ids, position_id, team_id, form_stats, attention_mask)
        # enc_out (1, seq_len, embed_size)
        if embed_size is None:
            embed_size = enc_out.shape[-1]
        seq_len = enc_out.shape[1]
        mask = batch["attention_mask"]
        ids = batch["input_ids"]
        for i in range(seq_len):
            if mask[i].item() != 1:
                continue
            pid = ids[i].item()
            emb = enc_out[0, i].cpu().numpy().astype(np.float32)
            list_player_season_emb.append((pid, season_name, emb))

    if not list_player_season_emb:
        return np.zeros((0, embed_size or 1), dtype=np.float32), pd.DataFrame(
            columns=["player_id", "season_name", "overall"]
        )

    # Группируем по (player_id, season_name), усредняем
    from collections import defaultdict
    agg: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    for pid, season, emb in list_player_season_emb:
        agg[(pid, season)].append(emb)
    mean_embeddings = []
    meta_rows = []
    for (pid, season), embs in sorted(agg.items()):
        overall = player_id_to_overall.get(pid)
        if overall is None:
            continue
        mean_emb = np.stack(embs, axis=0).mean(axis=0)
        mean_embeddings.append(mean_emb)
        meta_rows.append({"player_id": pid, "season_name": season, "overall": overall})

    if not mean_embeddings:
        return np.zeros((0, embed_size), dtype=np.float32), pd.DataFrame(
            columns=["player_id", "season_name", "overall"]
        )
    embeddings = np.stack(mean_embeddings, axis=0)
    meta = pd.DataFrame(meta_rows)
    return embeddings, meta


class SofifaAggregatedDataset(Dataset):
    """Датасет для задачи 2: усреднённый по сезону эмбеддинг → overall.

    Возвращает (aggregated_embedding, overall). Голова получает уже один вектор на сэмпл.
    """

    def __init__(self, embeddings: np.ndarray, meta: pd.DataFrame):
        """
        Args:
            embeddings: (N, embed_size) float32.
            meta: DataFrame с колонкой overall (N строк), порядок как в embeddings.
        """
        assert len(meta) == len(embeddings)
        self.embeddings = embeddings
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "aggregated_embedding": torch.from_numpy(self.embeddings[idx].copy()),
            "overall": torch.tensor(self.meta.iloc[idx]["overall"], dtype=torch.float32),
        }
