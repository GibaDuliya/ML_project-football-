"""
Baselines for statistics prediction.

AverageBaseline
    Placeholder for NMSP \"average-of-last-N\" baseline (team-level stats).

compute_stats_baseline_mse_per_stat
    Simple baseline for per-player stats prediction (39 статов):
    predicts each stat as its mean value on the evaluation set and
    returns per-stat MSE. Used as the \"baseline (b)\" column in the
    stats-head notebook table.
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


class AverageBaseline:
    """Baseline that predicts team stats as the mean over last N matches.

    NOTE: this is a placeholder for the NMSP baseline described in the paper.
    It is not used yet in the codebase; when NMSP pipeline is implemented,
    this class can be filled in to operate on team-level aggregated stats.

    Args:
        window_size: number of previous matches to average (default 5).
        stat_columns: list of stat column names to predict.
    """

    def __init__(self, window_size: int = 5, stat_columns: Optional[list[str]] = None):
        self.window_size = window_size
        self.stat_columns = stat_columns or []

    def predict(
        self,
        df: pd.DataFrame,
        match_id_col: str = "match_id",
        team_name_col: str = "team_name",
    ) -> pd.DataFrame:
        """Placeholder for future NMSP baseline.

        For now, raises NotImplementedError to make it explicit that this
        baseline is not wired into the NMSP pipeline yet.
        """
        raise NotImplementedError("AverageBaseline.predict is not implemented yet.")

    def evaluate(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> dict[str, float]:
        """Placeholder for future NMSP baseline evaluation."""
        raise NotImplementedError("AverageBaseline.evaluate is not implemented yet.")


def compute_stats_baseline_mse_per_stat(
    eval_loader: DataLoader,
    num_stats: int,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Baseline MSE per stat for player-level stats prediction.

    For each stat we predict a constant equal to its mean over all
    non-padding player tokens in the evaluation set. The returned
    values correspond to MSE of this constant predictor — i.e. the
    dispersion of each stat around its mean.

    Args:
        eval_loader: DataLoader yielding dicts with keys
            \"form_stats\" (B, S, F) and \"attention_mask\" (B, S).
        num_stats: number of stat dimensions (e.g. 39).
        device: optional device; if provided, tensors are moved there
            before computation (useful when eval_loader yields CUDA tensors).

    Returns:
        np.ndarray of shape (num_stats,) with baseline MSE per stat.
    """
    total_sum = np.zeros(num_stats, dtype=np.float64)
    total_sq = np.zeros(num_stats, dtype=np.float64)
    total_n = np.zeros(num_stats, dtype=np.float64)

    dev = device or torch.device("cpu")

    with torch.no_grad():
        for batch in eval_loader:
            form_stats = batch["form_stats"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)

            mask_exp = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            weighted = form_stats * mask_exp  # zero out padding

            total_sum += weighted.sum(dim=(0, 1)).cpu().numpy()
            total_sq += (form_stats * weighted).sum(dim=(0, 1)).cpu().numpy()
            total_n += mask_exp.sum(dim=(0, 1)).cpu().numpy()

    mean_per_stat = total_sum / np.maximum(total_n, 1)
    # MSE(constant = mean) = E[(y - mean)^2] = E[y^2] - mean^2
    mse = total_sq / np.maximum(total_n, 1) - mean_per_stat**2
    return mse.astype(np.float32)