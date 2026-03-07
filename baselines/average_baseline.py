"""
Average-of-last-N baseline for NMSP.

This is the "strong baseline" from the paper: predict next match team stats
as the mean of the previous 5 matches' stats. Used as the benchmark in Table 2.
"""

import numpy as np
import pandas as pd


class AverageBaseline:
    """Baseline that predicts team stats as the mean over last N matches.

    Args:
        window_size: number of previous matches to average (default 5).
        stat_columns: list of stat column names to predict.
    """

    def __init__(self, window_size: int = 5, stat_columns: list[str] | None = None):
        ...

    def predict(
        self,
        df: pd.DataFrame,
        match_id_col: str = "match_id",
        team_name_col: str = "team_name",
    ) -> pd.DataFrame:
        """Generate predictions for every match in the DataFrame.

        For each match and each team, compute the mean of stat_columns
        over the previous `window_size` matches for that team.

        Args:
            df: DataFrame sorted by kickoff date, one row per player-match.
            match_id_col: column name for match ID.
            team_name_col: column name for team.

        Returns:
            DataFrame with columns: match_id, team_name, + predicted stat columns.
        """
        ...

    def evaluate(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute MSE, RMSE, dispersion coefficient vs ground truth.

        Args:
            predictions: output of predict().
            ground_truth: actual team-level stats per match.

        Returns:
            Dict with global_mse, per-stat rmse and delta values.
        """
        ...

### в точности не очень понятно как должны работать методы этого класса
### но возможно позже будет проще 