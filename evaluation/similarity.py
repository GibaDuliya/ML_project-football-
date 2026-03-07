"""
Player similarity analysis: retrieval, team cohesion, dissimilarity heatmaps.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between rows of A and B.

    Args:
        A: (n, d) matrix.
        B: (m, d) matrix.

    Returns:
        (n, m) similarity matrix with values in [-1, 1].
    """
    ...


def find_similar_players(
    query_player_id: int,
    player_embeddings: np.ndarray,
    id2player_name: dict[int, str],
    top_k: int = 10,
    exclude_ids: Optional[set[int]] = None,
) -> list[tuple[str, float]]:
    """Find top-K most similar players to a query player by cosine similarity.

    Args:
        query_player_id: integer ID of the query player.
        player_embeddings: (vocab_size, embed_size) matrix.
        id2player_name: mapping id → player name string.
        top_k: number of similar players to return.
        exclude_ids: set of player IDs to exclude from results.

    Returns:
        List of (player_name, similarity_score) tuples, sorted descending.
    """
    ...


def compute_team_cohesion(
    team_player_ids: list[int],
    player_embeddings: np.ndarray,
) -> float:
    """Compute team cohesion factor as mean pairwise cosine similarity.

    For all players in a squad, compute cumulative similarity with all
    teammates and average by squad size. (Section 4.2.1 of the paper.)

    Args:
        team_player_ids: list of player IDs in the squad.
        player_embeddings: (vocab_size, embed_size) full embedding matrix.

    Returns:
        Scalar cohesion score.
    """
    ...


def plot_dissimilarity_heatmap(
    player_ids_team1: list[int],
    player_ids_team2: list[int],
    player_embeddings: np.ndarray,
    id2player_name: dict[int, str],
    title: str = "Dissimilarity Heatmap",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plot dissimilarity heatmap between two teams' players.

    As in Figures 5-6 of the paper (Barcelona vs Real Madrid).

    Args:
        player_ids_team1: player IDs for team 1.
        player_ids_team2: player IDs for team 2.
        player_embeddings: full embedding matrix.
        id2player_name: id → name mapping.
        title: plot title.
        ax: optional Axes.

    Returns:
        matplotlib Figure.
    """
    ...
