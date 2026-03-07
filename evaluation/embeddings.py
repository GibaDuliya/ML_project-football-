"""
Embedding extraction and visualization utilities.

extract_embeddings        — pull player/position embedding matrices from a checkpoint.
cluster_positions         — KMeans clustering of position embeddings (cosine space).
plot_positions_on_pitch   — draw clustered positions on a football pitch diagram.
plot_player_position_fit  — bar charts: learned vs native positions for a player.
"""

from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


def extract_embeddings(
    checkpoint_path: str,
    model_class: type,
    model_kwargs: dict,
) -> dict[str, np.ndarray]:
    """Load a model checkpoint and extract embedding matrices.

    Args:
        checkpoint_path: path to model .safetensors or .pt file.
        model_class: MaskedPlayerModel or DownstreamModel class.
        model_kwargs: constructor kwargs for model_class.

    Returns:
        Dict with keys:
            "players": np.ndarray (vocab_size, embed_size)
            "positions": np.ndarray (n_positions, embed_size)
            "teams": np.ndarray (n_teams, embed_size) or None
    """
    ...


def cluster_positions(
    position_embeddings: np.ndarray,
    n_clusters: int = 2,
    n_init: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster position embeddings using KMeans in cosine similarity space.

    Args:
        position_embeddings: (n_positions, embed_size) matrix.
        n_clusters: number of clusters (2 or 3 as in paper).
        n_init: number of KMeans initializations.

    Returns:
        (cluster_labels, cluster_centers) — labels per position, centroids.
    """
    ...


def plot_positions_on_pitch(
    position_names: list[str],
    cluster_labels: np.ndarray,
    cluster_colors: Optional[list[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Draw a football pitch and overlay position names colored by cluster.

    Args:
        position_names: list of position abbreviations (GK, RB, CB, ...).
        cluster_labels: cluster assignment per position.
        cluster_colors: optional color per cluster.
        ax: optional matplotlib Axes to draw on.

    Returns:
        matplotlib Figure.
    """
    ...


def plot_player_position_fit(
    player_name: str,
    player_embedding: np.ndarray,
    position_embeddings: np.ndarray,
    position_names: list[str],
    native_positions: dict[str, int],
    top_k: int = 3,
) -> plt.Figure:
    """Side-by-side bar chart: top-K learned positions vs native position frequency.

    As in Figure 4 of the paper.

    Args:
        player_name: display name.
        player_embedding: (embed_size,) vector.
        position_embeddings: (n_positions, embed_size) matrix.
        position_names: list of position abbreviations.
        native_positions: dict {position_name: count} from actual match data.
        top_k: how many top positions to show.

    Returns:
        matplotlib Figure.
    """
    ...
