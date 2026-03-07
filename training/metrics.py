"""
Metric functions for HuggingFace Trainer's compute_metrics callback.

Each function receives an EvalPrediction(predictions, label_ids) and returns
a dict of metric_name → float.
"""

import numpy as np


def compute_metrics_mpp(eval_pred) -> dict[str, float]:
    """Compute top-1 and top-3 accuracy for Masked Player Prediction.

    Args:
        eval_pred: EvalPrediction with:
            predictions: tuple (logits, hidden_states, attentions)
                         logits shape: (n_batches, seq_len, vocab_size)
            label_ids:   (n_batches, seq_len) with -100 for non-masked tokens.

    Steps:
        1. Flatten logits and labels to (N_total_tokens, ...).
        2. Filter to only masked tokens (labels != -100).
        3. top1_accuracy = mean(argmax(logits) == labels).
        4. top3_accuracy = mean(labels in top-3 of logits).

    Returns:
        {"accuracy_top1": float, "accuracy_top3": float}
    """
    ...


def compute_metrics_nmsp(eval_pred) -> dict[str, float]:
    """Compute MSE and per-stat RMSE for Next Match Statistics Prediction.

    Args:
        eval_pred: EvalPrediction with:
            predictions: (n_samples, 2 * num_stats)
            label_ids:   (n_samples, 2 * num_stats)

    Returns:
        {"global_mse": float, "rmse_mean": float}
    """
    ...


def compute_dispersion_coefficient(
    rmse_values: np.ndarray,
    mean_values: np.ndarray,
) -> np.ndarray:
    """Compute dispersion coefficient δ = RMSE / mean for each statistic.

    This is the scale-independent metric used in Table 3 of the paper.

    Args:
        rmse_values: array of RMSE per statistic.
        mean_values: array of mean value per statistic.

    Returns:
        Array of dispersion coefficients.
    """
    ...
