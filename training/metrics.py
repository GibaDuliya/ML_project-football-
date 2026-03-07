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
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # иногда HF возвращает tuple (logits, ...)
    if isinstance(logits, tuple):
        logits = logits[0]

    vocab_size = logits.shape[-1]

    # flatten
    logits = logits.reshape(-1, vocab_size)
    labels = labels.reshape(-1)

    # keep only masked tokens
    mask = labels != -100
    logits = logits[mask]
    labels = labels[mask]

    if len(labels) == 0:
        return {"accuracy_top1": 0.0, "accuracy_top3": 0.0}

    # top-1
    top1_preds = np.argmax(logits, axis=1)
    acc_top1 = np.mean(top1_preds == labels)

    # top-3
    top3_preds = np.argsort(logits, axis=1)[:, -3:]
    acc_top3 = np.mean([label in row for label, row in zip(labels, top3_preds)])

    return {
        "accuracy_top1": float(acc_top1),
        "accuracy_top3": float(acc_top3),
    }


def compute_metrics_nmsp(eval_pred) -> dict[str, float]:
    """Compute MSE and per-stat RMSE for Next Match Statistics Prediction.

    Args:
        eval_pred: EvalPrediction with:
            predictions: (n_samples, 2 * num_stats)
            label_ids:   (n_samples, 2 * num_stats)

    Returns:
        {"global_mse": float, "rmse_mean": float}
    """
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    mse = np.mean((preds - labels) ** 2)

    rmse_per_stat = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
    rmse_mean = np.mean(rmse_per_stat)

    return {
        "global_mse": float(mse),
        "rmse_mean": float(rmse_mean),
    }


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
    rmse_values = np.asarray(rmse_values)
    mean_values = np.asarray(mean_values)

    # avoid division by zero
    eps = 1e-8

    dispersion = rmse_values / (mean_values + eps)

    return dispersion
