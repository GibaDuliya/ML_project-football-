"""
Training utilities — wrappers around HuggingFace Trainer.

build_training_args() — create TrainingArguments from config dict.
build_trainer()       — create Trainer with model, datasets, metrics.
"""

from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments


def build_training_args(config: dict) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from a config dict.

    The config dict should contain keys matching TrainingArguments fields,
    e.g.: output_dir, num_train_epochs, learning_rate, etc.

    Args:
        config: dict with training hyperparameters (from YAML "training" section).

    Returns:
        TrainingArguments instance.
    """
    args_dict = dict(config)
    return TrainingArguments(**args_dict)



def build_trainer(
    model: torch.nn.Module,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    compute_metrics: Optional[Callable] = None,
    data_collator: Optional[Callable] = None,
) -> Trainer:
    """Build a HuggingFace Trainer.

    Args:
        model: the PyTorch model (MaskedPlayerModel or DownstreamModel).
        args: TrainingArguments.
        train_dataset: training dataset.
        eval_dataset: validation dataset.
        compute_metrics: function (EvalPrediction) → dict of metric names/values.
        data_collator: optional collator (if dataset returns unbatched items).

    Returns:
        Configured Trainer instance.
    """
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if compute_metrics is not None else None,
        data_collator= data_collator if data_collator else None,
    )

