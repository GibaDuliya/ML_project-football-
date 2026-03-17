"""
Custom HuggingFace Trainer callbacks.

BestNCheckpointCallback — keeps top-N checkpoints for each of several metrics,
stored in dedicated folders per metric inside the main output directory.
"""

import json
import shutil
from pathlib import Path
from dataclasses import dataclass

from transformers import TrainerCallback


@dataclass
class _TrackedCheckpoint:
    """A saved checkpoint with its metric value."""
    value: float
    step: int
    path: str


class BestNCheckpointCallback(TrainerCallback):
    """Keep top-N checkpoints per metric in separate folders, delete the rest.

    After each evaluation the callback:
    1. Checks whether the current model is in the top-N for any tracked metric.
    2. If yes — saves a checkpoint into a per-metric folder (e.g.
       ``best_eval_loss_1/``, ``best_eval_loss_2/``).
    3. Writes a ``metrics.json`` file with the step number and all eval metrics.

    Args:
        metrics: dict mapping metric name (as it appears in eval logs,
                 e.g. ``"eval_loss"``) to ``"min"`` or ``"max"``.
        top_n:   how many best checkpoints to keep per metric (default 2).

    Usage::

        cb = BestNCheckpointCallback(
            metrics={"eval_loss": "min", "eval_accuracy_top3": "max"},
            top_n=2,
        )
        trainer = Trainer(..., callbacks=[cb])
        cb.set_trainer(trainer)   # required before trainer.train()
        trainer.train()
    """

    def __init__(
        self,
        metrics: dict[str, str],
        top_n: int = 2,
    ):
        super().__init__()
        self.metrics = metrics
        self.top_n = top_n
        self._boards: dict[str, list[_TrackedCheckpoint]] = {m: [] for m in metrics}
        self._trainer = None

    def set_trainer(self, trainer) -> None:
        """Store a reference to the Trainer (call after Trainer is created)."""
        self._trainer = trainer

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self._trainer is None:
            return

        for metric_name, direction in self.metrics.items():
            value = metrics.get(metric_name)
            if value is None:
                continue

            board = self._boards[metric_name]

            # Check if current value qualifies for top-N
            if len(board) >= self.top_n and not self._is_better(value, board[-1].value, direction):
                continue

            # Add new entry
            board.append(_TrackedCheckpoint(value=value, step=state.global_step, path=""))
            reverse = (direction == "max")
            board.sort(key=lambda c: c.value, reverse=reverse)
            if len(board) > self.top_n:
                board[:] = board[:self.top_n]

            # Re-save all checkpoints for this metric with correct rank folders
            for rank, entry in enumerate(board, start=1):
                folder_name = f"best_{metric_name}_{rank}"
                ckpt_dir = str(Path(args.output_dir) / folder_name)

                # Remove old contents if folder already exists
                if Path(ckpt_dir).exists():
                    shutil.rmtree(ckpt_dir, ignore_errors=True)

                # Save model
                self._trainer.save_model(ckpt_dir)
                entry.path = ckpt_dir

                # Write metrics.json
                metrics_info = {
                    "global_step": entry.step,
                    "rank": rank,
                    "tracked_metric": metric_name,
                    "tracked_value": entry.value,
                    **{k: v for k, v in metrics.items() if k.startswith("eval_")},
                }
                # Overwrite eval metrics with the ones from the entry's step
                # (current metrics are only correct for the entry that was just added)
                metrics_info["global_step"] = entry.step
                metrics_info["tracked_value"] = entry.value
                with open(Path(ckpt_dir) / "metrics.json", "w") as f:
                    json.dump(metrics_info, f, indent=2)

            print(f"  [BestN] {metric_name}: updated top-{self.top_n} "
                  f"(step {state.global_step}, {metric_name}={value:.4f})")

    def get_best_checkpoint_path(self, metric_name: str) -> str | None:
        """Return the path of the best checkpoint for the given metric."""
        board = self._boards.get(metric_name, [])
        return board[0].path if board else None

    def summary(self) -> str:
        """Return a human-readable summary of all tracked best checkpoints."""
        lines = []
        for metric_name, board in self._boards.items():
            direction = self.metrics[metric_name]
            lines.append(f"{metric_name} ({direction}):")
            for i, c in enumerate(board):
                lines.append(f"  #{i+1}  step={c.step}  {metric_name}={c.value:.4f}  -> {c.path}")
        return "\n".join(lines)

    @staticmethod
    def _is_better(new: float, old: float, direction: str) -> bool:
        if direction == "min":
            return new < old
        return new > old
