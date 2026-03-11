"""
Trainer для головы предсказания статистик (39 статов) по эмбеддингам после трансформера.

StatsHeadTrainer — обучение только головы при замороженном энкодере;
MSE/MAE по не-padding позициям; сохранение головы и evaluation с разбивкой по статкам.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE только по позициям с mask=1. pred/target (B, S, F), mask (B, S)."""
    mask_exp = mask.unsqueeze(-1).float()
    se = (pred - target) ** 2
    return (se * mask_exp).sum() / mask_exp.sum().clamp(min=1e-9)


def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MAE только по позициям с mask=1."""
    mask_exp = mask.unsqueeze(-1).float()
    ae = (pred - target).abs()
    return (ae * mask_exp).sum() / mask_exp.sum().clamp(min=1e-9)


class StatsHeadTrainer:
    """Обучает голову предсказания 39 статов; энкодер заморожен.

    Батч из DataLoader: dict с ключами input_ids, position_id, team_id,
    form_stats, attention_mask (как у MatchDatasetMPP).
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        head: torch.nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        *,
        output_dir: str | Path = ".",
        num_epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        device: Optional[torch.device] = None,
        logging_steps: int = 1,
        save_best: bool = True,
    ):
        self.encoder = encoder
        self.head = head
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.save_best = save_best

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.encoder = self.encoder.to(self.device)
        self.head = self.head.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.best_eval_mse = float("inf")

    def _forward_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = batch["input_ids"].to(self.device)
        position_id = batch["position_id"].to(self.device)
        team_id = batch["team_id"].to(self.device)
        form_stats = batch["form_stats"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            hidden, _ = self.encoder(
                input_ids, position_id, team_id, form_stats, attention_mask
            )
        pred = self.head(hidden)
        return pred, form_stats, attention_mask

    def train(self) -> float:
        """Запуск обучения. Возвращает лучший eval MSE."""
        self.head.train()
        for epoch in range(self.num_epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in self.train_loader:
                pred, form_stats, attention_mask = self._forward_batch(batch)
                loss = masked_mse(pred, form_stats, attention_mask)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / n_batches

            self.head.eval()
            eval_mse, _ = self.evaluate()
            self.head.train()

            if self.save_best and eval_mse < self.best_eval_mse:
                self.best_eval_mse = eval_mse
                self.save_head(self.output_dir / "stats_head.pt")

            if (epoch + 1) % self.logging_steps == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}  "
                    f"Train MSE: {avg_loss:.6f}  Eval MSE: {eval_mse:.6f}"
                )

        if self.save_best:
            print("Best eval MSE:", self.best_eval_mse)
            print("Голова сохранена:", self.output_dir / "stats_head.pt")
        return self.best_eval_mse

    def evaluate(self) -> tuple[float, float]:
        """Считает MSE и MAE на eval_loader. Возвращает (mse, mae)."""
        self.head.eval()
        total_mse = 0.0
        total_mae = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in self.eval_loader:
                pred, form_stats, attention_mask = self._forward_batch(batch)
                total_mse += masked_mse(pred, form_stats, attention_mask).item()
                total_mae += masked_mae(pred, form_stats, attention_mask).item()
                n_batches += 1
        n_batches = max(n_batches, 1)
        return total_mse / n_batches, total_mae / n_batches

    def evaluate_per_stat(self, num_stats: int) -> np.ndarray:
        """MSE по каждой из num_stats статок. Возвращает массив (num_stats,)."""
        self.head.eval()
        total_se = np.zeros(num_stats, dtype=np.float64)
        total_n = np.zeros(num_stats, dtype=np.float64)
        with torch.no_grad():
            for batch in self.eval_loader:
                pred, form_stats, attention_mask = self._forward_batch(batch)
                mask = attention_mask.unsqueeze(-1).float()
                se = (
                    ((pred - form_stats) ** 2 * mask)
                    .sum(dim=(0, 1))
                    .cpu()
                    .numpy()
                )
                n = mask.sum(dim=(0, 1)).cpu().numpy()
                total_se += se
                total_n += n
        return (total_se / np.maximum(total_n, 1)).astype(np.float32)

    def save_head(self, path: str | Path) -> None:
        """Сохранить state_dict головы."""
        torch.save(self.head.state_dict(), path)
