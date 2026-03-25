"""
Обучение регрессионной головы рейтинга SoFIFA по агрегированным по сезону эмбеддингам.

RatingHeadTrainer — обучает голову по батчам (aggregated_embedding, overall), MSE, сохранение rating_head.pt.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class RatingHeadTrainer:
    """Обучает голову регрессии по агрегированным по сезону эмбеддингам.

    Батч: aggregated_embedding (batch, embed_size), overall (batch,).
    Голова: head(emb.unsqueeze(1), mask) → один скаляр на сэмпл. Сохраняет rating_head.pt.
    """

    def __init__(
        self,
        head: torch.nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        *,
        output_dir: str | Path = ".",
        num_epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        device: Optional[torch.device] = None,
        logging_steps: int = 10,
        save_best: bool = True,
        target_mean: float | None = None,
        target_std: float | None = None,
    ):
        self.head = head
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.save_best = save_best
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = self.head.to(self.device)
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_eval_rmse = float("inf")  # RMSE in original rating scale
        self.train_ds_len = len(train_loader.dataset)
        self.eval_ds_len = len(eval_loader.dataset)
        self.target_mean = float(target_mean) if target_mean is not None else None
        self.target_std = float(target_std) if target_std is not None else None
        if (self.target_mean is None) ^ (self.target_std is None):
            raise ValueError("Provide both target_mean and target_std, or neither.")
        if self.target_std is not None and self.target_std <= 0:
            raise ValueError("target_std must be > 0.")

    def _normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        if self.target_mean is None:
            return y
        return (y - self.target_mean) / self.target_std

    def _denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        if self.target_mean is None:
            return y
        return y * self.target_std + self.target_mean

    def _forward_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        emb = batch["aggregated_embedding"].to(self.device)  # (batch, embed_size)
        overall = batch["overall"].to(self.device)
        overall_n = self._normalize_y(overall)
        # head ожидает (batch, seq_len, embed_size); seq_len=1
        enc_out = emb.unsqueeze(1)
        attn = torch.ones(enc_out.size(0), 1, dtype=torch.long, device=self.device)
        pred = self.head(enc_out, attn).squeeze(-1)
        return pred, overall_n

    def train(self) -> float:
        for epoch in range(self.num_epochs):
            self.head.train()
            train_loss = 0.0
            for batch in self.train_loader:
                pred, overall_n = self._forward_batch(batch)
                loss = F.mse_loss(pred, overall_n)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch["overall"].size(0)
            train_loss /= self.train_ds_len

            self.head.eval()
            # Compute RMSE in original rating scale (even if training uses normalized targets)
            se = 0.0
            n_total = 0
            with torch.no_grad():
                for batch in self.eval_loader:
                    pred_n, overall_n = self._forward_batch(batch)
                    pred = self._denormalize_y(pred_n)
                    overall = self._denormalize_y(overall_n)
                    se += ((pred - overall) ** 2).sum().item()
                    n_total += overall.numel()
            eval_rmse = (se / max(1, n_total)) ** 0.5

            if self.save_best and eval_rmse < self.best_eval_rmse:
                self.best_eval_rmse = eval_rmse
                torch.save(self.head.state_dict(), self.output_dir / "rating_head.pt")

            if (epoch + 1) % self.logging_steps == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}  "
                    f"train_loss={train_loss:.4f}  eval_rmse={eval_rmse:.4f}"
                )
        if self.save_best:
            print("Best eval RMSE:", self.best_eval_rmse)
            print("Голова сохранена:", self.output_dir / "rating_head.pt")
        return self.best_eval_rmse


class RatingBinHeadTrainer:
    """Классификация бина рейтинга по агрегированному эмбеддингу (CE loss, accuracy на eval)."""

    def __init__(
        self,
        head: torch.nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        *,
        output_dir: str | Path = ".",
        num_epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        device: Optional[torch.device] = None,
        logging_steps: int = 10,
        save_best: bool = True,
    ):
        self.head = head
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.save_best = save_best
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head = self.head.to(self.device)
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_eval_acc = 0.0
        self.train_ds_len = len(train_loader.dataset)
        self.eval_ds_len = len(eval_loader.dataset)

    def _forward_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        emb = batch["aggregated_embedding"].to(self.device)
        y = batch["class_id"].to(self.device)
        enc_out = emb.unsqueeze(1)
        attn = torch.ones(enc_out.size(0), 1, dtype=torch.long, device=self.device)
        logits = self.head(enc_out, attn)
        return logits, y

    def train(self) -> float:
        for epoch in range(self.num_epochs):
            self.head.train()
            train_loss = 0.0
            for batch in self.train_loader:
                logits, y = self._forward_batch(batch)
                loss = F.cross_entropy(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch["class_id"].size(0)
            train_loss /= max(1, self.train_ds_len)

            self.head.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in self.eval_loader:
                    logits, y = self._forward_batch(batch)
                    pred = logits.argmax(dim=-1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
            eval_acc = correct / max(1, total)

            if self.save_best and eval_acc > self.best_eval_acc:
                self.best_eval_acc = eval_acc
                torch.save(self.head.state_dict(), self.output_dir / "rating_class_head.pt")

            if (epoch + 1) % self.logging_steps == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}  "
                    f"train_loss={train_loss:.4f}  eval_acc={eval_acc:.4f}"
                )
        if self.save_best:
            print("Best eval accuracy:", self.best_eval_acc)
            print("Голова сохранена:", self.output_dir / "rating_class_head.pt")
        return float(self.best_eval_acc)
