"""
Обучение регрессионной головы рейтинга (SoFIFA) поверх замороженного энкодера.

RatingHeadTrainer — цикл train/eval, MSE loss, сохранение лучшей головы по eval RMSE.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class RatingHeadTrainer:
    """Обучает только голову регрессии рейтинга; энкодер заморожен.

    Голова получает эмбеддинги с выхода трансформера (после attention), а не сырые
    player embeddings. Для каждого сэмпла строится один токен (player_id, position=0,
    team=0, form_stats=0), прогоняется полный encoder, выход (batch, 1, embed_size)
    подаётся в голову.

    Батч из DataLoader: dict с ключами "player_id", "overall".
    """

    def __init__(
        self,
        model: torch.nn.Module,
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
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.save_best = save_best

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.best_eval_rmse = float("inf")
        self.train_ds_len = len(train_loader.dataset)
        self.eval_ds_len = len(eval_loader.dataset)

    def _forward_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        player_id = batch["player_id"].to(self.device)  # (batch,)
        overall = batch["overall"].to(self.device)
        batch_size = player_id.size(0)
        form_stats_size = self.model.encoder.form_embeddings.in_features

        # Один токен на игрока: полный forward энкодера → эмбеддинги после attention
        input_ids = player_id.unsqueeze(1)  # (batch, 1)
        position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        team_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
        form_stats = torch.zeros(batch_size, 1, form_stats_size, device=self.device)
        attention_mask = torch.ones(batch_size, 1, dtype=torch.long, device=self.device)

        enc_out, _ = self.model.encoder(
            input_ids, position_ids, team_ids, form_stats, attention_mask
        )
        pred = self.model.head(enc_out, attention_mask).squeeze(-1)
        return pred, overall

    def train(self) -> float:
        """Запуск обучения. Возвращает лучший eval RMSE."""
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch in self.train_loader:
                pred, overall = self._forward_batch(batch)
                loss = F.mse_loss(pred, overall)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch["player_id"].size(0)
            train_loss /= self.train_ds_len

            self.model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for batch in self.eval_loader:
                    pred, overall = self._forward_batch(batch)
                    loss = F.mse_loss(pred, overall)
                    eval_loss += loss.item() * batch["player_id"].size(0)
            eval_loss /= self.eval_ds_len
            eval_rmse = eval_loss ** 0.5

            if self.save_best and eval_rmse < self.best_eval_rmse:
                self.best_eval_rmse = eval_rmse
                torch.save(
                    self.model.head.state_dict(),
                    self.output_dir / "rating_head.pt",
                )

            if (epoch + 1) % self.logging_steps == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}  "
                    f"train_loss={train_loss:.4f}  eval_rmse={eval_rmse:.4f}"
                )

        if self.save_best:
            print("Best eval RMSE:", self.best_eval_rmse)
            print("Голова сохранена:", self.output_dir / "rating_head.pt")
        return self.best_eval_rmse
