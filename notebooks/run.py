import sys
from pathlib import Path

ROOT = Path(".").resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon
else:
    device = torch.device("cpu")
print("Device:", device)
# AMD GPU: нужен PyTorch с ROCm — https://pytorch.org/get-started/locally/ (выбрать ROCm)

from data.preprocessing import preprocess_raw_csv, build_vocab_mappings

raw_path = ROOT / "dataset" / "data_with_dates.csv"
sample_path = ROOT / "notebooks" / "train_sample_raw.csv"
output_dir = str(ROOT / "notebooks" / "train_sample_processed")

df_raw = pd.read_csv(raw_path)
df_raw.to_csv(sample_path, index=False)
df = preprocess_raw_csv(str(sample_path), output_dir)
vocab = build_vocab_mappings(df, output_dir)

print("Матчей (уникальных match_id):", df["match_id"].nunique())
print("players_vocab_size (pad_idx+1):", vocab["player_pad_token_id"] + 1)
print("Число классов для MPP (реальные игроки):", vocab["player_pad_token_id"] - 1)


import numpy as np
from torch.utils.data import DataLoader

from data.dataset import MatchDatasetMPP, PreCollatedDataset
from data.collator import DataCollatorMPP, DataCollatorPreCollated

max_seq_length = 36
sample_batch_size = 256   # как в risingBALLER config
repeat = 20               # проходов DataLoader с разным shuffle/маскированием
dev_ratio = 0.05
seed = 42

ds_full = MatchDatasetMPP(
    df,
    player_name2id=vocab["player_name2id"],
    team_name2id=vocab["team_name2id"],
    max_seq_length=max_seq_length,
    player_pad_token_id=vocab["player_pad_token_id"],
    team_pad_token_id=vocab["team_pad_token_id"],
    position_pad_token_id=25,
)

collator = DataCollatorMPP(
    player_mask_token_id=vocab["player_mask_token_id"],
    mask_percentage=0.25,
)

# Repeat: несколько проходов по датасету с shuffle → разные маски в каждом батче
def _collate_filter_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return collator(batch)

dataloader_build = DataLoader(
    ds_full,
    batch_size=sample_batch_size,
    shuffle=True,
    collate_fn=_collate_filter_none,
    drop_last=True,
)

all_batches = []
for _ in range(repeat):
    for batch in dataloader_build:
        if batch is not None:
            all_batches.append(batch)

np.random.seed(seed)
n_batches = len(all_batches)
dev_size = max(1, int(n_batches * dev_ratio))
dev_idx = np.random.choice(n_batches, size=dev_size, replace=False)
train_idx = np.array([i for i in range(n_batches) if i not in dev_idx])

train_batches = [all_batches[i] for i in train_idx]
dev_batches = [all_batches[i] for i in dev_idx]

train_dataset = PreCollatedDataset(train_batches)
eval_dataset = PreCollatedDataset(dev_batches)
collator_for_trainer = DataCollatorPreCollated()

print("Пресобранных батчей (repeat):", n_batches, "train:", len(train_batches), "eval:", len(dev_batches))
print("Шагов за эпоху:", len(train_batches), "эффективный батч:", sample_batch_size)

from models.pretrain import MaskedPlayerModel
from training.trainer import build_training_args, build_trainer
from training.metrics import compute_metrics_mpp

embed_size = 128
model = MaskedPlayerModel(
    embed_size=embed_size,
    num_layers=1,
    heads=2,
    forward_expansion=4,
    dropout=0.05,
    form_stats_size=39,
    players_vocab_size=vocab["player_pad_token_id"],
    teams_vocab_size=vocab["team_pad_token_id"],
    positions_vocab_size=25,
    use_teams_embeddings=False,
    position_enc_type="learned",
)
model = model.to(device)

# Параметры как в risingBALLER: каждый шаг = один пресобранный батч (256 матчей)
training_config = {
    "output_dir": str(ROOT / "notebooks" / "mpp_mini_output"),
    "num_train_epochs": 2000,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "learning_rate": 1e-4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "lr_scheduler_type": "linear",
    "logging_steps": 1000,
    "eval_strategy": "steps",
    "eval_steps": 1000,
    "save_strategy": "steps",
    "save_steps": 10000,
    "save_total_limit": 3,
    "report_to": "tensorboard",
    "seed": seed,
}

train_args = build_training_args(training_config)
trainer = build_trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics_mpp,
    data_collator=collator_for_trainer,
)

print("Trainer готов. Параметров модели:", sum(p.numel() for p in model.parameters()))

final_dir = ROOT / "notebooks" / "mpp_mini_output" / "final_model"
try:
    trainer.train()
finally:
    # Всегда сохраняем модель в конце (даже при прерывании)
    trainer.save_model(str(final_dir))
    print("Финальная модель сохранена в:", final_dir)

# Сохраняем метрики в CSV (из trainer.state.log_history)
if trainer.state.log_history:
    metrics_path = ROOT / "notebooks" / "mpp_mini_output" / "metrics.csv"
    pd.DataFrame(trainer.state.log_history).to_csv(metrics_path, index=False)
    print("Метрики сохранены в:", metrics_path)

# Финальная валидация по eval_steps. Отдельно evaluate() после train() в Jupyter ломает NotebookProgressCallback.
# Опционально: повторный eval после train. Сначала убираем NotebookProgressCallback (он обнуляется в on_train_end).
from transformers.utils.notebook import NotebookProgressCallback
trainer.remove_callback(NotebookProgressCallback)
trainer.evaluate()