"""
Run MPP (Masked Player Prediction) training for any compatible model.

Supports:
    - Main transformer model (models.pretrain.MaskedPlayerModel)
    - MLP baseline (baselines.MLP_baseline.MLP_baseline.MLPMaskedPlayerModel)

Usage examples:
    # Train main transformer model
    python run/run_mpp.py --data dataset/data_with_dates.csv --output outputs/mpp_main

    # Train MLP baseline
    python run/run_mpp.py --data dataset/data_with_dates.csv --output outputs/mpp_mlp \
        --model mlp_baseline --mlp_num_layers 2 --mlp_hidden_mult 4

    # Small test run
    python run/run_mpp.py --data dataset/data_with_dates.csv --output outputs/mpp_test \
        --embed_size 32 --epochs 10 --batch_size 32
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.preprocessing import preprocess_raw_csv, build_vocab_mappings
from data.dataset import MatchDatasetMPP, PreCollatedDataset
from data.collator import DataCollatorMPP, DataCollatorPreCollated
from training.trainer import build_training_args, build_trainer
from training.metrics import compute_metrics_mpp


def parse_args():
    p = argparse.ArgumentParser(description="MPP training for RisingBALLER models")

    # Data
    p.add_argument("--data", type=str, required=True,
                    help="Path to raw CSV (e.g. dataset/data_with_dates.csv)")
    p.add_argument("--output", type=str, default="outputs/mpp",
                    help="Output directory for checkpoints and logs")
    p.add_argument("--processed_dir", type=str, default=None,
                    help="Dir to store processed data (default: <output>/processed)")

    # Model selection
    p.add_argument("--model", type=str, default="transformer",
                    choices=["transformer", "mlp_baseline"],
                    help="Model architecture to train")

    # Common model hyperparameters
    p.add_argument("--embed_size", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1,
                    help="Number of transformer blocks (transformer) or MLP blocks (mlp_baseline)")
    p.add_argument("--heads", type=int, default=2, help="Attention heads (transformer only)")
    p.add_argument("--forward_expansion", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--use_teams_embeddings", action="store_true")
    p.add_argument("--position_enc_type", type=str, default="learned",
                    choices=["learned", "sinusoidal"])

    # MLP baseline specific
    p.add_argument("--mlp_num_layers", type=int, default=None,
                    help="Number of MLP blocks (overrides --num_layers for mlp_baseline)")
    p.add_argument("--mlp_hidden_mult", type=int, default=4,
                    help="Hidden dim multiplier for MLP blocks")

    # Data pipeline
    p.add_argument("--max_seq_length", type=int, default=36)
    p.add_argument("--mask_percentage", type=float, default=0.25)
    p.add_argument("--dev_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    # Training mode
    p.add_argument("--precollate", action="store_true",
                    help="Use pre-collation approach (as in kaggle notebook)")
    p.add_argument("--precollate_batch_size", type=int, default=256,
                    help="Batch size for pre-collation")
    p.add_argument("--precollate_repeat", type=int, default=20,
                    help="Number of augmentation repeats for pre-collation")

    # Training hyperparameters
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64,
                    help="Batch size (ignored if --precollate)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.0)
    p.add_argument("--scheduler", type=str, default="linear")
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=10000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--report_to", type=str, default="none",
                    choices=["none", "tensorboard", "wandb"])

    return p.parse_args()


def load_data(args):
    """Load and preprocess data, build vocab."""
    processed_dir = args.processed_dir or str(Path(args.output) / "processed")
    Path(processed_dir).mkdir(parents=True, exist_ok=True)

    print(f"Preprocessing data from {args.data}...")
    df = preprocess_raw_csv(args.data, processed_dir)
    vocab = build_vocab_mappings(df, processed_dir)

    print(f"  Matches: {df['match_id'].nunique()}")
    print(f"  Players vocab size: {vocab['players_vocab_size']}")
    print(f"  Player pad token id: {vocab['player_pad_token_id']}")
    return df, vocab


def build_datasets_direct(df, vocab, args):
    """Build train/eval datasets with on-the-fly masking (DataCollatorMPP)."""
    ds_full = MatchDatasetMPP(
        df,
        player_name2id=vocab["player_name2id"],
        team_name2id=vocab["team_name2id"],
        max_seq_length=args.max_seq_length,
        player_pad_token_id=vocab["player_pad_token_id"],
        team_pad_token_id=vocab["team_pad_token_id"],
        position_pad_token_id=25,
    )

    n = len(ds_full)
    n_val = max(1, int(n * args.dev_ratio))
    n_train = n - n_val

    np.random.seed(args.seed)
    indices = np.random.permutation(n)
    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:].tolist()

    train_dataset = Subset(ds_full, train_idx)
    eval_dataset = Subset(ds_full, val_idx)

    collator = DataCollatorMPP(
        player_mask_token_id=vocab["player_mask_token_id"],
        mask_percentage=args.mask_percentage,
    )

    print(f"  Train matches: {n_train}, Eval matches: {n_val}")
    return train_dataset, eval_dataset, collator


def build_datasets_precollated(df, vocab, args):
    """Build pre-collated train/eval datasets (kaggle approach)."""
    ds_full = MatchDatasetMPP(
        df,
        player_name2id=vocab["player_name2id"],
        team_name2id=vocab["team_name2id"],
        max_seq_length=args.max_seq_length,
        player_pad_token_id=vocab["player_pad_token_id"],
        team_pad_token_id=vocab["team_pad_token_id"],
        position_pad_token_id=25,
    )

    collator = DataCollatorMPP(
        player_mask_token_id=vocab["player_mask_token_id"],
        mask_percentage=args.mask_percentage,
    )

    def _collate_filter_none(batch):
        batch = [b for b in batch if b is not None]
        return collator(batch) if batch else None

    dataloader = DataLoader(
        ds_full,
        batch_size=args.precollate_batch_size,
        shuffle=True,
        collate_fn=_collate_filter_none,
        drop_last=True,
    )

    print(f"  Building pre-collated batches (repeat={args.precollate_repeat})...")
    all_batches = []
    for _ in range(args.precollate_repeat):
        for batch in dataloader:
            if batch is not None:
                all_batches.append(batch)

    np.random.seed(args.seed)
    n_batches = len(all_batches)
    dev_size = max(1, int(n_batches * args.dev_ratio))
    dev_idx = set(np.random.choice(n_batches, size=dev_size, replace=False).tolist())
    train_batches = [all_batches[i] for i in range(n_batches) if i not in dev_idx]
    dev_batches = [all_batches[i] for i in range(n_batches) if i in dev_idx]

    train_dataset = PreCollatedDataset(train_batches)
    eval_dataset = PreCollatedDataset(dev_batches)
    trainer_collator = DataCollatorPreCollated()

    print(f"  Total batches: {n_batches}, Train: {len(train_batches)}, Eval: {len(dev_batches)}")
    return train_dataset, eval_dataset, trainer_collator


def build_model(args, vocab):
    """Instantiate the model based on --model flag."""
    players_vocab_size = vocab["player_pad_token_id"]
    teams_vocab_size = vocab["team_pad_token_id"]

    if args.model == "transformer":
        from models.pretrain import MaskedPlayerModel
        model = MaskedPlayerModel(
            embed_size=args.embed_size,
            num_layers=args.num_layers,
            heads=args.heads,
            forward_expansion=args.forward_expansion,
            dropout=args.dropout,
            form_stats_size=39,
            players_vocab_size=players_vocab_size,
            teams_vocab_size=teams_vocab_size,
            positions_vocab_size=25,
            use_teams_embeddings=args.use_teams_embeddings,
            position_enc_type=args.position_enc_type,
        )
    elif args.model == "mlp_baseline":
        from baselines.MLP_baseline.MLP_baseline import MLPMaskedPlayerModel
        mlp_layers = args.mlp_num_layers if args.mlp_num_layers is not None else args.num_layers
        model = MLPMaskedPlayerModel(
            embed_size=args.embed_size,
            num_layers=mlp_layers,
            forward_expansion=args.mlp_hidden_mult,
            dropout=args.dropout,
            form_stats_size=39,
            players_vocab_size=players_vocab_size,
            teams_vocab_size=teams_vocab_size,
            positions_vocab_size=25,
            use_teams_embeddings=args.use_teams_embeddings,
            position_enc_type=args.position_enc_type,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model


def main():
    args = parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Data
    df, vocab = load_data(args)

    # Datasets
    if args.precollate:
        train_dataset, eval_dataset, collator = build_datasets_precollated(df, vocab, args)
        effective_batch_size = 1  # each item is already a full batch
    else:
        train_dataset, eval_dataset, collator = build_datasets_direct(df, vocab, args)
        effective_batch_size = args.batch_size

    # Model
    model = build_model(args, vocab)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")

    # Training config
    training_config = {
        "output_dir": args.output,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": effective_batch_size,
        "per_device_eval_batch_size": effective_batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.scheduler,
        "logging_steps": args.logging_steps,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,
        "seed": args.seed,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy_top1",
        "greater_is_better": True,
    }

    train_args = build_training_args(training_config)

    trainer = build_trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_mpp,
        data_collator=collator,
    )

    # Train
    print("\nStarting training...")
    best_model_dir = Path(args.output) / "best_model"
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Save best model (trainer loads best model at end if load_best_model_at_end=True)
        trainer.save_model(str(best_model_dir))
        print(f"Best model saved to: {best_model_dir}")

    # Final evaluation
    print("\nFinal evaluation on validation set:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Save metrics log
    if trainer.state.log_history:
        metrics_path = Path(args.output) / "metrics.csv"
        pd.DataFrame(trainer.state.log_history).to_csv(metrics_path, index=False)
        print(f"Training metrics saved to: {metrics_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
