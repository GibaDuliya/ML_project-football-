"""
CLI: Run Masked Player Prediction (MPP) pre-training.

Usage:
    python scripts/pretrain.py --config configs/pretrain_mpp.yaml

Steps:
    1. Load data config + training config from YAML.
    2. Load processed data and vocab mappings.
    3. Create MatchDatasetMPP + DataCollatorMPP.
    4. Build augmented dataset (repeat with different masks).
    5. Split into train/val.
    6. Instantiate MaskedPlayerModel.
    7. Build HF Trainer and train.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    ...


def main():
    parser = argparse.ArgumentParser(description="MPP Pre-training")
    parser.add_argument("--config", type=str, default="configs/pretrain_mpp.yaml")
    args = parser.parse_args()

    ...


if __name__ == "__main__":
    main()
