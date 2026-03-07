"""
CLI: Fine-tune a pre-trained RisingBALLER encoder on a downstream task.

Usage:
    python scripts/finetune.py --config configs/finetune_nmsp.yaml
    python scripts/finetune.py --config configs/finetune_position.yaml

Steps:
    1. Load config (data + model + head + training).
    2. Load processed data.
    3. Create appropriate Dataset (NMSP, classification, etc.).
    4. Instantiate DownstreamModel with encoder config + head config.
    5. Load pre-trained encoder weights from checkpoint.
    6. Optionally freeze encoder.
    7. Train with HF Trainer.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Downstream fine-tuning")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    ...


if __name__ == "__main__":
    main()
