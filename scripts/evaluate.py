"""
CLI: Evaluate a trained model and compare with baselines.

Usage:
    python scripts/evaluate.py --config configs/finetune_nmsp.yaml \
        --checkpoint outputs/nmsp/best_model --baseline average

Supports:
    - NMSP evaluation: MSE, RMSE, dispersion coefficient (Table 2-3 of paper).
    - MPP evaluation: top-1, top-3 accuracy (Table 1).
    - Baseline comparison: average-of-5 baseline for NMSP.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["average", "pca", None],
                        help="Baseline to compare against")
    parser.add_argument("--split", type=str, default="val",
                        choices=["train", "val"])
    args = parser.parse_args()

    ...


if __name__ == "__main__":
    main()
