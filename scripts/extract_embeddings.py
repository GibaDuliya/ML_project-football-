"""
CLI: Extract and save player/position embeddings from a trained checkpoint.

Usage:
    python scripts/extract_embeddings.py \
        --config configs/pretrain_mpp.yaml \
        --checkpoint outputs/mpp/checkpoint-342000 \
        --output embeddings/

Saves:
    players_embeddings.npy   — (vocab_size, embed_size)
    positions_embeddings.npy — (n_positions, embed_size)
    teams_embeddings.npy     — (n_teams, embed_size)  [if use_teams_embeddings]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="embeddings/")
    args = parser.parse_args()

    ...


if __name__ == "__main__":
    main()
