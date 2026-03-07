"""
CLI: Preprocess raw CSV → processed dataset + vocabulary mappings.

Usage:
    python scripts/preprocess.py --csv dataset/raw.csv --output dataset/processed/
    python scripts/preprocess.py --csv dataset/raw.csv --output dataset/processed/ \
        --seasons "2015/2016" "2016/2017"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocessing import preprocess_raw_csv, build_vocab_mappings


def main():
    parser = argparse.ArgumentParser(description="Preprocess StatsBomb CSV data")
    parser.add_argument("--csv", type=str, required=True, help="Path to raw CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--seasons", nargs="*", default=None, help="Filter by season names")
    parser.add_argument("--competitions", nargs="*", default=None, help="Filter by competition names")
    args = parser.parse_args()

    ...


if __name__ == "__main__":
    main()
