# Comparative Analysis of RisingBALLER and gMLP for Football Player Representation Learning

**Team RisingBALLER** — Skoltech Machine Learning 2026 Course Final Project

| Member | Contribution |
|--------|-------------|
| Ajnaz Gibadullin | Core model code (attention, gMLP), ablation experiments, repo setup (40%) |
| Vasilii Liamin | Data pipeline, MPP training/evaluation, SoFIFA downstream pipeline (40%) |
| Alexander Konkin | Auxiliary experiments, final report writing and editing (20%) |

---

## Overview

This project learns **contextual football player representations** via **Masked Player Prediction (MPP)** — treating each match as a sequence of player tokens and training the model to reconstruct masked player identities from the surrounding match context.

We reproduce the [RisingBALLER](https://statsbomb.com/) transformer-based framework and compare it against a **gMLP** alternative that replaces self-attention with static spatial token mixing through a Spatial Gating Unit (SGU). Along the way we identify and correct a **data leakage issue** in the original RisingBALLER evaluation protocol (augmenting before splitting), yielding more realistic metrics.

### Key Findings

- **Data leakage correction** drops RisingBALLER Top-3 accuracy from the originally reported >95% to ~66%.
- **gMLP consistently outperforms** all RisingBALLER variants on MPP under the corrected protocol.
- **Transfer to SoFIFA rating prediction** remains modest (~40% accuracy with 5-point bins) for all models, indicating that strong MPP performance does not imply reliable global player quality prediction.

## Results

### Masked Player Prediction (corrected protocol, no team embeddings)

| Model | Eval Loss | Top-1 Acc | Top-3 Acc |
|-------|-----------|-----------|-----------|
| RB-128 sinusoidal | 3.386 | 0.376 | 0.663 |
| RB-128 learned | 2.812 | 0.322 | 0.624 |
| RB-64 sinusoidal | 3.111 | 0.287 | 0.573 |
| RB-64 learned | 3.368 | 0.300 | 0.582 |
| **gMLP-128** | **2.218** | **0.606** | **0.881** |
| gMLP-64 | 2.145 | 0.584 | 0.858 |

### SoFIFA Next-Year Rating Classification (season-averaged embeddings)

| Model | NN | AdaBoost | XGBoost |
|-------|----|----------|---------|
| gMLP-64 | **0.401** | 0.254 | 0.349 |
| RB-128 learned | 0.394 | 0.238 | 0.378 |
| gMLP-128 | 0.391 | 0.221 | 0.322 |
| RB-128 sinusoidal | 0.391 | 0.238 | 0.352 |

## Data

We use publicly available **StatsBomb open event data** for the 2015–2016 season across the top five European leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1), providing ~1,000 matches. Each match is transformed into a player-centric representation with **39 aggregated event statistics** covering passing, shooting, interceptions, dribbling, fouls, goalkeeping, and more.

Each match is augmented 10x with different random mask configurations. Under our corrected protocol, the raw match list is split into train/validation **before** augmentation, ensuring no match leaks across splits.

## Methods

### RisingBALLER (Transformer)

Each player token combines a player identity embedding, projected match statistics, and a positional embedding (sinusoidal or learned). The sequence is processed by a single transformer layer with 2 attention heads and shared Q/K/V projections.

### gMLP

Player tokens are placed into a **fixed 50-slot sequence** (slots 0–24 for team 0, slots 25–49 for team 1), encoding positional and team information implicitly through slot ordering. Two gMLP blocks with Spatial Gating Units learn a static 50x50 token-mixing matrix — directly interpretable as a positional interaction map.

### Downstream Transfer

Frozen pretrained encoders produce match-level embeddings averaged per player-season. A neural network head, AdaBoost, and XGBoost are evaluated on 5-point SoFIFA rating bin classification.

## Project Structure

```
├── configs/                 # YAML configs for data, pretraining, and finetuning
├── data/                    # Dataset classes, collators, preprocessing, SoFIFA utils
├── evaluation/              # Embedding extraction and similarity utilities
├── models/
│   ├── gmlp/                # gMLP encoder, block, and pretrain wrapper
│   └── transformer/         # Transformer encoder, attention, and pretrain wrapper
├── notebooks/
│   ├── gMLP/                # gMLP MPP training/testing notebooks
│   ├── RisingBaller/        # RisingBALLER MPP training/testing notebooks
│   ├── next_year_rating_prediction/   # SoFIFA downstream task notebook
│   └── dataset_exps/        # Data exploration and preprocessing experiments
├── parsers/                 # SoFIFA web scraping utilities
├── run/
│   └── run_mpp.py           # Main MPP pretraining entry point
├── training/                # Trainer, callbacks, metrics
├── requirements.txt
├── Dockerfile / Dockerfile.gpu
├── docker-compose.yml / docker-compose.gpu.yml
└── Launch.md                # Setup and run instructions
```

## Getting Started

For detailed **setup, data preparation, and run instructions** (local Python, Docker CPU/GPU, evaluation, downstream notebooks), see **[Launch.md](Launch.md)**.

Quick start:

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Prepare data
mkdir -p dataset outputs
# Place your StatsBomb CSV into dataset/

# 3. Run MPP pretraining
python run/run_mpp.py \
  --data dataset/data_with_dates.csv \
  --output outputs/mpp_run \
  --epochs 10
```

## Technologies

- **PyTorch** + HuggingFace **Transformers** / **Accelerate** for model training
- **scikit-learn**, **XGBoost** for downstream classifiers
- **TensorBoard** for training visualization
- **Docker** / Docker Compose for reproducible environments
- **StatsBomb Open Data** as the primary data source
- **SoFIFA** ratings for downstream evaluation

## References

1. Adjileye, A. A. *RisingBALLER: A player is a token, a match is a sentence.* StatsBomb Conference 2024.
2. Liu, H. et al. *Pay Attention to MLPs.* NeurIPS 2021.
3. Devlin, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL-HLT 2019.
4. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
5. [StatsBomb Open Data](https://github.com/statsbomb/open-data)
6. [SoFIFA Player Ratings](https://sofifa.com)

## License

This project was developed as part of the Skoltech Machine Learning 2026 course.
