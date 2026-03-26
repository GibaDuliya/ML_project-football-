# Launch guide

How to run **MPP pre-training** (`run/run_mpp.py`) locally or in Docker, where to put data, and how evaluation works.

---

## 1. Data

Training expects a **raw match CSV** compatible with `data/preprocessing.py` (see project configs under `configs/` for schema notes).

1. Create the folder at the **repository root** (if it does not exist):

   ```bash
   mkdir -p dataset outputs
   ```

2. Copy your CSV into `dataset/`, e.g. `dataset/data_with_dates.csv`.

3. Point `--data` to that file (host path when running locally; `/app/dataset/...` inside the container when using Docker Compose as configured).

---

## 2. Local Python (no Docker)

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run training:

```bash
python run/run_mpp.py \
  --data dataset/data_with_dates.csv \
  --output outputs/mpp_run \
  --epochs 10
```

See all options:

```bash
python run/run_mpp.py --help
```

**Device:** the script uses CUDA if available, otherwise CPU. On Apple Silicon you can run locally with MPS only if you install PyTorch with MPS support and adjust the code; the stock `requirements.txt` + Docker path is CPU-oriented inside Linux containers.

---

## 3. Docker (CPU)

### Build

```bash
docker compose build
```

### Run training (override the default compose command)

The compose file defines a default `command`; override it for your CSV and hyperparameters:

```bash
docker compose run --rm train python run/run_mpp.py \
  --data /app/dataset/data_with_dates.csv \
  --output /app/outputs/mpp_docker \
  --epochs 10
```

**Smoke test (small model, one epoch)** — useful on Mac (CPU in container):

```bash
docker compose run --rm train python run/run_mpp.py \
  --data /app/dataset/data_with_dates.csv \
  --output /app/outputs/docker_smoke \
  --epochs 1 \
  --embed_size 32 \
  --num_layers 1 \
  --batch_size 16 \
  --eval_steps 20 \
  --logging_steps 20 \
  --save_steps 500
```

Artifacts appear under **`./outputs/...`** on the host (bind-mounted to `/app/outputs`).

### Docker image without Compose

```bash
docker build -t ml-football-train .
docker run --rm \
  -v "$(pwd)/dataset:/app/dataset" \
  -v "$(pwd)/outputs:/app/outputs" \
  ml-football-train \
  python run/run_mpp.py --data /app/dataset/data_with_dates.csv --output /app/outputs/mpp_run --epochs 10
```

---

## 4. Docker on macOS: `operation not permitted` on volume mount

If you see errors like:

`mkdir /host_mnt/Users/.../Documents: operation not permitted`

1. Create `dataset/` and `outputs/` on the host before running Compose (see §1).
2. **Docker Desktop → Settings → Resources → File sharing:** add your project path, or `/Users/<you>`, or `/Users/<you>/Documents`.
3. **System Settings → Privacy & Security → Full Disk Access:** enable **Docker** / **Docker Desktop**.
4. If it still fails, move the repo out of `Documents` (e.g. `~/dev/...`) and retry.

---

## 5. Docker (GPU)

Requires an NVIDIA GPU, drivers, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the **host** (Linux typical; not Apple Silicon).

Build and run:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm train \
  python run/run_mpp.py \
  --data /app/dataset/data_with_dates.csv \
  --output /app/outputs/mpp_gpu \
  --epochs 10
```

---

## 6. Evaluation (`eval`)

There is **no separate “eval-only” CLI** in this repo for MPP.

`run/run_mpp.py` already:

- runs **validation during training** (`eval_strategy: "steps"`, controlled by `--eval_steps`);
- runs a **final validation** after training (`trainer.evaluate()`), printed as **“Final evaluation on validation set”** with metrics (e.g. accuracy).

So **the same Docker/local command that runs training** performs eval at the end (and at intermediate steps). To “eval again” without a dedicated script, you would need a small custom script that loads `best_model` from `--output` and calls `trainer.evaluate()` — not shipped in this repository.

Checkpoints and the best model are written under your `--output` directory (e.g. `best_model/`).

---

## 7. Other workflows

- **SoFIFA / next-year rating (Jupyter):** the notebook `notebooks/next_year_rating_prediction/finetune_rating_next_year.ipynb` trains **downstream heads** on top of a **frozen** pre-trained encoder. It loads checkpoints from `mpp_mini_output/` (or your own), builds **aggregated embeddings** per **(player, season)** from match data filtered like `eda_data_with_dates.ipynb` (e.g. calendar years 2015–2016), and maps targets from **`dataset/sofifa_ratings_by_season.csv`** (`player_name`, `rating_year`, `overall`). The notebook compares three classifiers on **8 rating bins** (5-point bands from 50–90): a small PyTorch head (`ClassificationHead` + `RatingBinHeadTrainer`), **AdaBoost**, and **XGBoost**; metrics and saved models go under `notebooks/rating_head_output/next_year/` (or paths set in the notebook). Run it with a local Jupyter/venv environment — the default **Docker** image is set up for `run/run_mpp.py`, not for these notebooks.

- **Dependencies:** see `requirements.txt`. Optional parser features may need extra setup (e.g. Selenium for JS-rendered pages in `parsers/sofifa_by_year.py`).

---

## 8. Quick reference

| Goal | Command |
|------|--------|
| CLI help | `python run/run_mpp.py --help` |
| Local train | `python run/run_mpp.py --data dataset/your.csv --output outputs/run` |
| Docker build | `docker compose build` |
| Docker train | `docker compose run --rm train python run/run_mpp.py --data /app/dataset/your.csv --output /app/outputs/run ...` |
