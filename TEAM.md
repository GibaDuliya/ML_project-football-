# Распределение задач: эмбеддинги футболистов

Проект разбит на **3 роли** и **2 волны**. В первой волне все работают параллельно (после согласования контрактов ниже). Во второй волне подключаются скрипты и оценка, с учётом зависимостей.

---

## Контракты между ролями (согласовать в начале)

- **Формат обработанных данных:** выход `preprocess_raw_csv` — CSV в `configs/data.yaml → processed_dir`, колонки по `id_columns` + 39 стат-колонок. Метаданные (словари имя↔id) — в `metadata_dir` в виде pickle (`player_name2id`, `id2player_name`, `team_name2id`, `id2team_name` и т.д.).
- **Словари:** после препроцессинга в конфиг (или отдельный маленький yaml) подставляются `players_vocab_size`, `teams_vocab_size`, `player_mask_token_id`, `player_pad_token_id`, `team_pad_token_id`.
- **Датасеты:** `MatchDatasetMPP` и `MatchDatasetNMSP` отдают объекты, которые коллаторы превращают в батчи; формат ключей (например `input_ids`, `stats`, `labels`) — по docstring в `data/dataset.py` и `data/collator.py`.

---

## Роль 1 — Data Pipeline (препроцессинг и утилиты)

**Цель:** сырые CSV → обработанные данные + словари; утилиты для паддинга и масок.

### Волна 1 (параллельно с остальными)


| Файл                    | Задачи                                                                                                               |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `data/preprocessing.py` | `preprocess_raw_csv`, `build_vocab_mappings`, `_validate_columns`, `_save_pickle`, `_load_pickle`                    |
| `data/utils.py`         | `pad_sequence_1d`, `pad_sequence_2d`, `build_attention_mask`, `aggregate_player_stats_for_nmsp`, `custom_collate_fn` |
| `scripts/preprocess.py` | Реализовать `main()`: загрузка конфига, вызов препроцессинга, сохранение в `processed_dir` и `metadata_dir`          |


**Результат волны 1:** можно запустить `python scripts/preprocess.py` и получить `dataset/processed/` и `dataset/metadata/`, обновлённые размеры словарей в конфиге (или отдельном файле).

### Волна 2

- Проверка пайплайна на реальных данных, правки по обратной связи от Роли 2 и 3.
- При необходимости: небольшая документация по формату обработанных данных в README или в комментариях в `configs/data.yaml`.

---

## Роль 2 — Models & Training (модели и обучение)

**Цель:** энкодер игроков, внимание, головы MPP/NMSP/classification/regression, претренинг и файнтюнинг, тренировер и метрики.

### Волна 1 (параллельно с остальными)


| Файл                  | Задачи                                                                                                                                               |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `models/encoder.py`   | `SinusoidalEncoding`, `PlayerEncoder` (все методы: `_encode_player`, `_encode_stats`, `_encode_position`, `_encode_team`, `forward`, `get_*_weight`) |
| `models/attention.py` | `PlayerSelfAttention`, `PlayerTransformerBlock`                                                                                                      |
| `models/heads.py`     | `MPPHead`, `NMSPHead`, `ClassificationHead`, `RegressionHead`, `build_head`                                                                          |
| `models/pretrain.py`  | `MaskedPlayerModel` (`__init__`, `forward`, `get_encoder`)                                                                                           |
| `models/finetune.py`  | `DownstreamModel` (загрузка претрена, freeze/unfreeze, `forward`)                                                                                    |


**Результат волны 1:** модели собираются и считают forward по dummy-тензорам (без реальных данных).

### Волна 2 (после готовности датасетов и коллаторов от Роли 3)


| Файл                  | Задачи                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| `training/trainer.py` | `build_training_args`, `build_trainer` (интеграция с Transformers/Accelerate)                     |
| `training/metrics.py` | `compute_metrics_mpp`, `compute_metrics_nmsp`, `compute_dispersion_coefficient`                   |
| `scripts/pretrain.py` | `load_config`, `main()`: датасет MPP, коллатор, MaskedPlayerModel, trainer, сохранение чекпоинтов |
| `scripts/finetune.py` | `main()`: датасет NMSP (или другой downstream), DownstreamModel, trainer                          |


**Результат волны 2:** можно запустить претренинг и файнтюнинг по конфигам `configs/pretrain_mpp.yaml`, `configs/finetune_nmsp.yaml`, `configs/finetune_position.yaml`.

---

## Роль 3 — Datasets, Baselines & Evaluation (датасеты, бейзлайны, эвалюация)

**Цель:** датасеты и коллаторы под MPP/NMSP, бейзлайн, извлечение эмбеддингов и скрипты оценки.

### Волна 1 (параллельно с остальными)


| Файл               | Задачи                                                                                             |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| `data/dataset.py`  | `MatchDatasetMPP` (`__init__`, `__len__`, `__getitem__`), `MatchDatasetNMSP`, `PreCollatedDataset` |
| `data/collator.py` | `DataCollatorMPP` (`__init__`, `__call__`, `_mask_single_match`), `DataCollatorNMSP.__call__`      |


Ориентир по формату данных — `configs/data.yaml` и контракты выше (формат processed CSV и metadata).

**Результат волны 1:** из `dataset/processed/` и `metadata/` можно итерировать батчи MPP и NMSP через DataLoader.

### Волна 2 (после готовности чекпоинтов от Роли 2)


| Файл                            | Задачи                                                                                                    |
| ------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `baselines/average_baseline.py` | `AverageBaseline` (`__init__`, `predict`, `evaluate`)                                                     |
| `evaluation/embeddings.py`      | `extract_embeddings`, `cluster_positions`, `plot_positions_on_pitch`, `plot_player_position_fit`          |
| `evaluation/similarity.py`      | `cosine_similarity_matrix`, `find_similar_players`, `compute_team_cohesion`, `plot_dissimilarity_heatmap` |
| `scripts/evaluate.py`           | `main()`: загрузка модели/чекпоинта, сравнение с бейзлайном, вывод метрик                                 |
| `scripts/extract_embeddings.py` | `main()`: загрузка чекпоинта, сохранение эмбеддингов (например, в `.npy` или CSV)                         |


**Результат волны 2:** скрипты `evaluate.py` и `extract_embeddings.py` работают поверх обученных моделей; визуализации и метрики схожести доступны из `evaluation/`.

---

## Сводка по волнам


| Волна | Роль 1                                | Роль 2                                         | Роль 3                                                                    |
| ----- | ------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------- |
| **1** | preprocessing, utils, `preprocess.py` | encoder, attention, heads, pretrain, finetune  | dataset, collator                                                         |
| **2** | проверка пайплайна, док               | trainer, metrics, `pretrain.py`, `finetune.py` | baselines, embeddings, similarity, `evaluate.py`, `extract_embeddings.py` |


Зависимости: волна 2 у Роли 2 и Роли 3 опирается на выход волны 1 (данные + словари у Роли 1, батчи у Роли 3; чекпоинты от Роли 2 для скриптов Роли 3).

---

## Датасеты (напоминание)

- Сырые данные: `dataset/df_raw_counts_players_matches.csv` (уже есть в репозитории/диске).
- После препроцессинга: `dataset/processed/`, `dataset/metadata/` (создаёт Роль 1).
- Параметры: `team_max_players: 18`, `max_seq_length: 36`, `form_stats_size: 39` — в `configs/data.yaml`.

