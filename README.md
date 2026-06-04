# Cross-lingual Cultural Sentiment Study

Quantitative study of how Chinese, Spanish, and English online communities express sentiment toward global tech products — with explicit methodology to separate genuine cultural difference from model bias.

## Research questions

- **RQ1:** Do ZH/ES/EN communities express systematically different sentiment toward shared tech topics?
- **RQ2:** How much observed asymmetry is cultural vs. model bias (back-translation, gold sets, LLM baseline, calibration)?
- **RQ3:** How does Chinese subword tokenization affect sentiment predictions?

## Stack

Python 3.10+, PyTorch, XLM-RoBERTa, HuggingFace, Polars, FastAPI, Streamlit, MLflow.

## Setup

```bash
uv sync
```

## Phase 1 — dataset inspection (complete)

Baseline corpora (4 total: 2 train ZH/EN + 2 validation ZH/EN):

| Corpus | HF ID (inspection) | Role |
|--------|-------------------|------|
| weibo_senti_100k | `dirtycomputer/weibo_senti_100k` | ZH train |
| sentiment140 | `Roh2014/sentiment140_10k_tweets` (Phase 2: 100k from `adilbekovich/Sentiment140Twitter`) | EN train |
| ChnSentiCorp | `lansinuote/ChnSentiCorp` | ZH validation |
| SST-2 | `nyu-mll/glue` (config: `sst2`) | EN validation |

Spanish is **zero-shot** — no labeled ES training data. Peninsular ES validation uses study corpora in Phase 2.

### Cache XLM-R weights (run once, before travel)

```bash
python scripts/cache_xlmr.py
python scripts/cache_xlmr.py --verify-only
```

### Verify Phase 1 setup

```bash
# Offline checks: directories + cached model
python scripts/verify_phase1.py

# Include HuggingFace dataset smoke loads (requires network)
python scripts/verify_phase1.py --online
```

### Run baseline EDA notebook

```bash
jupyter notebook notebooks/01_baseline_datasets.ipynb
```

### Run tests

```bash
pytest tests/ -q
```

## Phase 2 — data pipeline and model (ready)

### 1. Collect Reddit study corpora

Requires Reddit API credentials in the environment:

```bash
export REDDIT_CLIENT_ID=...
export REDDIT_CLIENT_SECRET=...
export REDDIT_USER_AGENT="sentiment-study/0.1 by your_username"

python scripts/collect_reddit.py --language en --limit-per-subreddit 500
python scripts/collect_reddit.py --language es --limit-per-subreddit 500
```

Raw output: `data/raw/reddit/reddit_{en,es}.parquet`

### 2. Preprocess study corpora

```bash
python scripts/preprocess_corpus.py --input data/raw/reddit/reddit_en.parquet --lang en
python scripts/preprocess_corpus.py --input data/raw/reddit/reddit_es.parquet --lang es
```

Weibo ZH study pool is built from the training corpus via `load_weibo_study_corpus()` in code.

### 3. Fine-tune XLM-R (ZH + EN, zero-shot ES)

```bash
python scripts/train_model.py --output-dir models/xlmr-sentiment
```

Smoke test (tiny sample, 5 steps):

```bash
python scripts/smoke_phase2.py
```

Checkpoints export to `{output_dir}/best/` with MLflow metrics logged under experiment `sentiment-study`.

### 4. Batch inference

```bash
python scripts/run_inference.py \
  --input data/processed/study/study_en_processed.parquet \
  --model-path models/xlmr-sentiment/best \
  --output data/outputs/predictions/study_en_predictions.parquet
```

Output schema: `text, lang, topic, sentiment_label, sentiment_score, confidence, timestamp`.

### Where artifacts are saved

| Step | Script | Output path |
|------|--------|-------------|
| Reddit scrape | `collect_reddit.py` | `data/raw/reddit/reddit_{en,es}.parquet` |
| Preprocess | `preprocess_corpus.py` | `data/processed/study/{name}_processed.parquet` |
| Train | `train_model.py` | `models/xlmr-sentiment/best/` (weights + tokenizer) |
| Inference | `run_inference.py` | `data/outputs/predictions/{name}_predictions.parquet` |
| MLflow metrics | (during train) | `mlruns/` at repo root |
| Smoke test | `smoke_phase2.py` | `models/smoke/best/`, `data/processed/study/smoke_zh.parquet`, `data/outputs/predictions/smoke_zh_predictions.parquet` |

The smoke test previously used a **temp directory** (`/tmp/...`) that was deleted on exit — that is why folders looked empty. Re-run `python scripts/smoke_phase2.py` to populate the paths above, or pass `--ephemeral` for the old isolated behaviour.

**IDE note:** imports resolve from `src/` via `pyrightconfig.json` and `.vscode/settings.json`. Reload the window if squiggles remain.

## Project layout

```
nlp/
├── src/
│   ├── data/          # loaders, preprocessing, back-translation
│   ├── models/        # train, predict, calibration
│   └── analysis/      # distributions, stats, disentangle, tokenization
├── data/raw|processed|gold|outputs/
├── tests/
├── api/
├── dashboard/
├── notebooks/
└── report/
```

## Phases

1. **Madrid (May–Jun):** reading, repo setup, dataset inspection, model weights cache
2. **Madrid (Jun):** data pipeline, fine-tuning, inference parquet export
3. **China (Jul–Aug):** manual PyTorch loop, tests, disentanglement, analysis, dashboard
4. **Post-China (Sep+):** aspect-based extension, paper submission

## Offline mode (pre-departure)

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```
