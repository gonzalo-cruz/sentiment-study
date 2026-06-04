"""Project configuration: paths, constants, and environment variables."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GOLD_DATA_DIR = DATA_DIR / "gold"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

MAX_SEQUENCE_LENGTH = 256
BASE_MODEL_ID = "xlm-roberta-base"
FROZEN_ENCODER_LAYERS = 6
TRAIN_SEED = 42

# Phase 1 notebook / smoke tests — keep small to avoid disk and RAM pressure
INSPECTION_SAMPLE_SIZE = 2_000
INSPECTION_SHUFFLE_SEED = 42

# Phase 2 training targets (see project plan)
SENTIMENT140_TRAIN_SUBSAMPLE = 100_000
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 3
TRAIN_LEARNING_RATE = 2e-5
TRAIN_WARMUP_RATIO = 0.1

# Phase 2 training: ZH + EN only. Spanish is zero-shot (manual validation in Phase 2).
TRAINING_DATASETS = {
    "zh": "dirtycomputer/weibo_senti_100k",
    "en": "adilbekovich/Sentiment140Twitter",
}

VALIDATION_DATASETS = {
    "zh": "lansinuote/ChnSentiCorp",
    "en": "nyu-mll/glue",  # config: sst2
}

# Phase 1 inspection mirrors (smaller / script-free where the plan ID is deprecated)
INSPECTION_DATASETS = {
    "zh_train": "dirtycomputer/weibo_senti_100k",
    "en_train": "Roh2014/sentiment140_10k_tweets",
    "zh_val": "lansinuote/ChnSentiCorp",
    "en_val": "nyu-mll/glue",  # config: sst2
}

# Study corpora — Reddit communities (peninsular ES scope documented in project plan)
REDDIT_EN_SUBREDDITS: tuple[str, ...] = ("apple", "technology", "ChatGPT")
REDDIT_ES_SUBREDDITS: tuple[str, ...] = ("es", "spain", "preguntareddit", "podemos")

SUBREDDIT_TOPICS: dict[str, str] = {
    "apple": "apple",
    "technology": "technology",
    "ChatGPT": "chatgpt",
    "es": "general",
    "spain": "general",
    "preguntareddit": "general",
    "podemos": "politics",
}

# Reddit API credentials (set in environment)
REDDIT_CLIENT_ID_ENV = "REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET_ENV = "REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT_ENV = "REDDIT_USER_AGENT"

DEFAULT_CHECKPOINT_DIR = MODELS_DIR / "xlmr-sentiment"
PROCESSED_CORPUS_DIR = PROCESSED_DATA_DIR / "study"
INFERENCE_OUTPUT_DIR = OUTPUT_DATA_DIR / "predictions"
