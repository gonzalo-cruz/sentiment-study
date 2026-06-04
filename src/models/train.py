"""XLM-R fine-tuning with HuggingFace Trainer and MLflow tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import (
    BASE_MODEL_ID,
    DEFAULT_CHECKPOINT_DIR,
    FROZEN_ENCODER_LAYERS,
    MAX_SEQUENCE_LENGTH,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LEARNING_RATE,
    TRAIN_SEED,
    TRAIN_WARMUP_RATIO,
)
from data.loaders import load_training_dataset, load_validation_datasets

logger = logging.getLogger(__name__)


def freeze_bottom_layers(model: AutoModelForSequenceClassification, n_frozen: int) -> None:
    """Freeze embeddings and the bottom ``n_frozen`` encoder layers.

    Args:
        model: XLM-R sequence classification model.
        n_frozen: Number of lower encoder layers to freeze.
    """
    for parameter in model.roberta.embeddings.parameters():
        parameter.requires_grad = False
    for layer_index in range(n_frozen):
        for parameter in model.roberta.encoder.layer[layer_index].parameters():
            parameter.requires_grad = False


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute accuracy and macro-F1 for Trainer evaluation.

    Args:
        eval_pred: Tuple of logits and labels from ``Trainer.evaluate``.

    Returns:
        Metric dictionary with ``accuracy`` and ``f1``.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> Dataset:
    """Tokenize a dataset with ``text`` and ``label`` columns.

    Args:
        dataset: Source dataset with string texts and integer labels.
        tokenizer: Loaded tokenizer matching the base model.
        max_length: Maximum sequence length.

    Returns:
        Tokenized dataset ready for ``Trainer``.
    """
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize, batched=True, desc="tokenize")
    tokenized = tokenized.rename_column("label", "labels")
    columns_to_keep = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in tokenized.column_names:
        columns_to_keep.add("token_type_ids")
    drop_cols = [col for col in tokenized.column_names if col not in columns_to_keep]
    return tokenized.remove_columns(drop_cols)


def evaluate_validation_sets(
    trainer: Trainer,
    validation_sets: dict[str, Dataset],
) -> dict[str, float]:
    """Evaluate the trainer on each language-specific validation split.

    Args:
        trainer: Fitted or in-progress ``Trainer`` instance.
        validation_sets: Mapping of language code to tokenized validation data.

    Returns:
        Flat dictionary of per-language metrics.
    """
    metrics: dict[str, float] = {}
    for language, dataset in validation_sets.items():
        result = trainer.evaluate(eval_dataset=dataset)
        for key, value in result.items():
            if isinstance(value, (int, float)):
                metrics[f"val_{language}_{key}"] = float(value)
    return metrics


def train_model(
    output_dir: Path | None = None,
    *,
    num_epochs: int = TRAIN_EPOCHS,
    learning_rate: float = TRAIN_LEARNING_RATE,
    batch_size: int = TRAIN_BATCH_SIZE,
    en_subsample: int | None = None,
    zh_max_rows: int | None = None,
    max_steps: int | None = None,
    experiment_name: str = "sentiment-study",
    run_name: str | None = None,
) -> Path:
    """Fine-tune xlm-roberta-base on mixed ZH/EN training data.

    Args:
        output_dir: Directory for checkpoints and MLflow artifacts.
        num_epochs: Training epochs.
        learning_rate: AdamW learning rate.
        batch_size: Per-device train/eval batch size.
        en_subsample: Optional cap on English training rows.
        zh_max_rows: Optional cap on Chinese training rows.
        max_steps: Optional step cap for smoke tests.
        experiment_name: MLflow experiment name.
        run_name: Optional MLflow run name.

    Returns:
        Path to the best saved checkpoint directory.
    """
    checkpoint_dir = output_dir or DEFAULT_CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs: dict[str, int | None] = {}
    if en_subsample is not None:
        train_kwargs["en_subsample"] = en_subsample
    if zh_max_rows is not None:
        train_kwargs["zh_max_rows"] = zh_max_rows

    train_raw = load_training_dataset(**train_kwargs)
    validation_raw = load_validation_datasets()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    train_tokenized = tokenize_dataset(train_raw, tokenizer)
    validation_tokenized = {
        lang: tokenize_dataset(dataset, tokenizer)
        for lang, dataset in validation_raw.items()
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=2,
    )
    freeze_bottom_layers(model, FROZEN_ENCODER_LAYERS)

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        fp16=use_fp16,
        warmup_ratio=TRAIN_WARMUP_RATIO,
        eval_strategy="epoch" if max_steps is None else "steps",
        eval_steps=50 if max_steps is not None else None,
        save_strategy="epoch" if max_steps is None else "steps",
        save_steps=50 if max_steps is not None else None,
        load_best_model_at_end=max_steps is None,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=25,
        save_total_limit=2,
        seed=TRAIN_SEED,
        report_to="none",
    )

    combined_validation = validation_tokenized["zh"]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=combined_validation,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_id": BASE_MODEL_ID,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "frozen_layers": FROZEN_ENCODER_LAYERS,
                "fp16": use_fp16,
                "train_rows": len(train_tokenized),
                "en_subsample": en_subsample,
                "zh_max_rows": zh_max_rows,
                "max_steps": max_steps,
            }
        )

        trainer.train()
        val_metrics = evaluate_validation_sets(trainer, validation_tokenized)
        mlflow.log_metrics(val_metrics)

        export_dir = checkpoint_dir / "best"
        export_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(export_dir))
        tokenizer.save_pretrained(str(export_dir))
        mlflow.log_artifact(str(export_dir / "config.json"))

        logger.info("Training complete. Exported checkpoint: %s", export_dir)
        logger.info("Validation metrics: %s", val_metrics)
        return export_dir
