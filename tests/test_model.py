"""Tests for training utilities."""

import numpy as np
import pytest
import torch
from transformers import AutoModelForSequenceClassification

from config import BASE_MODEL_ID, FROZEN_ENCODER_LAYERS
from models.train import compute_metrics, freeze_bottom_layers


def test_compute_metrics_returns_f1_and_accuracy() -> None:
    logits = np.array([[2.0, 0.5], [0.1, 3.0], [1.0, 0.2]])
    labels = np.array([0, 1, 0])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_freeze_bottom_layers_disables_gradients() -> None:
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_ID, num_labels=2)
    freeze_bottom_layers(model, FROZEN_ENCODER_LAYERS)

    embedding_grad = next(model.roberta.embeddings.parameters()).requires_grad
    lower_layer_grad = next(
        model.roberta.encoder.layer[0].parameters()
    ).requires_grad
    upper_layer_grad = next(
        model.roberta.encoder.layer[FROZEN_ENCODER_LAYERS].parameters()
    ).requires_grad

    assert embedding_grad is False
    assert lower_layer_grad is False
    assert upper_layer_grad is True
