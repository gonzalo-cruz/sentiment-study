"""Per-language temperature scaling for confidence calibration."""

import numpy as np


def fit_temperature_scaler(
    logits: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Fit a single temperature parameter on validation logits.

    Args:
        logits: Raw model logits, shape (n_samples, n_classes).
        labels: Ground-truth class indices, shape (n_samples,).

    Returns:
        Optimal temperature T > 0.

    Raises:
        NotImplementedError: Calibration not yet implemented (Phase 3).
    """
    raise NotImplementedError("Temperature scaling — Phase 3")
