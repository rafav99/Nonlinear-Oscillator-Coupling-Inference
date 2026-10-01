from __future__ import annotations

import numpy as np


def per_target_sre(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    scale = np.std(y_true, axis=0, ddof=0)
    return rmse / np.maximum(scale, 1e-12)


def mean_sre(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(per_target_sre(y_true, y_pred)))
