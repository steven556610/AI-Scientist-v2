"""
metrics.py — Evaluation functions for Anticipatory Fusion experiments.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error
from typing import Tuple


def compute_auroc_auprc(
    probs: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    """Compute AUROC and AUPRC for binary classification."""
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)

    if labels.sum() == 0 or labels.sum() == len(labels):
        # Degenerate case
        return 0.5, float(labels.mean())

    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    return float(auroc), float(auprc)


def compute_regression_metrics(
    preds: np.ndarray, targets: np.ndarray
) -> Tuple[float, float]:
    """Compute R² and MAE for regression."""
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    return float(r2), float(mae)


def compute_detection_map(
    scores: np.ndarray,
    gt_labels: np.ndarray,
    window: str = "onset",
) -> float:
    """
    Proxy mAP metric: mean of per-sequence detection scores during onset window.

    scores    : (N,) per-sequence fusion score
    gt_labels : (N, T) ground-truth labels (proxy)
    window    : 'onset' → evaluate at T//2 to T//2 + 10; 'full' → all T
    """
    N, T = gt_labels.shape
    mid = T // 2
    end = min(mid + 10, T)

    if window == "onset":
        gt_slice = gt_labels[:, mid:end].mean(axis=1)  # (N,)
    else:
        gt_slice = gt_labels.mean(axis=1)

    scores = np.asarray(scores, dtype=float)
    gt_slice = np.asarray(gt_slice, dtype=float)

    # Compute correlation as a proxy mAP
    correlation = np.corrcoef(scores, gt_slice)[0, 1]
    # Normalise to [0, 1] range
    proxy_map = (correlation + 1) / 2
    return float(proxy_map)
