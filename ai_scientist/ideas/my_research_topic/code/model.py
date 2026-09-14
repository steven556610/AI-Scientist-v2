"""
model.py — Baseline and anticipatory fusion models.

All models use only numpy / sklearn for portability and speed.
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Information Value computation
# ---------------------------------------------------------------------------

def compute_modality_information_value(
    sequences: Dict, seed: int = 42
) -> Dict:
    """
    Approximate V_m(t) = marginal contribution of modality m.

    We use a lightweight proxy:
        V_m(t) ≈ 1 - degradation_level(t)   (already available in sequences)

    Future labels:
        future_drop_labels[horizon][i] = 1 if victim modality IV drops >30%
                                         within the next `horizon` steps
        future_iv[horizon][i]          = mean IV of victim modality over next
                                         `horizon` steps (regression target)

    Returns a dict with:
        iv_camera, iv_lidar, iv_radar : (N, T) current IV arrays
        future_drop_labels            : {horizon: (N,) bool array}
        future_iv                     : {horizon: (N,) float array}
        gt_labels                     : (N, T) detection labels (proxy)
    """
    N, T = sequences["camera_deg"].shape

    # Current IV = 1 - degradation
    iv_cam = 1 - sequences["camera_deg"]     # (N, T)
    iv_lid = 1 - sequences["lidar_deg"]
    iv_rad = 1 - sequences["radar_deg"]

    # Victim IV at each timestep
    victim_iv = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        vm = sequences["victim_modality"][i]
        if vm == "camera":
            victim_iv[i] = iv_cam[i]
        elif vm == "lidar":
            victim_iv[i] = iv_lid[i]
        else:
            victim_iv[i] = iv_rad[i]

    # Build future-horizon targets using the midpoint of each sequence
    # (to avoid boundary issues)
    mid = T // 2
    future_drop_labels = {}
    future_iv = {}

    for h in [5, 10, 20, 40]:
        safe_h = min(h, T - mid - 1)
        # For each sequence, look at IV around the midpoint
        iv_now = victim_iv[:, mid]          # (N,)
        iv_future = victim_iv[:, min(mid + safe_h, T - 1)]  # (N,)
        drop_frac = (iv_now - iv_future) / (iv_now + 1e-6)
        future_drop_labels[h] = (drop_frac > 0.30).astype(np.int32)
        future_iv[h] = iv_future

    # Proxy detection labels: high IV → good detection
    rng = np.random.default_rng(seed)
    gt_labels = (victim_iv > 0.5).astype(np.float32)
    gt_labels += rng.standard_normal((N, T)) * 0.05  # tiny noise
    gt_labels = np.clip(gt_labels, 0, 1)

    return {
        "iv_camera": iv_cam,
        "iv_lidar": iv_lid,
        "iv_radar": iv_rad,
        "victim_iv": victim_iv,
        "future_drop_labels": future_drop_labels,
        "future_iv": future_iv,
        "gt_labels": gt_labels,
    }


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_temporal_features(sequences: Dict, t: int = None) -> np.ndarray:
    """
    Build a feature vector per sequence using a fixed time-window centred at `t`.
    If t is None, use the middle of the sequence.
    """
    N, T, D = sequences["camera_features"].shape
    if t is None:
        t = T // 2
    t = min(max(t, 0), T - 1)

    window = 5
    t_start = max(0, t - window)
    t_end = min(T, t + window + 1)

    def _window_feats(arr):
        # arr: (N, T, D)
        segment = arr[:, t_start:t_end, :]          # (N, W, D)
        return np.concatenate([
            segment.mean(axis=1),                   # (N, D)
            segment.std(axis=1),                    # (N, D)
        ], axis=1)                                  # (N, 2D)

    cam_f = _window_feats(sequences["camera_features"])
    lid_f = _window_feats(sequences["lidar_features"])
    rad_f = _window_feats(sequences["radar_features"])

    return np.concatenate([cam_f, lid_f, rad_f], axis=1)   # (N, 6D)


def _extract_self_modal_features(sequences: Dict, victim_col: str, t: int = None) -> np.ndarray:
    """Features from the victim modality only (ablation baseline)."""
    N, T, D = sequences["camera_features"].shape
    if t is None:
        t = T // 2
    t = min(max(t, 0), T - 1)
    window = 5
    t_start = max(0, t - window)
    t_end = min(T, t + window + 1)
    key = f"{victim_col}_features"
    segment = sequences[key][:, t_start:t_end, :]
    return np.concatenate([segment.mean(axis=1), segment.std(axis=1)], axis=1)


# ---------------------------------------------------------------------------
# Anticipatory Cross-Modal Fuser
# ---------------------------------------------------------------------------

class AnticipatoryCrossModalFuser:
    """
    Lightweight GBT-based anticipatory fusion model.

    Trains separate classifiers and regressors per forecast horizon.
    """

    def __init__(
        self,
        forecast_horizons: List[int] = None,
        seed: int = 42,
        use_cross_modal: bool = True,
    ):
        self.forecast_horizons = forecast_horizons or [5, 10, 20, 40]
        self.seed = seed
        self.use_cross_modal = use_cross_modal
        self.classifiers: Dict = {}
        self.regressors: Dict = {}
        self.scalers: Dict = {}
        self.fusion_weights: Optional[np.ndarray] = None

    def _get_features(self, sequences: Dict) -> np.ndarray:
        N = sequences["camera_features"].shape[0]
        victim_keys = {
            "camera": "camera",
            "lidar": "lidar",
            "radar": "radar",
        }
        # Default: cross-modal (all three)
        if self.use_cross_modal:
            return _extract_temporal_features(sequences)
        else:
            # Use only camera self-modal features as representative
            return _extract_self_modal_features(sequences, "camera")

    def fit(self, sequences: Dict, iv_data: Dict) -> "AnticipatoryCrossModalFuser":
        X = self._get_features(sequences)
        for h in self.forecast_horizons:
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            self.scalers[h] = scaler

            y_cls = iv_data["future_drop_labels"][h]
            clf = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=self.seed
            )
            clf.fit(X_s, y_cls)
            self.classifiers[h] = clf

            y_reg = iv_data["future_iv"][h]
            reg = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=self.seed
            )
            reg.fit(X_s, y_reg)
            self.regressors[h] = reg

        # Simple fusion weight: modality IV at midpoint
        N, T, _ = sequences["camera_features"].shape
        mid = T // 2
        cam_iv = (1 - sequences["camera_deg"][:, mid])
        lid_iv = (1 - sequences["lidar_deg"][:, mid])
        rad_iv = (1 - sequences["radar_deg"][:, mid])
        stack = np.stack([cam_iv, lid_iv, rad_iv], axis=1)  # (N, 3)
        self.fusion_weights = stack / (stack.sum(axis=1, keepdims=True) + 1e-6)
        return self

    def predict(self, sequences: Dict, horizon: int) -> Dict:
        X = self._get_features(sequences)
        X_s = self.scalers[horizon].transform(X)
        probs = self.classifiers[horizon].predict_proba(X_s)[:, 1]
        iv_forecast = self.regressors[horizon].predict(X_s)
        return {"probs": probs, "iv_forecast": iv_forecast}

    def predict_fusion(
        self,
        sequences: Dict,
        use_forecasts: bool = True,
        use_oracle: bool = False,
        horizon: int = 10,
    ) -> np.ndarray:
        """
        Return per-sequence detection score (proxy for mAP).
        Higher = better fusion during onset window.
        """
        N, T, _ = sequences["camera_features"].shape
        mid = T // 2

        if use_oracle:
            # Oracle: know future IV exactly
            iv_cam = 1 - sequences["camera_deg"][:, min(mid + horizon, T - 1)]
            iv_lid = 1 - sequences["lidar_deg"][:, min(mid + horizon, T - 1)]
            iv_rad = 1 - sequences["radar_deg"][:, min(mid + horizon, T - 1)]
        elif use_forecasts and self.regressors:
            # Learned forecast
            pred = self.predict(sequences, horizon=horizon)
            iv_cam = np.clip(pred["iv_forecast"], 0, 1)
            iv_lid = np.clip(pred["iv_forecast"] * 0.9, 0, 1)  # simplified
            iv_rad = np.clip(pred["iv_forecast"] * 0.85, 0, 1)
        else:
            # Reactive: use current IV
            iv_cam = 1 - sequences["camera_deg"][:, mid]
            iv_lid = 1 - sequences["lidar_deg"][:, mid]
            iv_rad = 1 - sequences["radar_deg"][:, mid]

        stack = np.stack([iv_cam, iv_lid, iv_rad], axis=1)
        weights = stack / (stack.sum(axis=1, keepdims=True) + 1e-6)
        # Detection score = weighted sum of modality IVs (proxy for mAP)
        scores = (weights * stack).sum(axis=1)
        return scores


# ---------------------------------------------------------------------------
# Reactive Baseline Fuser
# ---------------------------------------------------------------------------

class ReactiveBaselineFuser:
    """
    Reactive baseline: fuses modalities based on *current* (not forecast) IV.
    Analogous to DynaFuser / RoCaRS-style entropy-weighted fusion.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, sequences: Dict, iv_data: Dict) -> "ReactiveBaselineFuser":
        return self

    def predict_fusion(self, sequences: Dict) -> np.ndarray:
        N, T, _ = sequences["camera_features"].shape
        mid = T // 2
        iv_cam = 1 - sequences["camera_deg"][:, mid]
        iv_lid = 1 - sequences["lidar_deg"][:, mid]
        iv_rad = 1 - sequences["radar_deg"][:, mid]
        stack = np.stack([iv_cam, iv_lid, iv_rad], axis=1)
        weights = stack / (stack.sum(axis=1, keepdims=True) + 1e-6)
        return (weights * stack).sum(axis=1)
