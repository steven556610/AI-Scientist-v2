"""
data_utils.py — Synthetic data generation for Anticipatory Fusion experiments.

Generates multimodal time-series sequences with physics-inspired gradual
degradation ramps (fog, soiling, rain, blur) for camera, LiDAR, and radar.
No external downloads required.
"""

import numpy as np
from typing import Dict, List, Optional


def _fog_profile(t: np.ndarray, onset: float, severity: float) -> np.ndarray:
    """Exponential fog ramp starting at onset."""
    profile = np.zeros_like(t, dtype=float)
    mask = t >= onset
    profile[mask] = severity * (1 - np.exp(-0.3 * (t[mask] - onset)))
    return np.clip(profile, 0, 1)


def _soiling_profile(t: np.ndarray, onset: float, severity: float) -> np.ndarray:
    """Linear soiling ramp."""
    profile = np.zeros_like(t, dtype=float)
    mask = t >= onset
    profile[mask] = np.minimum(severity * (t[mask] - onset) / (t[-1] - onset + 1e-6), severity)
    return np.clip(profile, 0, 1)


def _rain_profile(t: np.ndarray, onset: float, severity: float) -> np.ndarray:
    """Sigmoid rain accumulation."""
    k = 0.5
    profile = severity / (1 + np.exp(-k * (t - (onset + 5))))
    profile[t < onset] = 0
    return np.clip(profile, 0, 1)


def _blur_profile(t: np.ndarray, onset: float, severity: float) -> np.ndarray:
    """Step + linear blur."""
    profile = np.zeros_like(t, dtype=float)
    mask = t >= onset
    profile[mask] = np.minimum(severity * 0.5 + severity * 0.5 * (t[mask] - onset) / 10, severity)
    return np.clip(profile, 0, 1)


DEGRADATION_FNS = {
    "fog": _fog_profile,
    "soiling": _soiling_profile,
    "rain": _rain_profile,
    "blur": _blur_profile,
}


def _make_modality_features(
    seq_len: int,
    degradation_profile: np.ndarray,
    modality: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a (seq_len, feature_dim) array for one modality.
    Features encode: signal quality, feature-space entropy, and raw statistics.
    """
    feature_dim = 16
    t = np.arange(seq_len, dtype=float)

    # Base signal (clean)
    base = rng.standard_normal((seq_len, feature_dim)) * 0.1

    # Corruption: degrade feature quality proportionally
    quality = 1 - degradation_profile  # 1 = clean, 0 = fully degraded
    corrupted = base * quality[:, None] + rng.standard_normal((seq_len, feature_dim)) * (1 - quality[:, None]) * 0.5

    # Add modality-specific signal patterns
    if modality == "camera":
        corrupted[:, 0] += quality * np.sin(t / 5)           # texture richness
        corrupted[:, 1] += quality * np.cos(t / 3)           # edge density
    elif modality == "lidar":
        corrupted[:, 0] += quality * (1 - 0.5 * np.sin(t / 8))  # return intensity
        corrupted[:, 2] += quality * np.exp(-t / 30)             # point cloud density
    elif modality == "radar":
        corrupted[:, 0] += quality * 0.8                      # Doppler consistency
        corrupted[:, 3] += rng.standard_normal(seq_len) * 0.05  # clutter

    return corrupted.astype(np.float32)


def generate_degradation_sequences(
    n_sequences: int = 200,
    seq_len: int = 50,
    degradation_types: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict:
    """
    Generate a dataset of multimodal degradation sequences.

    Returns a dict with keys:
        camera_features  : (N, T, D)
        lidar_features   : (N, T, D)
        radar_features   : (N, T, D)
        camera_deg       : (N, T)  ground-truth degradation profile
        lidar_deg        : (N, T)
        radar_deg        : (N, T)
        victim_modality  : (N,)   which modality degrades
        deg_type         : (N,)   name of degradation
        onset_time       : (N,)   index when degradation begins
    """
    if degradation_types is None:
        degradation_types = ["fog", "soiling", "rain", "blur"]

    rng = np.random.default_rng(seed)
    t = np.arange(seq_len, dtype=float)

    out = {
        "camera_features": [],
        "lidar_features": [],
        "radar_features": [],
        "camera_deg": [],
        "lidar_deg": [],
        "radar_deg": [],
        "victim_modality": [],
        "deg_type": [],
        "onset_time": [],
    }

    for _ in range(n_sequences):
        deg_type = rng.choice(degradation_types)
        victim = rng.choice(["camera", "lidar", "radar"])
        onset = float(rng.integers(5, seq_len // 2))
        severity = float(rng.uniform(0.4, 0.95))

        deg_fn = DEGRADATION_FNS[deg_type]
        deg_profile = deg_fn(t, onset, severity)

        # Victim modality degrades; others show precursors (weaker correlated signal)
        precursor_strength = rng.uniform(0.05, 0.25)
        cam_deg = deg_profile if victim == "camera" else deg_profile * precursor_strength * rng.uniform(0, 1, seq_len)
        lid_deg = deg_profile if victim == "lidar" else deg_profile * precursor_strength * rng.uniform(0, 1, seq_len)
        rad_deg = deg_profile if victim == "radar" else deg_profile * precursor_strength * rng.uniform(0, 1, seq_len)

        out["camera_features"].append(_make_modality_features(seq_len, cam_deg, "camera", rng))
        out["lidar_features"].append(_make_modality_features(seq_len, lid_deg, "lidar", rng))
        out["radar_features"].append(_make_modality_features(seq_len, rad_deg, "radar", rng))
        out["camera_deg"].append(cam_deg.astype(np.float32))
        out["lidar_deg"].append(lid_deg.astype(np.float32))
        out["radar_deg"].append(rad_deg.astype(np.float32))
        out["victim_modality"].append(victim)
        out["deg_type"].append(deg_type)
        out["onset_time"].append(onset)

    for key in ["camera_features", "lidar_features", "radar_features",
                "camera_deg", "lidar_deg", "radar_deg"]:
        out[key] = np.array(out[key])

    out["victim_modality"] = np.array(out["victim_modality"])
    out["deg_type"] = np.array(out["deg_type"])
    out["onset_time"] = np.array(out["onset_time"], dtype=np.float32)

    return out
