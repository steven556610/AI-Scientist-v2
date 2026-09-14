"""
runfile.py — Entry-point for AI-Scientist to run and iterate on.
================================================================
Cross-Modal Canaries: Anticipatory Multimodal Fusion Experiment Scaffold

This scaffold implements a lightweight, self-contained simulation of the
core hypothesis: cross-modal sensor reading (from Camera / LiDAR / Radar)
can be used to **anticipate** future degradation of one modality before it
visually manifests, enabling anticipatory fusion re-routing.

All data is synthetic (no external downloads needed).
Results are saved to results/results.json for AI-Scientist parsing.

Usage:
    python runfile.py --out_dir results/ --config_path configs/default.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from data_utils import generate_degradation_sequences
from model import (
    ReactiveBaselineFuser,
    AnticipatoryCrossModalFuser,
    compute_modality_information_value,
)
from metrics import (
    compute_auroc_auprc,
    compute_regression_metrics,
    compute_detection_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Anticipatory Fusion Experiment")
    parser.add_argument(
        "--out_dir", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/default.yaml",
        help="Path to experiment config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load config ────────────────────────────────────────────────────────
    with open(args.config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    n_sequences = cfg.get("n_sequences", 200)
    sequence_len = cfg.get("sequence_len", 50)
    forecast_horizons = cfg.get("forecast_horizons", [5, 10, 20, 40])
    degradation_types = cfg.get("degradation_types", ["fog", "soiling", "rain", "blur"])

    print(f"[runfile] Config: {cfg}")
    print(f"[runfile] Generating {n_sequences} synthetic sequences …")

    # ── Synthetic data ─────────────────────────────────────────────────────
    t0 = time.time()
    sequences = generate_degradation_sequences(
        n_sequences=n_sequences,
        seq_len=sequence_len,
        degradation_types=degradation_types,
        seed=seed,
    )
    print(f"[runfile] Data generated in {time.time()-t0:.1f}s")

    # ── Compute modality information values ────────────────────────────────
    print("[runfile] Computing modality information values (V_m(t)) …")
    iv_data = compute_modality_information_value(sequences, seed=seed)

    # ── Experiments ────────────────────────────────────────────────────────
    results = {}

    # Experiment 1: Precursor Forecasting (the core hypothesis)
    print("\n[runfile] Experiment 1 — Precursor Existence & Forecasting")
    anticipatory = AnticipatoryCrossModalFuser(
        forecast_horizons=forecast_horizons,
        seed=seed,
        use_cross_modal=True,
    )
    anticipatory.fit(sequences, iv_data)

    # Self-modal-only ablation
    self_modal = AnticipatoryCrossModalFuser(
        forecast_horizons=forecast_horizons,
        seed=seed,
        use_cross_modal=False,
    )
    self_modal.fit(sequences, iv_data)

    exp1_results = {}
    for horizon in forecast_horizons:
        preds_cross = anticipatory.predict(sequences, horizon=horizon)
        preds_self = self_modal.predict(sequences, horizon=horizon)
        targets = iv_data["future_drop_labels"][horizon]

        auroc_cross, auprc_cross = compute_auroc_auprc(preds_cross["probs"], targets)
        auroc_self, auprc_self = compute_auroc_auprc(preds_self["probs"], targets)
        r2_cross, mae_cross = compute_regression_metrics(
            preds_cross["iv_forecast"], iv_data["future_iv"][horizon]
        )
        r2_self, mae_self = compute_regression_metrics(
            preds_self["iv_forecast"], iv_data["future_iv"][horizon]
        )
        exp1_results[f"horizon_{horizon}"] = {
            "cross_modal": {
                "AUROC": auroc_cross,
                "AUPRC": auprc_cross,
                "R2": r2_cross,
                "MAE": mae_cross,
            },
            "self_modal_only": {
                "AUROC": auroc_self,
                "AUPRC": auprc_self,
                "R2": r2_self,
                "MAE": mae_self,
            },
            "cross_modal_advantage_AUROC": auroc_cross - auroc_self,
        }
    results["exp1_precursor_forecasting"] = exp1_results

    # Experiment 2: Detection mAP during degradation onset window
    print("[runfile] Experiment 2 — Detection mAP during onset window")
    reactive = ReactiveBaselineFuser(seed=seed)
    reactive.fit(sequences, iv_data)

    map_reactive = compute_detection_map(
        reactive.predict_fusion(sequences), iv_data["gt_labels"], window="onset"
    )
    map_anticipatory = compute_detection_map(
        anticipatory.predict_fusion(sequences, use_forecasts=True),
        iv_data["gt_labels"],
        window="onset",
    )
    map_oracle = compute_detection_map(
        anticipatory.predict_fusion(sequences, use_forecasts=False, use_oracle=True),
        iv_data["gt_labels"],
        window="onset",
    )
    results["exp2_detection_map"] = {
        "reactive_baseline": map_reactive,
        "anticipatory_learned": map_anticipatory,
        "anticipatory_oracle_upper_bound": map_oracle,
        "anticipatory_improvement": map_anticipatory - map_reactive,
    }

    # ── Save results ───────────────────────────────────────────────────────
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[runfile] Results saved to {results_path}")

    # Print summary table
    print("\n=== SUMMARY ===")
    for h in forecast_horizons:
        k = f"horizon_{h}"
        r = results["exp1_precursor_forecasting"][k]
        print(
            f"  Horizon={h:>2}  AUROC cross={r['cross_modal']['AUROC']:.4f}  "
            f"self={r['self_modal_only']['AUROC']:.4f}  "
            f"advantage={r['cross_modal_advantage_AUROC']:+.4f}"
        )
    print(
        f"\n  mAP onset:  reactive={results['exp2_detection_map']['reactive_baseline']:.4f}  "
        f"anticipatory={results['exp2_detection_map']['anticipatory_learned']:.4f}  "
        f"oracle={results['exp2_detection_map']['anticipatory_oracle_upper_bound']:.4f}"
    )


if __name__ == "__main__":
    main()
