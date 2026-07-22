"""Data-drift report: production inputs vs training distribution.

Reads the requests logged by the API (logs/predictions.jsonl), runs a
two-sample Kolmogorov-Smirnov test per feature against the training
reference sample, and prints a report. Exits with code 1 if any feature
drifts, which makes it usable as a CI/cron gate.

Usage:
    python -m src.drift [--alpha 0.01] [--min-samples 30]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
PREDICTIONS_LOG = Path(__file__).resolve().parent.parent / "logs" / "predictions.jsonl"


def load_production_features(log_path: Path = PREDICTIONS_LOG) -> np.ndarray:
    if not log_path.exists():
        raise FileNotFoundError(f"No prediction log found at {log_path}. Run the API first.")
    rows = [json.loads(line)["features"] for line in log_path.read_text().splitlines() if line]
    return np.array(rows)


def drift_report(
    production: np.ndarray, alpha: float = 0.01
) -> list[dict]:
    reference = np.load(ARTIFACTS_DIR / "reference_sample.npy")
    feature_names = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())

    report = []
    for i, name in enumerate(feature_names):
        stat, p_value = ks_2samp(reference[:, i], production[:, i])
        report.append(
            {
                "feature": name,
                "ks_statistic": round(float(stat), 4),
                "p_value": float(p_value),
                "drifted": bool(p_value < alpha),
            }
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.01, help="significance level")
    parser.add_argument(
        "--min-samples", type=int, default=30, help="minimum logged requests required"
    )
    args = parser.parse_args()

    production = load_production_features()
    if len(production) < args.min_samples:
        print(f"Only {len(production)} logged requests (< {args.min_samples}); skipping test.")
        return 0

    report = drift_report(production, alpha=args.alpha)
    drifted = [r for r in report if r["drifted"]]

    print(f"Compared {len(production)} production requests against training reference.")
    print(f"{len(drifted)}/{len(report)} features show significant drift (alpha={args.alpha}).\n")
    for r in sorted(report, key=lambda r: r["p_value"])[:10]:
        flag = "DRIFT" if r["drifted"] else "ok"
        print(f"  [{flag:>5}] {r['feature']:<28} KS={r['ks_statistic']:.4f} p={r['p_value']:.2e}")

    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())
