"""Train the breast-cancer diagnosis model and export all serving artifacts.

Artifacts written to ./artifacts:
- model.joblib          fitted sklearn Pipeline (scaler + gradient boosting)
- metrics.json          hold-out and cross-validation metrics + metadata
- reference_sample.npy  training features used as reference for drift detection
- feature_names.json    ordered list of input feature names
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()

    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "holdout": {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        },
        "cross_validation": {
            "roc_auc_mean": round(cv_auc.mean(), 4),
            "roc_auc_std": round(cv_auc.std(), 4),
            "n_folds": 5,
        },
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
            "n_features": int(X.shape[1]),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
        },
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifacts_dir / "model.joblib")
    np.save(artifacts_dir / "reference_sample.npy", X_train)
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifacts_dir / "feature_names.json").write_text(json.dumps(list(data.feature_names)))

    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
