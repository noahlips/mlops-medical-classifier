"""Train the breast-cancer diagnosis model and export all serving artifacts.

Artifacts written to ./artifacts:
- model.joblib          fitted sklearn Pipeline (scaler + gradient boosting)
- metrics.json          hold-out and cross-validation metrics + metadata
- reference_sample.npy  training features used as reference for drift detection
- feature_names.json    ordered list of input feature names

Each run is also recorded in MLflow (parameters, metrics and the model), so
successive trainings can be compared. Browse them with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
TRACKING_URI = f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "breast-cancer-classifier"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.9,
}


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(**MODEL_PARAMS, random_state=RANDOM_STATE)),
        ]
    )


def _log_to_mlflow(tracking_uri: str, pipeline: Pipeline, metrics: dict, artifacts_dir: Path) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {**MODEL_PARAMS, "random_state": RANDOM_STATE, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS}
        )
        mlflow.log_metrics({f"holdout_{name}": value for name, value in metrics["holdout"].items()})
        mlflow.log_metrics(
            {
                "cv_roc_auc_mean": metrics["cross_validation"]["roc_auc_mean"],
                "cv_roc_auc_std": metrics["cross_validation"]["roc_auc_std"],
            }
        )
        mlflow.set_tags(
            {
                "sklearn_version": metrics["metadata"]["sklearn_version"],
                "python_version": metrics["metadata"]["python_version"],
            }
        )
        mlflow.log_artifact(str(artifacts_dir / "metrics.json"))
        mlflow.log_artifact(str(artifacts_dir / "feature_names.json"))
        mlflow.sklearn.log_model(pipeline, name="model")
        return run.info.run_id


def train(artifacts_dir: Path = ARTIFACTS_DIR, tracking_uri: str | None = TRACKING_URI) -> dict:
    """Train, evaluate and export the model.

    Pass ``tracking_uri=None`` to skip MLflow, for example in fast unit tests.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()

    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc")

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
            "n_folds": CV_FOLDS,
        },
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
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

    if tracking_uri is not None:
        metrics["metadata"]["mlflow_run_id"] = _log_to_mlflow(tracking_uri, pipeline, metrics, artifacts_dir)

    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
