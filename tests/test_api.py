import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer


@pytest.fixture(scope="module")
def client():
    # Ensure artifacts exist before the app module loads them.
    from src.train import ARTIFACTS_DIR, train

    if not (ARTIFACTS_DIR / "model.joblib").exists():
        train()

    from src.app import app

    return TestClient(app)


@pytest.fixture(scope="module")
def sample_features():
    data = load_breast_cancer()
    return data.data[0].tolist(), int(data.target[0])


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_exposes_metrics(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert len(body["feature_names"]) == 30
    assert body["metrics"]["holdout"]["roc_auc"] > 0.95


def test_predict_returns_consistent_probabilities(client, sample_features):
    features, expected = sample_features
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200
    body = response.json()

    assert body["prediction"] == expected
    assert body["label"] in {"benign", "malignant"}
    total = body["probability_benign"] + body["probability_malignant"]
    assert total == pytest.approx(1.0, abs=1e-3)


def test_predict_rejects_wrong_feature_count(client):
    response = client.post("/predict", json={"features": [1.0, 2.0]})
    assert response.status_code == 422


def test_predict_rejects_non_numeric(client):
    response = client.post("/predict", json={"features": ["a"] * 30})
    assert response.status_code == 422


def test_predictions_are_logged(client, sample_features, tmp_path, monkeypatch):
    import src.app as app_module

    log_file = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(app_module, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(app_module, "PREDICTIONS_LOG", log_file)

    features, _ = sample_features
    client.post("/predict", json={"features": features})

    assert log_file.exists()
    assert len(log_file.read_text().splitlines()) == 1


def test_drift_detects_shifted_distribution(client, tmp_path):
    from src.drift import drift_report

    rng = np.random.default_rng(0)
    reference = np.load(
        __import__("src.train", fromlist=["ARTIFACTS_DIR"]).ARTIFACTS_DIR
        / "reference_sample.npy"
    )
    shifted = reference[:100] * 3.0 + rng.normal(size=(100, reference.shape[1]))

    report = drift_report(shifted)
    drifted = [r for r in report if r["drifted"]]
    assert len(drifted) > 20
