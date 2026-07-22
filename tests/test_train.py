import numpy as np

from src.train import ARTIFACTS_DIR, build_pipeline, train


def test_pipeline_learns_something():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    pipeline = build_pipeline()
    pipeline.fit(data.data[:400], data.target[:400])
    acc = pipeline.score(data.data[400:], data.target[400:])
    assert acc > 0.9


def test_train_writes_artifacts_and_reports_metrics(tmp_path):
    metrics = train(artifacts_dir=tmp_path)

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "feature_names.json").exists()

    reference = np.load(tmp_path / "reference_sample.npy")
    assert reference.shape[1] == 30

    assert metrics["holdout"]["roc_auc"] > 0.95
    assert metrics["cross_validation"]["roc_auc_mean"] > 0.95


def test_default_artifacts_dir_is_repo_level():
    assert ARTIFACTS_DIR.name == "artifacts"
