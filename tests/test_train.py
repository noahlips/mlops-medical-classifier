import numpy as np
from mlflow.tracking import MlflowClient

from src.train import ARTIFACTS_DIR, EXPERIMENT_NAME, build_pipeline, train


def test_pipeline_learns_something():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    pipeline = build_pipeline()
    pipeline.fit(data.data[:400], data.target[:400])
    acc = pipeline.score(data.data[400:], data.target[400:])
    assert acc > 0.9


def test_train_writes_artifacts_and_reports_metrics(tmp_path):
    metrics = train(artifacts_dir=tmp_path, tracking_uri=None)

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "feature_names.json").exists()

    reference = np.load(tmp_path / "reference_sample.npy")
    assert reference.shape[1] == 30

    assert metrics["holdout"]["roc_auc"] > 0.95
    assert metrics["cross_validation"]["roc_auc_mean"] > 0.95
    assert "mlflow_run_id" not in metrics["metadata"]


def test_train_records_run_in_mlflow(tmp_path):
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    metrics = train(artifacts_dir=tmp_path / "artifacts", tracking_uri=tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(metrics["metadata"]["mlflow_run_id"])
    experiment = client.get_experiment(run.info.experiment_id)

    assert experiment.name == EXPERIMENT_NAME
    assert run.data.params["n_estimators"] == "300"
    assert run.data.metrics["holdout_roc_auc"] == metrics["holdout"]["roc_auc"]
    assert run.data.metrics["cv_roc_auc_mean"] == metrics["cross_validation"]["roc_auc_mean"]


def test_default_artifacts_dir_is_repo_level():
    assert ARTIFACTS_DIR.name == "artifacts"
