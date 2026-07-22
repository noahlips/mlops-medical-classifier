# MLOps Medical Classifier

End-to-end MLOps project: a breast-cancer diagnosis model (scikit-learn) served through a production-grade stack — versioned training artifacts, FastAPI service, automated tests, Docker image, CI pipeline and data-drift monitoring.

The model itself is deliberately simple (Gradient Boosting on the Wisconsin Diagnostic dataset): the point of this repo is everything **around** the model.

```
train.py ──> artifacts/ ──> FastAPI /predict ──> logs/predictions.jsonl ──> drift.py
   │              │                                                            │
   └── metrics.json + reference sample                        KS-test vs training data
                        CI: ruff + pytest + docker build + container smoke test
```

## Results

| Metric (hold-out, n=114) | Value |
|--------------------------|-------|
| ROC-AUC | **0.992** |
| Accuracy | 0.956 |
| Recall (benign) | 0.986 |
| F1 | 0.966 |
| 5-fold CV ROC-AUC | 0.990 ± 0.016 |

## Features

- **Reproducible training** — `python -m src.train` exports the fitted pipeline, metrics, feature names and a training reference sample, all versioned together
- **Typed REST API** — FastAPI + Pydantic validation (`/predict`, `/health`, `/model/info`), every request logged as JSONL
- **Drift monitoring** — `python -m src.drift` runs a per-feature two-sample Kolmogorov–Smirnov test between logged production inputs and the training distribution; non-zero exit code on drift makes it cron/CI-friendly
- **Tests** — 10 pytest cases covering training quality gates, API contract, request logging and drift detection
- **CI (GitHub Actions)** — lint (ruff), tests, Docker build and a container smoke test on every push
- **Docker** — the model is trained *at build time*, so each image ships a reproducible, self-contained model version

## Quickstart

```bash
pip install -r requirements-dev.txt
python -m src.train          # train + export artifacts
uvicorn src.app:app          # serve
pytest                       # run the test suite
python -m src.drift          # drift report (after some requests)
```

Or with Docker:

```bash
docker build -t medical-classifier .
docker run -p 8000:8000 medical-classifier
```

Example request:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.0787, 1.095, 0.9053, 8.589, 153.4, 0.0064, 0.049, 0.0537, 0.0159, 0.03, 0.0062, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```

## Disclaimer

Educational project. Not a medical device — never use for actual diagnosis.
