# mlops-medical-classifier

A breast cancer classifier (Wisconsin dataset) wrapped in the tooling you would need to actually run it somewhere: exported artifacts, an API, tests, a container, CI, and drift checks.

The model is a plain Gradient Boosting and it is not the interesting part. Everything around it is.

## Pipeline

```
src/train.py  ->  artifacts/  ->  src/app.py  ->  logs/predictions.jsonl  ->  src/drift.py
```

`train.py` writes four things to `artifacts/`, all versioned together: the fitted pipeline, its metrics, the ordered feature names, and a sample of the training data. That last one exists because you cannot measure drift later without knowing what the original distribution looked like.

The scaler lives inside the sklearn `Pipeline` rather than next to it, so inference applies exactly the normalisation that training learned.

## Metrics

Hold-out set, 114 samples:

```
ROC-AUC     0.992
Accuracy    0.956
Recall      0.986
F1          0.966

5-fold CV ROC-AUC   0.990 +/- 0.016
```

Recall is the one to look at here. In screening, missing a malignant tumour costs far more than a false alarm.

## API

Three endpoints, FastAPI with Pydantic validation:

- `POST /predict` takes the 30 features and returns the class plus both probabilities
- `GET /model/info` returns feature names and training metrics
- `GET /health`

Every prediction request gets appended to `logs/predictions.jsonl`.

## Drift

`python -m src.drift` compares those logged requests against the training sample, running a two-sample Kolmogorov-Smirnov test per feature at alpha = 0.01. It exits with code 1 if anything drifted, so it can be dropped into a cron job or a CI step as-is.

KS is non-parametric, which matters because the features here are on very different scales and are not all normal. The limitation is that it treats each feature independently, so it will not catch a change in how features correlate.

## Commands

```bash
pip install -r requirements-dev.txt
python -m src.train
uvicorn src.app:app
pytest
python -m src.drift
```

Docker builds train the model at image build time, so an image always ships a model that matches the code that produced it:

```bash
docker build -t medical-classifier .
docker run -p 8000:8000 medical-classifier
```

CI runs ruff, the 10 pytest cases, a Docker build, and a container smoke test on every push.

## Known gaps

The dataset is small, clean and nearly balanced, so it is nothing like real clinical data. There is no authentication on the API. There is no experiment tracking yet. And there is no monitoring of actual performance in production, because that would need real diagnoses coming back after the fact, which no public dataset gives you.

Educational project. Obviously not a medical device.
