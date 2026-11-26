# MP110 — PG Room Rent (G187)

Stage 2 MLOps tracking, cleaning, and training are preserved in this repository. Stage 3 adds the FastAPI service, Docker packaging, CI, Kubernetes manifests, drift analysis, retraining, promotion, and holdout prediction.

## Model loading

The production API loads `models:/pg-rent-g187@champion` through MLflow SQLite tracking:

```text
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_MODEL_URI=models:/pg-rent-g187@champion
```

The Docker image packages `mlflow.db` and `mlruns/` together. MLflow local artifact URIs must remain relative/container-mapped so the database and artifact tree are resolved consistently.

## Run the API locally

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"distance_to_campus_km":5.0,"room_area_sqft":300,"sharing":2,"meals_per_day":2,"building_age_yr":10,"wifi_mbps":100,"furnishing":"semi","locality":"Sector East"}'
```

## Tests and quality

```bash
PYTHONPATH=src .venv/bin/python -m pytest -v
.venv/bin/ruff check app src tests
.venv/bin/ruff format --check app src tests
```

## Docker

```bash
docker build -t mp110-api:1.0 .
docker run --rm -p 8000:8000 mp110-api:1.0
```

The Dockerfile uses Python 3.12 slim, a non-root user, the project virtual dependencies, port 8000, and packaged MLflow artifacts.

## Kubernetes

No Kubernetes cluster is configured in this workspace, so deployment is not claimed as successful. Validate or deploy with:

```bash
kubectl apply --dry-run=client --validate=false -f k8s/deployment.yaml -f k8s/service.yaml
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc
```

The Deployment uses `mp110-api`, 2 replicas, port 8000, and `/health` readiness/liveness probes. The Service `mp110-svc` exposes port 80 to target port 8000.

## Monitoring and retraining

`src/pg_rent/monitoring.py` calculates 10-bin PSI for every numeric feature, compares cleaned training and later data, calculates v1 MAE per locality, sorts later data chronologically, retrains v2 on train plus the first 70%, and evaluates v1/v2 on the final 30%. v2 is promoted only if relative MAE improvement is at least 15%.

Run it with:

```bash
PYTHONPATH=src .venv/bin/python -m pg_rent.monitoring
```

Outputs:

- `reports/psi_results.csv`
- `reports/locality_mae.csv`
- `predictions/holdout_predictions.csv`

## MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
