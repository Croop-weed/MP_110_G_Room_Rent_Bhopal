

ROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    MLFLOW_MODEL_URI=models:/pg-rent-g187@champion

WORKDIR /app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY mlflow.db .
COPY mlruns ./mlruns
COPY app ./app
COPY src ./src

ENV PYTHONPATH=/app/src

RUN python -c "import sqlite3; c=sqlite3.connect('/app/mlflow.db'); [(c.execute(f\"UPDATE {t} SET {col} = replace({col}, '/home/harshit/Downloads/G187_MP110/mlruns', '/app/mlruns')\")) for t,col in [('experiments','artifact_location'),('runs','artifact_uri'),('logged_models','artifact_location')]]; c.commit(); c.close()"

RUN mkdir -p /app/.mlflow && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
