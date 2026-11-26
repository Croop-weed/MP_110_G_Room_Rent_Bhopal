"""Rewrite MLflow SQLite artifact locations to container paths, then start Uvicorn."""

from __future__ import annotations

import sqlite3
import subprocess


DB_PATH = "/app/mlflow.db"


def rewrite_artifact_locations() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        for table, column in (
            ("experiments", "artifact_location"),
            ("runs", "artifact_uri"),
            ("logged_models", "artifact_location"),
        ):
            connection.execute(
                f"UPDATE {table} SET {column} = replace({column}, '/home/harshit/Downloads/G187_MP110/mlruns', '/app/mlruns')"
            )
        connection.commit()


if __name__ == "__main__":
    rewrite_artifact_locations()
    raise SystemExit(
        subprocess.call(
            [
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ]
        )
    )
