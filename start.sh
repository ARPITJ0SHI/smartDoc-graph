#!/bin/bash
set -e

echo "Starting FastAPI Server (Celery runs in-process)..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
