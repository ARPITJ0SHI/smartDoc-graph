#!/bin/bash
set -e

echo "Starting Celery Background Worker..."
# Start celery in the background
celery -A app.workers.celery_app worker --loglevel=info &

echo "Starting FastAPI Server..."
# Start uvicorn in the foreground (this will keep the container alive)
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
