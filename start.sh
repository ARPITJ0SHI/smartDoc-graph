#!/bin/bash
set -e

echo "Starting Celery Background Worker..."
# Start celery in the background with a single worker process to save memory
celery -A app.workers.celery_app worker --loglevel=info --concurrency=1 --pool=solo &

echo "Starting FastAPI Server..."
# Start uvicorn in the foreground (this will keep the container alive)
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
