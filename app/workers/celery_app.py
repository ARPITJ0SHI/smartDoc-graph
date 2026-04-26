"""Celery application instance.

Configured with Redis as both broker and result backend.
Run worker with: celery -A app.workers.celery_app worker --loglevel=info --concurrency=1
"""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "smartdoc",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# Auto-discover tasks in app.workers.tasks
celery_app.autodiscover_tasks(["app.workers"])
