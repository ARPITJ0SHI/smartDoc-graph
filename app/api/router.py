"""Main API router aggregator."""

from fastapi import APIRouter

from app.api.documents import router as documents_router
from app.api.chat import router as chat_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(documents_router)
api_router.include_router(chat_router)
