"""FastAPI application entry point.

Startup:
  - Initialize database tables
  - Preload FAISS index
  - Preload embedding model
  - Warm Redis cache connection

Middleware:
  - CORS
  - API key authentication
  - Rate limiting (slowapi)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database import init_db
from app.api.router import api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # --- Startup ---
    logger.info("Initializing database tables...")
    init_db()

    logger.info("Preloading FAISS index...")
    from app.services.vector_store import VectorStoreService
    VectorStoreService()

    logger.info("Preloading embedding model...")
    from app.services.embedding import EmbeddingService
    EmbeddingService()

    logger.info("Preloading LLM client...")
    from app.services.llm import get_llm_service
    get_llm_service()

    if settings.enable_reranker:
        logger.info("Preloading reranker model...")
        from app.services.reranker import RerankerService
        RerankerService()

    logger.info("Warming cache connection...")
    from app.services.cache import CacheService
    CacheService()

    logger.info("SmartDoc API ready.")
    yield
    # --- Shutdown ---
    logger.info("SmartDoc API shutting down.")


app = FastAPI(
    title="SmartDoc RAG API",
    description="Production-grade RAG system with async ingestion and graph-based retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---- Middleware ----

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )




# ---- Routes ----

app.include_router(api_router)


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint (no auth required)."""
    return {"status": "ok", "service": "smartdoc"}
