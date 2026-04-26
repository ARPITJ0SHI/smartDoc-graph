"""Chat endpoint with query caching and LangGraph RAG pipeline."""

import logging
import time
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse, SourceInfo
from app.services.cache import CacheService
from app.services.memory import get_or_create_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Send a query to the RAG pipeline.

    Flow:
    1. Check query cache → return if hit
    2. Get or create chat session
    3. Invoke LangGraph RAG workflow
    4. Cache response
    5. Return answer + sources
    """
    started_at = time.perf_counter()

    # Get or create session
    session = get_or_create_session(db, request.session_id)
    session_id = session.id

    # 1. Check query cache
    cache = CacheService()
    cached = cache.get_cached_response(request.query, session_id)
    if cached:
        logger.info("Cache hit for query in session %s", session_id)
        logger.info("chat latency cache_hit=%.3fs", time.perf_counter() - started_at)
        return ChatResponse(
            answer=cached["answer"],
            sources=[SourceInfo(**s) for s in cached.get("sources", [])],
            session_id=session_id,
        )

    # 2. Invoke LangGraph RAG pipeline
    from app.workflows.rag_graph import rag_app

    initial_state = {
        "query": request.query,
        "session_id": str(session_id),
    }

    # thread_id links this invocation to the session's conversation history
    # via LangGraph's MemorySaver checkpointer
    config = {"configurable": {"thread_id": str(session_id)}}

    try:
        graph_started_at = time.perf_counter()
        result = await rag_app.ainvoke(initial_state, config=config)
        logger.info("rag graph latency %.3fs", time.perf_counter() - graph_started_at)
    except Exception:
        logger.exception("RAG pipeline failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="Internal error processing query")

    answer = result.get("final_answer", "")
    sources_raw = result.get("sources", [])
    sources = [
        SourceInfo(
            document_id=s.get("document_id", ""),
            page=s.get("page"),
            chunk_id=s.get("chunk_id", ""),
        )
        for s in sources_raw
    ]

    # 3. Cache response
    cache.set_cached_response(
        request.query,
        session_id,
        {"answer": answer, "sources": [s.model_dump() for s in sources]},
    )

    response = ChatResponse(
        answer=answer,
        sources=sources,
        session_id=session_id,
    )
    logger.info("chat latency total=%.3fs", time.perf_counter() - started_at)
    return response
