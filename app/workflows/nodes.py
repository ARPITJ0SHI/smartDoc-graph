"""LangGraph node functions for the RAG pipeline.

Each node operates on RAGState and returns a partial state update dict.
All LLM calls use async agenerate() for non-blocking execution.

Short-term memory is managed by LangGraph's checkpointer. The `messages`
field in state accumulates across invocations automatically. Nodes read
`conversation_context` (formatted from messages) for prompt injection.
"""

import logging
import re
import time
import asyncio
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage

from app.config import settings
from app.services.llm import get_llm_service
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.reranker import RerankerService
from app.services.memory import format_messages_as_context, load_messages_from_db

logger = logging.getLogger(__name__)


def _merge_retrieval_results(*ranked_lists: List[Dict], top_k: int) -> List[Dict]:
    """Fuse multiple retrieval lists while preserving the strongest candidates."""
    fused: Dict[str, Dict] = {}

    for results in ranked_lists:
        for rank, chunk in enumerate(results, start=1):
            text_key = re.sub(r"\s+", " ", chunk.get("text", "")).strip().lower()[:500]
            key = text_key or chunk.get("chunk_id") or f"{chunk.get('document_id')}:{rank}"
            contribution = 1.0 / (rank + 60)

            if key not in fused:
                merged = dict(chunk)
                merged["retrieval_score"] = contribution
                merged["source_document_ids"] = [chunk.get("document_id")]
                fused[key] = merged
            else:
                fused[key]["retrieval_score"] += contribution
                fused[key]["score"] = max(float(fused[key].get("score", 0)), float(chunk.get("score", 0)))
                doc_id = chunk.get("document_id")
                if doc_id and doc_id not in fused[key]["source_document_ids"]:
                    fused[key]["source_document_ids"].append(doc_id)

    return sorted(
        fused.values(),
        key=lambda c: (c.get("retrieval_score", 0), c.get("score", 0)),
        reverse=True,
    )[:top_k]


# ---- Node 1: Load Memory ----

async def load_memory_node(state: dict) -> dict:
    """Build conversation context from accumulated messages.

    On first invocation (or after server restart when MemorySaver is empty),
    falls back to loading messages from the database.
    """
    messages = state.get("messages", [])

    logger.info("load_memory_node: %d messages in checkpointer state", len(messages))

    # If checkpointer has no messages (first run or server restart),
    # try recovering from the database
    if not messages:
        try:
            from app.database import SessionLocal
            import uuid as _uuid

            # Only attempt DB recovery if session_id is a valid UUID
            session_id = state.get("session_id", "")
            _uuid.UUID(session_id)  # validate

            db = SessionLocal()
            try:
                db_messages = load_messages_from_db(db, session_id)
                if db_messages:
                    logger.info("Recovered %d messages from DB for session %s",
                                len(db_messages), session_id)
                    context = format_messages_as_context(db_messages)
                    return {"messages": db_messages, "conversation_context": context}
            finally:
                db.close()
        except (ValueError, Exception) as e:
            logger.debug("DB recovery skipped: %s", e)

    context = format_messages_as_context(messages)
    logger.info("load_memory_node: conversation_context length = %d", len(context))
    return {"conversation_context": context}


# ---- Node 2: HyDE Generation ----

async def hyde_generation_node(state: dict) -> dict:
    """Generate a hypothetical answer to improve embedding quality."""
    if not settings.enable_hyde:
        return {"hyde_doc": state["query"]}

    llm = get_llm_service()
    query = state["query"]
    context = state.get("conversation_context", "")

    prompt = (
        f"Given the following question, write a short hypothetical answer "
        f"that would appear in a relevant document. Do not explain yourself, "
        f"just write the answer passage.\n\n"
        f"Conversation context:\n{context}\n\n"
        f"Question: {query}"
    )
    hyde_doc = await llm.agenerate(prompt, system_message="You are a helpful document writer.")
    return {"hyde_doc": hyde_doc}


# ---- Node 3: FAISS Retrieval ----

async def retrieve_faiss_node(state: dict) -> dict:
    """Retrieve with both the original query and HyDE, then fuse rankings."""
    started_at = time.perf_counter()
    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()

    query = state["query"]

    embedding_tasks = {
        "query": asyncio.to_thread(embedding_service.embed_query, query),
    }
    if settings.enable_hyde:
        embedding_tasks["hyde"] = asyncio.to_thread(embedding_service.embed_query, state["hyde_doc"])

    embedding_results = await asyncio.gather(*embedding_tasks.values())
    embeddings = dict(zip(embedding_tasks.keys(), embedding_results))

    query_results = vector_store.search(embeddings["query"], top_k=settings.top_k_retrieval)
    expanded_query_results = vector_store.expand_neighbors(
        query_results,
        window=settings.retrieval_neighbor_window,
    )
    if settings.enable_hyde:
        hyde_results = vector_store.search(embeddings["hyde"], top_k=settings.top_k_retrieval)
        expanded_hyde_results = vector_store.expand_neighbors(
            hyde_results,
            window=settings.retrieval_neighbor_window,
        )
        retrieved = _merge_retrieval_results(
            expanded_query_results,
            expanded_hyde_results,
            top_k=max(settings.top_k_retrieval, settings.top_k_rerank * 2),
        )
    else:
        retrieved = _merge_retrieval_results(
            expanded_query_results,
            top_k=max(settings.top_k_retrieval, settings.top_k_rerank * 2),
        )

    logger.info("retrieve_faiss_node latency %.3fs chunks=%d", time.perf_counter() - started_at, len(retrieved))
    return {"retrieved_chunks": retrieved}


# ---- Node 4: CRAG Evaluation ----

async def crag_evaluation_node(state: dict) -> dict:
    """Classify each retrieved chunk as RELEVANT/IRRELEVANT/AMBIGUOUS.

    Uses a single batched LLM call. Sets fallback=True if fewer than
    MIN_RELEVANT_CHUNKS are classified as RELEVANT or AMBIGUOUS.
    """
    started_at = time.perf_counter()
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])[:settings.crag_top_k]
    context = state.get("conversation_context", "")

    if not chunks:
        return {"fallback": True, "relevant_count": 0, "retrieved_chunks": []}

    if not settings.enable_crag:
        return {
            "retrieved_chunks": chunks,
            "relevant_count": len(chunks),
            "fallback": False,
        }

    llm = get_llm_service()

    # Build batch classification prompt
    chunk_lines = []
    for i, chunk in enumerate(chunks):
        text_preview = chunk["text"][:250]  # Reduced for faster latency
        chunk_lines.append(f"[{i+1}] {text_preview}")
    chunks_text = "\n".join(chunk_lines)

    prompt = (
        f"Query: {query}\n"
        f"Context: {context}\n\n"
        f"Chunks:\n{chunks_text}\n\n"
        f"Grade each chunk's relevance to the query (1:RELEVANT, 2:IRRELEVANT, 3:AMBIGUOUS).\n"
        f"Output ONLY format:\n1:RELEVANT\n2:IRRELEVANT\n"
    )

    response = await llm.agenerate(
        prompt,
        system_message="You are a strict relevance judge. Output only the requested list format."
    )

    # Parse labels
    labels = {}
    for line in response.strip().split("\n"):
        match = re.match(r"(\d+)\s*:\s*(RELEVANT|IRRELEVANT|AMBIGUOUS)", line.strip(), re.IGNORECASE)
        if match:
            idx = int(match.group(1)) - 1
            label = match.group(2).upper()
            labels[idx] = label

    # Filter chunks
    relevant_chunks = []
    for i, chunk in enumerate(chunks):
        label = labels.get(i, "IRRELEVANT")
        chunk["crag_label"] = label
        if label in ("RELEVANT", "AMBIGUOUS"):
            relevant_chunks.append(chunk)

    relevant_count = sum(1 for l in labels.values() if l == "RELEVANT")
    ambiguous_count = sum(1 for l in labels.values() if l == "AMBIGUOUS")
    evidence_count = relevant_count + ambiguous_count
    fallback = evidence_count < settings.min_relevant_chunks

    logger.info("CRAG: %d relevant, %d ambiguous, fallback=%s",
                relevant_count,
                ambiguous_count,
                fallback)
    logger.info("crag_evaluation_node latency %.3fs judged=%d", time.perf_counter() - started_at, len(chunks))

    return {
        "retrieved_chunks": relevant_chunks,
        "relevant_count": relevant_count,
        "fallback": fallback,
    }


# ---- Node 5: Re-rank ----

async def rerank_node(state: dict) -> dict:
    """Re-rank relevant chunks using cross-encoder."""
    started_at = time.perf_counter()
    chunks = state.get("retrieved_chunks", [])
    query = state["query"]

    if settings.enable_reranker:
        reranker = RerankerService()
        reranked = await asyncio.to_thread(
            reranker.rerank,
            query,
            chunks,
            settings.top_k_rerank,
        )
    else:
        reranked = sorted(
            chunks,
            key=lambda c: (c.get("retrieval_score", 0), c.get("score", 0)),
            reverse=True,
        )[:settings.top_k_rerank]
    logger.info("rerank_node latency %.3fs chunks=%d", time.perf_counter() - started_at, len(chunks))
    return {"reranked_chunks": reranked}


# ---- Node 6: Generate Answer ----

async def generate_answer_node(state: dict) -> dict:
    """Generate the final answer using reranked chunks + conversation context."""
    started_at = time.perf_counter()
    llm = get_llm_service()
    query = state["query"]
    chunks = state.get("reranked_chunks", [])
    context = state.get("conversation_context", "")

    # Build context from reranked chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        source_info = f"[Source: doc={chunk.get('document_id', 'unknown')}, chunk={chunk.get('chunk_id', i)}]"
        context_parts.append(f"{source_info}\n{chunk['text'][:1400]}")
    doc_context = "\n\n---\n\n".join(context_parts)

    # Build prompt with grounding instruction
    prompt = (
        f"Conversation history: {context}\n\n"
        f"Retrieved documents:\n{doc_context}\n\n"
        f"User question: {query}\n\n"
        f"Answer the question based ONLY on the provided documents. "
        f"When the user asks for a list, enumerate every distinct matching item "
        f"present in the retrieved documents; "
        f"do not provide only examples or the first few matches. "
        f"If the documents do not contain enough information, say so clearly. "
        f"Cite sources by referencing document IDs when possible."
    )

    answer = await llm.agenerate(
        prompt,
        system_message=(
            "You are a precise document assistant. Answer strictly from the "
            "provided context. Do not hallucinate or add information not present "
            "in the context. If unsure, say you don't have enough information."
        ),
    )

    # Extract source references
    sources = []
    seen = set()
    for chunk in chunks:
        doc_id = chunk.get("document_id", "")
        chunk_id = chunk.get("chunk_id", "")
        key = (doc_id, chunk_id)
        if key not in seen:
            seen.add(key)
            sources.append({
                "document_id": doc_id,
                "page": chunk.get("page"),
                "chunk_id": chunk_id,
            })

    logger.info("generate_answer_node latency %.3fs", time.perf_counter() - started_at)
    return {"final_answer": answer, "sources": sources}


# ---- Node 7: Update Memory ----

async def update_memory_node(state: dict) -> dict:
    """Save messages to DB and append to graph state for checkpointer.

    The checkpointer handles short-term memory automatically.
    DB persistence is for audit trail and recovery after restart.
    """
    from app.database import SessionLocal
    from app.services.memory import save_messages_to_db

    user_query = state["query"]
    answer = state.get("final_answer", "")

    # Persist to DB (audit trail — non-fatal)
    try:
        db = SessionLocal()
        try:
            save_messages_to_db(db, state["session_id"], user_query, answer)
        finally:
            db.close()
    except Exception:
        logger.warning("Failed to persist messages to DB for session %s", state["session_id"], exc_info=True)

    # Append to state — the checkpointer persists this automatically
    # via the add_messages reducer on the `messages` field
    new_messages = [
        HumanMessage(content=user_query),
        AIMessage(content=answer),
    ]
    return {"messages": new_messages}


# ---- Node 8: Fallback ----

async def fallback_node(state: dict) -> dict:
    """Deterministic fallback — no LLM call, no hallucination risk."""
    from app.database import SessionLocal
    from app.services.memory import save_messages_to_db

    fallback_answer = (
        "I don't have enough relevant information in the uploaded documents "
        "to answer your question. Please try rephrasing, or upload additional "
        "documents that may contain the answer."
    )

    user_query = state["query"]

    # Persist to DB (audit trail — non-fatal)
    try:
        db = SessionLocal()
        try:
            save_messages_to_db(db, state["session_id"], user_query, fallback_answer)
        finally:
            db.close()
    except Exception:
        logger.warning("Failed to persist fallback to DB for session %s", state["session_id"], exc_info=True)

    # Append to state for checkpointer
    new_messages = [
        HumanMessage(content=user_query),
        AIMessage(content=fallback_answer),
    ]
    return {"final_answer": fallback_answer, "sources": [], "messages": new_messages}
