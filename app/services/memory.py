"""Chat memory service — DB persistence + LangGraph state helpers.

Short-term memory is handled by LangGraph's checkpointer (MemorySaver).
This module provides:
  - Session CRUD (PostgreSQL)
  - DB message persistence (audit trail + recovery on server restart)
  - Helper to format LangChain messages as a context string for prompts
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from sqlalchemy.orm import Session as DBSession

from app.models.chat import ChatMessage, ChatSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def get_or_create_session(db: DBSession, session_id: Optional[uuid.UUID] = None) -> ChatSession:
    """Get existing session or create a new one."""
    if session_id:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            return session
    session = ChatSession()
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


# ---------------------------------------------------------------------------
# DB message persistence (audit trail + restart recovery)
# ---------------------------------------------------------------------------


def save_messages_to_db(
    db: DBSession,
    session_id: uuid.UUID,
    user_query: str,
    assistant_answer: str,
) -> None:
    """Persist user + assistant messages to PostgreSQL."""
    db.add(ChatMessage(session_id=session_id, role="user", content=user_query))
    db.add(ChatMessage(session_id=session_id, role="assistant", content=assistant_answer))
    db.commit()
    logger.info("Saved turn to DB for session %s", session_id)


def load_messages_from_db(db: DBSession, session_id: uuid.UUID) -> List[BaseMessage]:
    """Load full message history from DB as LangChain messages.

    Used to recover conversation state after server restart
    (when the in-memory checkpointer loses its data).
    """
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    messages: List[BaseMessage] = []
    for row in rows:
        if row.role == "user":
            messages.append(HumanMessage(content=row.content))
        else:
            messages.append(AIMessage(content=row.content))
    return messages


# ---------------------------------------------------------------------------
# Context formatting (for injection into RAG prompts)
# ---------------------------------------------------------------------------


def format_messages_as_context(messages: List[BaseMessage], max_messages: int = 6) -> str:
    """Format a list of LangChain messages into a readable context string.

    Args:
        messages: LangChain message objects (HumanMessage / AIMessage).
        max_messages: Maximum number of recent messages to include.

    Returns:
        A formatted context string, or empty string if no messages.
    """
    if not messages:
        return ""

    recent = messages[-max_messages:]
    lines = []
    for msg in recent:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            # Truncate long assistant responses in context
            content = msg.content[:500] if len(msg.content) > 500 else msg.content
            lines.append(f"Assistant: {content}")

    if not lines:
        return ""

    return (
        "\n\n--- Conversation History ---\n"
        + "\n".join(lines)
        + "\n----------------------------\n"
    )
