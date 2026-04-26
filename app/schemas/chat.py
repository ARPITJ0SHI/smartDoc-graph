"""Pydantic schemas for the chat endpoint."""

from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


class SourceInfo(BaseModel):
    """Source reference for a retrieved chunk."""
    document_id: str
    page: Optional[int] = None
    chunk_id: str


class ChatRequest(BaseModel):
    """Incoming chat request."""
    session_id: Optional[UUID] = None
    query: str


class ChatResponse(BaseModel):
    """Chat response with answer and sources."""
    answer: str
    sources: List[SourceInfo] = []
    session_id: UUID
