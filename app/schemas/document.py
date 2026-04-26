"""Pydantic schemas for document upload and task status endpoints."""

from uuid import UUID
from pydantic import BaseModel

from app.models.document import DocumentStatus


class UploadResponse(BaseModel):
    """Response returned after document upload."""
    document_id: UUID
    task_id: UUID
    status: DocumentStatus

    model_config = {"from_attributes": True}


class TaskStatusResponse(BaseModel):
    """Response returned when polling task status."""
    task_id: UUID
    status: DocumentStatus
    document_id: UUID

    model_config = {"from_attributes": True}
