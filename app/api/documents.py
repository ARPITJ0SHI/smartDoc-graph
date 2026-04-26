"""Document upload and task status endpoints."""

import os
import uuid
import logging

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.document import Document, Task, DocumentStatus
from app.schemas.document import UploadResponse, TaskStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a document for async ingestion.

    Saves the file, creates Document + Task records, and dispatches
    a Celery task for processing.
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save file to upload directory
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}{ext}"
    file_path = os.path.join(settings.upload_dir, safe_filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info("Saved uploaded file: %s (%d bytes)", safe_filename, len(content))

    # Create Document record
    doc = Document(
        filename=file.filename or safe_filename,
        file_path=file_path,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    db.flush()

    # Create Task record (separate from Document — architecture requirement)
    task = Task(
        document_id=doc.id,
        status=DocumentStatus.PENDING,
    )
    db.add(task)
    db.commit()
    db.refresh(doc)
    db.refresh(task)

    # Dispatch Celery task
    from app.workers.tasks import process_document
    process_document.delay(str(task.id), str(doc.id), file_path)
    logger.info("Dispatched ingestion task %s for document %s", task.id, doc.id)

    return UploadResponse(
        document_id=doc.id,
        task_id=task.id,
        status=DocumentStatus.PENDING,
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: uuid.UUID, db: Session = Depends(get_db)):
    """Poll the status of a document processing task."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatusResponse(
        task_id=task.id,
        status=task.status,
        document_id=task.document_id,
    )
