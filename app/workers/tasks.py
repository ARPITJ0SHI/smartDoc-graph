"""Celery task definitions for async document ingestion."""

import logging
import os
import uuid
from datetime import datetime, timezone

from app.workers.celery_app import celery_app
from app.config import settings
from app.database import SessionLocal
from app.models.document import Document, Task, DocumentStatus

logger = logging.getLogger(__name__)


def _extract_document_blocks(file_path: str) -> list[dict]:
    """Extract text from PDF, DOCX, or TXT files and convert to Markdown."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        import pymupdf4llm
        md_text = pymupdf4llm.to_markdown(file_path)
        return [{"text": md_text, "page": None, "block_type": "markdown"}]

    elif ext == ".docx":
        import mammoth
        import markdownify
        
        with open(file_path, "rb") as docx_file:
            result = mammoth.convert_to_html(docx_file)
            html = result.value
            md_text = markdownify.markdownify(html, heading_style="ATX")
        return [{"text": md_text, "page": None, "block_type": "markdown"}]

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return [{"text": f.read(), "page": None, "block_type": "markdown"}]

    else:
        raise ValueError(f"Unsupported file type: {ext}")


@celery_app.task(bind=True, name="app.workers.tasks.process_document", max_retries=2)
def process_document(self, task_id: str, document_id: str, file_path: str) -> dict:
    """Ingestion pipeline: extract → chunk → embed → store in FAISS.

    Args:
        task_id: UUID string for the processing task.
        document_id: UUID string for the document entity.
        file_path: Path to the uploaded file on disk.
    """
    db = SessionLocal()
    try:
        # 1. Update status → PROCESSING
        task = db.query(Task).filter(Task.id == task_id).first()
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not task or not doc:
            raise ValueError(f"Task {task_id} or Document {document_id} not found")

        task.status = DocumentStatus.PROCESSING
        doc.status = DocumentStatus.PROCESSING
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Processing document %s (task %s)", document_id, task_id)

        # 2. Extract structure-aware text blocks
        document_blocks = _extract_document_blocks(file_path)
        total_chars = sum(len(block.get("text", "")) for block in document_blocks)
        if total_chars == 0:
            raise ValueError("No text content extracted from document")
        logger.info(
            "Extracted %d characters across %d blocks from %s",
            total_chars,
            len(document_blocks),
            file_path,
        )

        # 3. Chunk text while retaining page/block metadata
        from app.services.chunking import chunk_document_blocks
        chunks = chunk_document_blocks(document_blocks)
        logger.info("Created %d chunks", len(chunks))

        # 4. Generate embeddings
        from app.services.embedding import EmbeddingService
        embedding_service = EmbeddingService()
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_service.embed_texts(chunk_texts)

        # 5. Build metadata & store in FAISS
        metadata_list = []
        for i, chunk in enumerate(chunks):
            # Base metadata
            meta = {
                "document_id": document_id,
                "chunk_id": f"{document_id}_{i}",
                "text": chunk["text"],
            }
            # Add all other metadata from chunk (page, block_type, Header 1, etc.)
            for k, v in chunk.items():
                if k not in meta:
                    meta[k] = v
            
            metadata_list.append(meta)

        from app.services.vector_store import VectorStoreService
        vector_store = VectorStoreService()
        vector_store.add_documents(embeddings, metadata_list)

        # 6. Update status → COMPLETED
        task.status = DocumentStatus.COMPLETED
        doc.status = DocumentStatus.COMPLETED
        task.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("Document %s processing completed.", document_id)

        return {"status": "COMPLETED", "chunks": len(chunks)}

    except Exception as exc:
        # Update status → FAILED
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            doc = db.query(Document).filter(Document.id == document_id).first()
            if task:
                task.status = DocumentStatus.FAILED
                task.updated_at = datetime.now(timezone.utc)
            if doc:
                doc.status = DocumentStatus.FAILED
                doc.error_message = str(exc)[:1000]
            db.commit()
        except Exception:
            db.rollback()
        logger.exception("Document processing failed: %s", exc)
        raise
    finally:
        db.close()
