"""FAISS vector store with concurrency-safe write operations.

Single-writer pattern: only Celery workers write; FastAPI only reads.
A threading.Lock guards all write operations. Saves use atomic os.replace().
"""

import logging
import os
import pickle
import re
import tempfile
import threading
from typing import Dict, List, Optional

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Manages a FAISS IndexFlatIP with persistent metadata."""

    _instance: Optional["VectorStoreService"] = None
    _write_lock = threading.Lock()

    def __new__(cls) -> "VectorStoreService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.index_path = settings.faiss_index_path
        self.metadata_path = settings.faiss_metadata_path
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict] = []  # [{document_id, page, chunk_id, text}, ...]
        self._ensure_dirs()
        self.load()
        self._initialized = True

    def _ensure_dirs(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

    # ---- Read/Write Lifecycle ----

    def load(self) -> None:
        """Load FAISS index + metadata from disk. Creates empty index if missing."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("Loading FAISS index from %s", self.index_path)
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info("Loaded %d vectors.", self.index.ntotal)
            self._normalize_loaded_index_if_needed()
        else:
            logger.info("No existing FAISS index found — creating new.")
            self.index = None  # Created lazily on first add
            self.metadata = []

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """Return L2-normalized vectors for cosine similarity via IndexFlatIP."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _reconstruct_all_vectors(self) -> Optional[np.ndarray]:
        """Best-effort vector extraction for normalizing older persisted indexes."""
        if self.index is None or self.index.ntotal == 0:
            return None

        try:
            return self.index.reconstruct_n(0, self.index.ntotal).astype(np.float32)
        except Exception:
            try:
                vectors = np.vstack([
                    self.index.reconstruct(i) for i in range(self.index.ntotal)
                ])
                return vectors.astype(np.float32)
            except Exception:
                logger.warning("Could not reconstruct FAISS vectors for normalization.", exc_info=True)
                return None

    def _normalize_loaded_index_if_needed(self) -> None:
        """Normalize legacy indexes that were saved with raw embedding magnitudes."""
        if self.index is None or self.index.ntotal == 0:
            return

        vectors = self._reconstruct_all_vectors()
        if vectors is None:
            return

        norms = np.linalg.norm(vectors, axis=1)
        if np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
            return

        logger.info(
            "Normalizing existing FAISS index vectors for cosine search "
            "(norm range %.4f-%.4f).",
            float(norms.min()),
            float(norms.max()),
        )
        normalized = self._normalize_vectors(vectors)
        new_index = faiss.IndexFlatIP(normalized.shape[1])
        new_index.add(normalized)
        self.index = new_index
        self.save()

    def save(self) -> None:
        """Atomically persist FAISS index + metadata to disk (crash-safe)."""
        if self.index is None:
            return
        self._ensure_dirs()
        # Write to temp files first, then atomically replace
        dir_name = os.path.dirname(self.index_path)

        tmp_index = tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix=".faiss.tmp")
        tmp_meta = tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix=".pkl.tmp")
        try:
            faiss.write_index(self.index, tmp_index.name)
            tmp_index.close()

            with open(tmp_meta.name, "wb") as f:
                pickle.dump(self.metadata, f)
            tmp_meta.close()

            os.replace(tmp_index.name, self.index_path)
            os.replace(tmp_meta.name, self.metadata_path)
            logger.info("FAISS index saved (%d vectors).", self.index.ntotal)
        except Exception:
            # Cleanup temp files on failure
            for tmp in (tmp_index.name, tmp_meta.name):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            raise

    # ---- Write Operations (Lock-Protected) ----

    def add_documents(self, embeddings: np.ndarray, metadata_list: List[Dict]) -> None:
        """Add vectors + metadata under write lock.

        Args:
            embeddings: (N, D) float32 array of normalized vectors.
            metadata_list: List of dicts with keys: document_id, page, chunk_id, text.
        """
        with self._write_lock:
            if self.index is None:
                dim = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dim)
                logger.info("Created new FAISS IndexFlatIP with dimension %d.", dim)

            normalized_embeddings = self._normalize_vectors(embeddings)
            self.index.add(normalized_embeddings)
            self.metadata.extend(metadata_list)
            self.save()

    # ---- Read Operations (No Lock Needed) ----

    def search(self, query_embedding: np.ndarray, top_k: int = 15) -> List[Dict]:
        """Search for nearest neighbors. Returns metadata dicts with scores.

        Args:
            query_embedding: (D,) float32 normalized query vector.
            top_k: Number of results to return.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query = self._normalize_vectors(query_embedding)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            entry["_vector_index"] = int(idx)
            results.append(entry)
        return results

    def expand_neighbors(self, chunks: List[Dict], window: int = 2) -> List[Dict]:
        """Return retrieved chunks plus nearby chunks from the same document.

        PDF-extracted tables and lists often split one logical section across
        adjacent chunks. Neighbor expansion preserves that local context before
        CRAG and reranking trim the final answer context.
        """
        if not chunks or not self.metadata:
            return chunks

        by_doc_and_chunk_number: Dict[tuple, int] = {}
        for idx, entry in enumerate(self.metadata):
            chunk_id = str(entry.get("chunk_id", ""))
            match = re.search(r"_(\d+)$", chunk_id)
            if match:
                by_doc_and_chunk_number[(entry.get("document_id"), int(match.group(1)))] = idx

        expanded: List[Dict] = []
        seen: set[int] = set()
        for chunk in chunks:
            vector_index = chunk.get("_vector_index")
            if isinstance(vector_index, int) and vector_index not in seen:
                seen.add(vector_index)
                expanded.append(chunk)

            chunk_id = str(chunk.get("chunk_id", ""))
            match = re.search(r"_(\d+)$", chunk_id)
            if not match:
                continue

            document_id = chunk.get("document_id")
            chunk_number = int(match.group(1))
            for neighbor_number in range(chunk_number - window, chunk_number + window + 1):
                neighbor_index = by_doc_and_chunk_number.get((document_id, neighbor_number))
                if neighbor_index is None or neighbor_index in seen:
                    continue
                neighbor = dict(self.metadata[neighbor_index])
                neighbor["_vector_index"] = neighbor_index
                neighbor["score"] = float(chunk.get("score", 0))
                neighbor["neighbor_of"] = chunk.get("chunk_id")
                seen.add(neighbor_index)
                expanded.append(neighbor)

        return expanded
