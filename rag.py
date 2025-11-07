"""Lightweight Retrieval-Augmented Generation utilities.

The module implements a minimal offline-friendly pipeline that can:
- load local learning materials (text/markdown/json files),
- split them into overlapping chunks,
- store embeddings in ChromaDB (with an in-memory fallback), and
- expose a conversational retrieval chain that can be wired into the
  GPT4All tutor orchestration layer.

All heavy dependencies are optional to keep the project runnable in
restricted environments. Tests rely on the lightweight fallback
implementation to avoid large model downloads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

from engines.caching import DocumentCache, EmbeddingCache

try:  # pragma: no cover - optional dependency
    import chromadb
    from chromadb.api.models.Collection import Collection
except Exception:  # pragma: no cover - executed when chromadb missing
    chromadb = None
    Collection = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - executed when dependency missing
    SentenceTransformer = None  # type: ignore

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


class EmbeddingBackend(Protocol):
    """Simple protocol implemented by embedding backends."""

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError("EmbeddingBackend implementations must define embed().")


class SentenceTransformerBackend:
    """Wrapper around `sentence-transformers` with lazy initialisation."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed; provide a custom embedding backend."
            )
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:  # pragma: no cover - heavy dependency
        vector = self._model.encode([text], convert_to_numpy=True)[0]
        return vector.astype(float).tolist()


@dataclass
class Document:
    source: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DocumentChunk:
    id: str
    source: str
    index: int
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievedChunk:
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


def load_learning_materials(base_path: str, patterns: Sequence[str] = (".md", ".txt", ".json")) -> List[Document]:
    """Recursively load learning materials from a directory or JSON file."""

    documents: List[Document] = []
    root = Path(base_path)
    if not root.exists():
        return documents

    paths: Iterable[Path]
    if root.is_file():
        paths = [root]
    else:
        paths = (path for path in root.rglob("*") if path.is_file())

    for file_path in paths:
        suffix = file_path.suffix.lower()
        if patterns and suffix not in patterns:
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        if suffix == ".json":
            documents.extend(_documents_from_json(content, source=str(file_path)))
            continue

        documents.append(Document(source=str(file_path), content=content))
    return documents


def split_documents(
    documents: Iterable[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[DocumentChunk]:
    """Split documents using a recursive character strategy."""

    chunks: List[DocumentChunk] = []
    for doc in documents:
        text = doc.content.strip()
        if not text:
            continue
        start = 0
        index = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            chunk_id = _make_chunk_id(doc.source, index, chunk_text)
            base_metadata: Dict[str, Any] = {"source": doc.source, "index": index}
            if doc.metadata:
                base_metadata.update(doc.metadata)
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    source=doc.source,
                    index=index,
                    content=chunk_text,
                    metadata=base_metadata,
                )
            )
            start = max(end - chunk_overlap, end)
            index += 1
    return chunks


@dataclass
class VectorStore:
    """Simple vector store that can persist embeddings to ChromaDB."""

    def __init__(
        self, backend: EmbeddingBackend, collection: Optional[Collection] = None,
        doc_cache_size: int = 10000, embed_cache_size: int = 5000,
        _fallback_store: Optional[List[tuple[DocumentChunk, List[float]]]] = None
    ) -> None:
        self.backend = backend
        self.collection = collection
        self._doc_cache = DocumentCache()
        self._embed_cache = EmbeddingCache(max_size=embed_cache_size)
        self._fallback_store = _fallback_store

        # Register cleanup
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        if self.collection is not None:
            try:
                self.collection.persist()
            except Exception:
                pass  # Best effort persistence

    def query(self, query: str, k: int = 4) -> List[RetrievedChunk]:
        """Query the vector store for similar chunks."""
        if self.collection is not None:
            # Use ChromaDB
            query_embedding = self.backend.embed(query)
            results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
            chunks = []
            for i, (id_, content, metadata, distance) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                chunks.append(RetrievedChunk(
                    id=id_,
                    content=content,
                    metadata=metadata,
                    score=1.0 - distance  # Convert distance to similarity
                ))
            return chunks
        elif self._fallback_store is not None:
            # Use fallback in-memory store
            query_embedding = self.backend.embed(query)
            scored = []
            for chunk, embedding in self._fallback_store:
                similarity = _cosine_similarity(query_embedding, embedding)
                scored.append((chunk, similarity))
            scored.sort(key=lambda x: x[1], reverse=True)
            chunks = []
            for chunk, score in scored[:k]:
                chunks.append(RetrievedChunk(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk.metadata or {},
                    score=score
                ))
            return chunks
        else:
            return []


def build_vector_store(
    chunks: Sequence[DocumentChunk],
    persist_directory: Optional[str] = None,
    collection_name: str = "learning_materials",
    embedding_backend: Optional[EmbeddingBackend] = None,
) -> VectorStore:
    """Create a vector store from the provided chunks."""

    backend = embedding_backend or default_embedding_backend()

    collection: Optional[Collection] = None
    fallback: Optional[List[tuple[DocumentChunk, List[float]]]] = None

    if chromadb is not None and persist_directory:
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_or_create_collection(collection_name)
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        embeddings = [backend.embed(chunk.content) for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = {"source": chunk.source, "index": chunk.index}
            if chunk.metadata:
                metadata.update(chunk.metadata)
            metadatas.append(metadata)
        if ids:
            collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    else:
        fallback = []
        for chunk in chunks:
            fallback.append((chunk, backend.embed(chunk.content)))

    return VectorStore(backend=backend, collection=collection, _fallback_store=fallback)


class HashEmbeddingBackend:
    """Deterministic fallback embedding using hashed token frequencies."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = max(8, dimensions)

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in text.lower().split() if token]

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokenize(text):
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


def default_embedding_backend() -> EmbeddingBackend:
    """Resolve the default embedding backend with a lightweight fallback."""

    if SentenceTransformer is not None:
        try:
            return SentenceTransformerBackend()
        except Exception as exc:  # pragma: no cover - safeguard for runtime failures
            logging.getLogger(__name__).warning("Falling back to hash embeddings: %s", exc)
    return HashEmbeddingBackend()


class ConversationalRetrievalChain:
    """Minimal conversational retrieval wrapper."""

    def __init__(
        self,
        store: VectorStore,
        llm: Callable[[str], str],
        system_prompt: Optional[str] = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.system_prompt = system_prompt or (
            "You are an offline learning assistant. Use the provided context to answer."
        )

    def invoke(
        self,
        question: str,
        chat_history: Optional[Sequence[tuple[str, str]]] = None,
        k: int = 4,
    ) -> dict:
        retrieved = self.store.query(question, k=k)
        context = "\n\n".join(chunk.content for chunk in retrieved)
        history_lines = []
        for speaker, text in chat_history or []:
            history_lines.append(f"{speaker.upper()}: {text}")
        history_text = "\n".join(history_lines) if history_lines else "(none)"
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Context:\n{context if context else 'No relevant knowledge chunks found.'}\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Question: {question}\nAnswer:"
        )
        answer = self.llm(prompt)
        return {"answer": answer, "context": retrieved}


def _make_chunk_id(source: str, index: int, content: str) -> str:
    digest = hashlib.sha256(f"{source}:{index}:{content[:50]}".encode("utf-8")).hexdigest()
    return f"chunk-{index}-{digest[:12]}"


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    # Ensure equal length by truncation/extension with zeros.
    length = min(len(vec_a), len(vec_b))
    a = vec_a[:length]
    b = vec_b[:length]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _documents_from_json(payload: str, *, source: str) -> List[Document]:
    """Parse a JSON payload into documents.

    Supports arrays of objects/strings or single objects with a ``content``/``text`` field.
    Unknown structures fall back to the raw JSON string for backwards compatibility.
    """

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return [Document(source=source, content=payload)]

    documents: List[Document] = []

    def _build_document(content: str, metadata: Optional[Dict[str, Any]], *, idx: int) -> None:
        doc_source = source
        if metadata and "id" in metadata:
            doc_source = f"{source}#{metadata['id']}"
        elif metadata and "slug" in metadata:
            doc_source = f"{source}#{metadata['slug']}"
        elif idx >= 0:
            doc_source = f"{source}#{idx}"
        documents.append(Document(source=doc_source, content=content, metadata=metadata))

    if isinstance(data, list):
        for idx, entry in enumerate(data):
            if isinstance(entry, str):
                _build_document(entry, None, idx=idx)
            elif isinstance(entry, dict):
                text = _extract_text(entry)
                if not text:
                    continue
                metadata = {k: v for k, v in entry.items() if k not in {"content", "body", "text"}}
                metadata.setdefault("position", idx)
                _build_document(text, metadata, idx=idx)
    elif isinstance(data, dict):
        text = _extract_text(data)
        if text:
            metadata = {k: v for k, v in data.items() if k not in {"content", "body", "text"}}
            _build_document(text, metadata, idx=-1)
    else:
        documents.append(Document(source=source, content=json.dumps(data)))

    return documents or [Document(source=source, content=payload)]


def _extract_text(entry: Dict[str, Any]) -> Optional[str]:
    for key in ("content", "body", "text"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def fetch_gpt4all_documents(base_url: str, timeout: float = 5.0) -> List[Document]:
    """Fetch uploaded documents from a GPT4All server.

    The endpoint shape mirrors the community server that exposes ``/v1/documents``.
    Failures are logged and result in an empty list to keep the pipeline resilient.
    """

    if not base_url:
        return []

    try:
        import requests
    except Exception:  # pragma: no cover - requests is part of the runtime requirements
        logging.getLogger(__name__).warning("requests module unavailable; cannot fetch GPT4All documents")
        return []

    endpoint = base_url.rstrip("/") + "/v1/documents"
    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:  # pragma: no cover - depends on external server availability
        logging.getLogger(__name__).warning("Failed to fetch GPT4All documents: %s", exc)
        return []

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - network responses
        logging.getLogger(__name__).warning("Invalid JSON payload from GPT4All: %s", exc)
        return []

    documents: List[Document] = []
    entries: Iterable[Any]

    if isinstance(payload, dict):
        entries = payload.get("documents") or payload.get("data") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        logging.getLogger(__name__).warning("Unexpected GPT4All payload type: %s", type(payload))
        return []

    for idx, entry in enumerate(entries):
        if isinstance(entry, dict):
            text = _extract_text(entry)
            if not text:
                continue
            metadata = {k: v for k, v in entry.items() if k not in {"content", "body", "text"}}
            metadata.setdefault("source", entry.get("source") or entry.get("document_id"))
            documents.append(
                Document(
                    source=metadata.get("source") or f"gpt4all://document/{idx}",
                    content=text,
                    metadata=metadata,
                )
            )
        elif isinstance(entry, str):
            documents.append(Document(source=f"gpt4all://document/{idx}", content=entry))

    return documents
