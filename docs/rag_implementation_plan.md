# RAG Implementation Plan (WP2 Refresh)

## Current Retrieval Flow Inventory
- **Document ingestion**: `rag.load_learning_materials` loads Markdown, text, and JSON sources from `_RAG_CORPUS_PATH` or remote GPT4All uploads via `rag.fetch_gpt4all_documents`. The FastAPI bootstrap `_ensure_rag_ready` merges local and remote documents before chunking.
- **Chunking**: `rag.split_documents` applies configurable chunk size/overlap (`RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`) used directly by `_ensure_rag_ready`.
- **Vector store construction**: `rag.build_vector_store` currently resolves the default embedding backend (hash fallback when `sentence-transformers` unavailable) and either persists to Chroma (when available) or stores embeddings in memory. `_ensure_rag_ready` reuses this helper.
- **Retrieval chain**: `rag.ConversationalRetrievalChain` wraps the vector store with a supplied LLM callable, invoked by `/rag/query`. The `/rag/documents` endpoint inspects `_RAG_DOCUMENTS` populated during `_ensure_rag_ready` to expose metadata.
- **Tests already covering the flow**:
  - `tests/test_rag_pipeline.py` validates ingest→chunk→store→query→chain using the lightweight fallback backend.
  - `tests/test_rag_endpoints.py` exercises `/rag/documents` and `/rag/query` with the FastAPI wiring.

## WP2 Enhancement Targets
WP2 enhancements must **wrap and extend** the current helpers instead of rebuilding them:
- **Ingestion CLI wrapper**: add a CLI entry point that orchestrates `load_learning_materials` and `split_documents`, reusing `_ensure_rag_ready`'s configuration defaults so manual rebuilds feed the same vector store. New flows should call `rag.default_embedding_backend` when selecting embeddings to guarantee parity with the running service.
- **Configurable embedding backends**: introduce configuration hooks (env vars or dependency injection) that pass a selected backend into `default_embedding_backend`/`build_vector_store` without modifying the store/query API surface. Additional backends must implement `EmbeddingBackend` so `VectorStore.query` continues serving `/rag/query` unchanged.
- **Persistence options**: expand the `build_vector_store` pipeline with toggles for disk persistence (e.g., Chroma directories) while keeping `VectorStore.query` untouched so endpoint logic remains stable for `_ensure_rag_ready` and the FastAPI routes.
- **Monitoring/telemetry hooks**: layer optional logging around `_RAG_CHAIN.invoke`/`VectorStore.query` without altering return payloads expected by existing routes and tests.

## Deliverables & Test Updates
Future deliverables must layer onto the current modules:
- Update `tests/test_rag_pipeline.py` with additional cases for new ingestion flags or backend selection by calling the existing helpers (no bespoke pipelines).
- Extend `tests/test_rag_endpoints.py` to cover new endpoint parameters while continuing to call `_ensure_rag_ready`-provisioned globals.
- Ensure any CLI or configuration additions import from `rag.py`—particularly `load_learning_materials`, `split_documents`, `build_vector_store`, and `default_embedding_backend`—and reuse `_ensure_rag_ready` where applicable rather than duplicating logic.
