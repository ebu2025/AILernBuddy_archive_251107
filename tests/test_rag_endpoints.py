import asyncio
"""FastAPI endpoint tests for the retrieval interface.

Follow-on work should add cases here by exercising new parameters or modes on
the existing `/rag/*` routes so the coverage evolves with the current wiring
instead of replacing it.
"""

import json
from typing import Optional
from urllib.parse import urlencode

import app
import rag


class DummyEmbedding(rag.EmbeddingBackend):
    def embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [
            float(lower.count("bpmn")),
            float(lower.count("math")),
            float(len(lower)),
        ]


def _serialize_response(messages):
    status = 500
    body_bytes = b""
    for message in messages:
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            body_bytes += message.get("body", b"")
    data = json.loads(body_bytes.decode("utf-8") or "{}")
    return status, data


def _run_app(method: str, path: str, *, payload: Optional[dict] = None, query: Optional[dict] = None):
    body = b""
    headers = [(b"host", b"testserver")]
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers.extend(
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ]
        )
    query_string = urlencode(query or {}, doseq=True).encode()
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method.upper(),
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "scheme": "http",
        "query_string": query_string,
        "headers": headers,
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "state": {},
    }

    messages = []

    async def receive():
        nonlocal body
        if body:
            chunk, body = body, b""
            return {"type": "http.request", "body": chunk, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    async def _call():
        await app.app(scope, receive, send)
        return _serialize_response(messages)

    return asyncio.run(_call())


def test_rag_documents_endpoint():
    original_state = (
        app._RAG_INITIALIZED,
        app._RAG_DOCUMENTS,
        app._RAG_REMOTE_DOCUMENTS,
        app._RAG_STORE,
        app._RAG_CHAIN,
    )
    try:
        documents = [
            rag.Document(
                source="local://bpmn",
                content="BPMN pools coordinate participants.",
                metadata={"id": "bpmn_pool", "title": "BPMN Pools", "tags": ["bpmn"]},
            )
        ]
        app._RAG_INITIALIZED = True
        app._RAG_DOCUMENTS = documents
        app._RAG_REMOTE_DOCUMENTS = []
        app._RAG_STORE = None
        app._RAG_CHAIN = None

        status, payload = _run_app("GET", "/rag/documents")
    finally:
        (
            app._RAG_INITIALIZED,
            app._RAG_DOCUMENTS,
            app._RAG_REMOTE_DOCUMENTS,
            app._RAG_STORE,
            app._RAG_CHAIN,
        ) = original_state

    assert status == 200
    assert payload["count"] == 1
    doc = payload["documents"][0]
    assert doc["title"] == "BPMN Pools"
    assert "pools" in doc["preview"].lower()


def test_rag_query_endpoint():
    original_state = (
        app._RAG_INITIALIZED,
        app._RAG_DOCUMENTS,
        app._RAG_REMOTE_DOCUMENTS,
        app._RAG_STORE,
        app._RAG_CHAIN,
    )
    try:
        documents = [
            rag.Document(
                source="local://bpmn",
                content="Pools organise participants while tasks capture the work.",
                metadata={"id": "bpmn_pool"},
            )
        ]
        chunks = rag.split_documents(documents, chunk_size=80, chunk_overlap=0)
        store = rag.build_vector_store(chunks, embedding_backend=DummyEmbedding())
        chain = rag.ConversationalRetrievalChain(store, llm=lambda prompt: "Stub answer")

        app._RAG_INITIALIZED = True
        app._RAG_DOCUMENTS = documents
        app._RAG_REMOTE_DOCUMENTS = []
        app._RAG_STORE = store
        app._RAG_CHAIN = chain

        status, payload = _run_app(
            "POST",
            "/rag/query",
            payload={"question": "What are pools in BPMN?", "k": 1},
        )
    finally:
        (
            app._RAG_INITIALIZED,
            app._RAG_DOCUMENTS,
            app._RAG_REMOTE_DOCUMENTS,
            app._RAG_STORE,
            app._RAG_CHAIN,
        ) = original_state

    assert status == 200
    assert payload["answer"] == "Stub answer"
    assert payload["context"]
    assert payload["context"][0]["metadata"]["id"] == "bpmn_pool"
