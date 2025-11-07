"""Pipeline-level tests for the retrieval layer.

WP2 enhancements should build on these scenarios by extending the helpers in
``rag.py`` (``load_learning_materials``, ``split_documents``,
``build_vector_store``, ``default_embedding_backend``) and the FastAPI wiring
(``_ensure_rag_ready``, ``/rag/documents``, ``/rag/query``) rather than
introducing parallel ingestion/query flows.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import rag


class DummyEmbedding(rag.EmbeddingBackend):
    def embed(self, text: str) -> list[float]:
        lower = text.lower()
        features = [
            float(lower.count("pool")),
            float(lower.count("process")),
            float(lower.count("math")),
            float(len(lower)),
        ]
        return features


class RagPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.docs_dir = Path(self.tmpdir.name) / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        (self.docs_dir / "lesson1.txt").write_text(
            "BPMN pools organise participants in processes.",
            encoding="utf-8",
        )
        (self.docs_dir / "lesson2.md").write_text(
            "Mathematics K1 focuses on basic operations.",
            encoding="utf-8",
        )
        (self.docs_dir / "lessons.json").write_text(
            json.dumps(
                [
                    {
                        "id": "bpmn_intro",
                        "title": "BPMN intro",
                        "tags": ["bpmn"],
                        "content": "Pools represent participants in a collaboration diagram.",
                    },
                    {
                        "id": "math_addition",
                        "title": "Addition focus",
                        "content": "K1 learners add numbers within twenty by making tens.",
                    },
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_load_split_and_query(self):
        documents = rag.load_learning_materials(str(self.docs_dir))
        self.assertEqual(len(documents), 4)
        json_docs = [doc for doc in documents if doc.source.endswith("lessons.json#bpmn_intro")]
        self.assertTrue(json_docs)
        self.assertEqual(json_docs[0].metadata.get("title"), "BPMN intro")

        chunks = rag.split_documents(documents, chunk_size=40, chunk_overlap=10)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any("bpmn_intro" in chunk.metadata.get("id", "") for chunk in chunks if chunk.metadata))

        store = rag.build_vector_store(
            chunks,
            persist_directory=None,
            embedding_backend=DummyEmbedding(),
        )
        results = store.query("What are pools in BPMN?", k=2)
        self.assertTrue(results)
        self.assertTrue(any("pools" in chunk.content.lower() for chunk in results))

        chain = rag.ConversationalRetrievalChain(
            store,
            llm=lambda prompt: "Stub response",
        )
        response = chain.invoke("Explain pools", chat_history=[("user", "Hello")])
        self.assertIn("answer", response)
        self.assertEqual(response["answer"], "Stub response")
        self.assertTrue(response["context"])


if __name__ == "__main__":
    unittest.main()
