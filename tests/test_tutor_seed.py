import importlib
import json
import os
import tempfile
import unittest


class TutorSeedTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._prev_db_path = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = os.path.join(self._tmpdir.name, "seed.db")

        import db  # noqa: F401
        import tutor  # noqa: F401

        self.db = importlib.reload(db)
        self.tutor = importlib.reload(tutor)

    def tearDown(self):
        if self._prev_db_path is None:
            os.environ.pop("DB_PATH", None)
        else:
            os.environ["DB_PATH"] = self._prev_db_path

        import db  # noqa: F401
        import tutor  # noqa: F401

        importlib.reload(db)
        importlib.reload(tutor)
        self._tmpdir.cleanup()

    def test_ensure_seed_items_populates_database(self):
        # Should not raise even though the database file does not exist yet.
        self.tutor.ensure_seed_items()

        items = self.db.list_items(limit=10)
        self.assertGreater(len(items), 0)

        # Calling it again should be a no-op.
        self.tutor.ensure_seed_items()
        self.assertGreater(len(self.db.list_items(limit=10)), 0)

        item_bank_rows = self.db.list_item_bank(limit=10)
        self.assertGreater(len(item_bank_rows), 0)
        sample = dict(item_bank_rows[0])
        self.assertIn("bloom_level", sample)
        self.assertTrue(sample["bloom_level"])
        metadata_raw = sample.get("metadata_json")
        self.assertIsNotNone(metadata_raw)
        metadata = json.loads(metadata_raw)
        self.assertIn("topic", metadata)
        self.assertIn("tags", metadata)
        self.assertIsInstance(metadata["tags"], list)


if __name__ == "__main__":
    unittest.main()
