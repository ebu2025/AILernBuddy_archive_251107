import importlib
import os
import tempfile
import unittest


class MasteryDbTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._prev_db_path = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = os.path.join(self._tmpdir.name, "test.db")

        import db

        self.db = importlib.reload(db)
        self.db.init()

    def tearDown(self):
        if self._prev_db_path is None:
            os.environ.pop("DB_PATH", None)
        else:
            os.environ["DB_PATH"] = self._prev_db_path

        import db

        importlib.reload(db)
        self._tmpdir.cleanup()

    def test_upsert_reuses_single_row(self):
        self.db.upsert_mastery("alice", "math", 0.2)
        self.db.upsert_mastery("alice", "math", 0.8)

        rows = self.db.list_mastery("alice")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["level"], 0.8)

    def test_list_mastery_returns_latest_level(self):
        self.db.upsert_mastery("bob", "science", 0.1)
        first_rows = self.db.list_mastery("bob")
        self.assertEqual(len(first_rows), 1)
        self.assertAlmostEqual(first_rows[0]["level"], 0.1)

        self.db.upsert_mastery("bob", "science", 0.9)

        latest_rows = self.db.list_mastery("bob")
        self.assertEqual(len(latest_rows), 1)
        self.assertAlmostEqual(latest_rows[0]["level"], 0.9)

