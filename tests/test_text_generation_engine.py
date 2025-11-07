import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from bloom_levels import BLOOM_LEVELS


class TextGenerationEngineTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._prev_db_path = os.environ.get("DB_PATH")
        os.environ["DB_PATH"] = os.path.join(self._tmpdir.name, "test.db")

        import db  # noqa: F401

        self.db = importlib.reload(db)
        self.db.init()

        from engines.progression import ProgressionEngine, ensure_progress_record
        from engines.text_generation import TextGenerationProgressionEngine

        self.ensure_progress = ensure_progress_record
        self.progression_engine = ProgressionEngine(advance_threshold=0.8, window_size=3, min_attempts=1)
        self.engine = TextGenerationProgressionEngine(self.progression_engine)

        self.subject_id = "language_zh_en"
        self.user_id = "learner-1"
        self.db.upsert_subject(self.subject_id, "Mandarin Foundations", "language", "HSK practice")
        self.ensure_progress(self.user_id, self.subject_id)

    def tearDown(self):
        if self._prev_db_path is None:
            os.environ.pop("DB_PATH", None)
        else:
            os.environ["DB_PATH"] = self._prev_db_path

        import db  # noqa: F401

        importlib.reload(db)
        self._tmpdir.cleanup()

    def test_simple_generation_vocab_levels(self):
        content = self.engine.generate_activity(
            self.user_id,
            self.subject_id,
            skill_level="HSK1",
            lexicon_mode="simple",
        )

        self.assertEqual(content["skill_level"], "HSK1")
        self.assertEqual(content["language_pair"], "zh_en")
        self.assertTrue(all(entry["hsk_level"] <= 1 for entry in content["word_assignments"]))
        self.assertTrue(
            any(ex["bloom_level"] == BLOOM_LEVELS.lowest_level() for ex in content["exercises"])
        )
        self.assertIn("listening_comprehension", content["prompt_suite"])
        self.assertIn("dialogue_simulation", content["prompt_suite"])
        self.assertIn("source", content["strategy_metadata"])
        self.assertTrue(content["strategy_metadata"]["source"].endswith("hsk_levels.json"))

        listening = content["prompt_suite"]["listening_comprehension"]
        self.assertIn("audio", listening)
        self.assertTrue(
            {"asset", "duration_seconds", "transcript"}.issubset(listening["audio"].keys())
        )
        translation_prompt = content["prompt_suite"]["translation"]
        for key in ("source_text", "target_language", "reference_translation"):
            self.assertIn(key, translation_prompt)
        pronunciation_prompt = content["prompt_suite"]["pronunciation"]
        self.assertTrue(
            ("target_tones" in pronunciation_prompt)
            or ("stress_pattern" in pronunciation_prompt)
        )
        self.assertTrue(all("target_bloom_level" in task for task in content["tasks"]))

        events = self.db.list_learning_events(self.user_id, self.subject_id, limit=1)
        self.assertEqual(events[0]["event_type"], "text_generation")
        self.assertTrue(events[0]["skill_id"].startswith("zh_en:HSK1"))
        details = json.loads(events[0]["details"])
        self.assertEqual(details["lexicon_mode"], "simple")

    def test_broad_generation_introduces_higher_vocab(self):
        content = self.engine.generate_activity(
            self.user_id,
            self.subject_id,
            skill_level="HSK3",
            lexicon_mode="broad",
        )

        self.assertEqual(content["lexicon_mode"], "broad")
        self.assertTrue(any(entry["hsk_level"] > 3 for entry in content["word_assignments"]))
        self.assertEqual(content["strategy"], "hsk")

    def test_response_logging_and_progression(self):
        first_activity = self.engine.generate_activity(
            self.user_id,
            self.subject_id,
            skill_level="HSK1",
            lexicon_mode="simple",
        )
        self.engine.evaluate_response(
            self.user_id,
            self.subject_id,
            activity_id=first_activity["activity_id"],
            score=0.9,
            response_text="我喜欢喝茶。",
            lexical_errors=[],
        )

        second_activity = self.engine.generate_activity(
            self.user_id,
            self.subject_id,
            skill_level="HSK1",
            lexicon_mode="simple",
        )
        result = self.engine.evaluate_response(
            self.user_id,
            self.subject_id,
            activity_id=second_activity["activity_id"],
            score=0.92,
            response_text="天气虽然冷，但是我很开心。",
            lexical_errors=["tone"],
        )

        progress = self.db.get_user_progress(self.user_id, self.subject_id)
        self.assertNotEqual(progress["current_level"], BLOOM_LEVELS.lowest_level())

        events = self.db.list_learning_events(self.user_id, self.subject_id, limit=4)
        self.assertIn("text_response", {row["event_type"] for row in events})
        self.assertTrue(all(row["skill_id"].startswith("zh_en") for row in events))

        self.assertIn("tone", " ".join(step["action"] for step in result["next_steps"]))
        self.assertTrue(self.engine.interaction_log)

    def test_cefr_strategy_and_prompts(self):
        german_subject = "language_de_en"
        self.db.upsert_subject(german_subject, "Deutsch Brückenkurs", "language", "CEFR practice")
        self.ensure_progress(self.user_id, german_subject)

        content = self.engine.generate_activity(
            self.user_id,
            german_subject,
            skill_level="A2",
            lexicon_mode="broad",
            language_pair="de_en",
        )

        self.assertEqual(content["strategy"], "cefr")
        self.assertEqual(content["language_pair"], "de_en")
        self.assertTrue(all("cefr_level" in entry for entry in content["word_assignments"]))
        self.assertIn("translation", content["prompt_suite"])
        self.assertTrue(content["prompt_suite"]["translation"]["reference_translation"].startswith("We"))
        self.assertIn("source", content["strategy_metadata"])
        self.assertTrue(
            content["strategy_metadata"]["source"].endswith("cefr_de_en_templates.json")
        )
        listening = content["prompt_suite"]["listening_comprehension"]
        self.assertIn("keywords", listening)
        self.assertGreater(len(listening["keywords"]), 0)
        pronunciation_prompt = content["prompt_suite"]["pronunciation"]
        self.assertTrue(
            ("target_tones" in pronunciation_prompt)
            or ("stress_pattern" in pronunciation_prompt)
        )

    def test_assessment_pipelines_log_skill_specific_events(self):
        activity = self.engine.generate_activity(
            self.user_id,
            self.subject_id,
            skill_level="HSK2",
            lexicon_mode="simple",
        )

        listening = self.engine.assess_listening_comprehension(
            self.user_id,
            self.subject_id,
            activity_id=activity["activity_id"],
            learner_response="我喝了咖啡然后去超市",
        )
        self.assertGreaterEqual(listening["score"], 0.5)

        pronunciation = self.engine.assess_pronunciation(
            self.user_id,
            self.subject_id,
            activity_id=activity["activity_id"],
            audio_metadata={"clarity": 0.8, "phoneme_accuracy": 0.7, "pacing": 0.6},
        )
        self.assertAlmostEqual(pronunciation["score"], 0.72)

        translation = self.engine.assess_translation(
            self.user_id,
            self.subject_id,
            activity_id=activity["activity_id"],
            learner_translation="I drank a cup of coffee and then went to the supermarket.",
        )
        self.assertAlmostEqual(translation["score"], 1.0)

        events = self.db.list_learning_events(self.user_id, self.subject_id, limit=6)
        event_types = {row["event_type"] for row in events}
        self.assertIn("listening_assessment", event_types)
        self.assertIn("pronunciation_assessment", event_types)
        self.assertIn("translation_assessment", event_types)
        self.assertTrue(all(row["skill_id"].startswith("zh_en:HSK2") for row in events if row["event_type"].endswith("assessment")))

    def test_language_strategy_templates_conform_to_schema(self):
        from engines import text_generation as tg

        strategies = tg._LANGUAGE_STRATEGIES
        self.assertGreaterEqual(len(strategies), 2)

        required_prompts = {
            "listening_comprehension",
            "translation",
            "dialogue_simulation",
            "pronunciation",
        }

        for pair, strategy in strategies.items():
            self.assertIn("strategy", strategy)
            self.assertIn("levels", strategy)

            metadata = strategy.get("metadata", {})
            source_path = metadata.get("source")
            self.assertIsInstance(source_path, str)

            payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
            self.assertEqual(payload.get("language_pair"), pair)

            levels = payload.get("levels", {})
            self.assertIsInstance(levels, dict)
            self.assertGreater(len(levels), 0)

            for level_id, level_data in levels.items():
                with self.subTest(language_pair=pair, level=level_id):
                    self.assertIsInstance(level_data.get("default_bloom"), str)
                    self.assertIsInstance(level_data.get("grammar_focus", []), list)
                    self.assertIsInstance(level_data.get("competencies", []), list)

                    for mode in ("simple", "broad"):
                        self.assertIn(mode, level_data)
                        mode_block = level_data[mode]
                        self.assertIsInstance(mode_block, dict)
                        for key in ("text_tokens", "vocab", "exercises", "tasks"):
                            self.assertIn(key, mode_block)

                    prompts = level_data.get("prompts", {})
                    self.assertIsInstance(prompts, dict)
                    self.assertTrue(required_prompts.issubset(prompts.keys()))

                    listening = prompts["listening_comprehension"]
                    self.assertIn("audio", listening)
                    self.assertIsInstance(listening.get("keywords", []), list)
                    audio = listening["audio"]
                    self.assertTrue({"asset", "duration_seconds", "transcript"}.issubset(audio.keys()))

                    pronunciation = prompts["pronunciation"]
                    self.assertTrue(
                        ("target_tones" in pronunciation)
                        or ("stress_pattern" in pronunciation)
                    )


if __name__ == "__main__":
    unittest.main()
