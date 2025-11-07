import json
import math
import os
import sqlite3
import statistics
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Sequence, Union, cast

from bloom_levels import BLOOM_LEVELS
from schemas import LearnerModel
from db_pool import SQLiteConnectionPool

try:
    _LOWEST_BLOOM_LEVEL = BLOOM_LEVELS.lowest_level()
except Exception:
    _LOWEST_BLOOM_LEVEL = "K1"

_DIAGNOSIS_VALUES = {"conceptual", "procedural", "careless", "none"}

if TYPE_CHECKING:
    from schemas import AssessmentErrorPattern, AssessmentResult, AssessmentStepEvaluation

DB_PATH = os.getenv("DB_PATH", "data.db")

# Initialize connection pool
_pool = SQLiteConnectionPool(DB_PATH, max_connections=10)


def _conn():
    """Return a context manager for acquiring a pooled SQLite connection."""
    return _pool.get_connection()


def _exec(sql: str, params: Iterable = ()):
    with _pool.get_connection() as con:
        cur = con.execute(sql, params)
        con.commit()
        return cur

def _query(sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
    with _pool.get_connection() as con:
        cur = con.execute(sql, params)
        return cur.fetchall()

def _init_tables() -> None:
    """Initialize required tables if they don't exist."""
    _exec(
        """
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            intervention_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            context TEXT NOT NULL,
            intervention TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    _exec(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            profile TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

def log_intervention(
    user_id: str,
    intervention_type: str,
    confidence: float,
    context: Dict[str, Any],
    intervention: Dict[str, Any]
) -> None:
    """Log an intervention event to the database."""
    _init_tables()
    _exec(
        """
        INSERT INTO interventions
        (user_id, intervention_type, confidence, context, intervention)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            intervention_type,
            confidence,
            json.dumps(context),
            json.dumps(intervention)
        )
    )


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Return the stored profile for ``user_id`` if it exists."""
    _init_tables()
    rows = _query("SELECT profile FROM users WHERE id = ?", [user_id])
    if not rows:
        return None

    profile = rows[0]["profile"]
    if not profile:
        return {}

    try:
        return json.loads(profile)
    except json.JSONDecodeError:
        return {}


def update_user_profile(user_id: str, profile: Dict[str, Any]) -> None:
    """Upsert ``profile`` for ``user_id`` in the ``users`` table."""
    _init_tables()
    _exec(
        """
        INSERT INTO users (id, profile, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            profile = excluded.profile,
            updated_at = CURRENT_TIMESTAMP
        """,
        [user_id, json.dumps(profile)]
    )

# -------------- schema helpers --------------
def _add_column_if_missing(con: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    try:
        info = con.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row[1] for row in info}
        if column not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    except sqlite3.OperationalError:
        # Older SQLite builds may raise when table missing; safe to ignore here.
        pass


def _table_has_level_check(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    if not row or not row[0]:
        return False
    sql = row[0].upper()
    return "CHECK" in sql and "LEVEL" in sql


def _recreate_table(
    con: sqlite3.Connection,
    table: str,
    definition: str,
    columns: Sequence[str],
    indexes: Sequence[str] | None = None,
) -> None:
    column_list = ", ".join(columns)
    con.execute("PRAGMA foreign_keys=OFF")
    try:
        con.execute(f"ALTER TABLE {table} RENAME TO {table}_old")
        con.execute(definition)
        con.execute(
            f"INSERT INTO {table} ({column_list}) SELECT {column_list} FROM {table}_old"
        )
        con.execute(f"DROP TABLE {table}_old")
        for index_sql in indexes or ():
            con.execute(index_sql)
    finally:
        con.execute("PRAGMA foreign_keys=ON")


def _migrate_level_checks(con: sqlite3.Connection) -> None:
    migrations = {
        "modules": (
            """
            CREATE TABLE modules (
              module_id    TEXT PRIMARY KEY,
              subject_id   TEXT NOT NULL,
              title        TEXT NOT NULL,
              k_level      TEXT NOT NULL,
              description  TEXT,
              position     INTEGER DEFAULT 0,
              created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
            )
            """,
            [
                "module_id",
                "subject_id",
                "title",
                "k_level",
                "description",
                "position",
                "created_at",
            ],
            ["CREATE INDEX IF NOT EXISTS idx_modules_subject ON modules(subject_id)"]
        ),
        "activities": (
            """
            CREATE TABLE activities (
              activity_id   TEXT PRIMARY KEY,
              lesson_id     TEXT NOT NULL,
              activity_type TEXT NOT NULL,
              content       TEXT NOT NULL,
              target_level  TEXT,
              metadata      TEXT,
              created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(lesson_id) REFERENCES lessons(lesson_id) ON DELETE CASCADE
            )
            """,
            [
                "activity_id",
                "lesson_id",
                "activity_type",
                "content",
                "target_level",
                "metadata",
                "created_at",
            ],
            ["CREATE INDEX IF NOT EXISTS idx_activities_lesson ON activities(lesson_id)"]
        ),
        "user_progress": (
            """
            CREATE TABLE user_progress (
              user_id       TEXT NOT NULL,
              subject_id    TEXT NOT NULL,
              current_level TEXT NOT NULL,
              confidence    REAL DEFAULT 0.0,
              updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (user_id, subject_id),
              FOREIGN KEY(subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
            )
            """,
            ["user_id", "subject_id", "current_level", "confidence", "updated_at"],
            [],
        ),
    }

    for table, (definition, columns, indexes) in migrations.items():
        if _table_has_level_check(con, table):
            _recreate_table(con, table, definition, columns, indexes)

# -------------- schema --------------
_MASTERY_CONSTRAINT_AVAILABLE: Optional[bool] = None


def init():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.executescript(
            f"""
            PRAGMA foreign_keys = ON;
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS users (
              user_id     TEXT PRIMARY KEY,
              email       TEXT UNIQUE,
              pw_hash     TEXT NOT NULL,
              pw_salt     TEXT,
              created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS prompts (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              topic       TEXT NOT NULL,
              prompt_text TEXT NOT NULL,
              source      TEXT DEFAULT 'seed',
              created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_prompts_topic ON prompts(topic);

            CREATE TABLE IF NOT EXISTS journey_log (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id     TEXT NOT NULL,
              op          TEXT NOT NULL,
              payload     TEXT,
              created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_journey_user ON journey_log(user_id);

            CREATE TABLE IF NOT EXISTS chat_ops_log (
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id       TEXT NOT NULL,
              topic         TEXT,
              question      TEXT,
              answer        TEXT,
              raw_response  TEXT,
              response_json TEXT,
              applied_ops   TEXT,
              pending_ops   TEXT,
              created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chat_ops_user ON chat_ops_log(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS items (
              id          TEXT PRIMARY KEY,
              skill       TEXT NOT NULL,
              difficulty  REAL DEFAULT 0.0,
              body        TEXT NOT NULL,
              created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_items_skill ON items(skill);

            CREATE TABLE IF NOT EXISTS item_bank (
              id              TEXT PRIMARY KEY,
              domain          TEXT NOT NULL,
              skill_id        TEXT NOT NULL,
              bloom_level     TEXT NOT NULL,
              stimulus        TEXT NOT NULL,
              elo_target      REAL NOT NULL,
              answer_key      TEXT,
              rubric_id       TEXT,
              difficulty      REAL,
              exposure_limit  INTEGER DEFAULT 0,
              metadata_json   TEXT,
              references_json TEXT,
              created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_item_bank_domain ON item_bank(domain, skill_id);

            CREATE TABLE IF NOT EXISTS item_exposure (
              item_id      TEXT PRIMARY KEY,
              served_count INTEGER NOT NULL DEFAULT 0,
              last_served  TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS subjects (
              subject_id   TEXT PRIMARY KEY,
              name         TEXT NOT NULL,
              theme        TEXT NOT NULL,
              description  TEXT,
              created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS modules (
              module_id    TEXT PRIMARY KEY,
              subject_id   TEXT NOT NULL,
              title        TEXT NOT NULL,
              k_level      TEXT NOT NULL,
              description  TEXT,
              position     INTEGER DEFAULT 0,
              created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_modules_subject ON modules(subject_id);

            CREATE TABLE IF NOT EXISTS lessons (
              lesson_id    TEXT PRIMARY KEY,
              module_id    TEXT NOT NULL,
              title        TEXT NOT NULL,
              summary      TEXT,
              position     INTEGER DEFAULT 0,
              created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(module_id) REFERENCES modules(module_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_lessons_module ON lessons(module_id);

            CREATE TABLE IF NOT EXISTS activities (
              activity_id   TEXT PRIMARY KEY,
              lesson_id     TEXT NOT NULL,
              activity_type TEXT NOT NULL,
              content       TEXT NOT NULL,
              target_level  TEXT,
              metadata      TEXT,
              created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(lesson_id) REFERENCES lessons(lesson_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_activities_lesson ON activities(lesson_id);

            CREATE TABLE IF NOT EXISTS mastery (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id     TEXT NOT NULL,
              skill       TEXT NOT NULL,
              level       REAL DEFAULT 0.0,
              updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(user_id, skill)
            );

            CREATE INDEX IF NOT EXISTS idx_mastery_user ON mastery(user_id);

            CREATE TABLE IF NOT EXISTS user_progress (
              user_id       TEXT NOT NULL,
              subject_id    TEXT NOT NULL,
              current_level TEXT NOT NULL,
              confidence    REAL DEFAULT 0.0,
              band_lower    TEXT,
              band_upper    TEXT,
              target_probability REAL,
              ci_lower      REAL,
              ci_upper      REAL,
              ci_width      REAL,
              updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (user_id, subject_id),
              FOREIGN KEY(subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS learning_events (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id     TEXT NOT NULL,
              subject_id  TEXT NOT NULL,
              lesson_id   TEXT,
              event_type  TEXT NOT NULL,
              score       REAL,
              details     TEXT,
              created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
              FOREIGN KEY(lesson_id) REFERENCES lessons(lesson_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_learning_events_user ON learning_events(user_id, subject_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS quiz_attempts (
              id              INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id         TEXT NOT NULL,
              subject_id      TEXT NOT NULL,
              activity_id     TEXT NOT NULL,
              score           REAL NOT NULL,
              max_score       REAL NOT NULL,
              normalized_score REAL NOT NULL,
              pass_threshold  REAL NOT NULL,
              passed          INTEGER NOT NULL,
              confidence      REAL DEFAULT 1.0,
              path            TEXT DEFAULT 'direct',
              diagnosis       TEXT,
              self_assessment TEXT,
              created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
              FOREIGN KEY(activity_id) REFERENCES activities(activity_id) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_quiz_attempts_user ON quiz_attempts(user_id, subject_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS xapi_statements (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id     TEXT NOT NULL,
              verb        TEXT NOT NULL,
              object_id   TEXT NOT NULL,
              score       REAL,
              success     INTEGER,
              response    TEXT,
              context     TEXT,
              created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_xapi_statements_user ON xapi_statements(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS assessment_results (
              id             INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id        TEXT NOT NULL,
              domain         TEXT NOT NULL,
              item_id        TEXT NOT NULL,
              bloom_level    TEXT NOT NULL,
              response       TEXT NOT NULL,
              self_assessment TEXT,
              score          REAL NOT NULL,
              rubric_criteria TEXT NOT NULL,
              model_version  TEXT NOT NULL,
              prompt_version TEXT NOT NULL,
              latency_ms     INTEGER,
              tokens_in      INTEGER,
              tokens_out     INTEGER,
              confidence     REAL DEFAULT 0.0,
              source         TEXT NOT NULL DEFAULT 'direct',
              created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_assessment_user ON assessment_results(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS assessment_step_results (
              id             INTEGER PRIMARY KEY AUTOINCREMENT,
              assessment_id  INTEGER NOT NULL,
              step_id        TEXT NOT NULL,
              subskill       TEXT,
              outcome        TEXT NOT NULL,
              score_delta    REAL,
              hint           TEXT,
              feedback       TEXT,
              diagnosis      TEXT,
              created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(assessment_id) REFERENCES assessment_results(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_assessment_step_results_assessment
              ON assessment_step_results(assessment_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS assessment_error_patterns (
              id             INTEGER PRIMARY KEY AUTOINCREMENT,
              assessment_id  INTEGER NOT NULL,
              pattern_code   TEXT NOT NULL,
              description    TEXT,
              subskill       TEXT,
              occurrences    INTEGER DEFAULT 1,
              created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(assessment_id) REFERENCES assessment_results(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_assessment_error_patterns_assessment
              ON assessment_error_patterns(assessment_id);

            CREATE TABLE IF NOT EXISTS llm_metrics (
              id             INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id        TEXT,
              model_id       TEXT NOT NULL,
              prompt_version TEXT NOT NULL,
              prompt_variant TEXT,
              latency_ms     INTEGER NOT NULL,
              tokens_in      INTEGER,
              tokens_out     INTEGER,
              path_taken     TEXT,
              json_validated INTEGER DEFAULT 0,
              created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_llm_metrics_user ON llm_metrics(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS assessment_followups (
              user_id              TEXT NOT NULL,
              topic                TEXT NOT NULL,
              needs_assessment     INTEGER NOT NULL DEFAULT 0,
              microcheck_question  TEXT,
              microcheck_answer_key TEXT,
              microcheck_rubric    TEXT,
              microcheck_source    TEXT,
              created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (user_id, topic)
            );

            CREATE TABLE IF NOT EXISTS learning_path_state (
              user_id     TEXT NOT NULL,
              subject_id  TEXT NOT NULL,
              state_json  TEXT NOT NULL,
              updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (user_id, subject_id)
            );

            CREATE TABLE IF NOT EXISTS learning_path_events (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id     TEXT NOT NULL,
              subject_id  TEXT NOT NULL,
              bloom_level TEXT NOT NULL,
              action      TEXT NOT NULL,
              reason_code TEXT,
              reason      TEXT,
              confidence  REAL,
              evidence    TEXT,
              created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_learning_path_events_user
              ON learning_path_events(user_id, subject_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS learner_profile (
              user_id           TEXT PRIMARY KEY,
              goals_json        TEXT NOT NULL DEFAULT '[]',
              preferences_json  TEXT,
              history_summary   TEXT,
              created_at        TIMESTAMP NOT NULL,
              updated_at        TIMESTAMP NOT NULL
            );

            CREATE TABLE IF NOT EXISTS learner_priors (
              user_id     TEXT NOT NULL,
              skill_id    TEXT NOT NULL,
              proficiency REAL,
              bloom_low   TEXT,
              bloom_high  TEXT,
              created_at  TIMESTAMP NOT NULL,
              updated_at  TIMESTAMP NOT NULL,
              PRIMARY KEY (user_id, skill_id)
            );

            CREATE INDEX IF NOT EXISTS idx_learner_priors_user
              ON learner_priors(user_id);

            CREATE TABLE IF NOT EXISTS learner_confidence (
              user_id    TEXT NOT NULL,
              skill_id   TEXT NOT NULL,
              confidence REAL,
              created_at TIMESTAMP NOT NULL,
              updated_at TIMESTAMP NOT NULL,
              PRIMARY KEY (user_id, skill_id)
            );

            CREATE INDEX IF NOT EXISTS idx_learner_confidence_user
              ON learner_confidence(user_id);

            CREATE TABLE IF NOT EXISTS learner_misconceptions (
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id       TEXT NOT NULL,
              skill_id      TEXT,
              description   TEXT NOT NULL,
              severity      TEXT,
              evidence_json TEXT,
              last_seen     TIMESTAMP,
              created_at    TIMESTAMP NOT NULL,
              updated_at    TIMESTAMP NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_learner_misconceptions_user
              ON learner_misconceptions(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS answer_feedback (
              id         INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id    TEXT,
              answer_id  TEXT NOT NULL,
              rating     TEXT NOT NULL,
              comment    TEXT,
              confidence REAL,
              tags       TEXT,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_answer_feedback_answer
              ON answer_feedback(answer_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS teacher_analytics (
              user_id      TEXT NOT NULL,
              subject_id   TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY (user_id, subject_id)
            );

            CREATE TABLE IF NOT EXISTS user_consent (
              user_id      TEXT PRIMARY KEY,
              consented    INTEGER NOT NULL,
              consent_text TEXT,
              consented_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS pending_ops (
              id         INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id    TEXT NOT NULL,
              topic      TEXT,
              payload    TEXT NOT NULL,
              status     TEXT NOT NULL DEFAULT 'open',
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_pending_ops_status ON pending_ops(status, created_at);

            CREATE TABLE IF NOT EXISTS copilot_plans (
              id             INTEGER PRIMARY KEY AUTOINCREMENT,
              teacher_id     TEXT NOT NULL,
              topic          TEXT,
              objectives     TEXT,
              plan_json      TEXT NOT NULL,
              bloom_alignment TEXT,
              provenance     TEXT,
              status         TEXT NOT NULL DEFAULT 'draft',
              created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              updated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS copilot_moderation (
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              plan_id       INTEGER NOT NULL,
              moderator_id  TEXT NOT NULL,
              decision      TEXT NOT NULL,
              rationale     TEXT,
              flags         TEXT,
              created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(plan_id) REFERENCES copilot_plans(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_copilot_plans_teacher ON copilot_plans(teacher_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_copilot_moderation_plan ON copilot_moderation(plan_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS eval_reports (
              id         INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id     TEXT NOT NULL,
              probe_id   TEXT NOT NULL,
              category   TEXT NOT NULL,
              metrics    TEXT NOT NULL,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_eval_reports_run ON eval_reports(run_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS eval_instruments (
              id               INTEGER PRIMARY KEY AUTOINCREMENT,
              instrument_id    TEXT NOT NULL,
              topic            TEXT NOT NULL,
              stage            TEXT NOT NULL,
              version          TEXT,
              title            TEXT,
              description      TEXT,
              items            TEXT NOT NULL,
              metadata         TEXT,
              created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(instrument_id, stage)
            );

            CREATE INDEX IF NOT EXISTS idx_eval_instruments_topic_stage
              ON eval_instruments(topic, stage, updated_at DESC);

            CREATE TABLE IF NOT EXISTS eval_session_instruments (
              id             INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id     TEXT NOT NULL,
              stage          TEXT NOT NULL,
              instrument_ref INTEGER NOT NULL,
              attached_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(session_id, stage),
              FOREIGN KEY(instrument_ref) REFERENCES eval_instruments(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS eval_pretest_attempts (
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              learner_id    TEXT NOT NULL,
              topic         TEXT NOT NULL,
              score         REAL NOT NULL,
              max_score     REAL NOT NULL,
              attempt_id    TEXT,
              strategy      TEXT,
              metadata      TEXT,
              session_id    TEXT,
              instrument_ref INTEGER,
              instrument_id TEXT,
              instrument_version TEXT,
              attempted_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS eval_posttest_attempts (
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              learner_id    TEXT NOT NULL,
              topic         TEXT NOT NULL,
              score         REAL NOT NULL,
              max_score     REAL NOT NULL,
              attempt_id    TEXT,
              strategy      TEXT,
              metadata      TEXT,
              session_id    TEXT,
              instrument_ref INTEGER,
              instrument_id TEXT,
              instrument_version TEXT,
              attempted_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_eval_pretest_lookup ON eval_pretest_attempts(learner_id, topic, id DESC);
            CREATE INDEX IF NOT EXISTS idx_eval_posttest_lookup ON eval_posttest_attempts(learner_id, topic, id DESC);

            CREATE TABLE IF NOT EXISTS bloom_score (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id     TEXT NOT NULL,
              topic       TEXT NOT NULL,
              level       TEXT NOT NULL,
              score       INTEGER NOT NULL,
              updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(user_id, topic)
            );

            CREATE TABLE IF NOT EXISTS bloom_progress (
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id       TEXT NOT NULL,
              topic         TEXT NOT NULL,
              current_level TEXT NOT NULL,
              updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(user_id, topic)
            );

            CREATE INDEX IF NOT EXISTS idx_bloom_progress_user ON bloom_progress(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS bloom_progress_history (
              id                   INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id              TEXT NOT NULL,
              topic                TEXT NOT NULL,
              previous_level       TEXT,
              new_level            TEXT NOT NULL,
              k_level              TEXT,
              reason               TEXT,
              average_score        REAL,
              attempts_considered  INTEGER,
              created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_bloom_progress_history_user
              ON bloom_progress_history(user_id, topic, created_at DESC);
            """
        )
        _add_column_if_missing(con, "learning_events", "skill_id", "TEXT")
        _migrate_level_checks(con)
        _ensure_mastery_unique_index(con)
        # Clean up potential duplicates before enforcing the unique index.
        con.execute(
            """
            DELETE FROM mastery
            WHERE id NOT IN (
                SELECT id FROM (
                    SELECT m1.id
                    FROM mastery AS m1
                    LEFT JOIN mastery AS m2
                      ON m1.user_id = m2.user_id
                     AND m1.skill = m2.skill
                     AND (m1.updated_at < m2.updated_at
                          OR (m1.updated_at = m2.updated_at AND m1.id < m2.id))
                    WHERE m2.id IS NULL
                )
            )
            """
        )

        con.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mastery_user_skill
            ON mastery(user_id, skill)
            """
        )

        con.commit()

        _add_column_if_missing(con, "learning_path_events", "reason_code", "TEXT")
        _add_column_if_missing(con, "assessment_results", "confidence", "REAL DEFAULT 0.0")
        _add_column_if_missing(con, "assessment_results", "source", "TEXT DEFAULT 'direct'")
        _add_column_if_missing(con, "assessment_results", "self_assessment", "TEXT")
        _add_column_if_missing(con, "quiz_attempts", "confidence", "REAL DEFAULT 1.0")
        _add_column_if_missing(con, "quiz_attempts", "path", "TEXT DEFAULT 'direct'")
        _add_column_if_missing(con, "quiz_attempts", "diagnosis", "TEXT")
        _add_column_if_missing(con, "quiz_attempts", "self_assessment", "TEXT")
        _add_column_if_missing(con, "item_bank", "elo_target", "REAL DEFAULT 0")
        _add_column_if_missing(con, "user_progress", "band_lower", "TEXT")
        _add_column_if_missing(con, "user_progress", "band_upper", "TEXT")
        _add_column_if_missing(con, "user_progress", "target_probability", "REAL")
        _add_column_if_missing(con, "user_progress", "ci_lower", "REAL")
        _add_column_if_missing(con, "user_progress", "ci_upper", "REAL")
        _add_column_if_missing(con, "user_progress", "ci_width", "REAL")
        _add_column_if_missing(con, "llm_metrics", "prompt_variant", "TEXT")
        _add_column_if_missing(con, "llm_metrics", "path_taken", "TEXT")
        _add_column_if_missing(con, "llm_metrics", "json_validated", "INTEGER DEFAULT 0")
        _add_column_if_missing(con, "chat_ops_log", "raw_response", "TEXT")
        _add_column_if_missing(con, "users", "pw_salt", "TEXT")

# -------------- users / auth --------------
def get_user_auth(user_id: str) -> Optional[sqlite3.Row]:
    rows = _query("SELECT id, pw_hash, pw_salt FROM users WHERE id = ?", (user_id,))
    return rows[0] if rows else None

def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    rows = _query("SELECT id FROM users WHERE email = ?", (email,))
    return rows[0] if rows else None

def create_user(user_id: str, email: Optional[str], pw_hash: str, pw_salt: Optional[str] = None):
    _exec(
        "INSERT INTO users(id, pw_hash, pw_salt) VALUES (?,?,?)",
        (user_id, pw_hash, pw_salt),
    )


def update_user_password(user_id: str, pw_hash: str, pw_salt: Optional[str]) -> None:
    _exec(
        "UPDATE users SET pw_hash = ?, pw_salt = ? WHERE id = ?",
        (pw_hash, pw_salt, user_id),
    )

def ensure_user(user_id: str):
    rows = _query("SELECT 1 FROM users WHERE id = ?", (user_id,))
    if not rows:
        # Minimal setup without email/password (for guests, etc.)
        _exec(
            "INSERT INTO users(id) VALUES (?)",
            (user_id,),
        )

# -------------- prompts --------------
def add_prompt(topic: str, prompt_text: str, source: str = "seed"):
    _exec(
        "INSERT INTO prompts(topic, prompt_text, source) VALUES (?,?,?)",
        (topic, prompt_text, source)
    )

def get_prompts_for_topic(topic: str, limit: int = 5) -> list[sqlite3.Row]:
    return _query(
        "SELECT id, topic, prompt_text, source, created_at FROM prompts WHERE topic = ? ORDER BY id DESC LIMIT ?",
        (topic, int(limit))
    )

def list_prompts(topic: Optional[str] = None, limit: int = 50) -> list[sqlite3.Row]:
    if topic:
        return _query(
            "SELECT id, topic, prompt_text, source, created_at FROM prompts WHERE topic = ? ORDER BY id DESC LIMIT ?",
            (topic, int(limit))
        )
    return _query(
        "SELECT id, topic, prompt_text, source, created_at FROM prompts ORDER BY id DESC LIMIT ?",
        (int(limit),)
    )

# -------------- journey / ops-log --------------
def log_journey_update(user_id: str, op: str, payload: Dict[str, Any]):
    _exec(
        "INSERT INTO journey_log(user_id, op, payload) VALUES (?,?,?)",
        (user_id, op, json_dumps(payload))
    )

def _decode_json_field(value: Optional[str]) -> Any:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    value = value.strip()
    if not value:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now()
    text = str(value).strip()
    if not text:
        return datetime.now()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.now()


def _coerce_to_utc(dt: Optional[datetime], fallback: Optional[datetime] = None) -> datetime:
    if dt is None:
        dt = fallback or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def list_journey(user_id: Optional[str] = None, limit: int = 100) -> list[Dict[str, Any]]:
    if user_id:
        rows = _query(
            "SELECT id, user_id, op, payload, created_at FROM journey_log WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, int(limit))
        )
    else:
        rows = _query(
            "SELECT id, user_id, op, payload, created_at FROM journey_log ORDER BY id DESC LIMIT ?",
            (int(limit),)
        )

    data: list[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["payload"] = _decode_json_field(item.get("payload"))
        data.append(item)
    return data


def list_recent_recommendations(user_id: str, limit: int = 5) -> list[Dict[str, Any]]:
    rows = _query(
        """
        SELECT payload, created_at
        FROM journey_log
        WHERE user_id = ? AND op = 'suggest_next_item'
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (user_id, int(limit)),
    )
    recommendations: list[Dict[str, Any]] = []
    for row in rows:
        payload = _decode_json_field(row["payload"]) or {}
        if isinstance(payload, dict):
            entry: Dict[str, Any] = {
                "skill": payload.get("skill") or payload.get("target_skill"),
                "activity": payload.get("activity") or payload.get("activity_id"),
                "modality": payload.get("modality"),
                "reason": payload.get("reason") or payload.get("note"),
                "created_at": row["created_at"],
            }
            plan = payload.get("plan")
            if isinstance(plan, dict):
                highlights = plan.get("preference_highlights")
                if isinstance(highlights, list) and highlights:
                    entry["preference_highlights"] = highlights
                matches = plan.get("preference_matches")
                if isinstance(matches, dict) and matches:
                    entry["preference_matches"] = matches
                applied = plan.get("preferences_applied") or plan.get("preferences")
                if isinstance(applied, dict) and applied:
                    entry["preferences_applied"] = applied
        else:
            entry = {"skill": None, "reason": payload, "created_at": row["created_at"]}
        recommendations.append(entry)
    return recommendations


def list_recent_assessments(user_id: str, topic: Optional[str] = None, limit: int = 5) -> list[Dict[str, Any]]:
    sql = (
        "SELECT id, domain, item_id, bloom_level, score, confidence, source, self_assessment, created_at "
        "FROM assessment_results WHERE user_id = ?"
    )
    params: list[Any] = [user_id]
    if topic:
        sql += " AND domain = ?"
        params.append(topic)
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(int(limit))
    rows = _query(sql, params)
    data: list[Dict[str, Any]] = []
    for row in rows:
        data.append(
            {
                "assessment_id": int(row["id"]),
                "domain": row["domain"],
                "item_id": row["item_id"],
                "bloom_level": row["bloom_level"],
                "score": float(row["score"]),
                "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
                "source": row["source"],
                "self_assessment": row["self_assessment"],
                "created_at": row["created_at"],
            }
        )
    return data


def batch_learner_profile_data(
    user_id: str,
    *,
    mastery_limit: int = 200,
    bloom_limit: int = 200,
    assessment_limit: int = 50,
) -> Dict[str, Any]:
    """Fetch learner profile data for ``profile`` in a single batched query.

    The returned structure mirrors the output of ``list_mastery``,
    ``list_bloom_progress`` and ``list_recent_assessments`` while also
    including follow-up state records indexed by topic.
    """

    if mastery_limit <= 0:
        mastery_limit = 1
    if bloom_limit <= 0:
        bloom_limit = 1
    if assessment_limit <= 0:
        assessment_limit = 1

    query = """
    WITH mastery_limited AS (
        SELECT id, user_id, skill, level, updated_at
        FROM mastery
        WHERE user_id = ?
        ORDER BY updated_at DESC, id DESC
        LIMIT ?
    ),
    bloom_limited AS (
        SELECT id, user_id, topic, current_level, updated_at
        FROM bloom_progress
        WHERE user_id = ?
        ORDER BY updated_at DESC, id DESC
        LIMIT ?
    ),
    recent_assessments AS (
        SELECT
            id,
            user_id,
            domain,
            item_id,
            bloom_level,
            score,
            confidence,
            source,
            self_assessment,
            created_at,
            ROW_NUMBER() OVER (ORDER BY created_at DESC, id DESC) AS global_rank
        FROM assessment_results
        WHERE user_id = ?
    ),
    topic_union AS (
        SELECT skill AS topic FROM mastery_limited
        UNION
        SELECT topic FROM bloom_limited
        UNION
        SELECT topic FROM assessment_followups WHERE user_id = ?
        UNION
        SELECT domain FROM recent_assessments WHERE global_rank <= ?
    )
    SELECT
        tu.topic AS topic,
        ml.id AS mastery_id,
        ml.user_id AS mastery_user_id,
        ml.skill AS mastery_skill,
        ml.level AS mastery_level,
        ml.updated_at AS mastery_updated_at,
        bp.id AS bloom_id,
        bp.user_id AS bloom_user_id,
        bp.topic AS bloom_topic,
        bp.current_level AS bloom_level,
        bp.updated_at AS bloom_updated_at,
        af.user_id AS followup_user_id,
        af.topic AS followup_topic,
        af.needs_assessment AS followup_needs_assessment,
        af.microcheck_question AS followup_microcheck_question,
        af.microcheck_answer_key AS followup_microcheck_answer_key,
        af.microcheck_rubric AS followup_microcheck_rubric,
        af.microcheck_source AS followup_microcheck_source,
        af.created_at AS followup_created_at,
        af.updated_at AS followup_updated_at,
        ra.id AS assessment_id,
        ra.domain AS assessment_domain,
        ra.item_id AS assessment_item_id,
        ra.bloom_level AS assessment_bloom_level,
        ra.score AS assessment_score,
        ra.confidence AS assessment_confidence,
        ra.source AS assessment_source,
        ra.self_assessment AS assessment_self_assessment,
        ra.created_at AS assessment_created_at,
        ra.global_rank AS assessment_rank
    FROM topic_union tu
    LEFT JOIN mastery_limited ml ON ml.skill = tu.topic
    LEFT JOIN bloom_limited bp ON bp.topic = tu.topic
    LEFT JOIN assessment_followups af ON af.user_id = ? AND af.topic = tu.topic
    LEFT JOIN (SELECT * FROM recent_assessments WHERE global_rank <= ?) ra ON ra.domain = tu.topic
    ORDER BY tu.topic ASC, assessment_rank ASC
    """

    params: list[Any] = [
        user_id,
        int(mastery_limit),
        user_id,
        int(bloom_limit),
        user_id,
        user_id,
        int(assessment_limit),
        user_id,
        int(assessment_limit),
    ]

    mastery_map: Dict[int, Dict[str, Any]] = {}
    bloom_map: Dict[int, Dict[str, Any]] = {}
    followups: Dict[str, Dict[str, Any]] = {}
    assessment_map: Dict[int, Dict[str, Any]] = {}

    with _conn() as con:
        cur = con.execute(query, params)
        rows = cur.fetchall()

    for row in rows:
        topic = row["topic"]

        mastery_id = row["mastery_id"]
        if mastery_id is not None and mastery_id not in mastery_map:
            mastery_map[int(mastery_id)] = {
                "id": int(mastery_id),
                "user_id": row["mastery_user_id"],
                "skill": row["mastery_skill"],
                "level": float(row["mastery_level"]) if row["mastery_level"] is not None else None,
                "updated_at": row["mastery_updated_at"],
            }

        bloom_id = row["bloom_id"]
        if bloom_id is not None and bloom_id not in bloom_map:
            bloom_map[int(bloom_id)] = {
                "id": int(bloom_id),
                "user_id": row["bloom_user_id"],
                "topic": row["bloom_topic"],
                "current_level": row["bloom_level"],
                "updated_at": row["bloom_updated_at"],
            }

        followup_topic = row["followup_topic"]
        if row["followup_user_id"] and followup_topic is not None:
            followups[followup_topic] = {
                "user_id": row["followup_user_id"],
                "topic": followup_topic,
                "needs_assessment": bool(row["followup_needs_assessment"]),
                "microcheck_question": row["followup_microcheck_question"],
                "microcheck_answer_key": row["followup_microcheck_answer_key"],
                "microcheck_rubric": _decode_json_field(row["followup_microcheck_rubric"]),
                "microcheck_source": row["followup_microcheck_source"],
                "created_at": row["followup_created_at"],
                "updated_at": row["followup_updated_at"],
            }

        assessment_id = row["assessment_id"]
        if assessment_id is not None:
            aid = int(assessment_id)
            if aid not in assessment_map:
                assessment_map[aid] = {
                    "assessment_id": aid,
                    "domain": row["assessment_domain"],
                    "item_id": row["assessment_item_id"],
                    "bloom_level": row["assessment_bloom_level"],
                    "score": float(row["assessment_score"]) if row["assessment_score"] is not None else None,
                    "confidence": float(row["assessment_confidence"]) if row["assessment_confidence"] is not None else None,
                    "source": row["assessment_source"],
                    "self_assessment": row["assessment_self_assessment"],
                    "created_at": row["assessment_created_at"],
                    "_rank": int(row["assessment_rank"]) if row["assessment_rank"] is not None else None,
                }

    mastery_rows = sorted(
        mastery_map.values(),
        key=lambda item: (item.get("updated_at") or "", item.get("id") or 0),
        reverse=True,
    )[: int(mastery_limit)]

    bloom_rows = sorted(
        bloom_map.values(),
        key=lambda item: (item.get("updated_at") or "", item.get("id") or 0),
        reverse=True,
    )[: int(bloom_limit)]

    sorted_assessments = sorted(
        assessment_map.values(),
        key=lambda item: item.get("_rank") if item.get("_rank") is not None else 10**9,
    )
    assessments = [
        {key: value for key, value in entry.items() if key != "_rank"}
        for entry in sorted_assessments
    ]

    return {
        "mastery": mastery_rows,
        "bloom_progress": bloom_rows,
        "followups": followups,
        "recent_assessments": assessments,
    }


def compute_spaced_reviews(user_id: str, limit: int = 10) -> list[Dict[str, Any]]:
    rows = _query(
        """
        SELECT id, domain, item_id, score, confidence, source, created_at
        FROM assessment_results
        WHERE user_id = ?
        ORDER BY domain ASC, created_at ASC, id ASC
        """,
        (user_id,),
    )
    states: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        domain = row["domain"]
        state = states.setdefault(domain, {
            "ef": 2.5,
            "interval": 1,
            "repetitions": 0,
            "last_seen": None,
        })
        created_at = _parse_timestamp(row["created_at"])
        quality = max(0, min(5, round(float(row["score"]) * 5)))
        if quality >= 3:
            if state["repetitions"] == 0:
                interval = 1
            elif state["repetitions"] == 1:
                interval = 6
            else:
                interval = max(1, int(round(state["interval"] * state["ef"])))
            state["repetitions"] += 1
            state["ef"] = max(1.3, state["ef"] + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
            state["interval"] = interval
        else:
            state["repetitions"] = 0
            state["interval"] = 1

        state["last_seen"] = created_at
        state["last_item"] = row["item_id"]
        state["last_score"] = float(row["score"])
        state["confidence"] = float(row["confidence"]) if row["confidence"] is not None else None
        state["source"] = row["source"]
        state["evidence_id"] = int(row["id"])

    schedule: list[Dict[str, Any]] = []
    now = datetime.now()
    for domain, state in states.items():
        last_seen: datetime = state.get("last_seen") or now
        interval_days = max(1, int(state.get("interval", 1)))
        due_at = last_seen + timedelta(days=interval_days)
        schedule.append(
            {
                "domain": domain,
                "due_at": due_at.isoformat(),
                "interval_days": interval_days,
                "ef": round(float(state.get("ef", 2.5)), 2),
                "last_item": state.get("last_item"),
                "last_score": state.get("last_score"),
                "confidence": state.get("confidence"),
                "source": state.get("source"),
                "evidence_id": state.get("evidence_id"),
            }
        )

    schedule.sort(key=lambda item: item["due_at"])
    return schedule[: max(0, int(limit))]


def save_copilot_plan(
    teacher_id: str,
    topic: Optional[str],
    objectives: list[str] | str | None,
    plan: Dict[str, Any],
    bloom_alignment: Optional[list[Dict[str, Any]]],
    provenance: Optional[Dict[str, Any]],
    *,
    status: str = "draft",
) -> Dict[str, Any]:
    stored_plan = json_dumps(plan)
    stored_objectives = json_dumps(objectives) if objectives is not None else None
    stored_alignment = json_dumps(bloom_alignment) if bloom_alignment is not None else None
    stored_provenance = json_dumps(provenance) if provenance is not None else None
    cur = _exec(
        """
        INSERT INTO copilot_plans(teacher_id, topic, objectives, plan_json, bloom_alignment, provenance, status)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            teacher_id,
            topic,
            stored_objectives,
            stored_plan,
            stored_alignment,
            stored_provenance,
            status,
        ),
    )
    plan_id = int(cur.lastrowid)
    return get_copilot_plan(plan_id)


def get_copilot_plan(plan_id: int) -> Dict[str, Any]:
    rows = _query(
        """
        SELECT id, teacher_id, topic, objectives, plan_json, bloom_alignment, provenance, status, created_at, updated_at
        FROM copilot_plans WHERE id = ?
        """,
        (int(plan_id),),
    )
    if not rows:
        raise ValueError("plan not found")
    row = dict(rows[0])
    for key in ("objectives", "plan_json", "bloom_alignment", "provenance"):
        row[key] = _decode_json_field(row.get(key))
    moderation_rows = _query(
        """
        SELECT id, moderator_id, decision, rationale, flags, created_at
        FROM copilot_moderation
        WHERE plan_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (int(plan_id),),
    )
    log: list[Dict[str, Any]] = []
    for entry in moderation_rows:
        item = dict(entry)
        item["flags"] = _decode_json_field(item.get("flags"))
        log.append(item)
    row["moderation_log"] = log
    return row


def list_copilot_plans(teacher_id: str, limit: int = 20) -> list[Dict[str, Any]]:
    rows = _query(
        """
        SELECT id, teacher_id, topic, objectives, plan_json, bloom_alignment, provenance, status, created_at, updated_at
        FROM copilot_plans
        WHERE teacher_id = ?
        ORDER BY updated_at DESC, id DESC
        LIMIT ?
        """,
        (teacher_id, int(limit)),
    )
    data: list[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in ("objectives", "plan_json", "bloom_alignment", "provenance"):
            item[key] = _decode_json_field(item.get(key))
        data.append(item)
    return data


def record_copilot_moderation(
    plan_id: int,
    moderator_id: str,
    decision: str,
    rationale: Optional[str],
    flags: Optional[list[str]],
    *,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    _exec(
        """
        INSERT INTO copilot_moderation(plan_id, moderator_id, decision, rationale, flags)
        VALUES (?,?,?,?,?)
        """,
        (
            int(plan_id),
            moderator_id,
            decision,
            rationale,
            json_dumps(flags) if flags is not None else None,
        ),
    )
    if status:
        _exec(
            "UPDATE copilot_plans SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, int(plan_id)),
        )
    return get_copilot_plan(plan_id)


def record_eval_result(run_id: str, probe_id: str, category: str, metrics: Dict[str, Any]) -> None:
    _exec(
        """
        INSERT INTO eval_reports(run_id, probe_id, category, metrics)
        VALUES (?,?,?,?)
        """,
        (run_id, probe_id, category, json_dumps(metrics)),
    )


def list_eval_results(run_id: Optional[str] = None, limit: int = 200) -> list[Dict[str, Any]]:
    if run_id:
        rows = _query(
            "SELECT id, run_id, probe_id, category, metrics, created_at FROM eval_reports WHERE run_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (run_id, int(limit)),
        )
    else:
        rows = _query(
            "SELECT id, run_id, probe_id, category, metrics, created_at FROM eval_reports ORDER BY created_at DESC, id DESC LIMIT ?",
            (int(limit),),
        )
    data: list[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["metrics"] = _decode_json_field(item.get("metrics"))
        data.append(item)
    return data


def _insert_eval_attempt(
    table: str,
    *,
    learner_id: str,
    topic: str,
    score: float,
    max_score: float,
    attempt_id: Optional[str] = None,
    strategy: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    instrument_ref: Optional[int] = None,
    instrument_id: Optional[str] = None,
    instrument_version: Optional[str] = None,
    attempted_at: Optional[str] = None,
) -> Dict[str, Any]:
    learner_id = (learner_id or "").strip()
    topic = (topic or "").strip()
    if not learner_id:
        raise ValueError("learner_id required")
    if not topic:
        raise ValueError("topic required")
    if max_score <= 0:
        raise ValueError("max_score must be positive")
    if score < 0:
        raise ValueError("score must be non-negative")
    if score > max_score:
        raise ValueError("score cannot exceed max_score")

    payload: Dict[str, Any] = {
        "learner_id": learner_id,
        "topic": topic,
        "score": float(score),
        "max_score": float(max_score),
        "attempt_id": attempt_id,
        "strategy": (strategy or None),
        "metadata": json_dumps(metadata) if metadata is not None else None,
    }

    columns = [
        "learner_id",
        "topic",
        "score",
        "max_score",
        "attempt_id",
        "strategy",
        "metadata",
    ]
    values = [
        payload["learner_id"],
        payload["topic"],
        payload["score"],
        payload["max_score"],
        payload["attempt_id"],
        payload["strategy"],
        payload["metadata"],
    ]

    if session_id:
        columns.append("session_id")
        values.append(session_id)

    if instrument_ref:
        columns.append("instrument_ref")
        values.append(int(instrument_ref))

    if instrument_id:
        columns.append("instrument_id")
        values.append(instrument_id)

    if instrument_version:
        columns.append("instrument_version")
        values.append(instrument_version)

    if attempted_at:
        when = _parse_timestamp(attempted_at).isoformat(timespec="seconds")
        columns.append("attempted_at")
        values.append(when)

    placeholders = ",".join(["?"] * len(columns))
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    cur = _exec(sql, values)
    attempt_id_val = cur.lastrowid
    return _get_eval_attempt(table, attempt_id_val)


def _get_eval_attempt(table: str, row_id: int) -> Dict[str, Any]:
    row = _query(
        f"""
        SELECT id, learner_id, topic, score, max_score, attempt_id, strategy,
               metadata, session_id, instrument_ref, instrument_id, instrument_version, attempted_at
        FROM {table}
        WHERE id = ?
        """,
        (int(row_id),),
    )
    if not row:
        raise ValueError("attempt not found")
    entry = dict(row[0])
    entry["metadata"] = _decode_json_field(entry.get("metadata"))
    instrument_ref = entry.get("instrument_ref")
    if instrument_ref:
        entry["instrument"] = get_eval_instrument_by_ref(int(instrument_ref))
    else:
        entry["instrument"] = None
    return entry


def _normalize_stage(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    if normalized not in {"pretest", "posttest"}:
        raise ValueError(f"unsupported evaluation stage: {stage}")
    return normalized


def ensure_eval_instrument(
    *,
    topic: str,
    stage: str,
    instrument_id: Optional[str] = None,
    instrument_version: Optional[str] = None,
    instrument: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    stage_key = _normalize_stage(stage)
    topic_key = (topic or "").strip()
    if not topic_key:
        raise ValueError("topic required for evaluation instrument")

    instrument_key = (instrument_id or "").strip() or None
    existing: Optional[Dict[str, Any]] = None
    if instrument_key:
        rows = _query(
            "SELECT id FROM eval_instruments WHERE instrument_id = ? AND stage = ?",
            (instrument_key, stage_key),
        )
        if rows:
            existing = get_eval_instrument_by_ref(int(rows[0]["id"]))

    if instrument is None:
        if existing:
            return existing
        if instrument_key:
            raise ValueError(f"evaluation instrument {instrument_key!r} not found for stage {stage_key}")
        return None

    if not isinstance(instrument, dict):
        raise ValueError("instrument payload must be a JSON object")

    items = instrument.get("items")
    if items is None:
        raise ValueError("instrument payload must include an 'items' collection")

    title = instrument.get("title")
    description = instrument.get("description")
    metadata = instrument.get("metadata")

    instrument_payload = {
        "title": title if title is not None else None,
        "description": description if description is not None else None,
        "items": json_dumps(items),
        "metadata": json_dumps(metadata) if metadata is not None else None,
    }

    version_value = instrument_version or instrument.get("version") or instrument.get("revision")
    if version_value is not None:
        version_value = str(version_value)

    if not instrument_key:
        instrument_key = f"{stage_key}:{uuid4().hex[:12]}"

    _exec(
        """
        INSERT INTO eval_instruments(
            instrument_id, topic, stage, version, title, description, items, metadata
        ) VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(instrument_id, stage) DO UPDATE SET
            topic=excluded.topic,
            version=excluded.version,
            title=excluded.title,
            description=excluded.description,
            items=excluded.items,
            metadata=excluded.metadata,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            instrument_key,
            topic_key,
            stage_key,
            version_value,
            instrument_payload["title"],
            instrument_payload["description"],
            instrument_payload["items"],
            instrument_payload["metadata"],
        ),
    )

    rows = _query(
        "SELECT id FROM eval_instruments WHERE instrument_id = ? AND stage = ?",
        (instrument_key, stage_key),
    )
    if not rows:
        raise ValueError("failed to persist evaluation instrument")
    return get_eval_instrument_by_ref(int(rows[0]["id"]))


def get_eval_instrument_by_ref(ref_id: int) -> Dict[str, Any]:
    rows = _query(
        """
        SELECT id, instrument_id, topic, stage, version, title, description, items, metadata, created_at, updated_at
        FROM eval_instruments
        WHERE id = ?
        """,
        (int(ref_id),),
    )
    if not rows:
        raise ValueError("instrument not found")
    entry = dict(rows[0])
    entry["items"] = _decode_json_field(entry.get("items"))
    entry["metadata"] = _decode_json_field(entry.get("metadata"))
    return entry


def get_eval_instrument(instrument_id: str, stage: str) -> Optional[Dict[str, Any]]:
    instrument_key = (instrument_id or "").strip()
    if not instrument_key:
        return None
    stage_key = _normalize_stage(stage)
    rows = _query(
        "SELECT id FROM eval_instruments WHERE instrument_id = ? AND stage = ?",
        (instrument_key, stage_key),
    )
    if not rows:
        return None
    return get_eval_instrument_by_ref(int(rows[0]["id"]))


def attach_eval_instrument_to_session(session_id: str, stage: str, instrument_ref: int) -> None:
    if not session_id or not instrument_ref:
        return
    stage_key = _normalize_stage(stage)
    _exec(
        """
        INSERT INTO eval_session_instruments(session_id, stage, instrument_ref)
        VALUES (?,?,?)
        ON CONFLICT(session_id, stage) DO UPDATE SET
            instrument_ref=excluded.instrument_ref,
            attached_at=CURRENT_TIMESTAMP
        """,
        (session_id, stage_key, int(instrument_ref)),
    )


def get_eval_session_instrument(session_id: str, stage: str) -> Optional[Dict[str, Any]]:
    if not session_id:
        return None
    stage_key = _normalize_stage(stage)
    rows = _query(
        """
        SELECT s.session_id, s.stage, s.instrument_ref,
               i.instrument_id, i.topic, i.stage AS instrument_stage,
               i.version, i.title, i.description, i.items, i.metadata, i.updated_at
        FROM eval_session_instruments s
        JOIN eval_instruments i ON i.id = s.instrument_ref
        WHERE s.session_id = ? AND s.stage = ?
        """,
        (session_id, stage_key),
    )
    if not rows:
        return None
    entry = dict(rows[0])
    entry["items"] = _decode_json_field(entry.get("items"))
    entry["metadata"] = _decode_json_field(entry.get("metadata"))
    return entry


def record_pretest_attempt(
    *,
    learner_id: str,
    topic: str,
    score: float,
    max_score: float,
    attempt_id: Optional[str] = None,
    strategy: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    instrument_id: Optional[str] = None,
    instrument_version: Optional[str] = None,
    instrument: Optional[Dict[str, Any]] = None,
    attempted_at: Optional[str] = None,
) -> Dict[str, Any]:
    instrument_record = ensure_eval_instrument(
        topic=topic,
        stage="pretest",
        instrument_id=instrument_id,
        instrument_version=instrument_version,
        instrument=instrument,
    )
    attempt = _insert_eval_attempt(
        "eval_pretest_attempts",
        learner_id=learner_id,
        topic=topic,
        score=score,
        max_score=max_score,
        attempt_id=attempt_id,
        strategy=strategy,
        metadata=metadata,
        session_id=session_id,
        instrument_ref=instrument_record.get("id") if instrument_record else None,
        instrument_id=instrument_record.get("instrument_id") if instrument_record else instrument_id,
        instrument_version=instrument_record.get("version") if instrument_record else instrument_version,
        attempted_at=attempted_at,
    )
    if session_id and instrument_record:
        attach_eval_instrument_to_session(session_id, "pretest", int(instrument_record["id"]))
    return attempt


def record_posttest_attempt(
    *,
    learner_id: str,
    topic: str,
    score: float,
    max_score: float,
    attempt_id: Optional[str] = None,
    strategy: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    instrument_id: Optional[str] = None,
    instrument_version: Optional[str] = None,
    instrument: Optional[Dict[str, Any]] = None,
    attempted_at: Optional[str] = None,
) -> Dict[str, Any]:
    instrument_record = ensure_eval_instrument(
        topic=topic,
        stage="posttest",
        instrument_id=instrument_id,
        instrument_version=instrument_version,
        instrument=instrument,
    )
    attempt = _insert_eval_attempt(
        "eval_posttest_attempts",
        learner_id=learner_id,
        topic=topic,
        score=score,
        max_score=max_score,
        attempt_id=attempt_id,
        strategy=strategy,
        metadata=metadata,
        session_id=session_id,
        instrument_ref=instrument_record.get("id") if instrument_record else None,
        instrument_id=instrument_record.get("instrument_id") if instrument_record else instrument_id,
        instrument_version=instrument_record.get("version") if instrument_record else instrument_version,
        attempted_at=attempted_at,
    )
    if session_id and instrument_record:
        attach_eval_instrument_to_session(session_id, "posttest", int(instrument_record["id"]))
    return attempt


def _latest_attempts_subquery(table: str) -> str:
    return f"""
        SELECT a.*
        FROM {table} a
        JOIN (
            SELECT learner_id, topic, MAX(id) AS latest_id
            FROM {table}
            GROUP BY learner_id, topic
        ) latest
          ON a.learner_id = latest.learner_id
         AND a.topic = latest.topic
         AND a.id = latest.latest_id
    """


def fetch_normalized_gains(
    *,
    learner_id: Optional[str] = None,
    topic: Optional[str] = None,
    strategy: Optional[str] = None,
) -> list[Dict[str, Any]]:
    pre_sql = _latest_attempts_subquery("eval_pretest_attempts")
    post_sql = _latest_attempts_subquery("eval_posttest_attempts")

    sql = f"""
        SELECT
            pre.learner_id,
            pre.topic,
            pre.score AS pre_score,
            pre.max_score AS pre_max_score,
            pre.strategy AS pre_strategy,
            pre.metadata AS pre_metadata,
            pre.session_id AS pre_session_id,
            pre.instrument_id AS pre_instrument_id,
            pre.instrument_version AS pre_instrument_version,
            pre.instrument_ref AS pre_instrument_ref,
            pre.attempted_at AS pre_attempted_at,
            post.score AS post_score,
            post.max_score AS post_max_score,
            post.strategy AS post_strategy,
            post.metadata AS post_metadata,
            post.session_id AS post_session_id,
            post.instrument_id AS post_instrument_id,
            post.instrument_version AS post_instrument_version,
            post.instrument_ref AS post_instrument_ref,
            post.attempted_at AS post_attempted_at
        FROM ({pre_sql}) pre
        JOIN ({post_sql}) post
          ON pre.learner_id = post.learner_id
         AND pre.topic = post.topic
    """

    conditions: list[str] = []
    params: list[Any] = []
    if learner_id:
        conditions.append("pre.learner_id = ?")
        params.append(learner_id)
    if topic:
        conditions.append("pre.topic = ?")
        params.append(topic)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    rows = _query(sql, params)

    results: list[Dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        entry["pre_metadata"] = _decode_json_field(entry.get("pre_metadata"))
        entry["post_metadata"] = _decode_json_field(entry.get("post_metadata"))
        chosen_strategy = entry.get("post_strategy") or entry.get("pre_strategy")
        entry["strategy"] = chosen_strategy

        pre_ref = entry.get("pre_instrument_ref")
        entry["pre_instrument"] = get_eval_instrument_by_ref(int(pre_ref)) if pre_ref else None
        post_ref = entry.get("post_instrument_ref")
        entry["post_instrument"] = get_eval_instrument_by_ref(int(post_ref)) if post_ref else None

        pre_score = entry.get("pre_score")
        post_score = entry.get("post_score")
        pre_max = entry.get("pre_max_score")
        post_max = entry.get("post_max_score")

        pre_norm: Optional[float]
        if pre_score is None or pre_max in (None, 0):
            pre_norm = None
        else:
            pre_norm = float(pre_score) / float(pre_max)

        post_norm: Optional[float]
        if post_score is None or post_max in (None, 0):
            post_norm = None
        else:
            post_norm = float(post_score) / float(post_max)

        delta: Optional[float]
        if pre_norm is None or post_norm is None:
            delta = None
        else:
            delta = post_norm - pre_norm

        gain: Optional[float]
        if pre_norm is None or delta is None:
            gain = None
        else:
            denom = 1.0 - pre_norm
            if denom <= 0:
                gain = None
            else:
                gain = delta / denom

        entry["pre_normalized"] = round(pre_norm, 4) if pre_norm is not None else None
        entry["post_normalized"] = round(post_norm, 4) if post_norm is not None else None
        entry["delta"] = round(delta, 4) if delta is not None else None
        if gain is not None:
            rounded_gain = round(gain, 4)
            entry["normalized_gain"] = rounded_gain
            entry["g"] = rounded_gain
        else:
            entry["normalized_gain"] = None
            entry["g"] = None

        if strategy and (chosen_strategy or "").strip() != strategy:
            continue

        results.append(entry)
    return results


def summarize_normalized_gains(
    *,
    learner_id: Optional[str] = None,
    topic: Optional[str] = None,
    strategy: Optional[str] = None,
) -> list[Dict[str, Any]]:
    pairs = fetch_normalized_gains(
        learner_id=learner_id,
        topic=topic,
        strategy=strategy,
    )
    buckets: dict[tuple[str, Optional[str]], Dict[str, Any]] = {}
    for pair in pairs:
        topic_key = pair["topic"]
        strat_key = pair.get("strategy") or None
        bucket = buckets.setdefault(
            (topic_key, strat_key),
            {
                "topic": topic_key,
                "strategy": strat_key,
                "pair_count": 0,
                "valid_gain_count": 0,
                "gain_sum": 0.0,
                "pre_sum": 0.0,
                "post_sum": 0.0,
                "delta_sum": 0.0,
                "gain_values": [],
            },
        )
        bucket["pair_count"] += 1
        pre_value = pair.get("pre_normalized")
        post_value = pair.get("post_normalized")
        delta_value = pair.get("delta")
        gain_value = pair.get("normalized_gain")

        if pre_value is not None:
            bucket["pre_sum"] += float(pre_value)
        if post_value is not None:
            bucket["post_sum"] += float(post_value)
        if delta_value is not None:
            bucket["delta_sum"] += float(delta_value)
        if gain_value is not None:
            bucket["valid_gain_count"] += 1
            bucket["gain_sum"] += float(gain_value)
            bucket["gain_values"].append(float(gain_value))

    summary: list[Dict[str, Any]] = []
    for (topic_key, strat_key), bucket in sorted(buckets.items()):
        count = bucket["pair_count"]
        valid_count = bucket["valid_gain_count"]
        avg_gain = None
        if valid_count:
            avg_gain = bucket["gain_sum"] / valid_count
        avg_pre = bucket["pre_sum"] / count if count else None
        avg_post = bucket["post_sum"] / count if count else None
        avg_delta = bucket["delta_sum"] / count if count else None
        ci = mean_confidence_interval(bucket["gain_values"]) if valid_count else None
        summary.append(
            {
                "topic": topic_key,
                "strategy": strat_key,
                "pair_count": count,
                "valid_gain_count": valid_count,
                "average_normalized_gain": round(avg_gain, 4) if avg_gain is not None else None,
                "average_pre": round(avg_pre, 4) if avg_pre is not None else None,
                "average_post": round(avg_post, 4) if avg_post is not None else None,
                "average_delta": round(avg_delta, 4) if avg_delta is not None else None,
                "gain_confidence_interval": ci,
            }
        )
    return summary


def mean_confidence_interval(values: list[float], confidence: float = 0.95) -> Optional[Dict[str, Any]]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance < 0:
        variance = 0.0
    std_dev = math.sqrt(variance)
    if std_dev == 0:
        margin = 0.0
    else:
        margin = 1.96 * (std_dev / math.sqrt(len(values)))
    lower = mean - margin
    upper = mean + margin
    return {
        "confidence_level": confidence,
        "margin": round(margin, 4),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "mean": round(mean, 4),
    }

def record_chat_ops(
    user_id: str,
    topic: Optional[str],
    question: str,
    answer: str,
    response_json: Optional[Dict[str, Any]],
    applied_ops: Iterable[Dict[str, Any]],
    pending_ops: Iterable[Dict[str, Any]],
    *,
    raw_response: Optional[str] = None,
):
    _exec(
        """
        INSERT INTO chat_ops_log(
            user_id, topic, question, answer, raw_response, response_json, applied_ops, pending_ops
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            topic,
            question,
            answer,
            raw_response,
            json_dumps(response_json) if response_json is not None else None,
            json_dumps(list(applied_ops) if applied_ops is not None else []),
            json_dumps(list(pending_ops) if pending_ops is not None else []),
        ),
    )


def _collect_rubric_terms(value: Any) -> list[str]:
    terms: list[str] = []
    if value is None:
        return terms
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            terms.append(cleaned)
        return terms
    if isinstance(value, Mapping):
        for nested in value.values():
            terms.extend(_collect_rubric_terms(nested))
        return terms
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for entry in value:
            terms.extend(_collect_rubric_terms(entry))
        return terms
    text = str(value).strip()
    if text:
        terms.append(text)
    return terms


def _normalize_rubric_payload(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        payload = {key: value for key, value in raw.items() if value is not None}
    else:
        payload = {}
    if isinstance(raw, (list, tuple, set)):
        criteria_source: Any = raw
    elif isinstance(raw, str):
        criteria_source = [raw]
    else:
        criteria_source = payload.get("criteria")
    criteria_terms = []
    seen: set[str] = set()
    for term in _collect_rubric_terms(criteria_source):
        lowered = term.lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            criteria_terms.append(term)
    if criteria_terms:
        payload["criteria"] = criteria_terms
    elif "criteria" in payload:
        payload.pop("criteria", None)

    keyword_terms: list[str] = []
    seen_keywords: set[str] = set()
    for term in _collect_rubric_terms(raw):
        lowered = term.lower()
        if lowered and lowered not in seen_keywords:
            seen_keywords.add(lowered)
            keyword_terms.append(lowered)
    if keyword_terms:
        payload["keywords"] = keyword_terms
    elif "keywords" in payload:
        payload.pop("keywords", None)
    return payload if payload else None


def set_needs_assessment(
    user_id: str,
    topic: str,
    needs_assessment: bool,
    *,
    microcheck: Optional[Dict[str, Any]] = None,
) -> None:
    question = None
    answer_key = None
    rubric_json = None
    source = None
    if microcheck:
        question = microcheck.get("question") or microcheck.get("microcheck_question")
        answer_key = (
            microcheck.get("answer_key")
            or microcheck.get("expected")
            or microcheck.get("microcheck_expected")
        )
        source = microcheck.get("source") or microcheck.get("microcheck_source") or "json"
        given = microcheck.get("given") or microcheck.get("microcheck_given")
        score = microcheck.get("score") or microcheck.get("microcheck_score")

        score_value: float | None = None
        if score is not None:
            try:
                candidate = float(score)
            except (TypeError, ValueError):
                candidate = math.nan
            if not math.isnan(candidate) and not math.isinf(candidate):
                score_value = max(0.0, min(1.0, candidate))

        raw_rubric = microcheck.get("rubric")
        rubric_payload = _normalize_rubric_payload(raw_rubric) or {}
        if answer_key and "expected" not in rubric_payload:
            rubric_payload["expected"] = answer_key
        if given is not None and "given" not in rubric_payload:
            rubric_payload["given"] = given
        if score_value is not None:
            rubric_payload["score"] = score_value
        rubric_json = json_dumps(rubric_payload) if rubric_payload else None
    _exec(
        """
        INSERT INTO assessment_followups(
            user_id, topic, needs_assessment, microcheck_question,
            microcheck_answer_key, microcheck_rubric, microcheck_source,
            created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, topic) DO UPDATE SET
            needs_assessment=excluded.needs_assessment,
            microcheck_question=excluded.microcheck_question,
            microcheck_answer_key=excluded.microcheck_answer_key,
            microcheck_rubric=excluded.microcheck_rubric,
            microcheck_source=excluded.microcheck_source,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            user_id,
            topic,
            1 if needs_assessment else 0,
            question,
            answer_key,
            rubric_json,
            source,
        ),
    )


def get_followup_state(user_id: str, topic: str) -> Optional[Dict[str, Any]]:
    rows = _query(
        """
        SELECT user_id, topic, needs_assessment, microcheck_question,
               microcheck_answer_key, microcheck_rubric, microcheck_source,
               created_at, updated_at
        FROM assessment_followups
        WHERE user_id = ? AND topic = ?
        """,
        (user_id, topic),
    )
    if not rows:
        return None
    row = dict(rows[0])
    row["needs_assessment"] = bool(row.get("needs_assessment"))
    rubric_raw = _decode_json_field(row.get("microcheck_rubric"))
    normalized_rubric = _normalize_rubric_payload(rubric_raw)
    row["microcheck_rubric"] = normalized_rubric or rubric_raw
    row["microcheck_expected"] = row.get("microcheck_answer_key")
    rubric_dict = row.get("microcheck_rubric")
    if isinstance(rubric_dict, dict):
        if "given" in rubric_dict:
            row["microcheck_given"] = rubric_dict.get("given")
        if "score" in rubric_dict:
            row["microcheck_score"] = rubric_dict.get("score")
    return row


def clear_microcheck(user_id: str, topic: str) -> None:
    _exec(
        """
        UPDATE assessment_followups
        SET microcheck_question = NULL,
            microcheck_answer_key = NULL,
            microcheck_rubric = NULL,
            microcheck_source = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ? AND topic = ?
        """,
        (user_id, topic),
    )


def clear_followup_state(user_id: str, topic: str) -> None:
    _exec(
        "DELETE FROM assessment_followups WHERE user_id = ? AND topic = ?",
        (user_id, topic),
    )


def save_pending_op(user_id: str, topic: Optional[str], payload: Dict[str, Any]) -> int:
    cur = _exec(
        """
        INSERT INTO pending_ops(user_id, topic, payload, status)
        VALUES (?,?,?, 'open')
        """,
        (
            user_id,
            topic,
            json_dumps(payload),
        ),
    )
    return int(cur.lastrowid)


def list_pending_ops(limit: int = 50) -> list[Dict[str, Any]]:
    rows = _query(
        """
        SELECT id, user_id, topic, payload, status, created_at, updated_at
        FROM pending_ops
        WHERE status = 'open'
        ORDER BY created_at ASC, id ASC
        LIMIT ?
        """,
        (int(limit),),
    )
    data: list[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["payload"] = _decode_json_field(item.get("payload"))
        data.append(item)
    return data


def resolve_pending_op(op_id: int) -> None:
    _exec(
        """
        UPDATE pending_ops
        SET status = 'resolved', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (int(op_id),),
    )


def list_chat_ops(user_id: Optional[str] = None, limit: int = 100) -> list[Dict[str, Any]]:
    if user_id:
        rows = _query(
            """
            SELECT id, user_id, topic, question, answer, raw_response, response_json, applied_ops, pending_ops, created_at
            FROM chat_ops_log
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        )
    else:
        rows = _query(
            """
            SELECT id, user_id, topic, question, answer, raw_response, response_json, applied_ops, pending_ops, created_at
            FROM chat_ops_log
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (int(limit),),
        )

    data: list[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["response_json"] = _decode_json_field(item.get("response_json"))
        for key in ("applied_ops", "pending_ops"):
            parsed = _decode_json_field(item.get(key))
            if parsed is None:
                parsed = []
            item[key] = parsed
        data.append(item)
    return data

# -------------- items (diagnostic/practice item bank) --------------
def upsert_item(item_id: str, skill: str, difficulty: float, body: str):
    _exec(
        """
        INSERT INTO items(id, skill, difficulty, body) VALUES (?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET skill=excluded.skill, difficulty=excluded.difficulty, body=excluded.body
        """,
        (item_id, skill, float(difficulty), body)
    )

def list_items(skill: Optional[str] = None, limit: int = 100) -> list[sqlite3.Row]:
    if skill:
        return _query(
            "SELECT id, skill, difficulty, body, created_at FROM items WHERE skill = ? ORDER BY created_at DESC LIMIT ?",
            (skill, int(limit))
        )
    return _query(
        "SELECT id, skill, difficulty, body, created_at FROM items ORDER BY created_at DESC LIMIT ?",
        (int(limit),)
    )


def upsert_item_bank_entries(items: Iterable[Dict[str, Any]]) -> None:
    to_store = list(items)
    if not to_store:
        return

    with _conn() as con:
        for item in to_store:
            item_id = item["id"]
            metadata_json = json_dumps(item.get("metadata")) if item.get("metadata") is not None else None
            references_json = json_dumps(item.get("references")) if item.get("references") is not None else None
            difficulty = item.get("difficulty")
            exposure_limit = item.get("exposure_limit")

            con.execute(
                """
                INSERT INTO item_bank(
                  id, domain, skill_id, bloom_level, stimulus,
                  elo_target, answer_key, rubric_id, difficulty, exposure_limit,
                  metadata_json, references_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  domain=excluded.domain,
                  skill_id=excluded.skill_id,
                  bloom_level=excluded.bloom_level,
                  stimulus=excluded.stimulus,
                  elo_target=excluded.elo_target,
                  answer_key=excluded.answer_key,
                  rubric_id=excluded.rubric_id,
                  difficulty=excluded.difficulty,
                  exposure_limit=excluded.exposure_limit,
                  metadata_json=excluded.metadata_json,
                  references_json=excluded.references_json
                """,
                (
                    item_id,
                    item.get("domain"),
                    item.get("skill_id"),
                    item.get("bloom_level"),
                    item.get("stimulus"),
                    float(item.get("elo_target")),
                    item.get("answer_key"),
                    item.get("rubric_id"),
                    None if difficulty is None else float(difficulty),
                    None if exposure_limit is None else int(exposure_limit),
                    metadata_json,
                    references_json,
                ),
            )

            con.execute(
                """
                INSERT INTO item_exposure(item_id, served_count, last_served)
                VALUES (?, 0, NULL)
                ON CONFLICT(item_id) DO UPDATE SET served_count = served_count
                """,
                (item_id,),
            )

        con.commit()


def list_item_bank(
    domain: Optional[str] = None,
    skill_id: Optional[str] = None,
    bloom_level: Optional[str] = None,
    limit: int = 200,
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[Any] = []

    if domain:
        clauses.append("domain = ?")
        params.append(domain)
    if skill_id:
        clauses.append("skill_id = ?")
        params.append(skill_id)
    if bloom_level:
        clauses.append("bloom_level = ?")
        params.append(bloom_level)

    where = ""
    if clauses:
        where = " WHERE " + " AND ".join(clauses)

    query = (
        "SELECT id, domain, skill_id, bloom_level, stimulus, answer_key, rubric_id, difficulty, exposure_limit, metadata_json, references_json "
        "FROM item_bank"
        f"{where} ORDER BY domain, skill_id LIMIT ?"
    )
    params.append(int(limit))
    return _query(query, params)


def get_item_bank_entry(item_id: str) -> Optional[sqlite3.Row]:
    rows = _query(
        """
        SELECT id, domain, skill_id, bloom_level, stimulus, answer_key, rubric_id,
               difficulty, exposure_limit, metadata_json, references_json
        FROM item_bank WHERE id = ?
        """,
        (item_id,),
    )
    return rows[0] if rows else None


def get_item_exposures(item_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    ids = [str(item_id) for item_id in item_ids]
    if not ids:
        return {}

    placeholders = ",".join("?" for _ in ids)
    rows = _query(
        f"SELECT item_id, served_count, last_served FROM item_exposure WHERE item_id IN ({placeholders})",
        ids,
    )
    exposures: Dict[str, Dict[str, Any]] = {row["item_id"]: dict(row) for row in rows}
    for item_id in ids:
        exposures.setdefault(item_id, {"item_id": item_id, "served_count": 0, "last_served": None})
    return exposures


def increment_item_exposure(item_id: str, increment: int = 1) -> None:
    step = max(int(increment), 0)
    if step == 0:
        return
    _exec(
        """
        INSERT INTO item_exposure(item_id, served_count, last_served)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(item_id) DO UPDATE SET
          served_count = served_count + ?,
          last_served = CURRENT_TIMESTAMP
        """,
        (item_id, step, step),
    )


# -------------- curriculum structure --------------
def upsert_subject(subject_id: str, name: str, theme: str, description: Optional[str] = None):
    _exec(
        """
        INSERT INTO subjects(subject_id, name, theme, description)
        VALUES (?,?,?,?)
        ON CONFLICT(subject_id) DO UPDATE SET
          name=excluded.name,
          theme=excluded.theme,
          description=excluded.description
        """,
        (subject_id, name, theme, description)
    )


def list_subjects(limit: int = 50) -> list[sqlite3.Row]:
    return _query(
        "SELECT subject_id, name, theme, description, created_at FROM subjects ORDER BY name ASC LIMIT ?",
        (int(limit),)
    )


def upsert_module(
    module_id: str,
    subject_id: str,
    title: str,
    k_level: str,
    description: Optional[str] = None,
    position: Optional[int] = None,
):
    insert_position = 0 if position is None else int(position)
    update_position = None if position is None else int(position)
    _exec(
        """
        INSERT INTO modules(module_id, subject_id, title, k_level, description, position)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(module_id) DO UPDATE SET
          subject_id=excluded.subject_id,
          title=excluded.title,
          k_level=excluded.k_level,
          description=excluded.description,
          position=COALESCE(?, modules.position)
        """,
        (module_id, subject_id, title, k_level, description, insert_position, update_position)
    )


def list_modules(subject_id: Optional[str] = None, limit: int = 100) -> list[sqlite3.Row]:
    if subject_id:
        return _query(
            """
            SELECT module_id, subject_id, title, k_level, description, position, created_at
            FROM modules
            WHERE subject_id = ?
            ORDER BY position ASC, created_at ASC
            LIMIT ?
            """,
            (subject_id, int(limit))
        )
    return _query(
        """
        SELECT module_id, subject_id, title, k_level, description, position, created_at
        FROM modules
        ORDER BY subject_id ASC, position ASC
        LIMIT ?
        """,
        (int(limit),)
    )


def upsert_lesson(
    lesson_id: str,
    module_id: str,
    title: str,
    summary: Optional[str] = None,
    position: Optional[int] = None,
):
    insert_position = 0 if position is None else int(position)
    update_position = None if position is None else int(position)
    _exec(
        """
        INSERT INTO lessons(lesson_id, module_id, title, summary, position)
        VALUES (?,?,?,?,?)
        ON CONFLICT(lesson_id) DO UPDATE SET
          module_id=excluded.module_id,
          title=excluded.title,
          summary=excluded.summary,
          position=COALESCE(?, lessons.position)
        """,
        (lesson_id, module_id, title, summary, insert_position, update_position)
    )


def list_lessons(
    module_id: Optional[str] = None,
    subject_id: Optional[str] = None,
    limit: int = 100,
) -> list[sqlite3.Row]:
    if module_id:
        return _query(
            """
            SELECT lesson_id, module_id, title, summary, position, created_at
            FROM lessons
            WHERE module_id = ?
            ORDER BY position ASC, created_at ASC
            LIMIT ?
            """,
            (module_id, int(limit))
        )
    if subject_id:
        return _query(
            """
            SELECT l.lesson_id, l.module_id, l.title, l.summary, l.position, l.created_at
            FROM lessons AS l
            JOIN modules AS m ON m.module_id = l.module_id
            WHERE m.subject_id = ?
            ORDER BY m.position ASC, l.position ASC
            LIMIT ?
            """,
            (subject_id, int(limit))
        )
    return _query(
        """
        SELECT lesson_id, module_id, title, summary, position, created_at
        FROM lessons
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (int(limit),)
    )


def upsert_activity(
    activity_id: str,
    lesson_id: str,
    activity_type: str,
    content: str,
    target_level: Optional[str] = None,
    metadata: Optional[Union[Dict[str, Any], str]] = None,
):
    stored_metadata = metadata
    if isinstance(metadata, dict):
        stored_metadata = json_dumps(metadata)
    _exec(
        """
        INSERT INTO activities(activity_id, lesson_id, activity_type, content, target_level, metadata)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(activity_id) DO UPDATE SET
          lesson_id=excluded.lesson_id,
          activity_type=excluded.activity_type,
          content=excluded.content,
          target_level=excluded.target_level,
          metadata=excluded.metadata
        """,
        (activity_id, lesson_id, activity_type, content, target_level, stored_metadata)
    )


def list_activities(
    lesson_id: Optional[str] = None,
    module_id: Optional[str] = None,
    subject_id: Optional[str] = None,
    limit: int = 100,
) -> list[sqlite3.Row]:
    if lesson_id:
        return _query(
            """
            SELECT activity_id, lesson_id, activity_type, content, target_level, metadata, created_at
            FROM activities
            WHERE lesson_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (lesson_id, int(limit))
        )
    if module_id:
        return _query(
            """
            SELECT a.activity_id, a.lesson_id, a.activity_type, a.content, a.target_level, a.metadata, a.created_at
            FROM activities AS a
            JOIN lessons AS l ON l.lesson_id = a.lesson_id
            WHERE l.module_id = ?
            ORDER BY a.created_at ASC
            LIMIT ?
            """,
            (module_id, int(limit))
        )
    if subject_id:
        return _query(
            """
            SELECT a.activity_id, a.lesson_id, a.activity_type, a.content, a.target_level, a.metadata, a.created_at
            FROM activities AS a
            JOIN lessons AS l ON l.lesson_id = a.lesson_id
            JOIN modules AS m ON m.module_id = l.module_id
            WHERE m.subject_id = ?
            ORDER BY m.position ASC, l.position ASC, a.created_at ASC
            LIMIT ?
            """,
            (subject_id, int(limit))
        )
    return _query(
        "SELECT activity_id, lesson_id, activity_type, content, target_level, metadata, created_at FROM activities ORDER BY created_at DESC LIMIT ?",
        (int(limit),)
    )


# -------------- progression tracking --------------
def upsert_user_progress(
    user_id: str,
    subject_id: str,
    current_level: str,
    confidence: Optional[float] = None,
    *,
    band_lower: Optional[str] = None,
    band_upper: Optional[str] = None,
    target_probability: Optional[float] = None,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    ci_width: Optional[float] = None,
):
    _exec(
        """
        INSERT INTO user_progress(
          user_id,
          subject_id,
          current_level,
          confidence,
          band_lower,
          band_upper,
          target_probability,
          ci_lower,
          ci_upper,
          ci_width,
          updated_at
        )
        VALUES (?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, subject_id) DO UPDATE SET
          current_level=excluded.current_level,
          confidence=COALESCE(excluded.confidence, user_progress.confidence),
          band_lower=COALESCE(excluded.band_lower, user_progress.band_lower),
          band_upper=COALESCE(excluded.band_upper, user_progress.band_upper),
          target_probability=COALESCE(excluded.target_probability, user_progress.target_probability),
          ci_lower=COALESCE(excluded.ci_lower, user_progress.ci_lower),
          ci_upper=COALESCE(excluded.ci_upper, user_progress.ci_upper),
          ci_width=COALESCE(excluded.ci_width, user_progress.ci_width),
          updated_at=CURRENT_TIMESTAMP
        """,
        (
            user_id,
            subject_id,
            current_level,
            None if confidence is None else float(confidence),
            band_lower,
            band_upper,
            None if target_probability is None else float(target_probability),
            None if ci_lower is None else float(ci_lower),
            None if ci_upper is None else float(ci_upper),
            None if ci_width is None else float(ci_width),
        )
    )


def get_user_progress(user_id: str, subject_id: str) -> Optional[sqlite3.Row]:
    rows = _query(
        """
        SELECT user_id, subject_id, current_level, confidence, band_lower, band_upper,
               target_probability, ci_lower, ci_upper, ci_width, updated_at
        FROM user_progress
        WHERE user_id = ? AND subject_id = ?
        """,
        (user_id, subject_id)
    )
    return rows[0] if rows else None


def list_user_progress(user_id: Optional[str] = None, limit: int = 100) -> list[sqlite3.Row]:
    if user_id:
        return _query(
            """
            SELECT user_id, subject_id, current_level, confidence, band_lower, band_upper,
                   target_probability, ci_lower, ci_upper, ci_width, updated_at
            FROM user_progress
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, int(limit))
        )
    return _query(
        """
        SELECT user_id, subject_id, current_level, confidence, band_lower, band_upper,
               target_probability, ci_lower, ci_upper, ci_width, updated_at
        FROM user_progress
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (int(limit),)
    )


def log_learning_event(
    user_id: str,
    subject_id: str,
    event_type: str,
    lesson_id: Optional[str] = None,
    score: Optional[float] = None,
    details: Optional[Union[Dict[str, Any], str]] = None,
    skill_id: Optional[str] = None,
) -> int:
    stored_details = details
    if isinstance(details, dict):
        stored_details = json_dumps(details)
    cur = _exec(
        """
        INSERT INTO learning_events(user_id, subject_id, lesson_id, event_type, score, details, skill_id)
        VALUES (?,?,?,?,?,?,?)
        """
,
        (
            user_id,
            subject_id,
            lesson_id,
            event_type,
            None if score is None else float(score),
            stored_details,
            skill_id,
        ),
    )
    return int(cur.lastrowid)

def list_learning_events(
    user_id: Optional[str] = None,
    subject_id: Optional[str] = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    if user_id and subject_id:
        return _query(
            """
            SELECT id, user_id, subject_id, lesson_id, event_type, score, details, skill_id, created_at
            FROM learning_events
            WHERE user_id = ? AND subject_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, subject_id, int(limit))
        )
    if user_id:
        return _query(
            """
            SELECT id, user_id, subject_id, lesson_id, event_type, score, details, skill_id, created_at
            FROM learning_events
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, int(limit))
        )
    if subject_id:
        return _query(
            """
            SELECT id, user_id, subject_id, lesson_id, event_type, score, details, skill_id, created_at
            FROM learning_events
            WHERE subject_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (subject_id, int(limit))
        )
    return _query(
        "SELECT id, user_id, subject_id, lesson_id, event_type, score, details, skill_id, created_at FROM learning_events ORDER BY created_at DESC LIMIT ?",
        (int(limit),)
    )


def record_quiz_attempt(
    user_id: str,
    subject_id: str,
    activity_id: str,
    score: float,
    max_score: float,
    pass_threshold: float = 0.8,
    *,
    confidence: float | None = None,
    path: str = "direct",
    diagnosis: str | None = None,
    self_assessment: str | None = None,
) -> Dict[str, Any]:
    max_score_value = float(max_score)
    if max_score_value <= 0:
        raise ValueError("max_score must be greater than zero")

    score_value = float(score)
    if score_value < 0:
        raise ValueError("score must be greater than or equal to zero")
    if score_value > max_score_value:
        raise ValueError("score cannot exceed max_score")

    normalized = max(0.0, min(score_value / max_score_value, 1.0))
    passed = 1 if normalized >= float(pass_threshold) else 0
    diagnosis_value: str | None = None
    if diagnosis:
        candidate = str(diagnosis).strip().lower()
        if candidate in _DIAGNOSIS_VALUES:
            diagnosis_value = None if candidate == "none" else candidate

    reflection: str | None = None
    if isinstance(self_assessment, str):
        stripped = self_assessment.strip()
        reflection = stripped or None

    cur = _exec(
        """
        INSERT INTO quiz_attempts(
            user_id, subject_id, activity_id, score, max_score, normalized_score,
            pass_threshold, passed, confidence, path, diagnosis, self_assessment
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            subject_id,
            activity_id,
            score_value,
            max_score_value,
            normalized,
            float(pass_threshold),
            passed,
            float(confidence if confidence is not None else 1.0),
            path,
            diagnosis_value,
            reflection,
        ),
    )
    attempt_id = int(cur.lastrowid)
    context_skill = f"{subject_id}.{activity_id}"
    bloom_level = _LOWEST_BLOOM_LEVEL
    try:
        activity_rows = _query(
            "SELECT target_level, metadata FROM activities WHERE activity_id = ?",
            (activity_id,)
        )
        if activity_rows:
            row = activity_rows[0]
            target_level = row["target_level"]
            if target_level:
                bloom_level = str(target_level)
            metadata_raw = row["metadata"]
            if metadata_raw:
                try:
                    metadata = json.loads(metadata_raw)
                    meta_skill = metadata.get("skill") or metadata.get("skills")
                    if isinstance(meta_skill, str):
                        context_skill = f"{subject_id}.{meta_skill}"
                    elif isinstance(meta_skill, list) and meta_skill:
                        context_skill = f"{subject_id}.{meta_skill[0]}"
                except Exception:
                    pass
    except Exception:
        pass
    try:
        from xapi import emit as emit_xapi

        emit_xapi(
            user_id=user_id,
            verb="http://adlnet.gov/expapi/verbs/answered",
            object_id=f"activity:{activity_id}",
            score=normalized,
            success=bool(passed),
            response=None,
            context={
                "bloom": bloom_level or _LOWEST_BLOOM_LEVEL,
                "skill": context_skill,
                "confidence": float(confidence if confidence is not None else 1.0),
                "path": path,
            },
        )
    except Exception:
        pass
    return {
        "attempt_id": attempt_id,
        "user_id": user_id,
        "subject_id": subject_id,
        "activity_id": activity_id,
        "score": score_value,
        "max_score": max_score_value,
        "normalized_score": normalized,
        "passed": bool(passed),
        "pass_threshold": float(pass_threshold),
        "confidence": float(confidence if confidence is not None else 1.0),
        "path": path,
        "diagnosis": diagnosis_value,
        "self_assessment": reflection,
    }


def record_assessment_steps(
    assessment_id: int,
    steps: Iterable[Mapping[str, Any] | "AssessmentStepEvaluation"],
) -> None:
    prepared: list[tuple[Any, ...]] = []
    for step in steps:
        payload: Mapping[str, Any]
        if hasattr(step, "model_dump"):
            payload = step.model_dump()
        elif isinstance(step, Mapping):
            payload = step
        else:
            continue

        step_id = str(payload.get("step_id") or payload.get("id") or "").strip()
        if not step_id:
            continue
        outcome_raw = str(payload.get("outcome") or "incorrect").strip().lower()
        outcome = outcome_raw if outcome_raw in {"correct", "incorrect", "hint", "skipped"} else "incorrect"
        subskill = payload.get("subskill")
        score_delta = payload.get("score_delta")
        hint = payload.get("hint")
        feedback = payload.get("feedback")
        diagnosis_value = payload.get("diagnosis")
        if isinstance(diagnosis_value, str):
            diagnosis_clean = diagnosis_value.strip().lower()
            diagnosis_value = diagnosis_clean or None
        else:
            diagnosis_value = None

        prepared.append(
            (
                int(assessment_id),
                step_id,
                None if subskill is None else str(subskill),
                outcome,
                None if score_delta is None else float(score_delta),
                None if hint is None else str(hint),
                None if feedback is None else str(feedback),
                diagnosis_value,
            )
        )

    if not prepared:
        return

    with _conn() as con:
        con.executemany(
            """
            INSERT INTO assessment_step_results(
              assessment_id, step_id, subskill, outcome, score_delta, hint, feedback, diagnosis
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            prepared,
        )
        con.commit()


def record_assessment_error_patterns(
    assessment_id: int,
    patterns: Iterable[Mapping[str, Any] | "AssessmentErrorPattern"],
) -> None:
    prepared: list[tuple[Any, ...]] = []
    for pattern in patterns:
        payload: Mapping[str, Any]
        if hasattr(pattern, "model_dump"):
            payload = pattern.model_dump()
        elif isinstance(pattern, Mapping):
            payload = pattern
        else:
            continue

        code = str(payload.get("code") or payload.get("pattern_code") or "").strip()
        if not code:
            continue
        description = payload.get("description")
        subskill = payload.get("subskill")
        occurrences_raw = payload.get("occurrences")
        if occurrences_raw is None:
            occurrences_raw = payload.get("count")
        try:
            occurrences = max(1, int(occurrences_raw)) if occurrences_raw is not None else 1
        except (TypeError, ValueError):
            occurrences = 1

        prepared.append(
            (
                int(assessment_id),
                code,
                None if description is None else str(description),
                None if subskill is None else str(subskill),
                occurrences,
            )
        )

    if not prepared:
        return

    with _conn() as con:
        con.executemany(
            """
            INSERT INTO assessment_error_patterns(
              assessment_id, pattern_code, description, subskill, occurrences
            ) VALUES (?,?,?,?,?)
            """,
            prepared,
        )
        con.commit()


def save_assessment_result(result: "AssessmentResult") -> int:
    rubric_json = json_dumps([criterion.model_dump() for criterion in result.rubric_criteria])
    created_at = result.created_at.isoformat()
    reflection: str | None = None
    if isinstance(result.self_assessment, str):
        stripped = result.self_assessment.strip()
        reflection = stripped or None
    cur = _exec(
        """
        INSERT INTO assessment_results(
            user_id, domain, item_id, bloom_level, response, self_assessment, score,
            rubric_criteria, model_version, prompt_version, latency_ms,
            tokens_in, tokens_out, confidence, source, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            result.user_id,
            result.domain,
            result.item_id,
            result.bloom_level,
            result.response,
            reflection,
            float(result.score),
            rubric_json,
            result.model_version,
            result.prompt_version,
            result.latency_ms,
            result.tokens_in,
            result.tokens_out,
            float(result.confidence or 0.0),
            str(result.source),
            created_at,
        ),
    )
    try:
        from xapi import emit as emit_xapi

        emit_xapi(
            user_id=result.user_id,
            verb="http://adlnet.gov/expapi/verbs/evaluated",
            object_id=f"assessment:{result.item_id}",
            score=float(result.score),
            success=None,
            response=result.response,
            context={
                "bloom": result.bloom_level,
                "skill": f"{result.domain}.{result.item_id}",
                "model_version": result.model_version,
                "confidence": float(result.confidence or 0.0),
                "source": str(result.source),
                "self_assessment": reflection,
            },
        )
    except Exception:
        pass
    assessment_id = int(cur.lastrowid)

    try:
        if result.step_evaluations:
            record_assessment_steps(assessment_id, result.step_evaluations)
        if result.error_patterns:
            record_assessment_error_patterns(assessment_id, result.error_patterns)
    except Exception:
        pass

    return assessment_id


def list_assessment_step_results(assessment_id: int) -> list[sqlite3.Row]:
    return _query(
        """
        SELECT id, assessment_id, step_id, subskill, outcome, score_delta, hint, feedback, diagnosis, created_at
        FROM assessment_step_results
        WHERE assessment_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (int(assessment_id),),
    )


def list_assessment_error_patterns(assessment_id: int) -> list[sqlite3.Row]:
    return _query(
        """
        SELECT id, assessment_id, pattern_code, description, subskill, occurrences, created_at
        FROM assessment_error_patterns
        WHERE assessment_id = ?
        ORDER BY created_at ASC, id ASC
        """,
        (int(assessment_id),),
    )


def list_recent_step_diagnostics(
    user_id: str,
    domain: str,
    limit: int = 25,
) -> list[Dict[str, Any]]:
    rows = _query(
        """
        SELECT
          sr.assessment_id,
          sr.step_id,
          sr.subskill,
          sr.outcome,
          sr.score_delta,
          sr.hint,
          sr.feedback,
          sr.diagnosis,
          sr.created_at,
          ar.item_id,
          ar.bloom_level,
          ar.score AS assessment_score,
          ar.confidence AS assessment_confidence
        FROM assessment_step_results AS sr
        JOIN assessment_results AS ar ON ar.id = sr.assessment_id
        WHERE ar.user_id = ? AND ar.domain = ?
        ORDER BY sr.created_at DESC, sr.id DESC
        LIMIT ?
        """,
        (user_id, domain, int(limit)),
    )
    return [dict(row) for row in rows]


def record_llm_metric(
    user_id: Optional[str],
    model_id: str,
    prompt_version: str,
    prompt_variant: Optional[str],
    latency_ms: int,
    tokens_in: Optional[int],
    tokens_out: Optional[int],
    *,
    path_taken: Optional[str] = None,
    json_validated: bool | int | None = None,
) -> None:
    _exec(
        """
        INSERT INTO llm_metrics(user_id, model_id, prompt_version, prompt_variant, latency_ms, tokens_in, tokens_out, path_taken, json_validated)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            model_id,
            prompt_version,
            prompt_variant,
            int(latency_ms),
            None if tokens_in is None else int(tokens_in),
            None if tokens_out is None else int(tokens_out),
            path_taken,
            int(bool(json_validated)) if json_validated is not None else 0,
        ),
    )


def mark_last_llm_metric_validated(user_id: str) -> None:
    """Mark the most recent llm_metrics row for the user as JSON-validated."""

    _exec(
        """
        UPDATE llm_metrics
           SET json_validated = 1
         WHERE id = (
            SELECT id
              FROM llm_metrics
             WHERE user_id IS ? OR user_id = ?
             ORDER BY created_at DESC, id DESC
             LIMIT 1
         )
        """,
        (user_id, user_id),
    )


def list_recent_quiz_attempts(
    user_id: str,
    subject_id: str,
    limit: int = 20,
) -> list[sqlite3.Row]:
    return _query(
        """
        SELECT id, user_id, subject_id, activity_id, score, max_score, normalized_score, pass_threshold, passed, confidence, path, diagnosis, self_assessment, created_at
        FROM quiz_attempts
        WHERE user_id = ? AND subject_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (user_id, subject_id, int(limit))
    )

# -------------- mastery / bloom --------------
def upsert_mastery(user_id: str, skill: str, level: float):
    _write_mastery(user_id, skill, float(level))

def get_theta(user_id: str, skill: str) -> float:
    rows = _query(
        "SELECT level FROM mastery WHERE user_id = ? AND skill = ?",
        (user_id, skill)
    )
    return float(rows[0]["level"]) if rows else 0.0

def set_theta(user_id: str, skill: str, value: float):
    _write_mastery(user_id, skill, float(value))

def list_mastery(user_id: Optional[str] = None, limit: int = 100) -> list[sqlite3.Row]:
    if user_id:
        return _query(
            "SELECT id, user_id, skill, level, updated_at FROM mastery WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, int(limit))
        )
    return _query(
        "SELECT id, user_id, skill, level, updated_at FROM mastery ORDER BY updated_at DESC LIMIT ?",
        (int(limit),)
    )

def upsert_bloom(user_id: str, topic: str, level: str, score: int):
    _exec(
        """
        INSERT INTO bloom_score(user_id, topic, level, score, updated_at)
        VALUES (?,?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, topic) DO UPDATE SET
          level=excluded.level,
          score=excluded.score,
          updated_at=CURRENT_TIMESTAMP
        """,
        (user_id, topic, level, int(score))
    )

def list_bloom(user_id: Optional[str] = None, limit: int = 100) -> list[sqlite3.Row]:
    if user_id:
        return _query(
            "SELECT id, user_id, topic, level, score, updated_at FROM bloom_score WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, int(limit))
        )
    return _query(
        "SELECT id, user_id, topic, level, score, updated_at FROM bloom_score ORDER BY updated_at DESC LIMIT ?",
        (int(limit),)
    )


def upsert_bloom_progress(
    user_id: str,
    topic: str,
    current_level: str,
    *,
    reason: Optional[str] = None,
    average_score: Optional[float] = None,
    attempts_considered: Optional[int] = None,
    k_level: Optional[str] = None,
):
    previous = get_bloom_progress(user_id, topic)

    _exec(
        """
        INSERT INTO bloom_progress(user_id, topic, current_level, updated_at)
        VALUES (?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, topic) DO UPDATE SET
          current_level=excluded.current_level,
          updated_at=CURRENT_TIMESTAMP
        """,
        (user_id, topic, current_level),
    )

    previous_level = previous["current_level"] if previous else None
    changed = previous is None or previous_level != current_level
    if not changed:
        return

    _exec(
        """
        INSERT INTO bloom_progress_history(
          user_id,
          topic,
          previous_level,
          new_level,
          k_level,
          reason,
          average_score,
          attempts_considered
        )
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            topic,
            previous_level,
            current_level,
            k_level,
            reason,
            average_score if average_score is None else float(average_score),
            attempts_considered if attempts_considered is None else int(attempts_considered),
        ),
    )


def get_bloom_progress(user_id: str, topic: str) -> Optional[sqlite3.Row]:
    rows = _query(
        "SELECT id, user_id, topic, current_level, updated_at FROM bloom_progress WHERE user_id = ? AND topic = ?",
        (user_id, topic),
    )
    return rows[0] if rows else None


def list_bloom_progress(user_id: Optional[str] = None, limit: int = 100) -> list[sqlite3.Row]:
    if user_id:
        return _query(
            "SELECT id, user_id, topic, current_level, updated_at FROM bloom_progress WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, int(limit)),
        )
    return _query(
        "SELECT id, user_id, topic, current_level, updated_at FROM bloom_progress ORDER BY updated_at DESC LIMIT ?",
        (int(limit),),
    )


def list_bloom_progress_history(
    user_id: Optional[str] = None,
    *,
    topic: Optional[str] = None,
    limit: int = 200,
) -> list[sqlite3.Row]:
    clauses = []
    params: list[Any] = []
    if user_id:
        clauses.append("user_id = ?")
        params.append(user_id)
    if topic:
        clauses.append("topic = ?")
        params.append(topic)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(int(limit))
    sql = (
        "SELECT id, user_id, topic, previous_level, new_level, k_level, reason, average_score, attempts_considered, created_at "
        f"FROM bloom_progress_history {where} ORDER BY created_at DESC, id DESC LIMIT ?"
    )
    return _query(sql, params)


# -------------- adaptive learning path --------------
def get_learning_path_state(user_id: str, subject_id: str) -> Optional[dict[str, Any]]:
    rows = _query(
        "SELECT state_json, updated_at FROM learning_path_state WHERE user_id = ? AND subject_id = ?",
        (user_id, subject_id),
    )
    if not rows:
        return None
    row = rows[0]
    state = _decode_json_field(row["state_json"]) or {}
    if isinstance(state, dict):
        state = {**state, "updated_at": row["updated_at"]}
    return state if isinstance(state, dict) else {"updated_at": row["updated_at"]}


def upsert_learning_path_state(user_id: str, subject_id: str, state: dict[str, Any]) -> None:
    payload = json_dumps(state)
    _exec(
        """
        INSERT INTO learning_path_state(user_id, subject_id, state_json, updated_at)
        VALUES (?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(user_id, subject_id) DO UPDATE SET
          state_json=excluded.state_json,
          updated_at=CURRENT_TIMESTAMP
        """,
        (user_id, subject_id, payload),
    )


def log_learning_path_event(
    user_id: str,
    subject_id: str,
    bloom_level: str,
    action: str,
    *,
    reason: Optional[str] = None,
    reason_code: Optional[str] = None,
    confidence: Optional[float] = None,
    evidence: Optional[dict[str, Any]] = None,
) -> None:
    _exec(
        """
        INSERT INTO learning_path_events(user_id, subject_id, bloom_level, action, reason_code, reason, confidence, evidence)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            user_id,
            subject_id,
            bloom_level,
            action,
            reason_code,
            reason,
            None if confidence is None else float(confidence),
            json_dumps(evidence) if evidence is not None else None,
        ),
    )


def list_learning_path_events(
    user_id: str,
    subject_id: Optional[str] = None,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    clauses = ["user_id = ?"]
    params: list[Any] = [user_id]
    if subject_id:
        clauses.append("subject_id = ?")
        params.append(subject_id)
    params.append(int(limit))
    where = " AND ".join(clauses)
    rows = _query(
        f"SELECT user_id, subject_id, bloom_level, action, reason_code, reason, confidence, evidence, created_at "
        f"FROM learning_path_events WHERE {where} ORDER BY created_at DESC, id DESC LIMIT ?",
        params,
    )
    events: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        entry["evidence"] = _decode_json_field(entry.get("evidence"))
        events.append(entry)
    return events


def list_learning_path_states(
    user_id: Optional[str] = None,
    subject_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    clauses = []
    params: list[Any] = []
    if user_id:
        clauses.append("user_id = ?")
        params.append(user_id)
    if subject_id:
        clauses.append("subject_id = ?")
        params.append(subject_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = _query(
        f"SELECT user_id, subject_id, state_json, updated_at FROM learning_path_state {where}",
        params,
    )
    states: list[dict[str, Any]] = []
    for row in rows:
        payload = _decode_json_field(row["state_json"]) or {}
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("levels", {})
        payload.setdefault("history", [])
        entry = {
            "user_id": row["user_id"],
            "subject_id": row["subject_id"],
            "state": payload,
            "updated_at": row["updated_at"],
        }
        states.append(entry)
    return states


def apply_learning_path_override(
    user_id: str,
    subject_id: str,
    *,
    target_level: Optional[str] = None,
    notes: Optional[str] = None,
    expires_at: Optional[str] = None,
    applied_by: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if not user_id or not subject_id:
        raise ValueError("user_id and subject_id are required")

    state_record = get_learning_path_state(user_id, subject_id) or {}
    state = dict(state_record)
    state.pop("updated_at", None)
    levels = state.setdefault("levels", {})
    history = state.setdefault("history", [])
    if not isinstance(levels, dict):
        levels = {}
        state["levels"] = levels
    if not isinstance(history, list):
        history = []
        state["history"] = history

    override_entry: dict[str, Any] = {
        "target_level": target_level,
        "notes": notes,
        "expires_at": expires_at,
        "applied_by": applied_by,
        "metadata": metadata or {},
        "created_at": datetime.utcnow().isoformat(),
    }

    if target_level:
        state["current_level"] = target_level
        levels.setdefault(target_level, float(levels.get(target_level, 0.0)))

    overrides = state.setdefault("manual_overrides", [])
    if not isinstance(overrides, list):
        overrides = []
        state["manual_overrides"] = overrides
    overrides.append(override_entry)
    state["manual_overrides"] = overrides[-10:]
    state["manual_override"] = override_entry

    upsert_learning_path_state(user_id, subject_id, state)
    refreshed = get_learning_path_state(user_id, subject_id) or {}
    return refreshed


def _detect_hint_signal(op: Optional[str], payload: Any) -> bool:
    if op and "hint" in op.lower():
        return True
    if isinstance(payload, dict):
        for key in ("event_type", "type", "action"):
            value = payload.get(key)
            if isinstance(value, str) and "hint" in value.lower():
                return True
        details = payload.get("details")
        if isinstance(details, dict):
            return _detect_hint_signal(None, details)
    if isinstance(payload, str) and "hint" in payload.lower():
        return True
    return False


def _subject_from_payload(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        for key in ("subject_id", "subject", "skill", "topic"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        details = payload.get("details")
        if isinstance(details, dict):
            return _subject_from_payload(details)
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            return _subject_from_payload(metadata)
    return None


def compute_teacher_analytics(window_days: int = 7) -> list[dict[str, Any]]:
    window_days = max(1, int(window_days))
    since = datetime.utcnow() - timedelta(days=window_days)
    since_text = since.strftime("%Y-%m-%d %H:%M:%S")
    generated_at = datetime.utcnow().isoformat()

    analytics: dict[tuple[str, str], dict[str, Any]] = {}

    for record in list_learning_path_states():
        user_id = record["user_id"]
        subject_id = record["subject_id"]
        state = record["state"]
        history = state.get("history") if isinstance(state.get("history"), list) else []
        history_tail = [item for item in history[-5:] if isinstance(item, dict)]
        entry = analytics.setdefault(
            (user_id, subject_id),
            {
                "user_id": user_id,
                "subject_id": subject_id,
                "window_days": window_days,
                "hint_count": 0,
                "low_confidence_count": 0,
                "history_tail": history_tail,
                "current_level": state.get("current_level"),
                "levels": state.get("levels") if isinstance(state.get("levels"), dict) else {},
                "manual_override": state.get("manual_override"),
                "manual_overrides": state.get("manual_overrides") if isinstance(state.get("manual_overrides"), list) else [],
                "state_updated_at": record.get("updated_at"),
                "hint_events": [],
                "last_assessment_at": None,
                "confidence_trend": None,
                "recent_confidence": None,
                "recent_score": None,
                "confidence_interval_lower": None,
                "confidence_interval_upper": None,
                "confidence_interval_mean": None,
                "confidence_interval_margin": None,
                "confidence_interval_width": None,
                "confidence_interval_confidence_level": None,
                "confidence_interval_sample_size": 0,
                "_history_for_calc": history_tail,
                "_confidences": [],
                "_scores": [],
            },
        )
        entry["history_tail"] = history_tail
        entry["levels"] = entry.get("levels") or {}
        if not isinstance(entry["levels"], dict):
            entry["levels"] = {}
        entry["manual_override"] = state.get("manual_override")
        overrides = state.get("manual_overrides")
        entry["manual_overrides"] = overrides if isinstance(overrides, list) else []
        entry["state_updated_at"] = record.get("updated_at")
        entry["current_level"] = state.get("current_level")
        entry["_history_for_calc"] = history_tail

    assessment_rows = _query(
        """
        SELECT user_id, domain, score, confidence, created_at
        FROM assessment_results
        WHERE created_at >= ?
        """,
        (since_text,),
    )
    for row in assessment_rows:
        user_id = row["user_id"]
        subject_id = row["domain"] or "general"
        key = (user_id, subject_id)
        entry = analytics.setdefault(
            key,
            {
                "user_id": user_id,
                "subject_id": subject_id,
                "window_days": window_days,
                "hint_count": 0,
                "low_confidence_count": 0,
                "history_tail": [],
                "current_level": None,
                "levels": {},
                "manual_override": None,
                "manual_overrides": [],
                "state_updated_at": None,
                "hint_events": [],
                "last_assessment_at": None,
                "confidence_trend": None,
                "recent_confidence": None,
                "recent_score": None,
                "confidence_interval_lower": None,
                "confidence_interval_upper": None,
                "confidence_interval_mean": None,
                "confidence_interval_margin": None,
                "confidence_interval_width": None,
                "confidence_interval_confidence_level": None,
                "confidence_interval_sample_size": 0,
                "_history_for_calc": [],
                "_confidences": [],
                "_scores": [],
            },
        )
        score = float(row["score"] or 0.0)
        confidence = float(row["confidence"] or 0.0)
        timestamp = _parse_timestamp(row["created_at"])
        entry["_scores"].append((timestamp, score))
        entry["_confidences"].append((timestamp, confidence))
        if confidence < 0.45:
            entry["low_confidence_count"] = int(entry.get("low_confidence_count", 0)) + 1
        last_at = entry.get("last_assessment_at")
        if not last_at or timestamp > _parse_timestamp(last_at):
            entry["last_assessment_at"] = timestamp.isoformat()

    hint_rows = _query(
        """
        SELECT user_id, op, payload, created_at
        FROM journey_log
        WHERE created_at >= ? AND (lower(op) LIKE '%hint%' OR payload LIKE '%hint%')
        """,
        (since_text,),
    )
    for row in hint_rows:
        payload = _decode_json_field(row["payload"]) or {}
        user_id = row["user_id"]
        subject_id = _subject_from_payload(payload) or "general"
        entry = analytics.setdefault(
            (user_id, subject_id),
            {
                "user_id": user_id,
                "subject_id": subject_id,
                "window_days": window_days,
                "hint_count": 0,
                "low_confidence_count": 0,
                "history_tail": [],
                "current_level": None,
                "levels": {},
                "manual_override": None,
                "manual_overrides": [],
                "state_updated_at": None,
                "hint_events": [],
                "last_assessment_at": None,
                "confidence_trend": None,
                "recent_confidence": None,
                "recent_score": None,
                "confidence_interval_lower": None,
                "confidence_interval_upper": None,
                "confidence_interval_mean": None,
                "confidence_interval_margin": None,
                "confidence_interval_width": None,
                "confidence_interval_confidence_level": None,
                "confidence_interval_sample_size": 0,
                "_history_for_calc": [],
                "_confidences": [],
                "_scores": [],
            },
        )
        if _detect_hint_signal(row["op"], payload):
            entry["hint_count"] = int(entry.get("hint_count", 0)) + 1
            entry.setdefault("hint_events", []).append(
                {
                    "op": row["op"],
                    "created_at": row["created_at"],
                    "subject_inferred": subject_id,
                }
            )
            entry["hint_events"] = entry["hint_events"][-10:]

    results: list[dict[str, Any]] = []
    for (user_id, subject_id), entry in analytics.items():
        history_for_calc = [item for item in entry.get("_history_for_calc", []) if isinstance(item, dict)]
        failure_signals = 0
        progress_markers = 0
        deltas: list[float] = []
        for item in history_for_calc:
            confidence_raw = item.get("confidence")
            confidence = float(confidence_raw) if confidence_raw is not None else 1.0
            delta_raw = item.get("delta")
            try:
                delta = float(delta_raw) if delta_raw is not None else 0.0
            except (TypeError, ValueError):
                delta = 0.0
            correct = bool(item.get("correct", True))
            if delta_raw is not None:
                try:
                    deltas.append(float(delta_raw))
                except (TypeError, ValueError):
                    pass
            if not correct or confidence < 0.45:
                failure_signals += 1
            if delta > 0:
                progress_markers += 1

        negative_deltas = [val for val in deltas if val < -0.01]
        recent_delta = deltas[-1] if deltas else None
        regression_detected = False
        if history_for_calc and len(history_for_calc) >= 3:
            if failure_signals >= len(history_for_calc) - 1 and progress_markers == 0:
                regression_detected = True
        if len(negative_deltas) >= 2 or (recent_delta is not None and recent_delta < -0.01):
            regression_detected = True

        confidences = sorted(entry.get("_confidences", []), key=lambda pair: pair[0])
        confidence_trend = None
        if confidences:
            conf_values = [float(val) for _, val in confidences]
            entry["confidence_interval_sample_size"] = len(conf_values)
            entry["recent_confidence"] = round(statistics.fmean(conf_values[-3:]), 3)
            if len(conf_values) >= 2:
                confidence_trend = round(conf_values[-1] - conf_values[0], 3)
                entry["confidence_trend"] = confidence_trend
            ci = mean_confidence_interval(conf_values)
            if ci:
                entry["confidence_interval_lower"] = ci.get("lower")
                entry["confidence_interval_upper"] = ci.get("upper")
                entry["confidence_interval_mean"] = ci.get("mean")
                entry["confidence_interval_margin"] = ci.get("margin")
                entry["confidence_interval_confidence_level"] = ci.get("confidence_level")
                width = None
                lower = ci.get("lower")
                upper = ci.get("upper")
                if lower is not None and upper is not None:
                    width = round(float(upper) - float(lower), 4)
                entry["confidence_interval_width"] = width
        else:
            entry["confidence_interval_sample_size"] = 0
            entry["confidence_interval_lower"] = None
            entry["confidence_interval_upper"] = None
            entry["confidence_interval_mean"] = None
            entry["confidence_interval_margin"] = None
            entry["confidence_interval_confidence_level"] = None
            entry["confidence_interval_width"] = None
        scores = sorted(entry.get("_scores", []), key=lambda pair: pair[0])
        if scores:
            score_values = [float(val) for _, val in scores]
            entry["recent_score"] = round(statistics.fmean(score_values[-3:]), 3)

        if confidence_trend is not None and confidence_trend <= -0.1:
            regression_detected = True

        low_confidence_hits = int(entry.get("low_confidence_count", 0) or 0)
        hint_hits = int(entry.get("hint_count", 0) or 0)
        recent_confidence = entry.get("recent_confidence")
        flag_low_confidence = bool(low_confidence_hits >= 3)
        if recent_confidence is not None:
            try:
                if float(recent_confidence) < 0.45:
                    flag_low_confidence = True
            except (TypeError, ValueError):
                pass
        flag_high_hints = bool(hint_hits >= 4)
        flag_regression = bool(regression_detected)

        flagged_reasons: list[str] = []
        if flag_low_confidence:
            flagged_reasons.append("low_confidence")
        if flag_high_hints:
            flagged_reasons.append("high_hints")
        if flag_regression:
            flagged_reasons.append("regression")

        entry["flag_low_confidence"] = flag_low_confidence
        entry["flag_high_hints"] = flag_high_hints
        entry["flag_regression"] = flag_regression
        entry["flagged_reasons"] = sorted(dict.fromkeys(flagged_reasons))
        entry["stuck_flag"] = bool(entry["flagged_reasons"])

        entry.pop("_confidences", None)
        entry.pop("_scores", None)
        entry.pop("_history_for_calc", None)

        entry["analytics_updated_at"] = generated_at

        payload = {k: v for k, v in entry.items() if k not in {"user_id", "subject_id"}}
        metrics_json = json_dumps(payload)
        _exec(
            """
            INSERT INTO teacher_analytics(user_id, subject_id, metrics_json, updated_at)
            VALUES (?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, subject_id) DO UPDATE SET
              metrics_json=excluded.metrics_json,
              updated_at=CURRENT_TIMESTAMP
            """,
            (user_id, subject_id, metrics_json),
        )
        results.append({"user_id": user_id, "subject_id": subject_id, **payload})

    existing_keys = {
        (row["user_id"], row["subject_id"])
        for row in _query("SELECT user_id, subject_id FROM teacher_analytics", ())
    }
    desired_keys = set(analytics.keys())
    for key in existing_keys - desired_keys:
        _exec(
            "DELETE FROM teacher_analytics WHERE user_id = ? AND subject_id = ?",
            key,
        )

    results.sort(
        key=lambda item: (
            0 if item.get("stuck_flag") else 1,
            -int(item.get("hint_count") or 0),
            -int(item.get("low_confidence_count") or 0),
            item.get("user_id") or "",
            item.get("subject_id") or "",
        )
    )
    return results


def list_teacher_analytics(
    *,
    subject_id: Optional[str] = None,
    only_flagged: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    rows = _query(
        "SELECT user_id, subject_id, metrics_json, updated_at FROM teacher_analytics ORDER BY updated_at DESC",
        (),
    )
    data: list[dict[str, Any]] = []
    for row in rows:
        metrics = _decode_json_field(row["metrics_json"]) or {}
        if not isinstance(metrics, dict):
            metrics = {}
        entry = {
            "user_id": row["user_id"],
            "subject_id": row["subject_id"],
            "updated_at": row["updated_at"],
            **metrics,
        }
        if subject_id and entry.get("subject_id") != subject_id:
            continue
        if only_flagged and not entry.get("stuck_flag"):
            continue
        data.append(entry)

    data.sort(
        key=lambda item: (
            0 if item.get("stuck_flag") else 1,
            -int(item.get("hint_count") or 0),
            -int(item.get("low_confidence_count") or 0),
            item.get("user_id") or "",
            item.get("subject_id") or "",
        )
    )
    if limit:
        data = data[: int(limit)]
    return data


# -------------- feedback & quality control --------------
def store_feedback(
    user_id: Optional[str],
    answer_id: str,
    rating: str,
    comment: str = "",
    *,
    confidence: Optional[float] = None,
    tags: Optional[Sequence[str]] = None,
) -> int:
    if not answer_id:
        raise ValueError("answer_id is required")
    normalized_rating = (rating or "").strip().lower()
    if normalized_rating not in {"up", "down", "flag"}:
        raise ValueError("rating must be 'up', 'down', or 'flag'")
    tag_payload: Optional[str] = None
    if tags:
        tag_payload = json_dumps([str(tag) for tag in tags if tag])
    cur = _exec(
        """
        INSERT INTO answer_feedback(user_id, answer_id, rating, comment, confidence, tags)
        VALUES (?,?,?,?,?,?)
        """,
        (
            user_id,
            answer_id,
            normalized_rating,
            comment.strip() or None,
            None if confidence is None else float(confidence),
            tag_payload,
        ),
    )
    return int(cur.lastrowid)


def aggregate_feedback(
    *,
    answer_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    clauses: list[str] = []
    params: list[Any] = []
    if answer_id:
        clauses.append("answer_id = ?")
        params.append(answer_id)
    if user_id:
        clauses.append("user_id = ?")
        params.append(user_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = _query(
        f"SELECT rating, COUNT(*) as count, AVG(COALESCE(confidence, 0.0)) as avg_confidence "
        f"FROM answer_feedback {where} GROUP BY rating",
        params,
    )
    summary = {"total": 0, "ratings": {}, "average_confidence": None}
    confidence_values: list[float] = []
    for row in rows:
        count = int(row["count"])
        rating_value = row["rating"]
        summary["ratings"][rating_value] = count
        summary["total"] += count
        avg_conf = row["avg_confidence"]
        if avg_conf is not None:
            confidence_values.append(float(avg_conf))
    if confidence_values:
        summary["average_confidence"] = round(sum(confidence_values) / len(confidence_values), 3)
    return summary


def list_feedback(answer_id: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if answer_id:
        clauses.append("answer_id = ?")
        params.append(answer_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(int(limit))
    rows = _query(
        f"SELECT id, user_id, answer_id, rating, comment, confidence, tags, created_at "
        f"FROM answer_feedback {where} ORDER BY created_at DESC, id DESC LIMIT ?",
        params,
    )
    results: list[dict[str, Any]] = []
    for row in rows:
        entry = dict(row)
        entry["tags"] = _decode_json_field(entry.get("tags"))
        results.append(entry)
    return results


# -------------- privacy & consent --------------
def record_privacy_consent(user_id: str, consented: bool, consent_text: Optional[str] = None) -> None:
    if not user_id:
        raise ValueError("user_id required")
    _exec(
        """
        INSERT INTO user_consent(user_id, consented, consent_text, consented_at)
        VALUES (?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
          consented=excluded.consented,
          consent_text=COALESCE(excluded.consent_text, user_consent.consent_text),
          consented_at=CURRENT_TIMESTAMP
        """,
        (user_id, int(bool(consented)), consent_text),
    )


def get_privacy_consent(user_id: str) -> Optional[dict[str, Any]]:
    if not user_id:
        return None
    rows = _query(
        "SELECT user_id, consented, consent_text, consented_at FROM user_consent WHERE user_id = ?",
        (user_id,),
    )
    if not rows:
        return None
    row = rows[0]
    return {
        "user_id": row["user_id"],
        "consented": bool(row["consented"]),
        "consent_text": row["consent_text"],
        "consented_at": row["consented_at"],
    }

# -------------- learner model --------------
def get_learner_model(user_id: str) -> LearnerModel:
    if not user_id:
        raise ValueError("user_id is required")

    profile_rows = _query(
        """
        SELECT goals_json, preferences_json, history_summary, updated_at
        FROM learner_profile
        WHERE user_id = ?
        """,
        (user_id,),
    )
    if profile_rows:
        profile_row = profile_rows[0]
        goals_raw = _decode_json_field(profile_row["goals_json"]) or []
        preferences_raw = _decode_json_field(profile_row["preferences_json"]) or {}
        history_summary = profile_row["history_summary"]
        updated_at = _coerce_to_utc(_parse_timestamp(profile_row["updated_at"]))
    else:
        goals_raw = []
        preferences_raw = {}
        history_summary = None
        updated_at = _coerce_to_utc(None)

    if not isinstance(goals_raw, list):
        goals_raw = [goals_raw]

    pref_dict: dict[str, Any]
    if isinstance(preferences_raw, dict):
        pref_dict = dict(preferences_raw)
    else:
        pref_dict = {"additional": preferences_raw}

    additional = pref_dict.get("additional")
    if not isinstance(additional, dict):
        additional = {} if additional is None else {"value": additional}

    known_pref_keys = {
        "modalities",
        "pacing",
        "language_level",
        "languages",
        "time_windows",
        "additional",
    }
    for key in list(pref_dict.keys()):
        if key not in known_pref_keys:
            additional[key] = pref_dict.pop(key)
    pref_dict["additional"] = additional

    skill_rows = _query(
        """
        SELECT skill_id, proficiency, bloom_low, bloom_high, updated_at
        FROM learner_priors
        WHERE user_id = ?
        ORDER BY skill_id
        """,
        (user_id,),
    )
    confidence_rows = _query(
        """
        SELECT skill_id, confidence, updated_at
        FROM learner_confidence
        WHERE user_id = ?
        """,
        (user_id,),
    )

    confidence_map: dict[str, tuple[float | None, datetime | None]] = {}
    for row in confidence_rows:
        confidence_map[row["skill_id"]] = (
            float(row["confidence"]) if row["confidence"] is not None else None,
            _coerce_to_utc(_parse_timestamp(row["updated_at"])) if row["updated_at"] else None,
        )

    skills: list[dict[str, Any]] = []
    for row in skill_rows:
        skill_id = row["skill_id"]
        bloom_band = None
        lower = row["bloom_low"]
        upper = row["bloom_high"]
        if lower or upper:
            bloom_band = {"lower": lower, "upper": upper}
        entry: dict[str, Any] = {
            "skill_id": skill_id,
            "proficiency": float(row["proficiency"]) if row["proficiency"] is not None else None,
            "bloom_band": bloom_band,
            "last_updated": _coerce_to_utc(_parse_timestamp(row["updated_at"])) if row["updated_at"] else None,
        }
        confidence_tuple = confidence_map.pop(skill_id, None)
        if confidence_tuple:
            value, conf_updated = confidence_tuple
            if value is not None:
                entry["confidence"] = value
            if conf_updated is not None:
                entry["confidence_updated_at"] = conf_updated
        skills.append(entry)

    for skill_id, (value, conf_updated) in confidence_map.items():
        skills.append(
            {
                "skill_id": skill_id,
                "proficiency": None,
                "bloom_band": None,
                "confidence": value,
                "confidence_updated_at": conf_updated,
            }
        )

    misconception_rows = _query(
        """
        SELECT id, skill_id, description, severity, evidence_json, last_seen, updated_at
        FROM learner_misconceptions
        WHERE user_id = ?
        ORDER BY updated_at DESC, id DESC
        """,
        (user_id,),
    )

    misconceptions: list[dict[str, Any]] = []
    for row in misconception_rows:
        evidence_raw = _decode_json_field(row["evidence_json"]) or []
        if not isinstance(evidence_raw, list):
            evidence_raw = [evidence_raw]
        misconceptions.append(
            {
                "misconception_id": str(row["id"]),
                "skill_id": row["skill_id"],
                "description": row["description"],
                "severity": row["severity"],
                "evidence": evidence_raw,
                "last_seen": _coerce_to_utc(_parse_timestamp(row["last_seen"])) if row["last_seen"] else None,
            }
        )

    payload = {
        "user_id": user_id,
        "goals": goals_raw,
        "preferences": pref_dict,
        "skills": skills,
        "misconceptions": misconceptions,
        "history_summary": history_summary,
        "updated_at": updated_at,
    }
    return LearnerModel.model_validate(payload)


def update_learner_model(model: LearnerModel) -> LearnerModel:
    if not model.user_id:
        raise ValueError("user_id is required")

    now = datetime.now(timezone.utc)
    normalized = model.model_copy(update={"updated_at": now})
    now_iso = now.isoformat()

    goals_payload = [
        goal.model_dump(mode="json", exclude_none=True) for goal in normalized.goals
    ]
    preferences_payload = normalized.preferences.model_dump(mode="json", exclude_none=True)

    with _conn() as con:
        con.execute(
            """
            INSERT INTO learner_profile (user_id, goals_json, preferences_json, history_summary, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              goals_json=excluded.goals_json,
              preferences_json=excluded.preferences_json,
              history_summary=excluded.history_summary,
              updated_at=excluded.updated_at
            """,
            (
                normalized.user_id,
                json_dumps(goals_payload),
                json_dumps(preferences_payload),
                normalized.history_summary,
                now_iso,
                now_iso,
            ),
        )

        skill_ids: set[str] = set()
        confidence_ids: set[str] = set()

        for skill in normalized.skills:
            skill_ids.add(skill.skill_id)
            skill_updated = _coerce_to_utc(skill.last_updated, now)
            skill_updated_iso = skill_updated.isoformat()
            bloom_band = skill.bloom_band
            lower = bloom_band.lower if bloom_band else None
            upper = bloom_band.upper if bloom_band else None
            con.execute(
                """
                INSERT INTO learner_priors (user_id, skill_id, proficiency, bloom_low, bloom_high, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, skill_id) DO UPDATE SET
                  proficiency=excluded.proficiency,
                  bloom_low=excluded.bloom_low,
                  bloom_high=excluded.bloom_high,
                  updated_at=excluded.updated_at
                """,
                (
                    normalized.user_id,
                    skill.skill_id,
                    skill.proficiency,
                    lower,
                    upper,
                    skill_updated_iso,
                    skill_updated_iso,
                ),
            )

            if skill.confidence is not None:
                confidence_ids.add(skill.skill_id)
                conf_updated = _coerce_to_utc(skill.confidence_updated_at, now)
                conf_iso = conf_updated.isoformat()
                con.execute(
                    """
                    INSERT INTO learner_confidence (user_id, skill_id, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, skill_id) DO UPDATE SET
                      confidence=excluded.confidence,
                      updated_at=excluded.updated_at
                    """,
                    (
                        normalized.user_id,
                        skill.skill_id,
                        float(skill.confidence),
                        conf_iso,
                        conf_iso,
                    ),
                )

        if skill_ids:
            placeholders = ",".join("?" for _ in skill_ids)
            con.execute(
                f"DELETE FROM learner_priors WHERE user_id = ? AND skill_id NOT IN ({placeholders})",
                (normalized.user_id, *skill_ids),
            )
        else:
            con.execute(
                "DELETE FROM learner_priors WHERE user_id = ?",
                (normalized.user_id,),
            )

        if confidence_ids:
            placeholders = ",".join("?" for _ in confidence_ids)
            con.execute(
                f"DELETE FROM learner_confidence WHERE user_id = ? AND skill_id NOT IN ({placeholders})",
                (normalized.user_id, *confidence_ids),
            )
        else:
            con.execute(
                "DELETE FROM learner_confidence WHERE user_id = ?",
                (normalized.user_id,),
            )

        con.execute("DELETE FROM learner_misconceptions WHERE user_id = ?", (normalized.user_id,))
        for entry in normalized.misconceptions:
            last_seen_dt = _coerce_to_utc(entry.last_seen, now)
            last_seen_iso = last_seen_dt.isoformat()
            evidence_payload = entry.evidence
            con.execute(
                """
                INSERT INTO learner_misconceptions (
                  user_id, skill_id, description, severity, evidence_json, last_seen, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized.user_id,
                    entry.skill_id,
                    entry.description,
                    entry.severity,
                    json_dumps(evidence_payload),
                    last_seen_iso,
                    last_seen_iso,
                    now_iso,
                ),
            )

    return get_learner_model(normalized.user_id)


# -------------- helpers --------------
def json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _ensure_mastery_unique_index(con: sqlite3.Connection):
    """Ensure a unique constraint exists for (user_id, skill) pairs.

    Older deployments created the mastery table without the UNIQUE constraint.
    We attempt to backfill by removing duplicate rows (keeping the newest) and
    adding an explicit unique index. Any failures leave the database unchanged.
    """

    try:
        con.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mastery_user_skill ON mastery(user_id, skill)"
        )
        return
    except sqlite3.IntegrityError:
        pass
    except sqlite3.OperationalError as exc:
        # If the table is missing entirely we bubble up the error, otherwise
        # fall through to deduplication retry.
        if "no such table" in str(exc).lower():
            raise

    # Remove duplicates, keeping the most recent entry per (user_id, skill).
    con.execute(
        """
        DELETE FROM mastery
        WHERE rowid NOT IN (
          SELECT MAX(rowid)
          FROM mastery
          GROUP BY user_id, skill
        )
        """
    )
    con.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_mastery_user_skill ON mastery(user_id, skill)"
    )


def _write_mastery(user_id: str, skill: str, value: float):
    global _MASTERY_CONSTRAINT_AVAILABLE

    if _MASTERY_CONSTRAINT_AVAILABLE is False:
        _legacy_mastery_write(user_id, skill, value)
        return

    try:
        _exec(
            """
            INSERT INTO mastery(user_id, skill, level, updated_at) VALUES (?,?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, skill) DO UPDATE SET
              level=excluded.level,
              updated_at=CURRENT_TIMESTAMP
            """,
            (user_id, skill, value)
        )
        _MASTERY_CONSTRAINT_AVAILABLE = True
    except sqlite3.OperationalError as exc:
        if "ON CONFLICT clause does not match any PRIMARY KEY or UNIQUE constraint" not in str(exc):
            raise
        _MASTERY_CONSTRAINT_AVAILABLE = False
        _legacy_mastery_write(user_id, skill, value)


def _legacy_mastery_write(user_id: str, skill: str, value: float):
    with _conn() as con:
        cur = con.execute(
            "SELECT id FROM mastery WHERE user_id = ? AND skill = ?",
            (user_id, skill)
        )
        row = cur.fetchone()
        if row:
            con.execute(
                "UPDATE mastery SET level = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (value, row["id"])
            )
        else:
            con.execute(
                "INSERT INTO mastery(user_id, skill, level, updated_at) VALUES (?,?,?,CURRENT_TIMESTAMP)",
                (user_id, skill, value)
            )
        con.commit()


def export_user_data(user_id: str) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {"user_id": user_id}

    def _rows(sql: str, params: Iterable) -> list[Dict[str, Any]]:
        return [dict(row) for row in _query(sql, params)]

    bundle["users"] = _rows("SELECT user_id, email, created_at FROM users WHERE user_id = ?", (user_id,))
    bundle["mastery"] = _rows("SELECT * FROM mastery WHERE user_id = ?", (user_id,))
    bundle["user_progress"] = _rows("SELECT * FROM user_progress WHERE user_id = ?", (user_id,))
    bundle["bloom_score"] = _rows("SELECT * FROM bloom_score WHERE user_id = ?", (user_id,))
    bundle["bloom_progress"] = _rows("SELECT * FROM bloom_progress WHERE user_id = ?", (user_id,))
    bundle["bloom_progress_history"] = _rows("SELECT * FROM bloom_progress_history WHERE user_id = ? ORDER BY created_at", (user_id,))
    bundle["quiz_attempts"] = _rows("SELECT * FROM quiz_attempts WHERE user_id = ?", (user_id,))

    bundle["learner_profile"] = []
    for row in _query("SELECT * FROM learner_profile WHERE user_id = ?", (user_id,)):
        entry = dict(row)
        for key in ("goals_json", "preferences_json"):
            entry[key] = _decode_json_field(entry.get(key))
        bundle["learner_profile"].append(entry)

    bundle["learner_priors"] = _rows("SELECT * FROM learner_priors WHERE user_id = ?", (user_id,))
    bundle["learner_confidence"] = _rows("SELECT * FROM learner_confidence WHERE user_id = ?", (user_id,))

    bundle["learner_misconceptions"] = []
    for row in _query(
        "SELECT * FROM learner_misconceptions WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
        (user_id,),
    ):
        entry = dict(row)
        entry["evidence_json"] = _decode_json_field(entry.get("evidence_json"))
        bundle["learner_misconceptions"].append(entry)

    bundle["eval_pretest_attempts"] = []
    for row in _query(
        "SELECT * FROM eval_pretest_attempts WHERE learner_id = ? ORDER BY attempted_at, id",
        (user_id,),
    ):
        entry = dict(row)
        entry["metadata"] = _decode_json_field(entry.get("metadata"))
        bundle["eval_pretest_attempts"].append(entry)

    bundle["eval_posttest_attempts"] = []
    for row in _query(
        "SELECT * FROM eval_posttest_attempts WHERE learner_id = ? ORDER BY attempted_at, id",
        (user_id,),
    ):
        entry = dict(row)
        entry["metadata"] = _decode_json_field(entry.get("metadata"))
        bundle["eval_posttest_attempts"].append(entry)

    bundle["learning_path_state"] = []
    for row in _query("SELECT * FROM learning_path_state WHERE user_id = ?", (user_id,)):
        entry = dict(row)
        entry["state_json"] = _decode_json_field(entry.get("state_json"))
        bundle["learning_path_state"].append(entry)

    bundle["learning_path_events"] = []
    for row in _query("SELECT * FROM learning_path_events WHERE user_id = ? ORDER BY created_at", (user_id,)):
        entry = dict(row)
        entry["evidence"] = _decode_json_field(entry.get("evidence"))
        bundle["learning_path_events"].append(entry)

    bundle["answer_feedback"] = []
    for row in _query("SELECT * FROM answer_feedback WHERE user_id = ? ORDER BY created_at", (user_id,)):
        entry = dict(row)
        entry["tags"] = _decode_json_field(entry.get("tags"))
        bundle["answer_feedback"].append(entry)

    bundle["user_consent"] = _rows("SELECT * FROM user_consent WHERE user_id = ?", (user_id,))

    bundle["learning_events"] = []
    for row in _query("SELECT * FROM learning_events WHERE user_id = ?", (user_id,)):
        item = dict(row)
        item["details"] = _decode_json_field(item.get("details"))
        bundle["learning_events"].append(item)

    bundle["journey_log"] = []
    for row in _query("SELECT * FROM journey_log WHERE user_id = ? ORDER BY id", (user_id,)):
        item = dict(row)
        item["payload"] = _decode_json_field(item.get("payload"))
        bundle["journey_log"].append(item)

    bundle["chat_ops_log"] = []
    for row in _query("SELECT * FROM chat_ops_log WHERE user_id = ? ORDER BY id", (user_id,)):
        item = dict(row)
        item["response_json"] = _decode_json_field(item.get("response_json"))
        for key in ("applied_ops", "pending_ops"):
            parsed = _decode_json_field(item.get(key))
            if parsed is None:
                parsed = []
            item[key] = parsed
        bundle["chat_ops_log"].append(item)

    bundle["assessment_followups"] = []
    for row in _query("SELECT * FROM assessment_followups WHERE user_id = ? ORDER BY topic", (user_id,)):
        item = dict(row)
        item["microcheck_rubric"] = _decode_json_field(item.get("microcheck_rubric"))
        bundle["assessment_followups"].append(item)

    bundle["assessment_results"] = []
    for row in _query("SELECT * FROM assessment_results WHERE user_id = ? ORDER BY id", (user_id,)):
        item = dict(row)
        raw_rubric = item.get("rubric_criteria")
        if isinstance(raw_rubric, (bytes, bytearray)):
            raw_rubric = raw_rubric.decode("utf-8", errors="ignore")
        if isinstance(raw_rubric, str):
            try:
                item["rubric_criteria"] = json.loads(raw_rubric)
            except Exception:
                pass
        bundle["assessment_results"].append(item)

    bundle["assessment_step_results"] = []
    for row in _query(
        """
        SELECT sr.*
        FROM assessment_step_results AS sr
        JOIN assessment_results AS ar ON ar.id = sr.assessment_id
        WHERE ar.user_id = ?
        ORDER BY sr.assessment_id, sr.id
        """,
        (user_id,),
    ):
        bundle["assessment_step_results"].append(dict(row))

    bundle["assessment_error_patterns"] = []
    for row in _query(
        """
        SELECT ep.*
        FROM assessment_error_patterns AS ep
        JOIN assessment_results AS ar ON ar.id = ep.assessment_id
        WHERE ar.user_id = ?
        ORDER BY ep.assessment_id, ep.id
        """,
        (user_id,),
    ):
        bundle["assessment_error_patterns"].append(dict(row))

    bundle["pending_ops"] = []
    for row in _query("SELECT * FROM pending_ops WHERE user_id = ? ORDER BY id", (user_id,)):
        item = dict(row)
        item["payload"] = _decode_json_field(item.get("payload"))
        bundle["pending_ops"].append(item)

    bundle["xapi_statements"] = []
    for row in _query("SELECT * FROM xapi_statements WHERE user_id = ? ORDER BY id", (user_id,)):
        item = dict(row)
        ctx = item.get("context")
        if isinstance(ctx, (bytes, bytearray)):
            ctx = ctx.decode("utf-8", errors="ignore")
        if isinstance(ctx, str):
            try:
                item["context"] = json.loads(ctx)
            except Exception:
                pass
        bundle["xapi_statements"].append(item)

    bundle["llm_metrics"] = _rows("SELECT * FROM llm_metrics WHERE user_id = ? ORDER BY id", (user_id,))

    return bundle


def delete_user_data(user_id: str) -> Dict[str, int]:
    user_tables = [
        "chat_ops_log",
        "assessment_followups",
        "journey_log",
        "learning_events",
        "quiz_attempts",
        "assessment_results",
        "xapi_statements",
        "pending_ops",
        "learning_path_events",
        "learning_path_state",
        "answer_feedback",
        "bloom_progress",
        "bloom_progress_history",
        "llm_metrics",
        "user_progress",
        "bloom_score",
        "mastery",
        "teacher_analytics",
        "learner_profile",
        "learner_priors",
        "learner_confidence",
        "learner_misconceptions",
    ]
    learner_tables = [
        "eval_pretest_attempts",
        "eval_posttest_attempts",
    ]
    counts: Dict[str, int] = {}
    with _conn() as con:
        for table in user_tables:
            cur = con.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
            counts[table] = cur.rowcount if cur is not None else 0
        for table in learner_tables:
            cur = con.execute(f"DELETE FROM {table} WHERE learner_id = ?", (user_id,))
            counts[table] = cur.rowcount if cur is not None else 0
        cur = con.execute("DELETE FROM user_consent WHERE user_id = ?", (user_id,))
        counts["user_consent"] = cur.rowcount if cur is not None else 0
        cur = con.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        counts["users"] = cur.rowcount if cur is not None else 0
        con.commit()
    return counts

