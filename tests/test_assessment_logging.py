from schemas import (
    AssessmentErrorPattern,
    AssessmentResult,
    AssessmentStepEvaluation,
    RubricCriterion,
)


def test_save_assessment_result_logs_steps_and_patterns(temp_db):
    import db

    db.upsert_subject("math_algebra", "Mathematik Algebra", "math", "Lineare Modelle")

    result = AssessmentResult(
        user_id="learner",
        domain="math",
        item_id="algebra.linear_equations",
        bloom_level="K3",
        response="Schritt 1: ...",
        score=0.4,
        rubric_criteria=[RubricCriterion(id="accuracy", score=0.4)],
        model_version="gpt-5-codex",
        prompt_version="v1",
        confidence=0.2,
        step_evaluations=[
            AssessmentStepEvaluation(
                step_id="translate",
                subskill="algebra.linear_equations.translation",
                outcome="incorrect",
                score_delta=0.0,
                hint="Welche Größe suchst du? Formuliere sie als Variable.",
                feedback="Übersetzung der Textaufgabe ist unvollständig",
                diagnosis="conceptual",
            ),
            AssessmentStepEvaluation(
                step_id="balance",
                subskill="algebra.linear_equations.balance",
                outcome="hint",
                score_delta=0.0,
                hint="Denke daran, auf beiden Seiten dieselbe Operation auszuführen.",
                diagnosis="procedural",
            ),
        ],
        error_patterns=[
            AssessmentErrorPattern(
                code="sign-flip",
                description="Vorzeichenwechsel beim Übertragen fehlt",
                subskill="algebra.linear_equations.balance",
                occurrences=2,
            )
        ],
    )

    assessment_id = db.save_assessment_result(result)
    assert assessment_id > 0

    steps = db.list_assessment_step_results(assessment_id)
    assert len(steps) == 2
    assert steps[0]["step_id"] == "translate"
    assert steps[1]["hint"].startswith("Denke daran")

    patterns = db.list_assessment_error_patterns(assessment_id)
    assert len(patterns) == 1
    assert patterns[0]["pattern_code"] == "sign-flip"
    assert patterns[0]["occurrences"] == 2

    diagnostics = db.list_recent_step_diagnostics("learner", "math", limit=5)
    assert diagnostics
    assert diagnostics[0]["subskill"] in {"algebra.linear_equations.translation", "algebra.linear_equations.balance"}
    assert any(entry["diagnosis"] == "conceptual" for entry in diagnostics)
