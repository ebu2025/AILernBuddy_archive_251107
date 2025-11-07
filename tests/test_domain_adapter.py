import pytest

import db
from engines.domain_adapter import DomainAdaptiveOrchestrator
from engines.progression import ProgressionEngine


@pytest.fixture
def orchestrator(temp_db):
    return DomainAdaptiveOrchestrator()


def test_prioritise_modalities_prefers_audio(orchestrator):
    priorities = orchestrator.prioritise_modalities("language", skill_id="zh-hsk1-lexis")
    assert priorities, "Expected modality priorities"
    assert priorities[0]["channel"] == "audio"


def test_progression_engine_per_domain(orchestrator):
    engine = orchestrator.progression_engine_for("business_process")
    assert engine.window_size == 4
    fallback = orchestrator.progression_engine_for("unknown")
    assert isinstance(fallback, ProgressionEngine)


def test_assessment_pipelines_persist_results(orchestrator):
    model_xml = """
    <definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">
      <bpmn:process id="Process_1">
        <bpmn:startEvent id="Start" />
        <bpmn:task id="Task_1" />
        <bpmn:exclusiveGateway id="Gateway_1" />
        <bpmn:endEvent id="End" />
        <bpmn:sequenceFlow id="f1" sourceRef="Start" targetRef="Task_1" />
        <bpmn:sequenceFlow id="f2" sourceRef="Task_1" targetRef="Gateway_1" />
        <bpmn:sequenceFlow id="f3" sourceRef="Gateway_1" targetRef="End" />
      </bpmn:process>
    </definitions>
    """
    bpmn_payload = {
        "model_xml": model_xml,
        "item_id": "demo-bpmn",
        "bloom_level": "K3",
        "skill_id": "modeling.basics",
    }
    result = orchestrator.evaluate_assessment("business_process", "learner-bpmn", bpmn_payload)
    assessment_id = result["assessment_id"]
    steps = db.list_assessment_step_results(assessment_id)
    assert steps and any(step["subskill"].startswith("bpmn") for step in steps)

    speech_payload = {
        "transcript": "Hallo ich komme aus Berlin",
        "item_id": "demo-language",
        "target_terms": ["hallo", "berlin"],
        "domain": "language",
        "skill_id": "de-cefr-a1-intro",
    }
    speech_result = orchestrator.evaluate_assessment("language", "learner-lang", speech_payload)
    speech_id = speech_result["assessment_id"]
    speech_steps = db.list_assessment_step_results(speech_id)
    assert any(step["subskill"].endswith("lexis") for step in speech_steps)

    math_payload = {
        "steps": [
            {"step_id": "1", "expression": "2+2", "subskill": "arith.add", "expected": 4},
            {"step_id": "2", "expression": "4*2", "subskill": "arith.multiply", "expected": 8},
        ],
        "expected_result": 8,
        "item_id": "demo-math",
        "domain": "mathematics",
        "skill_id": "arith_fluency_addition",
    }
    math_result = orchestrator.evaluate_assessment("mathematics", "learner-math", math_payload)
    math_id = math_result["assessment_id"]
    math_steps = db.list_assessment_step_results(math_id)
    assert len(math_steps) == 2
    patterns = db.list_assessment_error_patterns(math_id)
    assert isinstance(patterns, list)


def test_feedback_tone_adjustment(orchestrator):
    assert orchestrator.select_feedback_tone("language", {"confidence": 0.2, "recent_outcome": "failed"}) == "supportive"
    assert orchestrator.select_feedback_tone(
        "mathematics", {"confidence": 0.9, "recent_outcome": "success"}
    ) == "celebratory"
