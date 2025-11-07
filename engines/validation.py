"""Validation utilities for assessment results and related data structures."""

from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime

class ValidationError(Exception):
    """Base class for validation errors."""
    pass

class AssessmentValidationError(ValidationError):
    """Raised when assessment data fails validation."""
    pass

def validate_assessment_result(data: Dict[str, Any]) -> None:
    """Validate an assessment result object.
    
    Raises AssessmentValidationError if validation fails.
    """
    required_fields = {
        "answer": str,
        "bloom_level": str,
        "assessment": dict,
        "diagnosis": str,
        "confidence": float,
        "action": str,
    }
    
    # Check required fields
    for field, expected_type in required_fields.items():
        if field not in data:
            raise AssessmentValidationError(f"Missing required field: {field}")
        if not isinstance(data[field], expected_type):
            raise AssessmentValidationError(
                f"Field {field} has wrong type. Expected {expected_type.__name__}, "
                f"got {type(data[field]).__name__}"
            )
    
    # Validate assessment substructure
    assessment = data["assessment"]
    required_assessment_fields = {
        "item_id": str,
        "bloom_level": str,
        "score": float,
        "confidence": float,
        "response": str,
        "source": str,
    }
    
    for field, expected_type in required_assessment_fields.items():
        if field not in assessment:
            raise AssessmentValidationError(f"Missing required assessment field: {field}")
        if not isinstance(assessment[field], expected_type):
            raise AssessmentValidationError(
                f"Assessment field {field} has wrong type. Expected {expected_type.__name__}, "
                f"got {type(assessment[field]).__name__}"
            )
    
    # Validate value ranges
    if not (0 <= assessment["score"] <= 1):
        raise AssessmentValidationError("Assessment score must be between 0 and 1")
    if not (0 <= assessment["confidence"] <= 1):
        raise AssessmentValidationError("Assessment confidence must be between 0 and 1")
    if not (0 <= data["confidence"] <= 1):
        raise AssessmentValidationError("Confidence must be between 0 and 1")
    
    # Validate bloom level consistency
    if assessment["bloom_level"] != data["bloom_level"]:
        raise AssessmentValidationError(
            "Bloom level mismatch between assessment and top-level"
        )
    
    # Validate diagnosis values
    valid_diagnoses = {"conceptual", "procedural", "careless", "none"}
    if data["diagnosis"] not in valid_diagnoses:
        raise AssessmentValidationError(
            f"Invalid diagnosis value. Must be one of: {', '.join(valid_diagnoses)}"
        )
    
    # Validate action values
    valid_actions = {
        "progression:evaluate",
        "progression:advance",
        "progression:regress",
        "none"
    }
    if data["action"] not in valid_actions:
        raise AssessmentValidationError(
            f"Invalid action value. Must be one of: {', '.join(valid_actions)}"
        )

def validate_microcheck_data(data: Dict[str, Any]) -> None:
    """Validate microcheck-related fields in assessment data."""
    # All microcheck fields should be present or absent together
    microcheck_fields = {
        "microcheck_question",
        "microcheck_expected",
        "microcheck_given",
        "microcheck_score"
    }
    
    present_fields = {field for field in microcheck_fields if data.get(field) is not None}
    
    if present_fields and present_fields != microcheck_fields:
        raise AssessmentValidationError(
            "All microcheck fields must be present if any are provided"
        )
    
    if present_fields:
        if not isinstance(data["microcheck_score"], (int, float)):
            raise AssessmentValidationError("Microcheck score must be numeric")
        if not (0 <= data["microcheck_score"] <= 1):
            raise AssessmentValidationError("Microcheck score must be between 0 and 1")

def validate_assessment_step(step: Dict[str, Any]) -> None:
    """Validate an individual assessment step."""
    required_fields = {
        "step_id": str,
        "prompt": str,
        "response": str,
        "score": float,
        "bloom_level": str,
    }
    
    for field, expected_type in required_fields.items():
        if field not in step:
            raise AssessmentValidationError(f"Missing required step field: {field}")
        if not isinstance(step[field], expected_type):
            raise AssessmentValidationError(
                f"Step field {field} has wrong type. Expected {expected_type.__name__}, "
                f"got {type(step[field]).__name__}"
            )
    
    if not (0 <= step["score"] <= 1):
        raise AssessmentValidationError("Step score must be between 0 and 1")