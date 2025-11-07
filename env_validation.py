"""Environment variable validation and management."""

import os
import logging
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

class EnvironmentError(Exception):
    """Raised when required environment variables are missing or invalid."""
    pass

def validate_environment() -> None:
    """Validate critical environment variables.
    
    Raises EnvironmentError if validation fails.
    """
    # No environment variables are strictly required because the application
    # provides sensible fallbacks where needed. We keep the structure in case
    # future configuration items truly require explicit values.
    required_vars: Dict[str, str] = {}

    defaults = {
        "DB_PATH": os.getenv("DB_PATH") or "data.db",
    }

    # Apply defaults before validation so dependent modules see consistent values.
    for var, value in defaults.items():
        if not os.getenv(var):
            os.environ[var] = value
            logger.info("Environment variable %s not set; using default '%s'", var, value)
    
    optional_vars = {
        "RAG_CORPUS_PATH": "Path to RAG document corpus",
        "LRS_URL": "Learning Record Store URL",
        "LRS_AUTH": "Learning Record Store authentication",
        "PROMPT_VARIANT": "Active prompt variant name",
    }
    
    # Check required variables
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )
    
    # Validate URLs
    url_vars = {"GPT4ALL_URL", "LRS_URL"}
    for var in url_vars:
        value = os.getenv(var)
        if value and not (value.startswith("http://") or value.startswith("https://")):
            raise EnvironmentError(f"Invalid URL format for {var}: {value}")
    
    # Log optional variables status
    for var, description in optional_vars.items():
        if not os.getenv(var):
            logger.warning(f"Optional environment variable not set: {var} ({description})")

def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on", "enabled"}
