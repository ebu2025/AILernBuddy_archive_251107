"""Retry and error handling utilities for process model execution."""

import logging
import time
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar, cast

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ProcessExecutionError(Exception):
    """Base class for process execution errors."""
    pass

class ProcessTimeoutError(ProcessExecutionError):
    """Raised when a process execution times out."""
    pass

class ProcessRetryExhaustedError(ProcessExecutionError):
    """Raised when retry attempts are exhausted."""
    pass

def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying process model execution with exponential backoff."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"Process execution failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    
                    if attempt < max_retries - 1:
                        sleep_time = min(delay, max_delay)
                        time.sleep(sleep_time)
                        delay *= backoff_factor
                    
            raise ProcessRetryExhaustedError(
                f"Process execution failed after {max_retries} attempts"
            ) from last_exception
            
        return wrapper
    return decorator

def with_timeout(timeout: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for adding timeout to process execution."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import signal
            
            def handler(signum: int, frame: Any) -> None:
                raise ProcessTimeoutError(f"Process execution timed out after {timeout} seconds")
            
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(timeout))
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Restore previous handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
            return result
            
        return wrapper
    return decorator