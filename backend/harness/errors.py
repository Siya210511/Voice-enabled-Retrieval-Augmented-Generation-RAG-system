"""
Stage-specific exceptions + a retry decorator with exponential backoff.

Each stage in the harness catches its own exception type, retries a bounded
number of times with backoff, and converts a final failure into a structured
*Result object (see schemas.py) rather than letting a raw exception blow up
the whole request -- so one flaky STT call doesn't take down retrieval for
a request that could have used a cached/text query instead.
"""

import time
import functools
from typing import Callable, TypeVar

T = TypeVar("T")


class TranscriptionError(Exception):
    pass


class RetrievalError(Exception):
    pass


class GenerationError(Exception):
    pass


class GuardrailBlockedError(Exception):
    """Raised internally to short-circuit the pipeline when a guardrail fails.
    Caught by the harness and turned into a blocked PipelineResponse, not
    surfaced as a raw error to the caller."""
    pass


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay_sec: float = 0.2,
    exceptions: tuple = (Exception,),
):
    """Decorator: retries the wrapped function on the given exception types,
    with exponential backoff (base_delay * 2^attempt). Re-raises the last
    exception if all attempts are exhausted, tagged with attempt count via
    a `.attempts` attribute so callers can record it in the *Result object.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    result = fn(*args, **kwargs)
                    return result, attempt
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts:
                        time.sleep(base_delay_sec * (2 ** (attempt - 1)))
            raise last_exc
        return wrapper
    return decorator
