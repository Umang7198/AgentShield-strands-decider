"""Errors from talking to the Decider.

Every one of these means "we have no usable evidence", and the pipeline treats them all the same:
it falls back to :func:`agentshield.core.fallback_decision` (fail closed). The subclasses exist so
the audit log and metrics can say *why*.
"""

from __future__ import annotations


class DeciderError(Exception):
    """Base class: the Decider could not give us usable answers."""


class DeciderUnavailable(DeciderError):
    """Could not connect, or the server answered 5xx."""


class DeciderTimeout(DeciderError):
    """The request timed out."""


class DeciderRejected(DeciderError):
    """The server refused the request (4xx), e.g. it was malformed or too long."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Decider rejected the request ({status_code}): {detail}")
        self.status_code = status_code
        self.detail = detail


class DeciderBadResponse(DeciderError):
    """The response was not valid JSON, or did not contain a usable answer for every question."""
