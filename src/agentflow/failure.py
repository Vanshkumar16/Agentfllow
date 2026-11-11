"""Structured failure classification and recovery guidance."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum


class FailureCategory(str, Enum):
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    TEST_ASSERTION = "test_assertion_failure"
    RUNTIME_EXCEPTION = "runtime_exception"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "empty_or_malformed_response"
    SCHEMA_VALIDATION = "schema_validation_failure"
    TRUNCATED_RESPONSE = "truncated_response"
    INCOMPLETE_RESPONSE = "incomplete_response"
    REPEATED_IDENTICAL = "repeated_identical_failure"
    UNKNOWN = "unknown"


class RecoveryDecision(str, Enum):
    RETRY = "RETRY"
    REVISE = "REVISE"
    REPLAN = "REPLAN"
    STOP = "STOP"


_RECOVERY_ACTIONS = {
    FailureCategory.SYNTAX_ERROR: (
        "Revise the patch to fix the syntax error, then inspect the edited file "
        "before finishing."
    ),
    FailureCategory.IMPORT_ERROR: (
        "Inspect the failing import, project dependencies, module names, and "
        "package paths before changing behavior."
    ),
    FailureCategory.TEST_ASSERTION: (
        "Re-evaluate the expected behavior from the assertion and task; make a "
        "minimal correction that satisfies the stated contract."
    ),
    FailureCategory.RUNTIME_EXCEPTION: (
        "Inspect the traceback from the first relevant frame and correct the "
        "underlying runtime failure."
    ),
    FailureCategory.TIMEOUT: (
        "Stop the timed-out operation; only replan with a smaller bounded action "
        "if the remaining time and budgets permit."
    ),
    FailureCategory.MALFORMED_RESPONSE: (
        "Use controlled regeneration: return exactly the required JSON schema "
        "in one response, without prose or markdown outside the JSON object."
    ),
    FailureCategory.SCHEMA_VALIDATION: (
        "Regenerate the response using only the required fields and the exact "
        "tool-specific argument schema."
    ),
    FailureCategory.TRUNCATED_RESPONSE: (
        "Treat the response as provider-confirmed truncation; regenerate once "
        "within the remaining call, token, and time limits."
    ),
    FailureCategory.INCOMPLETE_RESPONSE: (
        "Use controlled regeneration and return exactly the required JSON "
        "schema. The response is invalid or incomplete, but provider metadata "
        "cannot confirm truncation."
    ),
    FailureCategory.REPEATED_IDENTICAL: (
        "Do not repeat the same repair. Stop and report the unchanged failure "
        "instead of issuing another identical retry."
    ),
    FailureCategory.UNKNOWN: (
        "Inspect the available failure evidence and choose a different, bounded "
        "diagnostic action before attempting another change."
    ),
}


@dataclass(frozen=True)
class FailureAssessment:
    category: FailureCategory
    recovery_action: str
    fingerprint: str
    evidence: str
    suggested_decision: RecoveryDecision


def classify_failure(evidence: str, *, timed_out: bool = False) -> FailureAssessment:
    normalized_evidence = evidence.strip()[-8_000:]
    lowered = normalized_evidence.lower()

    if timed_out or _contains_timeout(lowered):
        category = FailureCategory.TIMEOUT
    elif _contains_truncated_response(lowered):
        category = FailureCategory.TRUNCATED_RESPONSE
    elif _contains_incomplete_response(lowered):
        category = FailureCategory.INCOMPLETE_RESPONSE
    elif _contains_schema_failure(lowered):
        category = FailureCategory.SCHEMA_VALIDATION
    elif _contains_syntax_error(lowered):
        category = FailureCategory.SYNTAX_ERROR
    elif _contains_import_error(lowered):
        category = FailureCategory.IMPORT_ERROR
    elif _contains_malformed_response(lowered):
        category = FailureCategory.MALFORMED_RESPONSE
    elif _contains_assertion_failure(lowered):
        category = FailureCategory.TEST_ASSERTION
    elif _contains_runtime_exception(lowered):
        category = FailureCategory.RUNTIME_EXCEPTION
    else:
        category = FailureCategory.UNKNOWN

    return FailureAssessment(
        category=category,
        recovery_action=_RECOVERY_ACTIONS[category],
        fingerprint=_fingerprint(category, normalized_evidence),
        evidence=normalized_evidence,
        suggested_decision=_DECISIONS[category],
    )


def repeated_failure(assessment: FailureAssessment) -> FailureAssessment:
    return FailureAssessment(
        category=FailureCategory.REPEATED_IDENTICAL,
        recovery_action=_RECOVERY_ACTIONS[FailureCategory.REPEATED_IDENTICAL],
        fingerprint=assessment.fingerprint,
        evidence=assessment.evidence,
        suggested_decision=RecoveryDecision.STOP,
    )


def _contains_timeout(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "timed out",
            "timeout",
            "time budget exhausted",
            "timed_out=true",
        )
    )


def _contains_syntax_error(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "syntaxerror",
            "indentationerror",
            "taberror",
            "invalid syntax",
        )
    )


def _contains_import_error(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "modulenotfounderror",
            "no module named",
            "importerror",
            "cannot import name",
        )
    )


def _contains_assertion_failure(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "assertionerror",
            "assert ",
            "failed:",
            "failed ",
        )
    )


def _contains_runtime_exception(text: str) -> bool:
    return "traceback (most recent call last)" in text or any(
        marker in text
        for marker in (
            "typeerror:",
            "valueerror:",
            "keyerror:",
            "indexerror:",
            "attributeerror:",
            "zerodivisionerror:",
            "runtimeerror:",
        )
    )


def _contains_malformed_response(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "empty or malformed response",
            "empty model response",
            "malformed response:",
            "response is not valid json",
            "response must contain an 'action' object",
            "action requires a non-empty tool name",
            "action arguments must be a json object",
            "single-shot response is not valid json",
            "single-shot response requires",
        )
    )


def _contains_truncated_response(text: str) -> bool:
    return (
        "provider-confirmed truncated response" in text
        or "truncated_response=true" in text
    )


def _contains_incomplete_response(text: str) -> bool:
    return (
        "provider termination status is unknown" in text
        or "incomplete_response=true" in text
    )


def _contains_schema_failure(text: str) -> bool:
    return (
        "response schema validation failed" in text
        or "schema validation failed" in text
    )


_DECISIONS = {
    FailureCategory.SYNTAX_ERROR: RecoveryDecision.REVISE,
    FailureCategory.IMPORT_ERROR: RecoveryDecision.REVISE,
    FailureCategory.TEST_ASSERTION: RecoveryDecision.REVISE,
    FailureCategory.RUNTIME_EXCEPTION: RecoveryDecision.REVISE,
    FailureCategory.TIMEOUT: RecoveryDecision.REPLAN,
    FailureCategory.MALFORMED_RESPONSE: RecoveryDecision.RETRY,
    FailureCategory.SCHEMA_VALIDATION: RecoveryDecision.RETRY,
    FailureCategory.TRUNCATED_RESPONSE: RecoveryDecision.RETRY,
    FailureCategory.INCOMPLETE_RESPONSE: RecoveryDecision.RETRY,
    FailureCategory.REPEATED_IDENTICAL: RecoveryDecision.STOP,
    FailureCategory.UNKNOWN: RecoveryDecision.REPLAN,
}


def _fingerprint(category: FailureCategory, evidence: str) -> str:
    normalized = re.sub(
        r"(?m)^=+.*?\s+in\s+\d+(?:\.\d+)?s\s*=+\s*$",
        "",
        evidence,
    )
    normalized = re.sub(r"\b\d+(?:\.\d+)?s\b", "<duration>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    digest = hashlib.sha256(f"{category.value}:{normalized}".encode("utf-8")).hexdigest()
    return digest
