"""PHI masking utility. No real patient identifiers may pass into the optimization engine."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from typing import Any

# Generated once per process; rotate by restarting or injecting via env var.
_HMAC_KEY: bytes = secrets.token_bytes(32)

# HIPAA Safe Harbor identifiers (45 CFR 164.514(b)(2))
_PHI_PATTERNS: list[tuple[str, str]] = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "<SSN>"),                          # SSN
    (r"\b\d{10,}\b", "<MRN>"),                                     # MRN / long numeric IDs
    (r"\b[A-Z]{2}\d{6,}\b", "<MRN>"),                             # alphanumeric MRN patterns
    (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "<DOB>"),                    # dates (M/D/YY or M/D/YYYY)
    (r"\b\d{4}-\d{2}-\d{2}\b", "<DOB>"),                          # ISO dates
    (r"\b\d{5}(?:-\d{4})?\b", "<ZIP>"),                           # ZIP codes
    (r"(?i)\b[A-Z][a-z]+ [A-Z][a-z]+\b", "<NAME>"),              # FirstName LastName pattern
    (r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", "<EMAIL>"),  # email
    (r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "<PHONE>"),  # US phone
    (r"\b(?:Dr\.?|Mr\.?|Ms\.?|Mrs\.?)\s+[A-Z][a-z]+\b", "<NAME>"),  # titled names
]

_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat), repl) for pat, repl in _PHI_PATTERNS
]


def mask_string(value: str) -> str:
    """Replace PHI tokens in a free-text string with placeholder labels."""
    for pattern, replacement in _COMPILED:
        value = pattern.sub(replacement, value)
    return value


def pseudonymize(identifier: str) -> str:
    """Return a deterministic, keyed HMAC-SHA256 pseudonym for an identifier.

    Stable within a process lifetime; cannot be reversed without _HMAC_KEY.
    """
    digest = hmac.new(_HMAC_KEY, identifier.encode(), hashlib.sha256).hexdigest()
    return f"PID-{digest[:12].upper()}"


def sanitize_dict(record: dict[str, Any], phi_fields: set[str]) -> dict[str, Any]:
    """Return a copy of record with phi_fields pseudonymized and string values masked."""
    sanitized: dict[str, Any] = {}
    for key, value in record.items():
        if key in phi_fields:
            sanitized[key] = pseudonymize(str(value))
        elif isinstance(value, str):
            sanitized[key] = mask_string(value)
        else:
            sanitized[key] = value
    return sanitized


# Fields treated as direct identifiers in the optimization pipeline.
PATIENT_PHI_FIELDS: frozenset[str] = frozenset({
    "patient_id",
    "name",
    "mrn",
    "dob",
    "ssn",
    "address",
    "phone",
    "email",
    "next_of_kin",
    "insurance_id",
})
