"""Plan callout text. Only the recommended plan carries the word 'Recommended'."""

from __future__ import annotations


def callout_words(selected_idx: int, knee_idx: int) -> tuple[str, str]:
    """Return (label, verb) for the plan callout box."""
    if selected_idx == knee_idx:
        return f"Recommended Plan (Solution #{selected_idx}):", "the system recommends"
    return f"Plan you are inspecting (Solution #{selected_idx}):", "this plan makes"
