"""Serialises an ActionPlan to JSON or Markdown for UI consumption."""

from __future__ import annotations

import json
from typing import Any

from src.engine.directives import Action, ActionPlan, ActionType


def to_json(plan: ActionPlan, indent: int = 2) -> str:
    """Return a JSON string representation of the ActionPlan."""
    return json.dumps(_plan_to_dict(plan), indent=indent)


def to_markdown(plan: ActionPlan) -> str:
    """Return a Markdown table of directives with a summary header."""
    lines: list[str] = [
        "## Action Plan",
        "",
        f"**Pareto front size:** {plan.pareto_size}  "
        f"| **Knee solution:** #{plan.knee_idx}  "
        f"| **f1 (clustering):** {plan.knee_f[0]:.1f}  "
        f"| **f2 (vacancy):** {plan.knee_f[1]:.1f}  "
        f"| **f3 (stability):** {plan.knee_f[2]:.2f}",
        "",
        f"**Directives:** {plan.n_moves} MOVE | {plan.n_holds} HOLD | {plan.n_alerts} ALERT",
        "",
    ]

    if not plan.actions:
        lines.append("_No directives generated._")
        return "\n".join(lines)

    # Column order: type-specific fields first, then reason.
    header = "| Type | Patient | Subspecialty | From | To | Benefit | CMI | Reason |"
    sep = "|------|---------|-------------|------|----|---------|-----|--------|"
    lines += [header, sep]

    for a in plan.actions:
        row = (
            f"| **{a.action_type.value}** "
            f"| {a.patient_pseudo_id or ''} "
            f"| {a.subspecialty} "
            f"| {a.from_bed_id or ''} "
            f"| {a.to_bed_id or ''} "
            f"| {f'{a.clustering_benefit:.2f}' if a.clustering_benefit is not None else ''} "
            f"| {f'{a.cmi:.2f}' if a.cmi is not None else ''} "
            f"| {a.reason} |"
        )
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _action_to_dict(a: Action) -> dict[str, Any]:
    return {
        "action_type": a.action_type.value,
        "subspecialty": a.subspecialty,
        "reason": a.reason,
        "patient_pseudo_id": a.patient_pseudo_id,
        "from_bed_id": a.from_bed_id,
        "to_bed_id": a.to_bed_id,
        "clustering_benefit": a.clustering_benefit,
        "cmi": a.cmi,
    }


def _plan_to_dict(plan: ActionPlan) -> dict[str, Any]:
    return {
        "summary": {
            "pareto_size": plan.pareto_size,
            "knee_idx": plan.knee_idx,
            "knee_f": {
                "f1_clustering": float(plan.knee_f[0]),
                "f2_vacancy": float(plan.knee_f[1]),
                "f3_stability": float(plan.knee_f[2]),
            },
            "n_moves": plan.n_moves,
            "n_holds": plan.n_holds,
            "n_alerts": plan.n_alerts,
        },
        "actions": [_action_to_dict(a) for a in plan.actions],
    }
