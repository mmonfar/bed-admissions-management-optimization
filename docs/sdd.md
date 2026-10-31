# SDD: NSGA-II Bed Admissions Manager

Master spec-driven development contract. Full domain/architecture narrative lives in
[PROJECT.md](../PROJECT.md) and [README.md](../README.md) — this file is the audit
surface: requirement, acceptance criteria, and status.

Statuses: `DONE` | `PARTIAL` | `MISSING` | `DRIFT` (implemented but disagrees with spec).

---

## 1. Optimization Engine (`src/engine/`)

| Req | Acceptance Criteria | Status | Notes |
|---|---|---|---|
| Integer-vector encoding | `x[i]` indexes into `beds` list; `xl=0`, `xu=n_beds-1` | DONE | `problem.py` |
| No-duplicate-bed repair | `NoDuplicateBedRepair` runs post crossover/mutation, vectorized | DONE | `solver.py` |
| f1 Clustering penalty | Ladder 1/5/10 (ward/wing/floor), configurable via `settings.yaml` | DONE | `problem.py` |
| f2 Vacancy gap | Non-overflow beds unoccupied count | DONE | `problem.py` |
| f3 Stability cost | CMI-weighted move cost | DONE | `problem.py` |
| Transfer-cap constraint | `g0 = n_moves - transfer_cap <= 0` | DONE | `problem.py` |
| Knee-point selection | Min-L2 to utopian point after min-max norm | DONE | `directives.py::find_knee_point` |
| Directive generation | MOVE / HOLD / ALERT per rules in PROJECT.md | DONE | `directives.py` |
| Pre-assignment / discharge forecasting | Trim longest-LOS assigned patients before solve | DONE | `_trim_discharges` |

## 2. Data & Privacy (`src/utils/`)

| Req | Acceptance Criteria | Status | Notes |
|---|---|---|---|
| Pseudonymization | HMAC-based `pseudo_id`, never a real MRN | DONE | `sanitizer.py` |
| HIPAA Safe Harbor masking | Applied where patient data is exported/displayed | DONE | `sanitizer.py` |
| Mock scenario | 20-patient / 30-bed synthetic dataset for dev/demo | DONE | `mock_data.py` |
| Export formats | ActionPlan → JSON, Markdown | DONE | `exporters.py` |

## 3. Dashboard (`src/ui/dashboard.py`)

| Req | Acceptance Criteria | Status | Notes |
|---|---|---|---|
| Streamlit command center | Renders Pareto front, action table, heatmap | DONE | |
| Result caching | `@st.cache_data`; non-hashable args underscore-prefixed | DONE | |
| Manual re-run override | "Re-run Optimisation" clears cache explicitly | DONE | |
| Runtime overrides | Aggression multiplier, transfer cap slider, emergency department surge toggle | DONE | |

## 4. Configuration Surface

| Req | Acceptance Criteria | Status | Notes |
|---|---|---|---|
| `config/settings.yaml` | Penalty weights, CMI cost, default transfer cap | DONE | |
| `config/hospital_map.yaml` | Ward topology drives `AnchorMap` | DONE | |

## 5. Remotion Explainer Video (`remotion-video/`)

| Req | Acceptance Criteria | Status | Notes |
|---|---|---|---|
| 45-50s composition, 1920x1080 @ 30fps | `Root.tsx` registers correct duration/fps | DONE | see [project_remotion_video](../../.claude memory) |
| Conflict / Calculation / CTA sequence | 0-10s / 10-40s / 40-50s per PROJECT.md timing table | DONE | |
| Brand identity | Cream/anthracite/teal, Inter + JetBrains Mono | DONE | matches `brand_style.css` |

---

## Open Items / Ambiguities

None currently flagged. When a requirement here cannot be resolved without a product
decision (e.g. new objective definition, new constraint type, encoding change), add a
row under this section instead of guessing, and escalate to the user.

---

## Audit Log

| Date | Auditor | Result |
|---|---|---|
| 2026-08-16 | orchestrator scaffold | Initial contract derived from PROJECT.md + README.md; all core modules DONE, no drift detected |
| 2026-08-16 | janitor | Cleanup pass: moved 12 stray `__pycache__` .pyc files (src/, src/engine/, src/ui/, src/utils/) and stale `remotion-video/out/ClinicalFlow.mp4` (~12.9 MB, superseded) into /bin, paths preserved; nothing deleted, no tracked/in-progress files touched |
