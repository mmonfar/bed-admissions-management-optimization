# SDD: NSGA-II Bed Admissions Manager

Master spec-driven development contract. Full domain/architecture narrative lives in
[PROJECT.md](../PROJECT.md) and [README.md](../README.md) — this file is the audit
surface: requirement, acceptance criteria, and status.

## Purpose

Suggest which ward bed each patient should take, so that patients sit close to their
specialty ward, primary-ward beds are used, and few patients are moved. A planner
reads the suggestion; nothing moves a patient by itself.

## Scope

In scope: the NSGA-II engine (`src/engine/`), pseudonymisation and export helpers
(`src/utils/`), the Streamlit dashboard (`src/ui/`), synthetic demo data.
Out of scope: real patient data, clinical decisions, hospital system links.
Synthetic data only. Not a medical device. Not for clinical decision-making.

## Contract / interfaces

- `src.engine.models`: `Patient`, `Bed`, `Wing`, `DisplacementStatus`.
- `src.engine.problem.WardAssignmentProblem(patients, beds, anchor_map, penalties, cost_per_cmi, transfer_cap)`: 3 objectives (f1 clustering, f2 vacancy, f3 stability), 1 constraint (moves <= cap). Missing anchor or penalty key raises `KeyError`.
- `src.engine.solver`: `SolverConfig(pop_size=100, n_gen=100, seed=42)`, `run_optimization(problem, cfg)` returns a pymoo `Result`.
- `src.engine.directives`: `find_knee_point(F)`, `DirectiveGenerator(...).generate(result, solution_idx=None, move_cmi_threshold=0.5)` returns an `ActionPlan`; raises `ValueError` when there is no feasible solution.
- `src.utils.sanitizer`: `pseudonymize`, `mask_string`, `sanitize_dict`.
- `src.utils.exporters`: `to_json`, `to_markdown`.
- `src.utils.mock_data.make_mock_scenario()`: 20 patients, 30 beds.
- Run: `streamlit run src/ui/dashboard.py`.

## Acceptance criteria

1. Every Pareto solution assigns each patient a distinct bed and meets the transfer cap.
2. Objectives match the hand-computed values in `docs/specs/objectives-and-constraint.md`.
3. The knee point is the solution nearest the ideal point after min-max scaling.
4. A higher MOVE threshold never gives more MOVE directives.
5. Missing input (unknown specialty, missing penalty key, no feasible solution) raises an error and never returns a silent default.
6. Identifiers pass through `pseudonymize`; exports contain no raw identifier.
7. Known gap: the same seed does not give the same front (see Open Items). The test for it is marked `xfail`.

## Tests

Command: `.venv/Scripts/python.exe -m pytest tests` (run from the repo root).
Files: `tests/test_engine.py` (engine, knee point, directives), `tests/test_utils.py` (privacy, validation, export).
These are characterisation tests: they record what the code does now. No behaviour was changed to write them.
Last run (7 Nov 2026): 26 passed, 1 xfailed, about 7 s.

## Specs

See `docs/specs/`.

---

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

1. Seed is not reproducible: `NoDuplicateBedRepair` uses the global `np.random`, so `SolverConfig.seed` does not fix the front. Needs a product decision before any fix (CEO rule: no behaviour change in the SDD step).
2. With the demo cap of 5 transfers and a small run (pop 40, 15 generations), no feasible plan is found and `generate` raises. The dashboard default budget is larger; not re-checked here.
3. Brand findings in `src/ui/dashboard.py` and the missing wordmark wait for a design brief.

Other items: none. When a requirement here cannot be resolved without a product
decision (e.g. new objective definition, new constraint type, encoding change), add a
row under this section instead of guessing, and escalate to the user.

---

## Audit Log

| Date | Auditor | Result |
|---|---|---|
| 2026-08-16 | orchestrator scaffold | Initial contract derived from PROJECT.md + README.md; all core modules DONE, no drift detected |
| 2026-08-16 | janitor | Cleanup pass: moved 12 stray `__pycache__` .pyc files (src/, src/engine/, src/ui/, src/utils/) and stale `remotion-video/out/ClinicalFlow.mp4` (~12.9 MB, superseded) into /bin, paths preserved; nothing deleted, no tracked/in-progress files touched |
