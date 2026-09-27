# Bed Admissions Manager Optimization

NSGA-II multi-objective optimizer for hospital bed assignments. Returns a Pareto front of non-dominated solutions across three competing clinical objectives; the charge nurse selects the operating point that matches shift priorities.

---

## Problem

Assigning patients to beds requires balancing goals that conflict:

- Placing every patient near their clinical team (f1) pressures scarce anchor-ward beds.
- Filling primary beds before overflow (f2) may force cross-floor placements.
- Minimizing patient transfers (f3) constrains how much the plan can improve the above.

A weighted-sum formulation requires weights chosen in advance and produces one answer. NSGA-II surfaces the full trade-off space in a single run. The recommended solution is the **knee point** -- minimum L2 distance to the utopian point after min-max normalization, i.e. compromise programming with equal implicit weights.

---

## Objectives

All three are minimized (lower = better).

| ID | Dashboard Label | Definition |
|----|-----------------|------------|
| f1 | Clinical Home Alignment | Sum of radius penalties between each patient's assigned bed and their specialty anchor ward. Penalty ladder: same ward = 1 pt, different wing = 5 pt, different floor = 10 pt |
| f2 | Capacity Throughput | Count of primary (non-overflow) beds left unoccupied. Minimizing this maximizes ED intake capacity |
| f3 | Nursing Workload Protection | CMI-weighted cost of patient transfers. Each move is scaled by Case Mix Index to reflect clinical complexity |

Hard constraint: total moves <= transfer cap (default 5; adjustable at runtime).

---

## Repository Layout

```
config/
  settings.yaml          Penalty weights, displacement cost, transfer cap default
  hospital_map.yaml      Ward topology (floor, wing, specialty anchor)

src/
  engine/
    models.py            Patient, Bed, Wing, DisplacementStatus dataclasses
    problem.py           WardAssignmentProblem (pymoo ElementwiseProblem)
    solver.py            NoDuplicateBedRepair, SolverConfig, run_optimization
    directives.py        DirectiveGenerator, ActionPlan, find_knee_point
  utils/
    mock_data.py         Synthetic 20-patient / 30-bed scenario
    exporters.py         ActionPlan -> JSON / Markdown
    sanitizer.py         HMAC pseudonymization + HIPAA Safe Harbor masking
  ui/
    dashboard.py         Streamlit command center

brand_style.css          mm.farina brand identity (Anthracite / Teal / Inter)
remotion-video/          45-50s NSGA-II explainer video (React + Remotion + Three.js)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.11+ required.

**Run the dashboard:**

```bash
streamlit run src/ui/dashboard.py
```

**Remotion video (optional):**

```bash
cd remotion-video
npm install
npm start          # Remotion Studio with hot reload
npm run build      # Render to out/ClinicalFlow.mp4
```

---

## Configuration

Edit `config/settings.yaml` to tune penalties and solver defaults without touching code.

```yaml
optimization:
  penalties:
    same_floor: 1      # home ward penalty (f1 minimum unit)
    diff_floor: 5      # same floor, different wing
    diff_wing: 10      # different floor
  displacement:
    cost_per_cmi_point: 2.5   # f3 weight per CMI unit
    max_transfers_default: 5
```

Edit `config/hospital_map.yaml` to change ward topology:

```yaml
wards:
  - {id: "4A", specialty: "Cardiology", floor: 4, wing: "North"}
  ...
```

Runtime overrides are available from the dashboard sidebar (aggression multiplier, transfer cap slider, ED Surge Mode toggle).

---

## Architecture Notes

**Encoding:** Integer vector `x` of length `n_patients`; `x[i]` is the index into the `beds` list for patient `i`. Allows overflow reuse and unassigned slots, which a permutation encoding cannot.

**Repair operator:** `NoDuplicateBedRepair` enforces the no-duplicate-bed hard constraint after each crossover/mutation step using vectorized `np.unique` and `np.setdiff1d`.

**Caching:** `@st.cache_data` on `_run_cached`. Non-hashable arguments (`list[Patient]`, `list[Bed]`) are prefixed `_` to opt out of hashing; primitive parameters form the effective cache key.

**Pre-assignment mode:** When confirmed discharges are known, the longest-LOS assigned patients are removed from the optimization. Their beds are treated as available, enabling pre-assignment of incoming admissions.

**Directive categories:**
- `MOVE` -- transfer improves alignment by more than `move_cmi_threshold * cmi`
- `HOLD` -- vacant anchor-ward bed; specialty demand is displaced elsewhere; reserve it
- `ALERT` -- no anchor-ward bed available; displacement is structural, not resolvable by reshuffling

---

## Solver Scaling

| Patients | pop_size | n_gen | Approx. runtime |
|----------|----------|-------|-----------------|
| 20       | 100      | 100   | < 5 s           |
| 50       | 150      | 150   | ~20 s           |
| 100      | 200      | 200   | ~90 s           |
| 200+     | 300      | 300   | cache or async  |

---

## Brand

Anthracite `#36454F` / Teal `#008080` / Inter (body) / JetBrains Mono (code). Defined in `brand_style.css` and mirrored in `src/ui/dashboard.py` and `remotion-video/src/constants.ts`.

## Licence

Code is licensed under **AGPL-3.0-or-later** (see [`LICENSE`](LICENSE)); a commercial licence is available on request from the author via [LinkedIn](https://www.linkedin.com/in/martin-monteagudo-farina/). Non-code content is under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Details in [`LICENSING.md`](LICENSING.md).

## Disclaimer

Research and demonstration software. Not a medical device and not intended for clinical decision-making, diagnosis or treatment. Provided "as is", without warranty of any kind; the author accepts no liability for any use. Uses synthetic data only.
