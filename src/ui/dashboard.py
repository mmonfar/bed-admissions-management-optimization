"""Clinical Bed Command Center -- Streamlit dashboard."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.engine.directives import DirectiveGenerator, find_knee_point
from src.engine.models import Bed, Patient
from src.engine.problem import AnchorMap, WardAssignmentProblem
from src.engine.solver import SolverConfig, run_optimization
from src.utils.exporters import to_json, to_markdown
from src.utils.mock_data import (
    ANCHOR_MAP,
    BASE_COST_PER_CMI,
    BASE_PENALTIES,
    make_mock_scenario,
)

# ---------------------------------------------------------------------------
# Colour palette -- brand: Anthracite #36454F / Teal #008080 / Inter
# ---------------------------------------------------------------------------
_C = {
    "bg":       "#0E1117",
    "surface":  "#1A2230",
    "border":   "#36454F",  # brand anthracite
    "muted":    "#7A8EA0",
    "text":     "#E0E4F0",
    "accent":   "#008080",  # brand teal
    "move":     "#4A9EFF",
    "hold":     "#008080",  # brand teal
    "alert":    "#FF4B4B",
    "pareto":   "#4D5375",
    "selected": "#20B2AA",  # lighter teal
    "surge":    "#FF4B4B",
    "grade_a":  "#00C851",
    "grade_b":  "#7BC67E",
    "grade_c":  "#FFB400",
    "grade_d":  "#FF8800",
    "grade_f":  "#FF4B4B",
}

# Operational term map (label, tooltip)
_OBJ = {
    "f1": (
        "Clinical Home Alignment",
        "Measures how close patients are to their specialized clinical teams. "
        "Higher alignment reduces staff travel time and improves patient safety.",
    ),
    "f2": (
        "Capacity Throughput",
        "Optimizes bed availability to minimize Emergency Department boarding "
        "and maximize intake. Lower vacancy gap = faster admissions.",
    ),
    "f3": (
        "Nursing Workload Protection",
        "Prevents unnecessary patient transfers to protect nursing bandwidth "
        "and reduce medication-reconciliation risk from mid-shift moves.",
    ),
}

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical Bed Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)

_muted = _C["muted"]
_surface = _C["surface"]
_border = _C["border"]
_text = _C["text"]
_bg = _C["bg"]

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

      /* brand: Anthracite #36454F / Teal #008080 / Inter */
      /* Apply font via inheritance only -- wildcard would clobber Material Icons ligatures */
      .stApp {{ font-family: 'Inter', sans-serif; background-color: {_bg}; }}
      code {{ font-family: "Courier New", monospace; font-size: 0.82em; }}

      .metric-box {{
        background: {_surface}; border: 1px solid {_border};
        border-radius: 6px; padding: 12px 18px; text-align: center;
        cursor: default;
      }}
      .metric-label {{
        color: {_muted}; font-size: 0.72em;
        text-transform: uppercase; letter-spacing: 0.08em;
      }}
      .metric-value {{ color: {_text}; font-size: 1.6em; font-weight: 600; }}
      .metric-note  {{ color: {_muted}; font-size: 0.70em; margin-top: 2px; }}
      .grade-badge {{
        font-size: 2.8em; font-weight: 800; line-height: 1;
        text-align: center; padding: 8px 0;
      }}
      .narrative-box {{
        background: {_surface}; border-left: 3px solid {_C["accent"]};
        border-radius: 0 6px 6px 0; padding: 12px 18px;
        color: {_text}; font-size: 0.92em; line-height: 1.6;
      }}

      /* brand-style.css classes */
      .brand-v1 {{
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        font-family: 'Inter', sans-serif;
      }}
      .brand-v1 .name {{
        color: #36454F;
        font-size: 48px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 5px;
        border-bottom: 2px solid #36454F;
        padding-bottom: 4px;
      }}
      .brand-v1 .subtitle {{
        color: #008080;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 6px;
        margin-top: 5px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_scenario() -> tuple[list[Patient], list[Bed], AnchorMap, dict]:
    patients, beds, anchor_map, settings = make_mock_scenario()
    return patients, beds, anchor_map, settings


_patients, _beds, _anchor_map, _settings = _load_scenario()

# Theoretical objective bounds -- used in the Operational Logic reference tab.
# These are scenario-specific maximums, not Pareto-front maximums.
_n_non_overflow: int  = sum(1 for b in _beds if not b.is_overflow)
_max_f1: int          = len(_patients) * BASE_PENALTIES["diff_wing"]      # all patients on wrong floor
_max_f2: int          = _n_non_overflow                                    # all primary beds idle
_max_f3: float        = sum(p.cmi for p in _patients) * BASE_COST_PER_CMI # all patients moved

# ---------------------------------------------------------------------------
# Cached NSGA-II run
# ---------------------------------------------------------------------------

def _trim_discharges(patients: list[Patient], n: int) -> list[Patient]:
    """Remove the n longest-LOS assigned patients, simulating confirmed discharges."""
    if n <= 0:
        return patients
    candidates = sorted(
        (p for p in patients if p.current_bed_id is not None),
        key=lambda p: p.length_of_stay_days,
        reverse=True,
    )
    remove_ids = {p.pseudo_id for p in candidates[:n]}
    return [p for p in patients if p.pseudo_id not in remove_ids]


@st.cache_data(show_spinner=False)
def _run_cached(
    transfer_cap: int,
    penalties_flat: tuple[tuple[str, int], ...],
    cost_per_cmi: float,
    pop_size: int,
    n_gen: int,
    seed: int,
    n_confirmed: int,
    _patients: list[Patient],
    _beds: list[Bed],
    _anchor_map: AnchorMap,
) -> tuple[np.ndarray, np.ndarray]:
    opt_patients = _trim_discharges(_patients, n_confirmed)
    problem = WardAssignmentProblem(
        patients=opt_patients,
        beds=_beds,
        anchor_map=_anchor_map,
        penalties=dict(penalties_flat),
        cost_per_cmi=cost_per_cmi,
        transfer_cap=transfer_cap,
    )
    cfg = SolverConfig(pop_size=pop_size, n_gen=n_gen, seed=seed, verbose=False)
    result = run_optimization(problem, cfg)
    if result.X is None:
        raise RuntimeError("NSGA-II returned no feasible solutions. Relax constraints.")
    return result.X.astype(np.int32), result.F.astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers -- visualisation
# ---------------------------------------------------------------------------

def _build_density_heatmap(
    x: np.ndarray,
    patients: list[Patient],
    beds: list[Bed],
) -> go.Figure:
    ward_ids: list[str] = sorted({b.ward_id for b in beds})
    specialties: list[str] = sorted({p.subspecialty for p in patients})
    spec_idx = {s: i for i, s in enumerate(specialties)}
    ward_idx = {w: i for i, w in enumerate(ward_ids)}

    matrix = np.zeros((len(specialties), len(ward_ids)), dtype=int)
    for i, bed_idx in enumerate(x):
        matrix[spec_idx[patients[i].subspecialty], ward_idx[beds[bed_idx].ward_id]] += 1

    ward_labels = [
        f"{w}<br><sub>{next(b.specialty_anchor for b in beds if b.ward_id == w)}</sub>"
        for w in ward_ids
    ]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=ward_labels,
            y=specialties,
            colorscale=[[0, _C["surface"]], [0.5, "#2255AA"], [1, _C["accent"]]],
            showscale=True,
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 13, "color": _C["text"]},
            hoverongaps=False,
            hovertemplate="Ward: %{x}<br>Service: %{y}<br>Patients: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="Clinical Home Density by Ward",
            font=dict(color=_C["text"], size=14),
        ),
        template="plotly_dark",
        paper_bgcolor=_C["surface"],
        plot_bgcolor=_C["surface"],
        margin=dict(l=20, r=20, t=45, b=20),
        height=400,
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


def _build_pareto_scatter(
    F: np.ndarray,
    knee_idx: int,
    selected_idx: int,
) -> go.Figure:
    n = len(F)
    colors = [_C["pareto"]] * n
    sizes = [6] * n
    colors[knee_idx] = _C["accent"]
    sizes[knee_idx] = 14
    if selected_idx != knee_idx:
        colors[selected_idx] = _C["selected"]
        sizes[selected_idx] = 12

    hover = [
        (
            f"Sol #{i}<br>"
            f"Home Alignment: {F[i, 0]:.1f}<br>"
            f"Throughput: {F[i, 1]:.1f}<br>"
            f"Workload Risk: {F[i, 2]:.2f}"
            + (" [RECOMMENDED]" if i == knee_idx else "")
            + (" [SELECTED]" if i == selected_idx and i != knee_idx else "")
        )
        for i in range(n)
    ]
    fig = go.Figure(
        go.Scatter3d(
            x=F[:, 0],
            y=F[:, 1],
            z=F[:, 2],
            mode="markers",
            marker=dict(color=colors, size=sizes, opacity=0.85),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Trade-off Space (Pareto Front)", font=dict(color=_C["text"], size=14)),
        template="plotly_dark",
        paper_bgcolor=_C["surface"],
        scene=dict(
            xaxis=dict(title=dict(text="Home Alignment", font=dict(size=11))),
            yaxis=dict(title=dict(text="Throughput",     font=dict(size=11))),
            zaxis=dict(title=dict(text="Workload Risk",  font=dict(size=11))),
            bgcolor=_C["surface"],
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
        height=400,
    )
    return fig


def _style_action_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    color_map = {"MOVE": _C["move"], "HOLD": _C["hold"], "ALERT": _C["alert"]}

    def _type_color(val: str) -> str:
        c = color_map.get(val, _C["text"])
        return f"color: {c}; font-weight: 700"

    return (
        df.style
        .map(_type_color, subset=["Type"])
        .set_properties(**{"font-size": "0.82em"})
    )


# ---------------------------------------------------------------------------
# Helpers -- operational metrics
# ---------------------------------------------------------------------------

def _health_score(F_knee: np.ndarray, F_all: np.ndarray) -> tuple[str, str, str]:
    """Return (grade, description, hex_color) from knee-point objectives."""
    f_min = F_all.min(axis=0)
    f_max = F_all.max(axis=0)
    f_range = np.where(f_max > f_min, f_max - f_min, 1.0)
    norm = (F_knee - f_min) / f_range   # 0 = best, 1 = worst per objective
    score = float(norm.mean())
    if score < 0.15: return "A", "Excellent",         _C["grade_a"]
    if score < 0.30: return "B", "Good",              _C["grade_b"]
    if score < 0.50: return "C", "Moderate",          _C["grade_c"]
    if score < 0.70: return "D", "At Risk",           _C["grade_d"]
    return              "F", "Critical",          _C["grade_f"]


def _metric_html(
    label: str,
    value: str,
    note: str = "",
    tooltip: str = "",
) -> str:
    title_attr = f" title='{tooltip}'" if tooltip else ""
    note_html = f"<div class='metric-note'>{note}</div>" if note else ""
    return (
        f"<div class='metric-box'{title_attr}>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"{note_html}</div>"
    )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    f"<h2 style='color:{_C['text']}; margin-bottom:0;'>Clinical Bed Command Center</h2>"
    f"<p style='color:{_C['muted']}; margin-top:4px; font-size:0.85em;'>"
    "NSGA-II multi-objective optimisation &mdash; "
    f"{len(_patients)} patients &nbsp;/&nbsp; {len(_beds)} beds</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    tab_opt, tab_logic = st.tabs(["Optimizer", "Operational Logic"])

    # ------------------------------------------------------------------
    # Tab 1: Optimisation Controls
    # ------------------------------------------------------------------
    with tab_opt:
        st.markdown(
            f"<h4 style='color:{_C['text']};margin-top:8px;'>Controls</h4>",
            unsafe_allow_html=True,
        )

        transfer_cap: int = st.slider(
            "Max Transfers Per Shift",
            min_value=1,
            max_value=len(_patients),
            value=_settings["transfer_cap_default"],
            help=(
                "Hard ceiling on patient moves this shift (range: 1 to "
                f"{len(_patients)}). "
                "Solutions exceeding this count are marked infeasible by the "
                "engine. Lower = more stable shift. Higher = more clustering "
                "opportunity. Nursing Workload Protection (f3) best-case: 0 "
                "when no moves are needed."
            ),
        )

        aggression: float = st.slider(
            "Clinical Home Priority",
            min_value=0.1, max_value=3.0, value=1.0, step=0.1,
            help=(
                "Penalty multiplier on Clinical Home Alignment (range: 0.1x "
                "to 3.0x). At 1.0x the radius ladder is: same ward = 1 pt, "
                "different wing = 5 pt, different floor = 10 pt. "
                "At 3.0x those become 3 / 15 / 30 pt. "
                "f1 theoretical worst-case at 1.0x: "
                f"{_max_f1} pts ({len(_patients)} patients all on wrong floor). "
                "Lower f1 is always better."
            ),
        )

        ed_crisis: bool = st.toggle(
            "ED Surge Mode",
            value=False,
            help=(
                "Activates when the Emergency Department is at capacity. "
                "Flattens all Clinical Home penalties to 1 pt and lifts the "
                "transfer cap so Capacity Throughput (f2) drives the plan. "
                "f2 best-case: 0 (all primary beds occupied). "
                f"f2 worst-case: {_max_f2} (all {_n_non_overflow} primary beds idle)."
            ),
        )

        st.divider()

        with st.expander("Hospital Census Input", expanded=False):
            st.caption("Override defaults to reflect your live facility state.")
            total_beds_input: int = st.number_input(
                "Total Licensed Beds",
                min_value=1, max_value=500,
                value=len(_beds),
                help="All physical beds in the unit map, including blocked beds.",
            )
            blocked_beds_input: int = st.number_input(
                "Blocked Beds (Maintenance / Isolation)",
                min_value=0, max_value=total_beds_input,
                value=0,
                help=(
                    "Beds out of service for maintenance, deep-clean, or "
                    "contact-isolation holds. Reduces effective capacity."
                ),
            )
            staffed_beds_input: int = st.number_input(
                "Currently Staffed Beds",
                min_value=1, max_value=total_beds_input,
                value=total_beds_input,
                help="Beds with an active nursing assignment this shift.",
            )
            census_input: int = st.number_input(
                "Current Census (Patients in House)",
                min_value=0,
                max_value=total_beds_input,
                value=len(_patients),
                help=(
                    "Total number of patients currently occupying beds in the unit. "
                    "Overrides the model patient count for occupancy metrics."
                ),
            )
            available_beds_input: int = st.number_input(
                "Available Beds in Unit",
                min_value=0,
                max_value=total_beds_input,
                value=max(0, total_beds_input - len(_patients)),
                help=(
                    "Beds currently empty and ready to receive a patient. "
                    "Enter your live count; overrides the derived value."
                ),
            )
            st.markdown(
                f"<div style='color:{_C['muted']};font-size:0.75em;"
                "text-transform:uppercase;letter-spacing:0.06em;"
                "margin:10px 0 4px;'>Discharge Forecast</div>",
                unsafe_allow_html=True,
            )
            _n_assigned_total: int = sum(
                1 for p in _patients if p.current_bed_id is not None
            )
            confirmed_discharges: int = st.number_input(
                "Confirmed Discharges (TTOs ready)",
                min_value=0,
                max_value=_n_assigned_total,
                value=0,
                help=(
                    "Beds guaranteed empty within 4 hours -- "
                    "To-Take-Out (TTO) letter signed. "
                    "Used to compute projected available capacity."
                ),
            )
            estimated_discharges: int = st.number_input(
                "Estimated Discharges (Pending Review)",
                min_value=0,
                max_value=_n_assigned_total,
                value=0,
                help=(
                    "Beds likely to empty by the next shift. "
                    "Awaiting physician sign-off. "
                    "Included in the next-shift projection only."
                ),
            )
            use_projected: bool = st.checkbox(
                "Optimize for Projected Capacity",
                value=False,
                help=(
                    "When checked, confirmed-discharge patients (longest LOS) "
                    "are removed from the optimization. Their beds are treated "
                    "as available, allowing pre-assignment of incoming patients."
                ),
            )

        st.divider()

        st.markdown(
            f"<h4 style='color:{_C['text']}'>Solver Settings</h4>",
            unsafe_allow_html=True,
        )
        pop_size: int = st.select_slider(
            "Population Size", options=[20, 50, 100, 200], value=100,
        )
        n_gen: int = st.select_slider(
            "Generations", options=[20, 50, 100, 200], value=100,
        )
        run_btn: bool = st.button(
            "Re-run Optimisation", type="primary", use_container_width=True,
        )

    # ------------------------------------------------------------------
    # Tab 2: Operational Logic Reference
    # ------------------------------------------------------------------
    with tab_logic:
        st.markdown(
            f"<p style='color:{_C['muted']};font-size:0.72em;"
            "text-transform:uppercase;letter-spacing:0.08em;"
            "margin:8px 0 10px;'>Clinical Logic Reference</p>",
            unsafe_allow_html=True,
        )
        st.caption("All three objectives are minimised. Lower score = better clinical outcome.")

        def _ref_card(
            label: str,
            color: str,
            description: str,
            best_meaning: str,
            worst_val: str,
            worst_meaning: str,
        ) -> str:
            return (
                f"<div style='background:{_C['surface']};border:1px solid {_C['border']};"
                f"border-left:3px solid {color};border-radius:0 6px 6px 0;"
                f"padding:10px 12px;margin-bottom:10px;'>"
                f"<div style='color:{color};font-size:0.78em;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.06em;"
                f"margin-bottom:6px;'>{label}</div>"
                f"<div style='color:{_C['text']};font-size:0.78em;line-height:1.5;"
                f"margin-bottom:8px;'>{description}</div>"
                f"<table style='width:100%;font-size:0.74em;border-collapse:collapse;'>"
                f"<tr>"
                f"<td style='color:{_C['grade_a']};width:38%;padding:2px 4px 2px 0;"
                f"vertical-align:top;font-weight:600;'>Best (0)</td>"
                f"<td style='color:{_C['muted']};'>{best_meaning}</td>"
                f"</tr>"
                f"<tr>"
                f"<td style='color:{_C['alert']};padding:2px 4px 2px 0;"
                f"vertical-align:top;font-weight:600;'>Worst ({worst_val})</td>"
                f"<td style='color:{_C['muted']};'>{worst_meaning}</td>"
                f"</tr>"
                f"</table>"
                f"</div>"
            )

        st.markdown(
            _ref_card(
                label="Clinical Home Alignment",
                color=_C["move"],
                description=(
                    "Sum of radius penalties between each patient's assigned "
                    "bed and their Specialty Anchor ward. Penalty ladder: "
                    "same ward = 1 pt, different wing = 5 pt, "
                    "different floor = 10 pt."
                ),
                best_meaning="All patients in their home ward. Zero travel overhead for clinical teams.",
                worst_val=str(_max_f1),
                worst_meaning=(
                    f"All {len(_patients)} patients on a different floor "
                    f"from their anchor ({len(_patients)} x {BASE_PENALTIES['diff_wing']} pt)."
                ),
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            _ref_card(
                label="Capacity Throughput",
                color=_C["hold"],
                description=(
                    "Count of primary (non-overflow) beds left unoccupied "
                    "in the optimal plan. Reflects the hospital's ability to "
                    "accept new ED admissions without using overflow capacity."
                ),
                best_meaning="All primary beds occupied. Maximum intake capacity utilised.",
                worst_val=str(_max_f2),
                worst_meaning=(
                    f"All {_n_non_overflow} primary beds idle. "
                    "Patients pushed entirely to overflow."
                ),
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            _ref_card(
                label="Nursing Workload Protection",
                color=_C["alert"],
                description=(
                    "CMI-weighted cost of patient transfers recommended in the "
                    "plan. Each move is weighted by Case Mix Index to reflect "
                    "the clinical complexity of the disruption."
                ),
                best_meaning="No transfers recommended. Zero mid-shift disruption.",
                worst_val=f"{_max_f3:.1f}",
                worst_meaning=(
                    f"All {len(_patients)} patients moved. "
                    f"Total CMI {sum(p.cmi for p in _patients):.1f} x "
                    f"cost {BASE_COST_PER_CMI} per point."
                ),
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='color:{_C['muted']};font-size:0.72em;"
            f"border-top:1px solid {_C['border']};padding-top:8px;margin-top:4px;'>"
            "Penalty weights are configurable in "
            "<code>config/settings.yaml</code>. "
            "Aggression slider scales f1 penalties. "
            "ED Surge Mode flattens them to 1 pt, "
            "shifting optimisation priority to f2."
            "</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Derive effective parameters
# ---------------------------------------------------------------------------

if ed_crisis:
    effective_penalties: dict[str, int] = {k: 1 for k in BASE_PENALTIES}
    effective_cap: int = len(_patients)
    st.error(
        "ED SURGE MODE ACTIVE -- Clinical Home penalties suppressed. "
        "All transfers permitted. Capacity Throughput is the priority objective.",
    )
else:
    effective_penalties = {
        k: max(1, round(v * aggression)) for k, v in BASE_PENALTIES.items()
    }
    effective_cap = transfer_cap

penalties_flat = tuple(sorted(effective_penalties.items()))

n_confirmed: int = confirmed_discharges if use_projected else 0
opt_patients: list[Patient] = _trim_discharges(_patients, n_confirmed)

# ---------------------------------------------------------------------------
# Run optimisation
# ---------------------------------------------------------------------------

if run_btn:
    _run_cached.clear()

with st.spinner("Optimising bed assignments..."):
    X, F = _run_cached(
        transfer_cap=effective_cap,
        penalties_flat=penalties_flat,
        cost_per_cmi=BASE_COST_PER_CMI,
        pop_size=pop_size,
        n_gen=n_gen,
        seed=42,
        n_confirmed=n_confirmed,
        _patients=_patients,
        _beds=_beds,
        _anchor_map=_anchor_map,
    )

knee_idx: int = find_knee_point(F)
n_solutions: int = len(X)

if "selected_idx" not in st.session_state or run_btn:
    st.session_state["selected_idx"] = knee_idx

selected_idx: int = min(st.session_state["selected_idx"], n_solutions - 1)

# ---------------------------------------------------------------------------
# Derived clinical metrics (presentation layer only)
# ---------------------------------------------------------------------------

# Census / capacity -- census_input and available_beds_input override derived values
effective_capacity: int = max(1, total_beds_input - blocked_beds_input)
effective_staffed: int  = min(staffed_beds_input, effective_capacity)
n_placed: int           = census_input
n_assigned: int         = census_input
occupancy_pct: float    = n_placed / effective_capacity * 100
available_beds: int     = available_beds_input

# Discharge-forecast capacity projections
current_available_now: int  = max(0, effective_staffed - (n_assigned - confirmed_discharges))
projected_available: int    = max(0, current_available_now + estimated_discharges)
capacity_delta_now: int     = current_available_now - available_beds
capacity_delta_proj: int    = projected_available - available_beds

# Build generator using opt_patients to match the X array column count
gen = DirectiveGenerator(
    patients=opt_patients,
    beds=_beds,
    anchor_map=_anchor_map,
    penalties=effective_penalties,
)

proposed_pen: np.ndarray = gen._penalty_vec(X[selected_idx])
n_opt: int = len(opt_patients)
n_aligned: int  = int((proposed_pen == float(effective_penalties["same_floor"])).sum())
alignment_pct: float = n_aligned / n_opt * 100

overflow_used: int = int(gen._bed_is_overflow[X[selected_idx]].sum())
surge_active: bool = overflow_used > 0

grade, grade_desc, grade_color = _health_score(F[knee_idx], F)

# ---------------------------------------------------------------------------
# Live Capacity Dashboard
# ---------------------------------------------------------------------------

st.markdown(
    f"<p style='color:{_C['muted']};font-size:0.75em;"
    "text-transform:uppercase;letter-spacing:0.1em;"
    "margin-bottom:6px;'>Live Capacity Dashboard</p>",
    unsafe_allow_html=True,
)

cap1, cap2, cap3, cap4, cap5 = st.columns(5)

with cap1:
    st.metric(
        label="Bed Occupancy",
        value=f"{occupancy_pct:.1f}%",
        help="Patients placed / (Total Licensed - Blocked) x 100",
    )
with cap2:
    st.metric(
        label="Available Beds",
        value=str(available_beds),
        help="Effective capacity minus all placed patients.",
    )
with cap3:
    st.metric(
        label="Staffed Beds",
        value=str(effective_staffed),
        help="Beds with an active nursing assignment this shift.",
    )
with cap4:
    st.metric(
        label="Overflow Occupied",
        value=str(overflow_used),
        help=(
            "Overflow beds occupied in the recommended plan. "
            "Any value > 0 triggers Surge Protocol."
        ),
    )
with cap5:
    # Operational Health Score -- styled badge
    st.markdown(
        f"<div class='metric-box' title='Composite grade derived from all three "
        f"optimisation objectives at the recommended solution.'>"
        f"<div class='metric-label'>Operational Health</div>"
        f"<div class='grade-badge' style='color:{grade_color};'>{grade}</div>"
        f"<div class='metric-note'>{grade_desc}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

if surge_active:
    st.error(
        f"SURGE PROTOCOL ACTIVE -- {overflow_used} overflow bed(s) occupied "
        "in the recommended plan. Escalate to Nursing Supervisor.",
    )

# Discharge forecast row -- only rendered when discharge data is entered
if confirmed_discharges > 0 or estimated_discharges > 0:
    st.markdown(
        f"<p style='color:{_C['muted']};font-size:0.75em;"
        "text-transform:uppercase;letter-spacing:0.1em;"
        "margin:10px 0 4px;'>Discharge Forecast</p>",
        unsafe_allow_html=True,
    )
    dcol1, dcol2, dcol3, dcol4, _ = st.columns(5)
    with dcol1:
        st.metric(
            label="Current Available (Post-TTO)",
            value=str(current_available_now),
            delta=capacity_delta_now if capacity_delta_now != 0 else None,
            help=(
                "Staffed beds minus occupied beds, accounting for confirmed "
                "discharges. Formula: Staffed - (Occupied - Confirmed TTOs)."
            ),
        )
    with dcol2:
        st.metric(
            label="Projected Available (Next Shift)",
            value=str(projected_available),
            delta=capacity_delta_proj if capacity_delta_proj != 0 else None,
            help=(
                "Post-TTO available beds plus estimated next-shift discharges. "
                "Formula: Current Available + Estimated Discharges."
            ),
        )
    with dcol3:
        st.metric(
            label="Confirmed TTOs",
            value=str(confirmed_discharges),
            help="Beds emptying within 4 hours. Included in Current Available.",
        )
    with dcol4:
        st.metric(
            label="Estimated Discharges",
            value=str(estimated_discharges),
            help="Beds likely empty by next shift. Included in Projected Available.",
        )
    if use_projected and n_confirmed > 0:
        st.info(
            f"Pre-Assignment Mode: {n_confirmed} confirmed-discharge patient(s) "
            "removed from the optimisation. Their beds are available for "
            "incoming admissions in the plan below.",
        )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Executive Narrative
# ---------------------------------------------------------------------------

proxy = SimpleNamespace(X=X, F=F)
plan = gen.generate(proxy, solution_idx=selected_idx, move_cmi_threshold=0.5)

workload_status = (
    "within shift capacity"
    if plan.n_moves <= effective_cap
    else f"approaching the {effective_cap}-transfer shift limit"
)
alert_note = (
    f" {plan.n_alerts} patient(s) cannot be placed in their clinical home ward "
    "under current bed availability (flagged as Satellites)."
    if plan.n_alerts > 0
    else ""
)
hold_note = (
    f" {plan.n_holds} bed(s) are reserved for anticipated specialty admissions."
    if plan.n_holds > 0
    else ""
)

st.markdown(
    f"<div class='narrative-box'>"
    f"<strong>Recommended Plan (Solution #{selected_idx}):</strong> "
    f"To achieve <strong>{alignment_pct:.0f}% Clinical Home Alignment</strong>, "
    f"the system recommends <strong>{plan.n_moves} patient transfer(s)</strong>, "
    f"{workload_status}."
    f"{hold_note}"
    f"{alert_note}"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Executive View + Pareto Front
# ---------------------------------------------------------------------------

col_heat, col_pareto = st.columns(2, gap="medium")

with col_heat:
    st.plotly_chart(
        _build_density_heatmap(X[selected_idx], opt_patients, _beds),
        use_container_width=True,
    )

with col_pareto:
    st.plotly_chart(
        _build_pareto_scatter(F, knee_idx, selected_idx),
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Solution inspector
# ---------------------------------------------------------------------------

st.markdown(
    f"<p style='color:{_C['muted']}; font-size:0.8em; margin-bottom:4px;'>"
    f"Pareto front: {n_solutions} non-dominated solutions. "
    f"Gold marker = recommended (sol #{knee_idx}). "
    f"Cyan = selected.</p>",
    unsafe_allow_html=True,
)

if n_solutions > 1:
    new_idx: int = st.slider(
        "Inspect Pareto Solution",
        min_value=0, max_value=n_solutions - 1,
        value=selected_idx, key="pareto_slider",
    )
    if new_idx != selected_idx:
        st.session_state["selected_idx"] = new_idx
        selected_idx = new_idx
        st.rerun()

# Objective metric row -- operational labels with tooltips
sel_f = F[selected_idx]
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        _metric_html(
            _OBJ["f1"][0],
            f"{sel_f[0]:.1f}",
            note="penalty score (lower = better)",
            tooltip=_OBJ["f1"][1],
        ),
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        _metric_html(
            _OBJ["f2"][0],
            f"{sel_f[1]:.0f}",
            note="primary beds idle",
            tooltip=_OBJ["f2"][1],
        ),
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        _metric_html(
            _OBJ["f3"][0],
            f"{sel_f[2]:.2f}",
            note="CMI-weighted transfer cost",
            tooltip=_OBJ["f3"][1],
        ),
        unsafe_allow_html=True,
    )
with m4:
    knee_label = "Recommended plan" if selected_idx == knee_idx else f"Comparing vs sol #{knee_idx}"
    st.markdown(
        _metric_html("Solution", f"#{selected_idx}", note=knee_label),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Action Plan table
# ---------------------------------------------------------------------------

st.markdown(
    f"<h4 style='color:{_C['text']}'>Action Plan "
    f"<span style='color:{_C['muted']};font-size:0.7em;font-weight:400;'>"
    f"solution #{selected_idx}</span></h4>",
    unsafe_allow_html=True,
)

dir_col1, dir_col2, dir_col3 = st.columns(3)
with dir_col1:
    st.markdown(
        f"<div style='color:{_C['move']};font-size:1.1em;font-weight:700;'>"
        f"{plan.n_moves} MOVE</div>",
        unsafe_allow_html=True,
    )
with dir_col2:
    st.markdown(
        f"<div style='color:{_C['hold']};font-size:1.1em;font-weight:700;'>"
        f"{plan.n_holds} HOLD</div>",
        unsafe_allow_html=True,
    )
with dir_col3:
    st.markdown(
        f"<div style='color:{_C['alert']};font-size:1.1em;font-weight:700;'>"
        f"{plan.n_alerts} ALERT</div>",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

if plan.actions:
    records = [
        {
            "Type":      a.action_type.value,
            "Patient":   a.patient_pseudo_id or "--",
            "Service":   a.subspecialty,
            "From":      a.from_bed_id or "--",
            "To":        a.to_bed_id or "--",
            "Benefit":   f"{a.clustering_benefit:.2f}" if a.clustering_benefit is not None else "--",
            "CMI":       f"{a.cmi:.2f}" if a.cmi is not None else "--",
            "Reason":    a.reason,
        }
        for a in plan.actions
    ]
    df = pd.DataFrame(records)
    st.dataframe(
        _style_action_table(df),
        use_container_width=True,
        hide_index=True,
        height=min(400, 42 + 36 * len(records)),
    )
else:
    st.info("No directives for this solution. All patients are optimally placed.")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

with st.expander("Export Action Plan"):
    tab_md, tab_json = st.tabs(["Markdown", "JSON"])
    with tab_md:
        st.markdown(to_markdown(plan))
    with tab_json:
        st.code(to_json(plan), language="json")

st.caption(
    "Research and demonstration software. Not a medical device and not intended for "
    "clinical decision-making, diagnosis or treatment. Provided \"as is\", without warranty "
    "of any kind; the author accepts no liability for any use. Uses synthetic data only."
)
