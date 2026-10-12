"""
app.py - Sensitivity analysis web app for the 3-echelon supply chain (Streamlit)

Run:   streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import model as M

st.set_page_config(page_title="Sensitivity Analysis – 3-Echelon Supply Chain",
                   page_icon="📈", layout="wide")

# ----------------------------------------------------------------------
# Parameters shown as sliders: (key, label, unit)
# ----------------------------------------------------------------------
PARAM_GROUPS = [
    ("Key parameters", True, [
        ("L", "L – shelf life", "days"),
        ("Ex", "E[x] – expected survival rate", ""),
        ("S", "S – shelf capacity", "kg"),
        ("a", "a – demand scale", "kg/day"),
        ("b", "b – demand elasticity w.r.t. inventory", ""),
    ]),
    ("Farm", False, [
        ("w0", "w₀ – newborn weight", "kg"),
        ("w1", "w₁ – target weight", "kg"),
        ("alpha", "α – growth parameter", ""),
        ("beta", "β – growth parameter", ""),
        ("lam", "λ – growth parameter", "/day"),
        ("p_v", "p_v – chick purchase price", "$/kg"),
        ("K_f", "K_f – farm setup cost", "$"),
        ("c_f", "c_f – feed cost", "$/kg/day"),
        ("m_f", "m_f – mortality penalty", "$/kg"),
    ]),
    ("Processor", False, [
        ("K_p", "K_p – setup cost", "$"),
        ("h_p", "h_p – holding cost", "$/kg/day"),
        ("R", "R – processing rate", "kg/day"),
    ]),
    ("Retailer", False, [
        ("K_r", "K_r – ordering cost", "$"),
        ("h_r", "h_r – holding cost", "$/kg/day"),
        ("p_r", "p_r – retail price", "$/kg"),
    ]),
]
NUM_KEYS = list(M.BASE)
LABELS = {k: lab for _, _, items in PARAM_GROUPS for k, lab, _ in items}
MAIN_KEYS = [k for k, _, _ in PARAM_GROUPS[0][2]]
SLIDER_RANGE = 0.6          # sliders span -60% to +60% around the base value

for k in NUM_KEYS:          # initialise every slider at the base case
    st.session_state.setdefault(f"p_{k}", M.BASE[k])


def reset_all():
    for k in NUM_KEYS:
        st.session_state[f"p_{k}"] = M.BASE[k]


def pct_change(key):
    return (st.session_state[f"p_{key}"] / M.BASE[key] - 1.0) * 100.0


def fmt_pct(x):
    if abs(x) < 0.05:
        return "0%"
    return f"{x:+.1f}%"


def show(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                       # older Streamlit versions
        st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
# Cached computations
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_solve(params_tuple, constrained):
    return M.solve(dict(params_tuple), constrained)


@st.cache_data(show_spinner=False)
def run_sensitivity(center_tuple, constrained):
    center = dict(center_tuple)
    c = M.solve(center, constrained)
    cols = ["Parameter", "% change", "T* (days)", "T* % change", "n*", "n* % change",
            "Profit ($/day)", "Profit % change"]
    rows = [dict(zip(cols, ["Center", 0, c["T"], 0.0, c["n"], 0.0, c["profit"], 0.0]))] if c else []
    for key in MAIN_KEYS:
        for pc in (-40, -20, 20, 40):
            p = dict(center)
            p[key] = center[key] * (1 + pc / 100)
            s = M.solve(p, constrained)
            if s is None or c is None:
                continue
            rows.append(dict(zip(cols, [
                {"Ex": "E[x]"}.get(key, key), pc,
                s["T"], (s["T"] / c["T"] - 1) * 100,
                s["n"], (s["n"] / c["n"] - 1) * 100,
                s["profit"], (s["profit"] / c["profit"] - 1) * 100])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Sidebar: global settings
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    constrained = st.checkbox("Apply constraints  T_f ≤ nT  and  Q₁ ≤ S", value=True)
    st.button("↺ Reset to base case", on_click=reset_all, width="stretch")
    st.divider()
    st.subheader("Changed parameters")
    changed = [(k, pct_change(k)) for k in NUM_KEYS if abs(pct_change(k)) >= 0.05]
    if not changed:
        st.caption("No parameter changed yet (base case).")
    for k, pc in changed:
        old, new = M.BASE[k], st.session_state[f"p_{k}"]
        color = "green" if pc > 0 else "red"
        st.markdown(f"**{LABELS[k].split(' – ')[0]}**: :{color}[{fmt_pct(pc)}]  "
                    f"({old:.4g} → {new:.4g})")

# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------
st.title("📈 Sensitivity Analysis – 3-Echelon Supply Chain (zero ending inventory)")
st.caption("Drag a slider to change a parameter: the app shows the % change versus the base case, "
           "re-optimises (T*, n*) and redraws the profit charts.")

col_in, col_out = st.columns([4, 7], gap="large")

# ---- left column: sliders ----
with col_in:
    st.subheader("Parameters")
    for gname, expanded, items in PARAM_GROUPS:
        with st.expander(gname, expanded=expanded):
            for key, label, unit in items:
                base = M.BASE[key]
                st.slider(f"{label} {f'({unit})' if unit else ''}",
                          min_value=base * (1 - SLIDER_RANGE),
                          max_value=base * (1 + SLIDER_RANGE),
                          step=base / 100.0, format="%.4g", key=f"p_{key}")
                pc = pct_change(key)
                color = "gray" if abs(pc) < 0.05 else ("green" if pc > 0 else "red")
                st.caption(f"Change vs. base: :{color}[**{fmt_pct(pc)}**]  "
                           f"(base {base:.4g} → current {st.session_state[f'p_{key}']:.4g})")

# ---- current parameters ----
params = {k: float(st.session_state[f"p_{k}"]) for k in NUM_KEYS}
base_params = dict(M.BASE)
cur_t = tuple(sorted(params.items()))
base_t = tuple(sorted(base_params.items()))

with col_out:
    # validity checks
    if params["alpha"] <= params["w1"]:
        st.error("Need α > w₁ (otherwise the animals never reach the target weight). "
                 "Decrease w₁ or increase α.")
        st.stop()
    if params["Ex"] > 1.0:
        st.warning("E[x] > 1 has no probabilistic meaning (E[1−x] < 0). "
                   "This range is allowed for sensitivity testing only; interpret results with care.")

    sol = run_solve(cur_t, constrained)
    base_sol = run_solve(base_t, constrained)
    if sol is None:
        st.error("No feasible solution found for n ≤ 300. Increase S or reduce T_f "
                 "(change α, β, λ or w₁).")
        st.stop()

    # ---- headline results ----
    st.subheader("Optimal solution")
    d_profit = (sol["profit"] / base_sol["profit"] - 1) * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("T* (days)", f"{sol['T']:.2f}",
              f"{(sol['T'] / base_sol['T'] - 1) * 100:+.1f}% vs. base", delta_color="off")
    m2.metric("n*", f"{sol['n']}",
              f"{(sol['n'] / base_sol['n'] - 1) * 100:+.1f}% vs. base", delta_color="off")
    m3.metric("E[TPU*] ($/day)", f"{sol['profit']:.2f}", f"{d_profit:+.1f}% vs. base")
    m4.metric("Q₁* (kg)", f"{sol['Q1']:.1f}")

    cmp_df = pd.DataFrame({
        "": ["Base case", "Current", "% change"],
        "T* (days)": [f"{base_sol['T']:.2f}", f"{sol['T']:.2f}",
                      fmt_pct((sol['T'] / base_sol['T'] - 1) * 100)],
        "n*": [f"{base_sol['n']}", f"{sol['n']}",
               fmt_pct((sol['n'] / base_sol['n'] - 1) * 100)],
        "E[TPU*] ($/day)": [f"{base_sol['profit']:.2f}", f"{sol['profit']:.2f}", fmt_pct(d_profit)],
    })
    st.dataframe(cmp_df, hide_index=True, width="stretch")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            f"**Constraint T_f ≤ nT:** T_f = {sol['Tf']:.2f}, nT = {sol['nT']:.2f} days → "
            + ("🔴 **binding** (nT = T_f)" if sol["binding_Tf"] else
               ("🟢 satisfied" if sol["Tf"] <= sol["nT"] + 1e-6 else "⚠️ violated (constraints switched off)")))
    with a2:
        st.markdown(
            f"**Constraint Q₁ ≤ S:** Q₁ = {sol['Q1']:.1f}, S = {params['S']:.0f} kg → "
            + ("🔴 **binding** (Q₁ = S)" if sol["binding_S"] else
               ("🟢 satisfied" if sol["Q1"] <= params["S"] + 1e-6 else "⚠️ violated (constraints switched off)")))
    runner = sorted(sol["curve_n"], key=lambda r: -r[2])
    if len(runner) > 1 and sol["profit"] - runner[1][2] < 0.25:
        st.info(f"Near tie: n = {runner[1][0]} is only {sol['profit'] - runner[1][2]:.2f} $/day "
                f"below n* = {sol['n']}. Profit is almost flat around the optimum, so either "
                f"shipment count is a practical choice.")
    st.caption(f"Chicks ordered per processing cycle: ny ≈ {sol['ny']:.0f} · "
               f"surviving weight nQ₁ ≈ {sol['nQ1']:.0f} kg · "
               f"processing cycle nT ≈ {sol['nT']:.1f} days")

    tab_n, tab_T, tab_s = st.tabs(["📊 Profit vs. n", "📉 Profit vs. T", "📋 Sensitivity table"])

    # ---- profit vs n ----
    with tab_n:
        fig = go.Figure()
        rb = np.array(base_sol["curve_n"])
        rc = np.array(sol["curve_n"])
        lo_n, hi_n = max(1, sol["n"] - 25), sol["n"] + 25
        rb = rb[(rb[:, 0] >= lo_n) & (rb[:, 0] <= hi_n)]
        rc = rc[(rc[:, 0] >= lo_n) & (rc[:, 0] <= hi_n)]
        fig.add_trace(go.Scatter(x=rb[:, 0], y=rb[:, 2], mode="lines", name="Base case",
                                 line=dict(dash="dash", color="gray")))
        fig.add_trace(go.Scatter(x=rc[:, 0], y=rc[:, 2], mode="lines+markers", name="Current",
                                 customdata=rc[:, 1],
                                 hovertemplate="n = %{x}<br>profit = %{y:.2f} $/day"
                                               "<br>T*(n) = %{customdata:.3f} days<extra></extra>"))
        fig.add_trace(go.Scatter(x=[sol["n"]], y=[sol["profit"]], mode="markers", name="Optimum",
                                 marker=dict(size=15, symbol="star", color="red")))
        fig.update_layout(title="Expected profit vs. n (T re-optimised for each n)",
                          xaxis_title="n – number of shipments", yaxis_title="E[TPU_sc] ($/day)",
                          hovermode="x unified", height=430, margin=dict(t=60, b=40))
        show(fig)
        st.caption("Each point is the highest profit achievable for that n. "
                   "For small n, the constraint T_f ≤ nT forces a large T and Q₁ exceeds S, "
                   "so those points are excluded.")

    # ---- profit vs T ----
    with tab_T:
        use_opt = st.checkbox("Use optimal n = n*", value=True)
        if use_opt:
            n_plot = sol["n"]
        else:
            n_plot = int(st.number_input("n to plot", min_value=1, max_value=300,
                                         value=int(sol["n"]), step=1))
        Ts, pr = M.profit_vs_T(n_plot, params)
        Tsb, prb = M.profit_vs_T(base_sol["n"], base_params)
        lo, hi = M.T_bounds(n_plot, params, constrained)
        bt = M.best_T(n_plot, params, constrained)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Tsb, y=prb, mode="lines", name=f"Base case (n={base_sol['n']})",
                                 line=dict(dash="dash", color="gray")))
        fig.add_trace(go.Scatter(x=Ts, y=pr, mode="lines", name=f"Current (n={n_plot})"))
        if constrained:
            fig.add_vrect(x0=0, x1=lo, fillcolor="red", opacity=0.08, line_width=0,
                          annotation_text="violates T_f ≤ nT", annotation_position="top left")
            if hi < params["L"]:
                fig.add_vrect(x0=hi, x1=params["L"], fillcolor="red", opacity=0.08, line_width=0,
                              annotation_text="violates Q₁ ≤ S", annotation_position="top right")
        if bt is not None:
            fig.add_trace(go.Scatter(x=[bt[0]], y=[bt[1]], mode="markers", name="Optimal T",
                                     marker=dict(size=15, symbol="star", color="red")))
            ytop = bt[1] + 0.12 * abs(bt[1]) + 20
            ybot = bt[1] - max(150.0, 0.5 * abs(bt[1]))
            fig.update_yaxes(range=[ybot, ytop])
        fig.update_layout(title=f"Expected profit vs. T (n fixed at {n_plot})",
                          xaxis_title="T – retailer replenishment cycle (days)",
                          yaxis_title="E[TPU_sc] ($/day)", hovermode="x unified",
                          height=430, margin=dict(t=60, b=40))
        show(fig)
        if bt is None:
            st.warning(f"n = {n_plot} is infeasible under the current constraints.")
        else:
            st.caption(f"For n = {n_plot}: optimal T = {bt[0]:.3f} days, "
                       f"profit = {bt[1]:.2f} $/day. Red areas violate the constraints.")

    # ---- sensitivity table ----
    with tab_s:
        st.write("Sensitivity of T*, n* and profit to ±20% / ±40% changes in L, E[x], S, a and b, "
                 "**centered on the current slider values** "
                 "(with no slider changes, the center is the base case).")
        if st.button("Compute sensitivity table"):
            df = run_sensitivity(cur_t, constrained)
            st.dataframe(
                df.style.format({"T* (days)": "{:.2f}", "T* % change": "{:+.1f}", "n*": "{:d}",
                                 "n* % change": "{:+.1f}", "Profit ($/day)": "{:.2f}",
                                 "Profit % change": "{:+.1f}", "% change": "{:+d}"}),
                hide_index=True, width="stretch")
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               "sensitivity_table.csv", "text/csv")
