"""VoltGuard Module 3 dashboard.  Run:  streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from simulate_process import simulate
from alert_engine import (run_pipeline, build_incidents, evaluate, evaluate_by_attack,
                          review_queue, join_module2, LEVELS)
from physics_rules import RULES, CONFIG, RULE_THRESHOLDS, SEVERITY_NAMES

st.set_page_config(page_title="VoltGuard", page_icon="⚡", layout="wide")
st.title("⚡ VoltGuard – Physics-Aware ICS/SCADA Alert Dashboard")
st.caption("Academic prototype • offline/simulated data only • not a validated safety product")

COLORS = {"Normal": "#2ecc71", "Low": "#3498db", "Medium": "#f1c40f",
          "High": "#e67e22", "Critical": "#e74c3c"}
simulated = False

with st.sidebar:
    st.header("Data source")
    src = st.radio("Input", ["Simulated process", "Upload CSV (process + ai_score)",
                             "Upload process CSV + Module 2 predictions"])
    thr = st.slider("AI score threshold", 0.1, 0.95, 0.5, 0.05)
    min_sev = st.selectbox("Incident min severity", LEVELS[1:], index=1)
    gap = st.number_input("Merge gap (samples)", 0, 60, 5)
    try:
        if src.startswith("Sim"):
            seed = st.number_input("Random seed", 0, 9999, 42)
            df = simulate(seed=int(seed)); simulated = True
        elif "predictions" in src:
            p = st.file_uploader("Process CSV (timestamp + 7 process columns)", type="csv", key="p")
            m = st.file_uploader("Module 2 CSV (timestamp, ai_score)", type="csv", key="m")
            tol = st.text_input("Join tolerance", "1s")
            if not (p and m):
                st.info("Upload both files."); st.stop()
            df = join_module2(pd.read_csv(p, parse_dates=["timestamp"]),
                              pd.read_csv(m, parse_dates=["timestamp"]), tolerance=tol)
        else:
            up = st.file_uploader("CSV with timestamp, voltage_kv, current_a, frequency_hz, "
                                  "power_mw, power_factor, transformer_temp_c, breaker_status, ai_score",
                                  type="csv")
            if not up:
                st.info("Upload a CSV or switch to simulated data."); st.stop()
            df = pd.read_csv(up, parse_dates=["timestamp"])
        res = run_pipeline(df, thr)
    except ValueError as e:
        st.error(f"Input problem: {e}"); st.stop()
    if res.attrs.get("dropped_rows"):
        st.warning(f"Dropped {res.attrs['dropped_rows']} invalid rows.")

inc = build_incidents(res, min_sev, int(gap))
counts = res["final_severity"].value_counts()

if simulated:
    st.warning("The AI score here is a hand-crafted mock. Evaluation shows the design, "
               "not real model performance. Use Module 2's scores for real results.")

c = st.columns(5)
for col, lvl in zip(c, LEVELS[::-1]):
    col.metric(lvl, int(counts.get(lvl, 0)))

tab1, tab2, tab3, tab4 = st.tabs(["Signals", "Incidents & alerts", "Rules", "Evaluation"])

with tab1:
    sig = st.selectbox("Signal", ["current_a", "voltage_kv", "power_mw", "transformer_temp_c",
                                  "frequency_hz", "power_factor"])
    fig = go.Figure(go.Scatter(x=res["timestamp"], y=res[sig], mode="lines", name=sig,
                               line=dict(color="#7f8c8d", width=1)))
    for lvl in ["Low", "Medium", "High", "Critical"]:
        s = res[res["final_severity"] == lvl]
        fig.add_trace(go.Scatter(x=s["timestamp"], y=s[sig], mode="markers", name=lvl,
                                 marker=dict(color=COLORS[lvl], size=5),
                                 text=s["alert_reason"], hoverinfo="text+x+y"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    l, r = st.columns(2)
    with l:
        st.subheader("AI score vs time")
        f2 = px.scatter(res, x="timestamp", y="ai_score", color="final_severity",
                        color_discrete_map=COLORS, opacity=0.7, height=300)
        f2.add_hline(y=thr, line_dash="dash")
        st.plotly_chart(f2, use_container_width=True)
    with r:
        st.subheader("Rule violations")
        rc = {k: int(res["rules_triggered"].str.contains(k).sum()) for k in RULES}
        st.plotly_chart(px.bar(x=list(rc.values()), y=list(rc.keys()), orientation="h", height=300,
                               labels={"x": "samples", "y": ""}), use_container_width=True)

with tab2:
    st.subheader(f"Incidents ({min_sev}+, gaps ≤ {int(gap)} merged)")
    st.dataframe(inc, use_container_width=True, hide_index=True)
    st.download_button("Download incidents CSV", inc.to_csv(index=False), "incidents.csv")
    st.subheader("Alert log")
    sev_f = st.multiselect("Severity filter", LEVELS[1:], default=["Medium", "High", "Critical"])
    log = res[res["final_severity"].isin(sev_f)][
        ["timestamp", "final_severity", "ai_score", "physics_severity", "alert_reason"]]
    st.dataframe(log, use_container_width=True, hide_index=True)
    st.download_button("Download alerts CSV", res.to_csv(index=False), "alerts.csv")
    st.subheader("Review queue: AI-only alerts (physics normal)")
    rq = review_queue(res)
    st.caption(f"{len(rq)} possible false positives for analyst review.")
    st.dataframe(rq, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Documented operating rules")
    st.dataframe(pd.DataFrame([{"rule": k, "severity": f"{s} ({SEVERITY_NAMES[s]})",
                                "threshold": RULE_THRESHOLDS[k], "description": d, "rationale": why}
                               for k, (s, d, why) in RULES.items()]),
                 use_container_width=True, hide_index=True)
    st.subheader("Fusion matrix")
    st.dataframe(pd.DataFrame({
        "Physics": ["Critical (4)", "High (3)", "High (3)", "Medium (2)", "Medium (2)", "Normal", "Normal"],
        "AI flag": ["any", "yes", "no", "yes", "no", "yes", "no"],
        "Final": ["Critical", "Critical", "High", "High", "Medium", "Low", "Normal"]}),
        use_container_width=True, hide_index=True)
    with st.expander("Raw CONFIG values"):
        st.json(CONFIG)

with tab4:
    ev = evaluate(res)
    if ev is None:
        st.info("No ground-truth `true_attack` column in this data, so no evaluation is available.")
    else:
        st.subheader("Overall (per-sample)")
        st.dataframe(ev, use_container_width=True, hide_index=True)
        bt = evaluate_by_attack(res)
        if bt is not None:
            st.subheader("Recall by attack type")
            st.dataframe(bt, use_container_width=True, hide_index=True)
        st.caption("Limitations: synthetic data; rules were written with the attacks in mind, so "
                   "physics-only scores are optimistic; metrics are per-sample, not per-incident.")
