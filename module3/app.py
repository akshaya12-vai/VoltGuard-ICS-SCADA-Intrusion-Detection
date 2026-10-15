"""VoltGuard Module 3 dashboard.  Run:  streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from simulate_process import simulate
from alert_engine import run_pipeline, build_incidents, evaluate, LEVELS
from physics_rules import RULES, CONFIG

st.set_page_config(page_title="VoltGuard", page_icon="⚡", layout="wide")
st.title("⚡ VoltGuard – Physics-Aware ICS/SCADA Alert Dashboard")
st.caption("Academic prototype • offline/simulated data only • not a validated safety product")

COLORS = {"Normal": "#2ecc71", "Low": "#3498db", "Medium": "#f1c40f",
          "High": "#e67e22", "Critical": "#e74c3c"}

with st.sidebar:
    st.header("Data source")
    src = st.radio("Input", ["Simulated process", "Upload CSV (Module 2 output)"])
    thr = st.slider("AI score threshold", 0.1, 0.95, 0.5, 0.05)
    if src.startswith("Sim"):
        seed = st.number_input("Random seed", 0, 9999, 42)
        df = simulate(seed=int(seed))
    else:
        up = st.file_uploader("CSV with timestamp, voltage_kv, current_a, frequency_hz, "
                              "power_mw, power_factor, transformer_temp_c, breaker_status, ai_score",
                              type="csv")
        if not up:
            st.info("Upload a CSV or switch to simulated data."); st.stop()
        df = pd.read_csv(up, parse_dates=["timestamp"])
    st.header("Physics limits")
    for k, v in CONFIG.items():
        st.text(f"{k}: {v}")

res = run_pipeline(df, thr)
inc = build_incidents(res)
counts = res["final_severity"].value_counts()

c = st.columns(5)
for col, lvl in zip(c, LEVELS[::-1]):
    col.metric(lvl, int(counts.get(lvl, 0)))

st.subheader("Process variables with alerts")
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
st.plotly_chart(fig, width="stretch")

l, r = st.columns(2)
with l:
    st.subheader("AI score vs physics severity")
    f2 = px.scatter(res, x="timestamp", y="ai_score", color="final_severity",
                    color_discrete_map=COLORS, opacity=0.7, height=300)
    f2.add_hline(y=thr, line_dash="dash")
    st.plotly_chart(f2, width="stretch")
with r:
    st.subheader("Rule violations")
    rc = {k: int(res["rules_triggered"].str.contains(k).sum()) for k in RULES}
    st.plotly_chart(px.bar(x=list(rc.values()), y=list(rc.keys()), orientation="h", height=300,
                           labels={"x": "samples", "y": ""}), width="stretch")

st.subheader("Incidents")
st.dataframe(inc, width="stretch", hide_index=True)

st.subheader("Alert log")
sev_f = st.multiselect("Severity filter", LEVELS[1:], default=["Medium", "High", "Critical"])
log = res[res["final_severity"].isin(sev_f)][
    ["timestamp", "final_severity", "ai_score", "physics_severity", "alert_reason"]]
st.dataframe(log, width="stretch", hide_index=True)
st.download_button("Download alerts CSV", res.to_csv(index=False), "alerts.csv")

ev = evaluate(res)
if ev is not None:
    st.subheader("Evaluation vs ground truth (simulation)")
    st.dataframe(ev, width="stretch", hide_index=True)
    st.caption("Shows why combining AI + physics beats either alone.")
