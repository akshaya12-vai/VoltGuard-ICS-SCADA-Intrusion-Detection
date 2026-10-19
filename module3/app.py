"""VoltGuard Module 3 dashboard.  Run:  streamlit run app.py   (needs streamlit >= 1.50)"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from simulate_process import simulate
from alert_engine import (run_pipeline, build_incidents, evaluate, evaluate_by_attack,
                          review_queue, join_module2, LEVELS)
from physics_rules import RULES, CONFIG, RULE_THRESHOLDS, SEVERITY_NAMES

st.set_page_config(page_title="VoltGuard", page_icon="⚡", layout="wide")

# ---- colour themes (severity colours stay fixed so they always mean the same) ----
THEMES = {
    "Violet night": dict(bg="#1B1530", panel="#271F45", ink="#EEEAF8", muted="#A79FC4", grid="#3A3160", accent="#9B7BFF"),
    "Teal night":   dict(bg="#0F2A2E", panel="#163A40", ink="#E4F2F1", muted="#8FB5B3", grid="#234E55", accent="#2EC4B6"),
    "Ember":        dict(bg="#241816", panel="#33211E", ink="#F6ECE8", muted="#C2A59C", grid="#4A322D", accent="#FF8A5B"),
}
COLORS = {"Normal": "#4CAF8E", "Low": "#5B9BD5", "Medium": "#E3B341", "High": "#E8833A", "Critical": "#E5484D"}

with st.sidebar:
    theme = st.selectbox("Colour theme", list(THEMES))
T = THEMES[theme]
BG, PANEL, INK, MUTED, GRID, ACCENT = (T[k] for k in ("bg", "panel", "ink", "muted", "grid", "accent"))


def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@500;600;700&family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame {{ font-family: 'Source Sans 3', sans-serif; }}
h1, h2, h3, .vg-num, .vg-head {{ font-family: 'Barlow', sans-serif !important; letter-spacing: .01em; }}
.stApp {{ background:{BG}; color:{INK}; }}
[data-testid="stSidebar"] {{ background:{PANEL}; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-baseweb="tab-highlight"] {{ background-color:{ACCENT} !important; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:2rem; max-width:1400px; }}
.vg-title {{ display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:.5rem; }}
.vg-title h1 {{ margin:0; padding:0; font-size:2rem; }}
.vg-chip {{ background:{PANEL}; border:1px solid {ACCENT}; border-radius:999px; padding:.2rem .8rem; font-size:.85rem; color:{MUTED}; }}
.vg-banner {{ border-left:8px solid var(--c); background:var(--bg); border-radius:6px; padding:1rem 1.4rem; margin:1rem 0; }}
.vg-banner .vg-head {{ font-size:1.7rem; font-weight:600; line-height:1.2; }}
.vg-banner .vg-sub {{ color:{MUTED}; margin-top:.25rem; max-width:75ch; }}
.vg-tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:.75rem; margin-bottom:1rem; }}
.vg-tile {{ background:{PANEL}; border-radius:6px; padding:.7rem 1rem; border-bottom:4px solid var(--c); }}
.vg-tile .vg-num {{ font-size:2rem; font-weight:600; line-height:1.1; }}
.vg-tile .vg-lab {{ color:{MUTED}; font-size:.9rem; }}
.vg-note {{ border-left:4px solid {COLORS['Medium']}; background:{rgba(COLORS['Medium'], .1)}; padding:.6rem 1rem; border-radius:4px; margin-bottom:1rem; }}
button:focus-visible, [role="tab"]:focus-visible {{ outline:2px solid {ACCENT} !important; outline-offset:2px; }}
</style>
""", unsafe_allow_html=True)


def style_fig(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK, family="Source Sans 3, sans-serif"),
                      legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ---- sidebar: data and settings ------------------------------------------
simulated = False
with st.sidebar:
    st.header("Data")
    src = st.radio("Source", ["Simulated feeder", "One CSV with ai_score",
                              "Process CSV + Module 2 CSV"], label_visibility="collapsed")
    st.header("Alert settings")
    thr = st.slider("AI score threshold", 0.1, 0.95, 0.5, 0.05, help="Samples at or above this count as AI-flagged.")
    min_sev = st.selectbox("Incidents start at", LEVELS[1:], index=1, help="Lowest severity that opens an incident.")
    gap = st.number_input("Merge alerts closer than (samples)", 0, 60, 5)
    with st.expander("Adjust physics limits (what-if)"):
        tol = st.slider("Voltage tolerance (±%)", 2, 10, 5)
        i_max = st.slider("Current limit (A)", 250, 600, int(CONFIG["i_max"]), 10)
        t_max = st.slider("Temperature limit (°C)", 60, 110, int(CONFIG["temp_max"]))
        p_tol = st.slider("Power mismatch tolerance (MW)", 0.1, 1.0, CONFIG["power_tol_mw"], 0.05)
    cfg = dict(CONFIG, v_min=11 * (1 - tol / 100), v_max=11 * (1 + tol / 100),
               i_max=float(i_max), temp_max=float(t_max), power_tol_mw=float(p_tol))
    try:
        if src.startswith("Sim"):
            seed = st.number_input("Random seed", 0, 9999, 42)
            df = simulate(seed=int(seed)); simulated = True
        elif src.startswith("Process"):
            p = st.file_uploader("Process CSV", type="csv", key="p",
                                 help="timestamp, voltage_kv, current_a, frequency_hz, power_mw, "
                                      "power_factor, transformer_temp_c, breaker_status")
            m = st.file_uploader("Module 2 CSV (timestamp, ai_score)", type="csv", key="m")
            tmatch = st.text_input("Match timestamps within", "1s")
            if not (p and m):
                st.info("Upload both files to continue."); st.stop()
            df = join_module2(pd.read_csv(p, parse_dates=["timestamp"]),
                              pd.read_csv(m, parse_dates=["timestamp"]), tolerance=tmatch)
        else:
            up = st.file_uploader("CSV file", type="csv",
                                  help="timestamp, voltage_kv, current_a, frequency_hz, power_mw, "
                                       "power_factor, transformer_temp_c, breaker_status, ai_score")
            if not up:
                st.info("Upload a CSV or switch to the simulated feeder."); st.stop()
            df = pd.read_csv(up, parse_dates=["timestamp"])
        res = run_pipeline(df, thr, cfg)
    except ValueError as e:
        st.error(f"Could not read the data: {e}"); st.stop()
    if res.attrs.get("dropped_rows"):
        st.warning(f"Skipped {res.attrs['dropped_rows']} rows with missing or invalid values.")

n = len(res)
inc = build_incidents(res, min_sev, int(gap))
inc = inc.assign(status=[st.session_state.get(f"tri_{k}", "Not reviewed") for k in inc["incident"]])
counts = res["final_severity"].value_counts()
source_label = "Simulated 11 kV feeder" if simulated else "Uploaded data"
LIMITS = {"current_a": ["i_max"], "voltage_kv": ["v_min", "v_max"], "frequency_hz": ["f_min", "f_max"],
          "transformer_temp_c": ["temp_max"], "power_factor": ["pf_min", "pf_max"]}
SIGNALS = {"Current (A)": "current_a", "Voltage (kV)": "voltage_kv", "Power (MW)": "power_mw",
           "Transformer temperature (°C)": "transformer_temp_c", "Frequency (Hz)": "frequency_hz",
           "Power factor": "power_factor"}


def timeline_fig(d, sig, label, height):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.1, 0.9], vertical_spacing=0.03)
    fig.add_trace(go.Heatmap(
        x=d["timestamp"], y=["Severity"], z=[d["final_severity"].map(LEVELS.index).tolist()], zmin=0, zmax=4,
        showscale=False, colorscale=[[k / 4, COLORS[l]] for k, l in enumerate(LEVELS)],
        text=[d["final_severity"].tolist()], hovertemplate="%{x}<br>%{text}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["timestamp"], y=d[sig], mode="lines", name=label,
                             line=dict(color=MUTED, width=1.2)), row=2, col=1)
    for lvl in ["Low", "Medium", "High", "Critical"]:
        s = d[d["final_severity"] == lvl]
        if len(s):
            fig.add_trace(go.Scatter(x=s["timestamp"], y=s[sig], mode="markers", name=lvl,
                                     marker=dict(color=COLORS[lvl], size=6), text=s["alert_reason"],
                                     hovertemplate="%{x}<br>%{text}<extra></extra>"), row=2, col=1)
    for key in LIMITS.get(sig, []):
        fig.add_hline(y=cfg[key], line_dash="dash", line_color=COLORS["High"], opacity=0.7,
                      annotation_text=f"limit {cfg[key]:.2f}", annotation_font_color=MUTED, row=2, col=1)
    return style_fig(fig, height)


def gauge(title, val, lo, hi, ok_lo, ok_hi):
    bad, good = rgba(COLORS["Critical"], .35), rgba(COLORS["Normal"], .3)
    f = go.Figure(go.Indicator(mode="gauge+number", value=float(val), title={"text": title, "font": {"size": 14}},
                               gauge={"axis": {"range": [lo, hi]}, "bar": {"color": ACCENT},
                                      "steps": [{"range": [lo, ok_lo], "color": bad},
                                                {"range": [ok_lo, ok_hi], "color": good},
                                                {"range": [ok_hi, hi], "color": bad}]}))
    style_fig(f, 210)
    f.update_layout(margin=dict(l=25, r=25, t=50, b=10))
    return f


def report_md():
    L = ["# VoltGuard alert report", "",
         f"Data: {source_label}, {n:,} samples, {res['timestamp'].iloc[0]} to {res['timestamp'].iloc[-1]}",
         f"AI threshold {thr}; voltage ±{tol}%; current limit {i_max} A; temperature limit {t_max} C", "",
         "## Samples by severity"] + [f"- {l}: {int(counts.get(l, 0))}" for l in LEVELS[::-1]]
    L += ["", f"## Incidents ({min_sev} and above)"]
    L += [f"- Incident {r.incident}: {r.max_severity}, {r.start:%H:%M:%S} for {r.duration_s} s, "
          f"rules {r.rules}, status {r.status}" for r in inc.itertuples()] or ["- None"]
    return "\n".join(L)


# ---- header and status banner ---------------------------------------------
st.markdown(f"""
<div class="vg-title"><h1>⚡ VoltGuard</h1>
<span class="vg-chip">{source_label}, {n:,} samples, {res['timestamp'].iloc[0]:%d %b %H:%M} to {res['timestamp'].iloc[-1]:%H:%M}</span></div>
<div style="color:{MUTED}">Physics-aware alerts for ICS/SCADA feeders. Offline prototype, not a validated safety product.</div>
""", unsafe_allow_html=True)

if inc.empty:
    top, head, sub = "Normal", "No incidents found", "All process readings stayed within the documented limits."
else:
    top = max(inc["max_severity"], key=LEVELS.index)
    k = len(inc)
    head = f"{top}: {k} incident{'s' if k != 1 else ''} need{'' if k != 1 else 's'} review"
    worst = inc.iloc[[LEVELS.index(s) for s in inc["max_severity"]].index(LEVELS.index(top))]
    sub = (f"Most severe starts {worst['start']:%H:%M:%S} and lasts {worst['duration_s']} s. "
           f"Rules involved: {worst['rules']}.")
st.markdown(f"""<div class="vg-banner" style="--c:{COLORS[top]};--bg:{rgba(COLORS[top], .12)}">
<div class="vg-head">{head}</div><div class="vg-sub">{sub}</div></div>""", unsafe_allow_html=True)
if simulated:
    st.markdown("""<div class="vg-note">The AI score here is a hand-made mock, so the results show how the
design works, not real model accuracy. Load Module 2's scores for real results.</div>""", unsafe_allow_html=True)
tiles = "".join(f'<div class="vg-tile" style="--c:{COLORS[l]}"><div class="vg-num">{int(counts.get(l, 0)):,}</div>'
                f'<div class="vg-lab">{l} samples</div></div>' for l in LEVELS[::-1])
st.markdown(f'<div class="vg-tiles">{tiles}</div>', unsafe_allow_html=True)

tab_ov, tab_live, tab_tl, tab_inc, tab_rules, tab_ev = st.tabs(
    ["Overview", "Live replay", "Timeline", "Incidents", "Rules", "Evaluation"])

# ---- overview -------------------------------------------------------------
with tab_ov:
    a, b = st.columns(2)
    with a:
        st.subheader("Severity share")
        pie = go.Figure(go.Pie(labels=LEVELS, values=[int(counts.get(l, 0)) for l in LEVELS], hole=.55,
                               marker=dict(colors=[COLORS[l] for l in LEVELS]), sort=False))
        st.plotly_chart(style_fig(pie, 320), width="stretch")
    with b:
        st.subheader("Rule violations")
        rc = {key: int(res["rules_triggered"].str.contains(key).sum()) for key in RULES}
        bar = go.Figure(go.Bar(x=list(rc.values()), y=list(rc.keys()), orientation="h", marker_color=ACCENT))
        st.plotly_chart(style_fig(bar, 320).update_yaxes(autorange="reversed"), width="stretch")
    st.subheader("Where physics and AI agree")
    ct = pd.crosstab(res["physics_severity"].map(lambda s: SEVERITY_NAMES.get(s, "Normal")),
                     res["ai_flag"].map({True: "AI flags it", False: "AI quiet"}))
    ct = ct.reindex(index=["Normal", "Medium", "High", "Critical"], columns=["AI quiet", "AI flags it"], fill_value=0)
    hm = go.Figure(go.Heatmap(z=ct.values, x=ct.columns, y=ct.index, text=ct.values, texttemplate="%{text}",
                              colorscale=[[0, PANEL], [1, ACCENT]], showscale=False))
    st.plotly_chart(style_fig(hm, 260), width="stretch")
    st.caption("Off-diagonal cells are where only one detector fires, which is why the two are combined.")
    st.download_button("Download summary report", report_md(), "voltguard_report.md")

# ---- live replay (dynamic view) --------------------------------------------
with tab_live:
    st.session_state.setdefault("pos", 0)
    st.session_state.pos = min(st.session_state.pos, n - 1)
    c1, c2, c3, c4, c5 = st.columns([1, 1.3, 1.3, 1.6, 1.3])
    play = c1.toggle("Play", key="play")
    speed = c2.select_slider("Samples per second", [1, 5, 10, 25, 50, 100], value=10)
    win = c3.select_slider("Window (samples)", [100, 200, 300, 500, 800], value=300)
    live_label = c4.selectbox("Signal", list(SIGNALS), key="live_sig")
    c5.write("")
    if c5.button("Restart"):
        st.session_state.pos = 0
    if c5.button("Jump to next incident", disabled=inc.empty):
        later = inc[inc["start"] > res["timestamp"].iloc[st.session_state.pos]]
        target = (later if len(later) else inc).iloc[0]["start"]
        st.session_state.pos = max(int(res.index[res["timestamp"] == target][0]) - 30, 0)

    def live_view():
        if play:
            st.session_state.pos = min(st.session_state.pos + speed, n - 1)
        pos = st.session_state.pos
        row = res.iloc[pos]
        w = res.iloc[max(0, pos - win + 1): pos + 1]
        sev = row["final_severity"]
        st.markdown(f"""<div class="vg-banner" style="--c:{COLORS[sev]};--bg:{rgba(COLORS[sev], .12)}">
<div class="vg-head">{sev} at {row['timestamp']:%H:%M:%S}</div>
<div class="vg-sub">{row['alert_reason'] or 'All readings within limits.'}</div></div>""", unsafe_allow_html=True)
        st.progress(pos / max(n - 1, 1), text=f"Sample {pos + 1:,} of {n:,}" + ("  (end of data)" if pos >= n - 1 else ""))
        g = st.columns(4)
        g[0].plotly_chart(gauge("Voltage (kV)", row["voltage_kv"], 10, 12, cfg["v_min"], cfg["v_max"]), width="stretch")
        g[1].plotly_chart(gauge("Current (A)", row["current_a"], 0, 500, 0, cfg["i_max"]), width="stretch")
        g[2].plotly_chart(gauge("Temperature (°C)", row["transformer_temp_c"], 40, 110, 40, cfg["temp_max"]), width="stretch")
        g[3].plotly_chart(gauge("Frequency (Hz)", row["frequency_hz"], 49, 51, cfg["f_min"], cfg["f_max"]), width="stretch")
        st.plotly_chart(timeline_fig(w, SIGNALS[live_label], live_label, 380), width="stretch")
        feed = res.iloc[: pos + 1]
        feed = feed[feed["final_severity"] != "Normal"].tail(8)[["timestamp", "final_severity", "alert_reason"]]
        st.markdown("**Latest alerts**")
        st.dataframe(feed.iloc[::-1], width="stretch", hide_index=True)

    st.fragment(run_every=1.0 if play else None)(live_view)()

# ---- timeline --------------------------------------------------------------
with tab_tl:
    lab = st.selectbox("Signal", list(SIGNALS), key="tl_sig")
    st.plotly_chart(timeline_fig(res, SIGNALS[lab], lab, 520), width="stretch")
    st.caption("The colour strip shows alert severity over time. Dashed lines are the physics limits.")

# ---- incidents -------------------------------------------------------------
with tab_inc:
    st.subheader(f"Incidents ({min_sev} and above)")
    if inc.empty:
        st.info("No incidents. Lower the “Incidents start at” setting to see lower-severity alerts.")
    else:
        st.dataframe(inc, width="stretch", hide_index=True, column_config={
            "start": st.column_config.DatetimeColumn("Start", format="HH:mm:ss"),
            "end": st.column_config.DatetimeColumn("End", format="HH:mm:ss"),
            "duration_s": st.column_config.NumberColumn("Seconds"),
            "alert_samples": st.column_config.NumberColumn("Alert samples"),
            "max_severity": st.column_config.TextColumn("Severity"),
            "rules": st.column_config.TextColumn("Rules triggered", width="large"),
            "avg_ai_score": st.column_config.ProgressColumn("Avg AI score", min_value=0, max_value=1, format="%.2f"),
            "status": st.column_config.TextColumn("Analyst status")})
        st.download_button("Download incidents", inc.to_csv(index=False), "incidents.csv")

        st.subheader("Incident detail")
        pick = st.selectbox("Choose an incident", inc["incident"].tolist(),
                            format_func=lambda k: f"Incident {k}: {inc.loc[inc.incident == k, 'max_severity'].iloc[0]}, "
                                                  f"{inc.loc[inc.incident == k, 'start'].iloc[0]:%H:%M:%S}")
        st.selectbox("Analyst decision", ["Not reviewed", "Investigating", "Confirmed attack", "False positive"],
                     key=f"tri_{pick}", help="Saved for this session. Incident numbers change if you change settings.")
        row = inc[inc["incident"] == pick].iloc[0]
        pad = pd.Timedelta(seconds=60)
        w = res[(res["timestamp"] >= row["start"] - pad) & (res["timestamp"] <= row["end"] + pad)]
        inside = w[(w["timestamp"] >= row["start"]) & (w["timestamp"] <= row["end"])]
        panels = [("Current (A)", "current_a"), ("Voltage (kV)", "voltage_kv"),
                  ("Power (MW)", "power_mw"), ("Temperature (°C)", "transformer_temp_c")]
        f3 = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                           subplot_titles=[p[0] for p in panels])
        for k, (_, col) in enumerate(panels, 1):
            f3.add_trace(go.Scatter(x=w["timestamp"], y=w[col], mode="lines", showlegend=False,
                                    line=dict(color=INK, width=1.2)), row=k, col=1)
            if col == "power_mw":
                f3.add_trace(go.Scatter(x=w["timestamp"], y=w["power_calc_mw"], mode="lines", name="Expected from V×I×pf",
                                        line=dict(color=COLORS["Low"], width=1, dash="dot")), row=k, col=1)
            f3.add_vrect(x0=row["start"], x1=row["end"], fillcolor=COLORS[row["max_severity"]],
                         opacity=0.18, line_width=0, row=k, col=1)
        st.plotly_chart(style_fig(f3, 620), width="stretch")
        st.markdown("**Why it was flagged**")
        for r in [r for r in inside["physics_reason"].unique() if r]:
            st.markdown(f"- Physics: {r}")
        if inside["ai_flag"].any():
            st.markdown(f"- AI: suspicious traffic in {int(inside['ai_flag'].sum())} samples "
                        f"(highest score {inside['ai_score'].max():.2f})")

    st.subheader("Alert log")
    sev_f = st.multiselect("Show severities", LEVELS[1:], default=["Medium", "High", "Critical"])
    log = res[res["final_severity"].isin(sev_f)][["timestamp", "final_severity", "ai_score", "physics_severity", "alert_reason"]]
    st.dataframe(log, width="stretch", hide_index=True)
    st.download_button("Download full alert data", res.to_csv(index=False), "alerts.csv")
    rq = review_queue(res)
    with st.expander(f"Review queue: {len(rq)} AI-only alerts where physics looks normal"):
        st.caption("These may be false positives. An analyst should check them.")
        st.dataframe(rq, width="stretch", hide_index=True)

# ---- rules -----------------------------------------------------------------
with tab_rules:
    st.subheader("Physics rules")
    st.dataframe(pd.DataFrame([{"Rule": k, "Severity": f"{s} ({SEVERITY_NAMES[s]})", "Default threshold": RULE_THRESHOLDS[k],
                                "What it checks": d, "Why": why} for k, (s, d, why) in RULES.items()]),
                 width="stretch", hide_index=True)
    st.subheader("How physics and AI results combine")
    st.dataframe(pd.DataFrame({
        "Physics result": ["Critical", "High", "High", "Medium", "Medium", "Normal", "Normal"],
        "AI flags it": ["Either", "Yes", "No", "Yes", "No", "Yes", "No"],
        "Final alert": ["Critical", "Critical", "High", "High", "Medium", "Low (review)", "Normal"]}),
        width="stretch", hide_index=True)
    with st.expander("Limits in use now"):
        st.json(cfg)

# ---- evaluation ------------------------------------------------------------
with tab_ev:
    ev = evaluate(res)
    if ev is None:
        st.info("This data has no `true_attack` column, so there is nothing to score against.")
    else:
        st.subheader("Overall results (per sample)")
        st.dataframe(ev, width="stretch", hide_index=True)
        bt = evaluate_by_attack(res)
        if bt is not None:
            st.subheader("Share of each attack type detected")
            st.dataframe(bt, width="stretch", hide_index=True)
        st.caption("Limits: synthetic data, rules written with these attacks in mind (physics-only is "
                   "optimistic), and scores are per sample, not per incident.")