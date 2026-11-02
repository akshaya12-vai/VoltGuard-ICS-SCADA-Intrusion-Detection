"""VoltGuard control-room dashboard (Modules 1-3).  Run from module3/:  streamlit run web_app.py"""
import html
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="VoltGuard · Control Room", page_icon="⚡", layout="wide")

try:
    from simulate_process import simulate
    from alert_engine import (run_pipeline, build_incidents, evaluate, evaluate_by_attack,
                              evaluate_incidents, review_queue, threshold_sweep, LEVELS)
    from physics_rules import RULES, CONFIG, RULE_THRESHOLDS
    import integration as itg
    import incident_analysis as ia
    import tank_rules as tk
    import tank_data as td
except ImportError as e:
    st.error(f"Missing module: {e}. Copy all files from module3/ (integration.py, incident_analysis.py, tank_*.py).")
    st.stop()

AMBER, INFO, TEXT, MUTED = "#ffb020", "#38bdf8", "#d6e2ee", "#7f93a7"
COLORS = {"Normal": "#2dd4a0", "Low": "#38bdf8", "Medium": "#fbbf24", "High": "#fb923c", "Critical": "#f43f5e"}
STATUS = {"Normal": "ALL CLEAR", "Low": "ALL CLEAR", "Medium": "WARNING", "High": "ALERT", "Critical": "CRITICAL ALERT"}
SRC_M2 = "Module 2 output (agreed 11 kV dataset)"
SRC_SIM = "Simulated (mock AI)"
SRC_UP = "Upload CSV"
PROFILE_TANK = "Water tank (Module 1/2 data)"
PROFILE_FEEDER = "11 kV feeder (simulation)"

st.markdown("""<style>
.block-container{padding-top:1rem;max-width:1400px}
.hdr{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;
 background:linear-gradient(90deg,#111a23,#0d1a27);border:1px solid #223344;border-left:4px solid #ffb020;
 border-radius:10px;padding:14px 20px;margin-bottom:14px}
.hdr .t{font-size:1.35rem;font-weight:700;letter-spacing:.02em}.hdr .s{color:#7f93a7;font-size:.85rem}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{border:1px solid #223344;background:#0d151d;border-radius:999px;padding:3px 12px;font-size:.74rem;
 color:#9fb3c8;font-family:ui-monospace,Menlo,Consolas,monospace}
.led{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;box-shadow:0 0 8px currentColor}
.tile{background:#111a23;border:1px solid #223344;border-top:3px solid var(--c,#38bdf8);border-radius:10px;
 padding:12px 14px;height:100%}
.tile .k{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:#7f93a7}
.tile .v{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:1.7rem;font-weight:700;color:var(--c)}
.tile .s{font-size:.78rem;color:#587089}
.panel{background:#111a23;border:1px solid #223344;border-radius:10px;padding:12px 16px}
.alarm{display:flex;gap:12px;align-items:center;background:#0d151d;border:1px solid #223344;
 border-left:4px solid var(--c);border-radius:8px;padding:8px 12px;margin-bottom:8px}
.alarm .a1{font-weight:600;font-size:.92rem}.alarm .a2{color:#7f93a7;font-size:.76rem;margin-bottom:4px}
.alarm .n{font-family:ui-monospace,Menlo,monospace;font-size:1.25rem;font-weight:700;color:var(--c)}
.bar{height:5px;border-radius:4px;background:#1b2a38}.bar>div{height:5px;border-radius:4px;background:var(--c)}
.sec{font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;color:#7f93a7;margin:18px 0 8px;
 border-bottom:1px solid #223344;padding-bottom:4px}
.insight{background:#0d151d;border:1px solid #223344;border-left:3px solid #ffb020;border-radius:8px;
 padding:8px 12px;margin-bottom:6px;font-size:.9rem}
.banner{border-radius:10px;padding:10px 18px;margin-bottom:12px;font-weight:600;border:1px solid;
 font-family:ui-monospace,Menlo,monospace;letter-spacing:.05em}
div[data-testid="stTabs"] button{font-weight:600}
</style>""", unsafe_allow_html=True)


# ---------- helpers ----------
def esc(x):
    return html.escape(str(x))


def header(title, sub, chips=()):
    w = max(res.tail(60)["final_severity"], key=LEVELS.index)
    cs = "".join(f'<span class="chip">{esc(c)}</span>' for c in chips)
    st.markdown(
        f'<div class="hdr"><div><div class="t">{title}</div><div class="s">{sub}</div></div>'
        f'<div class="chips"><span class="chip" style="color:{COLORS[w]}"><span class="led" '
        f'style="color:{COLORS[w]};background:{COLORS[w]}"></span>{STATUS[w]}</span>{cs}</div></div>',
        unsafe_allow_html=True)


def tiles(items):
    for col, (k, v, s, *c) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="tile" style="--c:{c[0] if c else INFO}"><div class="k">{k}</div>'
                     f'<div class="v">{v}</div><div class="s">{s}</div></div>', unsafe_allow_html=True)


def section(t):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)


def style(fig, h=340):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=36, b=10), font=dict(color="#c9d6e3"),
                      legend=dict(orientation="h", y=1.13))
    fig.update_xaxes(gridcolor="#1a2836", zerolinecolor="#1a2836")
    fig.update_yaxes(gridcolor="#1a2836", zerolinecolor="#1a2836")
    return fig


@st.cache_data
def load_df(key):
    return itg.read_csv(key)


REQUIRED = ["timestamp", "voltage_kv", "current_a", "frequency_hz", "power_mw", "power_factor",
            "transformer_temp_c", "breaker_status", "ai_score"]


def check_schema(df):
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns {miss}")


@st.cache_data(show_spinner="Running detection pipeline...")
def compute(profile, src, thr, seed, split, stress, cfg_items, up_bytes):
    info, use_tank = {}, profile == PROFILE_TANK
    try:
        if use_tank:
            df, info = td.load(split, stress, seed)          # Module 1 files + Module 2 real model
        elif src == SRC_M2:
            df = pd.read_csv(itg.PATHS["m2out"], parse_dates=["timestamp"])
            check_schema(df)
        elif src == SRC_UP and up_bytes:
            df = pd.read_csv(io.BytesIO(up_bytes), parse_dates=["timestamp"])
            check_schema(df)
        else:
            df = simulate(seed=seed)
    except Exception as e:  # missing teammate files etc.
        df, info, use_tank = simulate(seed=seed), {"fallback": str(e)}, False
    cfg = dict(cfg_items) if (use_tank or profile != PROFILE_TANK) else None
    try:
        res = run_pipeline(df, thr, cfg, profile=tk if use_tank else None)
    except ValueError as e:
        df, info = simulate(seed=seed), {"fallback": str(e)}
        res = run_pipeline(df, thr)
    inc = build_incidents(res)
    return res, ia.enrich_incidents(inc, res), info


def fusion_table():
    t = pd.DataFrame({"AI says normal": ["Normal", "Medium", "High", "Critical"],
                      "AI flags attack": ["Low", "High", "Critical", "Critical"]},
                     index=["No physics violation", "Medium violation", "High violation", "Critical violation"])
    return t.style.map(lambda v: f"background-color:{COLORS[v]}44;color:white")


def metrics_at(y, score, t):
    p = score >= t
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=pr, recall=rc, f1=f1, accuracy=(tp + tn) / max(1, len(y)))


# ---------- sidebar / state ----------
with st.sidebar:
    st.markdown("### ⚡ VoltGuard")
    st.caption("Physics-aware ICS/SCADA intrusion detection · control room")
    profile = st.radio("Process", [PROFILE_TANK, PROFILE_FEEDER], key="profile",
                       help="Water tank = Module 1's dataset scored by Module 2's model.")
    split, up_bytes, stress, src = "test", None, False, SRC_SIM
    if profile == PROFILE_TANK:
        split = st.selectbox("Module 1 data", ["test", "train", "clean"], key="tsplit",
                             format_func={"test": "Test split (1000 rows)", "train": "Train split",
                                          "clean": "Cleaned dataset (all)"}.get)
        stress = st.checkbox("Add Module 3 stress-test attacks", value=False,
                             help="Adds 4 sustained, labelled attacks (inconsistent sensors, valve spoof, "
                                  "replay, overflow) to test the physics layer. Modules 1/2 files are not changed.")
        src = "Module 1 files + Module 2 model"
    else:
        src = st.radio("Data source", [SRC_M2, SRC_SIM, SRC_UP], key="src")
        if src == SRC_UP:
            up = st.file_uploader("CSV with process columns + ai_score", type="csv")
            up_bytes = up.getvalue() if up else None
            if not up:
                st.info("No file yet, showing simulated data.")
    thr = st.slider("AI score threshold", 0.1, 0.95, 0.5, 0.05, key="thr")
    seed = int(st.number_input("Random seed", 0, 9999, 42, key="seed"))
    with st.expander("Physics thresholds (what-if)"):
        if profile == PROFILE_TANK:
            vals = {k: float(st.slider(lbl, lo, hi, int(tk.CONFIG[k]), step)) for k, lbl, lo, hi, step in tk.WHATIF}
            cfg = dict(tk.CONFIG, **vals)
        else:
            tol = st.slider("Voltage tolerance ±%", 1.0, 10.0, 5.0, 0.5)
            imax = st.slider("Current limit (A)", 250, 600, int(CONFIG["i_max"]), 10)
            tmax = st.slider("Temperature limit (°C)", 60, 110, int(CONFIG["temp_max"]))
            ptol = st.slider("Power mismatch tolerance (MW)", 0.1, 1.5, float(CONFIG["power_tol_mw"]), 0.05)
            cfg = dict(CONFIG)
            cfg.update(v_min=11 * (1 - tol / 100), v_max=11 * (1 + tol / 100), i_max=float(imax),
                       temp_max=float(tmax), power_tol_mw=float(ptol))

res, ei, info = compute(profile, src, thr, seed, split, stress, tuple(sorted(cfg.items())), up_bytes)
TANK = "Tank_Water_Level" in res.columns          # which process the results actually are
RULES_ACTIVE = tk.RULES if TANK else RULES
THRESH_ACTIVE = tk.RULE_THRESHOLDS if TANK else RULE_THRESHOLDS
if "fallback" in info:
    st.sidebar.warning(f"Module 1/2 files not usable, using simulation. ({info['fallback'][:90]})")
if TANK and profile != PROFILE_TANK:
    pass
MEDIUM_UP = ["Medium", "High", "Critical"]


def chips_common():
    return [("Water tank" if TANK else "11 kV feeder"), f"{len(res):,} samples",
            f"{res.timestamp.min():%d %b %H:%M} → {res.timestamp.max():%H:%M}", f"AI ≥ {thr:.2f}"]


# ---------- plant mimic (HMI) ----------
def gauge(fig, col, value, title, rng, bands, limit=None):
    g = dict(axis=dict(range=list(rng), tickcolor="#587089"), bar=dict(color="#e6eef7", thickness=.25),
             bgcolor="#0d151d", borderwidth=0, steps=[dict(range=[lo, hi], color=c) for lo, hi, c in bands])
    if limit is not None:
        g["threshold"] = dict(line=dict(color="#f43f5e", width=3), thickness=.8, value=limit)
    fig.add_trace(go.Indicator(mode="gauge+number", value=float(value), title=dict(text=title, font=dict(size=13)),
                               number=dict(font=dict(size=26)), gauge=g), row=1, col=col)


AMB_D, GRN_D, RED_D = "#4a3a14", "#10382f", "#4a1d2a"


def mimic(r):
    if TANK:
        L, P, F, V = float(r.Tank_Water_Level), float(r.Pressure_Sensor), float(r.Flow_Rate), int(r.Valve_Status)
        fig = make_subplots(rows=1, cols=3, column_widths=[.34, .33, .33], horizontal_spacing=.05,
                            specs=[[{"type": "xy"}, {"type": "indicator"}, {"type": "indicator"}]])
        bad = L > cfg["level_high"] or L < cfg["level_low"]
        fill = "rgba(244,63,94,.6)" if bad else "rgba(251,191,36,.55)" if L > cfg["level_warn"] else "rgba(56,189,248,.55)"
        fig.add_shape(type="rect", x0=.2, x1=.8, y0=0, y1=100, line=dict(color="#587089", width=2), row=1, col=1)
        fig.add_shape(type="rect", x0=.2, x1=.8, y0=0, y1=float(np.clip(L, 0, 100)), line=dict(width=0),
                      fillcolor=fill, row=1, col=1)
        for yy, txt, c in ((cfg["level_high"], "HIGH", "#f43f5e"), (cfg["level_warn"], "WARN", "#fbbf24"),
                           (cfg["level_low"], "LOW", "#f43f5e")):
            fig.add_shape(type="line", x0=.15, x1=.85, y0=yy, y1=yy, line=dict(color=c, width=1, dash="dash"), row=1, col=1)
            fig.add_annotation(x=.92, y=yy, text=txt, showarrow=False, font=dict(size=9, color=c), row=1, col=1)
        fig.add_annotation(x=.5, y=min(max(L, 8), 92) , text=f"<b>{L:.1f}</b>", showarrow=False,
                           font=dict(size=22, color="#fff"), row=1, col=1)
        fig.add_shape(type="line", x0=.5, x1=.5, y0=0, y1=-14, line=dict(color="#587089", width=3), row=1, col=1)
        fig.add_trace(go.Scatter(x=[.5], y=[-20], mode="markers+text", text=["OPEN" if V else "CLOSED"],
                                 textposition="bottom center", textfont=dict(size=11, color="#d6e2ee"),
                                 marker=dict(size=24, symbol="diamond", color=COLORS["Normal"] if V else COLORS["Critical"]),
                                 showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=1)
        fig.update_yaxes(visible=False, range=[-36, 108], row=1, col=1)
        gauge(fig, 2, P, "Pressure (PSI)", (0, 70), [(0, cfg["pressure_low"], AMB_D), (cfg["pressure_low"], cfg["pressure_high"], GRN_D),
                                                    (cfg["pressure_high"], 70, RED_D)], cfg["pressure_high"])
        gauge(fig, 3, F, "Flow rate", (0, 25), [(0, cfg["flow_min"], AMB_D), (cfg["flow_min"], cfg["flow_max"], GRN_D),
                                                (cfg["flow_max"], 25, RED_D)])
    else:
        fig = make_subplots(rows=1, cols=3, specs=[[{"type": "indicator"}] * 3], horizontal_spacing=.05)
        gauge(fig, 1, r.voltage_kv, "Voltage (kV)", (9, 14), [(9, cfg["v_min"], AMB_D), (cfg["v_min"], cfg["v_max"], GRN_D),
                                                              (cfg["v_max"], 14, RED_D)], cfg["v_max"])
        gauge(fig, 2, r.current_a, "Current (A)", (0, 450), [(0, cfg["i_max"], GRN_D), (cfg["i_max"], 450, RED_D)], cfg["i_max"])
        gauge(fig, 3, r.power_mw, "Power (MW)", (0, 9), [(0, 9, GRN_D)])
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=330, margin=dict(l=10, r=10, t=30, b=10), font=dict(color="#c9d6e3"))
    return fig


def _set_pos(i):
    st.session_state["pos"] = int(i)


def insights():
    out = []
    n_inc = int(ei["max_severity"].isin(MEDIUM_UP).sum()) if len(ei) else 0
    if n_inc:
        top = ei.sort_values("max_risk" if "max_risk" in ei else "risk_score", ascending=False).iloc[0]
        out.append(f"<b>{n_inc}</b> incidents (Medium or higher) were raised from {len(res):,} samples. "
                   f"Highest risk: <b>{esc(top['attack_type'])}</b> at {top['start']:%H:%M:%S} (risk {int(top['risk_score'])}/100).")
    else:
        out.append(f"No incidents raised from {len(res):,} samples.")
    ev = evaluate(res)
    if ev is not None:
        g = ev.set_index("method")["recall"]
        out.append(f"Share of attack samples detected: AI alone <b>{g['AI only']:.0%}</b>, physics alone "
                   f"<b>{g['Physics only']:.0%}</b>, combined <b>{g['Combined (Medium+)']:.0%}</b>.")
        only_phys = int(((res.true_attack == 1) & ~res.ai_flag & (res.physics_severity >= 2)).sum())
        if only_phys:
            out.append(f"<b>{only_phys}</b> attack samples were invisible to the AI and caught only by the physics rules.")
        seg, summ = evaluate_incidents(res, ei)
        if summ["attacks"]:
            out.append(f"<b>{summ['detected']}</b> of {summ['attacks']} attacks opened an incident "
                       f"(mean delay {summ['mean_delay_s']:.0f} s), <b>{summ['correctly_named']}</b> were named correctly, "
                       f"and {summ['false_alarm_incidents']} incident(s) had no real attack behind them.")
    q = review_queue(res)
    if len(q):
        out.append(f"<b>{len(q)}</b> AI-only alerts (physics normal) went to the review queue as possible false positives.")
    return out


# ---------- pages ----------
def page_overview():
    header("Plant overview", "Live process view, active alarms and key findings", chips_common())
    worst = max(res.tail(60)["final_severity"], key=LEVELS.index)
    c = COLORS[worst]
    st.markdown(f'<div class="banner" style="background:{c}22;border-color:{c};color:{c}">● {STATUS[worst]} — '
                f'highest severity in the latest 60 samples: {worst}</div>', unsafe_allow_html=True)

    # replay position
    alerts = np.flatnonzero(res["final_severity"].map(LEVELS.index).to_numpy() >= 2)
    best = int(res["risk_score"].idxmax())
    if "pos" not in st.session_state or st.session_state["pos"] >= len(res):
        st.session_state["pos"] = best
    pos = st.session_state["pos"]
    prev_a, next_a = alerts[alerts < pos], alerts[alerts > pos]

    left, right = st.columns([1.75, 1])
    with left:
        section("Plant view · replay")
        b1, b2, b3, _ = st.columns([1, 1, 1, 2])
        b1.button("⏮ Highest risk", on_click=_set_pos, args=(best,), width="stretch")
        b2.button("◀ Previous alert", on_click=_set_pos, args=(prev_a[-1] if len(prev_a) else pos,),
                  disabled=not len(prev_a), width="stretch")
        b3.button("Next alert ▶", on_click=_set_pos, args=(next_a[0] if len(next_a) else pos,),
                  disabled=not len(next_a), width="stretch")
        st.slider("Replay position (sample)", 0, len(res) - 1, key="pos")
        r = res.iloc[st.session_state["pos"]]
        st.plotly_chart(mimic(r), width="stretch")
        sc = COLORS[r["final_severity"]]
        reason = r["alert_reason"] or "No rule violated, AI score below threshold."
        st.markdown(
            f'<div class="panel" style="border-left:4px solid {sc}"><span class="chip" style="color:{sc}">'
            f'{r["final_severity"].upper()}</span> <span class="chip">{r["timestamp"]:%Y-%m-%d %H:%M:%S}</span> '
            f'<span class="chip">risk {int(r["risk_score"])}/100</span> <span class="chip">AI score {r["ai_score"]:.2f}</span>'
            f'<div style="margin-top:8px;color:#9fb3c8;font-size:.88rem">{esc(reason)}</div></div>', unsafe_allow_html=True)
    with right:
        section("Active alarms · latest incidents")
        recent = ei[ei.max_severity.isin(MEDIUM_UP)].sort_values("start", ascending=False).head(6) if len(ei) else ei
        if len(recent) == 0:
            st.success("No active alarms.")
        for _, a in recent.iterrows():
            col = COLORS[a["max_severity"]]
            st.markdown(f'<div class="alarm" style="--c:{col}"><div style="flex:1"><div class="a1">{esc(a["attack_type"])}</div>'
                        f'<div class="a2">{a["start"]:%H:%M:%S} · {int(a["duration_s"])} s · {a["max_severity"]}</div>'
                        f'<div class="bar"><div style="width:{int(a["risk_score"])}%"></div></div></div>'
                        f'<div class="n">{int(a["risk_score"])}</div></div>', unsafe_allow_html=True)
        q = review_queue(res)
        st.markdown(f'<div class="insight">Review queue: <b>{len(q)}</b> AI-only alerts waiting for an analyst</div>',
                    unsafe_allow_html=True)

    section("Key numbers")
    phys_only = int(((res.true_attack == 1) & ~res.ai_flag & (res.physics_severity >= 2)).sum()) if "true_attack" in res else None
    tiles([("Samples", f"{len(res):,}", "process readings", INFO),
           ("Incidents", int(ei["max_severity"].isin(MEDIUM_UP).sum()) if len(ei) else 0, "Medium or higher", AMBER),
           ("Critical samples", int((res.final_severity == "Critical").sum()), "need immediate review", COLORS["Critical"]),
           ("Highest risk", int(ei["risk_score"].max()) if len(ei) else 0, "out of 100", COLORS["High"]),
           ("Physics-only catches", phys_only if phys_only is not None else "n/a", "attack samples the AI missed", "#a78bfa"),
           ("Review queue", len(review_queue(res)), "AI-only alerts", COLORS["Low"])])

    section("Key findings")
    for line in insights():
        st.markdown(f'<div class="insight">{line}</div>', unsafe_allow_html=True)
    if TANK and not stress:
        st.info("On Module 1's data alone, AI and physics both detect its simple attacks. Tick "
                "**Add Module 3 stress-test attacks** in the sidebar to see what the physics rules add.")

    section("Alert activity")
    l, rr = st.columns([1, 1.6])
    with l:
        cts = res["final_severity"].value_counts().reindex(LEVELS).dropna().reset_index()
        cts.columns = ["severity", "samples"]
        st.plotly_chart(style(px.pie(cts, names="severity", values="samples", hole=.6, color="severity",
                                     color_discrete_map=COLORS, title="Samples by severity"), 320), width="stretch")
    with rr:
        tmp = res[res.final_severity != "Normal"].assign(minute=lambda d: d["timestamp"].dt.floor("min"))
        if len(tmp):
            rate = tmp.groupby(["minute", "final_severity"]).size().reset_index(name="alerts")
            st.plotly_chart(style(px.bar(rate, x="minute", y="alerts", color="final_severity", color_discrete_map=COLORS,
                                         category_orders={"final_severity": LEVELS}, title="Alerts per minute"), 320),
                            width="stretch")
        else:
            st.info("No alerts to chart.")

    section("Pipeline connection check")
    chk = pd.DataFrame([{"": "✅" if ok else "❌", "Check": cname, "Detail": d} for cname, ok, d in itg.compatibility_report(tank=TANK)])
    st.dataframe(chk, width="stretch", hide_index=True)


def page_data():
    header("Module 1 · Data collection & preprocessing", "Dataset quality, class balance, features and documentation", chips_common())
    names = {"clean": "Cleaned dataset", "train": "Train split", "test": "Test split"}
    key = st.radio("Dataset", list(names), format_func=names.get, horizontal=True)
    df = load_df(key)
    if df is None:
        st.error(f"{itg.PATHS[key]} not found. Run git pull to get Module 1 files."); return
    lab = itg.find_label(df)
    miss, dup = int(df.isna().sum().sum()), int(df.duplicated().sum())
    quality = max(0, 100 - 5 * (miss > 0) - 5 * (dup > 0) - (0 if lab else 20))
    tiles([("Rows", f"{len(df):,}", names[key], INFO), ("Columns", df.shape[1], f"label: {lab or 'not found'}", INFO),
           ("Missing cells", miss, "after cleaning", AMBER), ("Duplicate rows", dup, "should be 0", COLORS["High"]),
           ("Attack share", f"{itg.is_attack(df[lab]).mean():.1%}" if lab else "n/a", "of rows", COLORS["Critical"]),
           ("Data quality", f"{quality}/100", "no gaps, no duplicates, labelled", COLORS["Normal"])])
    st.write("")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Class balance", "Features", "Time view", "Correlation", "Dictionary & report", "Preview"])
    num = [c for c in df.select_dtypes("number").columns if c != lab]
    with t1:
        if lab:
            vc = df[lab].astype(str).value_counts().reset_index(); vc.columns = [lab, "count"]
            a, b = st.columns(2)
            a.plotly_chart(style(px.bar(vc, x=lab, y="count", color=lab), 330), width="stretch")
            b.plotly_chart(style(px.pie(vc, names=lab, values="count", hole=.55), 330), width="stretch")
            if len(vc) == 2:
                st.caption(f"Imbalance ratio about {vc['count'].max() / max(1, vc['count'].min()):.1f} : 1 — "
                           "use precision/recall/F1, not accuracy alone.")
        else:
            st.info("No label column detected.")
    with t2:
        if num:
            f = st.selectbox("Feature", num)
            s = df.sample(min(len(df), 20000), random_state=0)
            cl = s[lab].astype(str) if lab else None
            a, b = st.columns(2)
            a.plotly_chart(style(px.histogram(s, x=f, color=cl, nbins=50, barmode="overlay", opacity=.65, title="Distribution"), 340), width="stretch")
            b.plotly_chart(style(px.box(s, x=cl, y=f, color=cl, title="Spread by class") if lab else px.box(s, y=f), 340), width="stretch")
            if lab:
                stats = df.assign(_cls=df[lab].astype(str)).groupby("_cls")[num].agg(["mean", "std", "min", "max"]).T
                st.markdown("**Operating ranges by class** (these inform the physics limits in Module 3)")
                st.dataframe(stats.round(2), width="stretch")
            else:
                st.dataframe(df[num].describe().T, width="stretch")
    with t3:
        tcol = next((c for c in df.columns if c.lower() == "timestamp"), None)
        if tcol and num:
            f = st.selectbox("Signal", num, key="tv")
            d2 = df.assign(**{tcol: pd.to_datetime(df[tcol])})
            fig = px.line(d2, x=tcol, y=f)
            fig.update_traces(line=dict(color="#587089", width=1))
            if lab:
                a_ = d2[itg.is_attack(d2[lab])]
                fig.add_trace(go.Scatter(x=a_[tcol], y=a_[f], mode="markers", name="attack", marker=dict(color=COLORS["Critical"], size=5)))
            st.plotly_chart(style(fig, 360), width="stretch")
        else:
            st.info("No timestamp column.")
    with t4:
        n_ = df.select_dtypes("number")
        if n_.shape[1] > 1:
            cols = n_.var().sort_values(ascending=False).head(12).index
            st.plotly_chart(style(px.imshow(n_[cols].corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                            text_auto=".2f", aspect="auto"), 480), width="stretch")
    with t5:
        dd = itg.read_csv("dictionary")
        q = st.text_input("Search data dictionary")
        if dd is not None:
            v = dd[dd.astype(str).apply(lambda r: q.lower() in " ".join(r).lower(), axis=1)] if q else dd
            st.dataframe(v, width="stretch", hide_index=True)
            st.download_button("Download dictionary", v.to_csv(index=False), "data_dictionary.csv")
        else:
            st.warning("data_dictionary.csv not found")
        with st.expander("Preprocessing report", expanded=True):
            st.code(itg.read_text("prep"), language="text")
    with t6:
        st.dataframe(df.head(300), width="stretch")
        st.download_button("Download this dataset (CSV)", df.to_csv(index=False), f"{key}.csv")


@st.cache_data(show_spinner="Scoring traffic with the Random Forest...")
def eval_model(split_):
    df = itg.read_csv(split_)
    if df is None:
        return None
    lab = itg.find_label(df)
    model = itg.load_model()
    feats = itg.model_features(model, df, lab)
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    meta = {"Model": type(est).__name__, "Features": len(feats)}
    for a in ("n_estimators", "max_depth", "criterion", "class_weight"):
        if hasattr(est, a):
            meta[a] = str(getattr(est, a))
    imp = est.feature_importances_.tolist() if hasattr(est, "feature_importances_") and \
        len(est.feature_importances_) == len(feats) else None
    return dict(y=itg.is_attack(df[lab]).astype(int).to_numpy() if lab else None,
                score=np.asarray(itg.score_traffic(model, df, lab)), feats=feats, imp=imp, meta=meta, label=lab)


def page_model():
    header("Module 2 · AI-based intrusion detection", "Random Forest performance, thresholds and blind spots", chips_common())
    sp = st.radio("Evaluate on", ["test", "train"], horizontal=True)
    try:
        E = eval_model(sp)
    except Exception as e:
        st.error(f"Could not load model / data: {e}"); return
    if E is None:
        st.error("Dataset not found. Run git pull."); return
    if E["y"] is None:
        st.warning("No label column found, cannot compute metrics."); return
    y, score = E["y"], E["score"]
    t = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05)
    m = metrics_at(y, score, t)
    auc = None
    try:
        from sklearn.metrics import roc_curve, roc_auc_score
        auc = roc_auc_score(y, score)
    except Exception:
        pass
    tiles([("Accuracy", f"{m['accuracy']:.3f}", "all samples", INFO), ("Precision", f"{m['precision']:.3f}", "alerts that were real", INFO),
           ("Recall", f"{m['recall']:.3f}", "attacks caught", COLORS["Normal"]), ("F1-score", f"{m['f1']:.3f}", "balance", AMBER),
           ("ROC AUC", f"{auc:.3f}" if auc else "n/a", "ranking quality", "#a78bfa"),
           ("Errors", f"{m['fp']} FP / {m['fn']} FN", "false alarms / misses", COLORS["High"])])
    st.write("")
    t1, t2, t3, t4, t5 = st.tabs(["Performance", "Thresholds & scores", "What the model uses", "Blind spots", "Try it · report"])
    with t1:
        a, b = st.columns(2)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        a.plotly_chart(style(px.imshow(cm, x=["Pred normal", "Pred attack"], y=["True normal", "True attack"], text_auto=True,
                                       color_continuous_scale="Blues", title="Confusion matrix"), 340), width="stretch")
        if auc:
            fpr, tpr, _ = roc_curve(y, score)
            f = go.Figure(go.Scatter(x=fpr, y=tpr, name="ROC", line=dict(color=AMBER)))
            f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="chance", line=dict(dash="dash", color="#587089")))
            f.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
            b.plotly_chart(style(f, 340), width="stretch")
        st.markdown("**Model card**  \n" + "  \n".join(f"- {k}: `{v}`" for k, v in E["meta"].items()))
    with t2:
        c, d = st.columns(2)
        sw = pd.DataFrame([{"threshold": x, **{k: metrics_at(y, score, x)[k] for k in ("precision", "recall", "f1")}}
                           for x in np.round(np.arange(.05, .96, .05), 2)])
        c.plotly_chart(style(px.line(sw, x="threshold", y=["precision", "recall", "f1"], title="Threshold trade-off"), 340), width="stretch")
        h = pd.DataFrame({"score": score, "class": np.where(y == 1, "attack", "normal")})
        d.plotly_chart(style(px.histogram(h, x="score", color="class", nbins=40, barmode="overlay", opacity=.65, title="Score distribution",
                                          color_discrete_map={"attack": COLORS["Critical"], "normal": COLORS["Normal"]}), 340), width="stretch")
    with t3:
        if E["imp"]:
            fi = pd.DataFrame({"feature": E["feats"], "importance": E["imp"]}).sort_values("importance")
            st.plotly_chart(style(px.bar(fi.tail(15), x="importance", y="feature", orientation="h", title="Feature importance"), 380), width="stretch")
            st.caption("The model only sees these columns. Anything outside them, such as sensor consistency or "
                       "valve/flow relationships, is invisible to it and is covered by Module 3's physics rules.")
        else:
            st.info("This model does not expose feature importances.")
    with t4:
        ba = evaluate_by_attack(res)
        if ba is not None and len(ba) > 1:
            mm = ba.melt(id_vars=["attack_type", "samples"], value_vars=["AI only", "Physics only", "Combined (Medium+)"])
            st.plotly_chart(style(px.bar(mm, x="attack_type", y="value", color="variable", barmode="group",
                                         title="Share of each attack type detected (current Module 3 data)"), 380), width="stretch")
            st.caption("Tick the stress-test option in the sidebar to add attack types the model cannot see.")
        else:
            st.info("Only one attack type in the current data, so there is no per-type comparison. "
                    "Enable **Add Module 3 stress-test attacks** in the sidebar.")
    with t5:
        if st.button("Pick a random row"):
            st.session_state["rid"] = int(np.random.default_rng().integers(len(y)))
        rid = st.session_state.get("rid")
        if rid is not None and rid < len(y):
            pred = "ATTACK" if score[rid] >= t else "NORMAL"
            truth = "attack" if y[rid] else "normal"
            ok = (pred == "ATTACK") == bool(y[rid])
            st.markdown(f"Row **{rid}** → model score **{score[rid]:.3f}** → **{pred}** · true label: **{truth}** "
                        f"{'✅ correct' if ok else '❌ wrong'}")
            st.dataframe(load_df(sp).iloc[[rid]], width="stretch")
        with st.expander("Evaluation report from Module 2", expanded=True):
            st.code(itg.read_text("eval"), language="text")


def page_alerts():
    header("Module 3 · Physics-aware analysis & alerts", "Rule violations, AI+physics fusion, incidents and response", chips_common())
    sev_n = res["final_severity"].value_counts()
    tiles([(l, int(sev_n.get(l, 0)), "samples", COLORS[l]) for l in LEVELS[::-1]])
    st.write("")
    t1, t2, t3, t4, t5, t6 = st.tabs(["Live timeline", "Incidents & response", "Attacks vs detection", "Physics rules",
                                      "Fusion & evaluation", "Review queue"])

    with t1:
        ts = res["timestamp"]
        lo, hi = st.slider("Time window", ts.min().to_pydatetime(), ts.max().to_pydatetime(),
                           (ts.min().to_pydatetime(), ts.max().to_pydatetime()))
        w = res[(ts >= lo) & (ts <= hi)]
        sig = (tk.SIGNALS if TANK else [("current_a", "Current (A)"), ("voltage_kv", "Voltage (kV)"),
                                        ("power_mw", "Power (MW)"), ("transformer_temp_c", "Temp (°C)")]) + [("ai_score", "AI score")]
        fig = make_subplots(rows=len(sig), cols=1, shared_xaxes=True, vertical_spacing=.025, subplot_titles=[s[1] for s in sig])
        for i, (c, _) in enumerate(sig, 1):
            fig.add_trace(go.Scatter(x=w.timestamp, y=w[c], line=dict(color="#587089", width=1), showlegend=False), row=i, col=1)
            for lvl in ["Low", "Medium", "High", "Critical"]:
                s = w[w.final_severity == lvl]
                if len(s):
                    fig.add_trace(go.Scatter(x=s.timestamp, y=s[c], mode="markers", name=lvl, legendgroup=lvl, showlegend=(i == 1),
                                             marker=dict(color=COLORS[lvl], size=4), text=s.alert_reason, hoverinfo="text+x+y"), row=i, col=1)
        oc, ec, on = tk.OVERLAY if TANK else ("power_mw", "power_calc_mw", "V·I·pf (expected)")
        fig.add_trace(go.Scatter(x=w.timestamp, y=w[ec], line=dict(color=AMBER, width=1, dash="dot"), name=on),
                      row=[s[0] for s in sig].index(oc) + 1, col=1)
        fig.add_hline(y=thr, line_dash="dash", line_color="#587089", row=len(sig), col=1)
        for _, r in ei[ei.max_severity.isin(MEDIUM_UP)].iterrows():
            if r["end"] >= lo and r["start"] <= hi:
                fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor=COLORS[r["max_severity"]], opacity=.13, line_width=0, row="all", col=1)
        st.plotly_chart(style(fig, 170 * len(sig)), width="stretch")
        st.caption("Shaded bands = Medium+ incidents. Dotted line = value the physics expects (power from V×I×pf, or pressure from tank level).")

    with t2:
        if ei.empty:
            st.success("No incidents.")
        else:
            show = st.multiselect("Severity", LEVELS[1:], default=MEDIUM_UP)
            v = ei[ei.max_severity.isin(show)].sort_values("risk_score", ascending=False)
            st.dataframe(v[["incident", "start", "end", "duration_s", "max_severity", "attack_type", "risk_score", "mitre_ics"]],
                         width="stretch", hide_index=True,
                         column_config={"risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d")})
            if len(v):
                pick = st.selectbox("Inspect incident", v["incident"].tolist(),
                                    format_func=lambda i: f"Incident {i} · " + v.loc[v.incident == i, "attack_type"].iloc[0])
                r = v[v.incident == pick].iloc[0]
                a, b = st.columns([1, 1.4])
                a.markdown(f"**{r['attack_type']}**  \nRisk **{r['risk_score']}/100** · {r['max_severity']}  \n"
                           f"MITRE ICS: {r['mitre_ics']}  \nRules: {r['rules']}")
                b.info(f"**Evidence:** {r['evidence']}\n\n**Recommended response:** {r['recommended_action']}")
            st.download_button("⬇ Download incident report (.md)", ia.generate_report(res, ei, src, thr), "voltguard_incident_report.md")
        with st.expander("Alert log"):
            sev = st.multiselect("Filter", LEVELS[1:], default=MEDIUM_UP, key="logsev")
            log = res[res.final_severity.isin(sev)][["timestamp", "final_severity", "ai_score", "physics_severity", "alert_reason"]]
            st.dataframe(log, width="stretch", hide_index=True)
            st.download_button("Download alerts CSV", res.to_csv(index=False), "alerts.csv")

    with t3:
        if "true_attack" not in res:
            st.info("This data has no ground-truth labels.")
        else:
            seg, summ = evaluate_incidents(res, ei)
            tiles([("True attacks", summ["attacks"], "contiguous attack segments", COLORS["Critical"]),
                   ("Detected", summ["detected"], "opened an incident", COLORS["Normal"]),
                   ("Named correctly", summ["correctly_named"], "attack type matched", INFO),
                   ("Mean delay", f"{summ['mean_delay_s']:.0f} s", "to first alert", AMBER),
                   ("False-alarm incidents", summ["false_alarm_incidents"], "no real attack", COLORS["High"])])
            if len(seg):
                span = max(60.0, (res.timestamp.max() - res.timestamp.min()).total_seconds())
                pad = pd.Timedelta(seconds=max(1.0, span / 250))
                gt = pd.DataFrame({"group": "True attacks", "label": seg["attack"], "Start": seg["start"], "Finish": seg["end"] + pad})
                parts = [gt]
                if len(ei):
                    parts.append(pd.DataFrame({"group": "Detected incidents", "label": ei["likely_attack"], "Start": ei["start"],
                                               "Finish": ei["end"] + pad}))
                tl = pd.concat(parts)
                st.plotly_chart(style(px.timeline(tl, x_start="Start", x_end="Finish", y="group", color="label",
                                                  title="Ground truth vs what Module 3 detected"), 300), width="stretch")
                st.dataframe(seg, width="stretch", hide_index=True)

    with t4:
        rows = []
        has_truth = "true_attack" in res
        for rid, (sv, desc, *_) in RULES_ACTIVE.items():
            m_ = res["rules_triggered"].str.contains(rid)
            row = {"rule": rid, "severity": sv, "threshold": THRESH_ACTIVE.get(rid, ""), "description": desc, "violations": int(m_.sum())}
            if has_truth:
                row["on normal rows (false alarms)"] = int((m_ & (res.true_attack == 0)).sum())
                row["on attack rows (caught)"] = int((m_ & (res.true_attack == 1)).sum())
            rows.append(row)
        rt = pd.DataFrame(rows)
        st.dataframe(rt, width="stretch", hide_index=True)
        if has_truth:
            mm = rt.melt(id_vars="rule", value_vars=["on normal rows (false alarms)", "on attack rows (caught)"])
            st.plotly_chart(style(px.bar(mm, x="value", y="rule", color="variable", orientation="h", barmode="group",
                                         color_discrete_sequence=[COLORS["High"], COLORS["Normal"]], title="Evidence per rule: false alarms vs real catches"), 420),
                            width="stretch")
        else:
            st.plotly_chart(style(px.bar(rt, x="violations", y="rule", orientation="h", color="severity", color_continuous_scale="OrRd"), 380), width="stretch")
        st.caption("Change limits in the sidebar (Physics thresholds) to see how the alerts react.")

    with t5:
        st.dataframe(fusion_table(), width="stretch")
        ev = evaluate(res)
        if ev is None:
            st.info("No ground-truth column (true_attack) in this data.")
        else:
            st.dataframe(ev, width="stretch", hide_index=True)
            mm = ev.melt(id_vars="method", value_vars=["precision", "recall", "f1"])
            a, b = st.columns(2)
            a.plotly_chart(style(px.bar(mm, x="method", y="value", color="variable", barmode="group", title="Per-sample metrics"), 340), width="stretch")
            sw = threshold_sweep(res)
            b.plotly_chart(style(px.line(sw, x="threshold", y="f1", color="method", title="F1 vs AI threshold"), 340), width="stretch")
            st.caption("Any alert = physics Medium+ or AI flag; High and above needs physics support. AI-only flags appear as Low alerts for review.")

    with t6:
        q = review_queue(res)
        tiles([("Review queue", len(q), "AI flagged, physics normal", COLORS["Low"])])
        st.write("")
        if len(q):
            st.dataframe(q, width="stretch", hide_index=True)
            st.download_button("Download review queue (CSV)", q.to_csv(index=False), "review_queue.csv")
            st.caption("These alerts have no physical violation behind them. An analyst should confirm or dismiss them; "
                       "repeated entries suggest raising the AI threshold.")
        else:
            st.success("Nothing to review: every AI alert is backed by a physics violation.")


def page_about():
    header("About & method", "How the three modules connect, how an alert is decided, and what the limits are", chips_common())
    section("System architecture")
    dot = '''digraph G { rankdir=LR; bgcolor="transparent"; nodesep=.5; ranksep=.6;
      node [shape=box, style="rounded,filled", fillcolor="#16222d", fontcolor="#d6e2ee", color="#223344", fontname="Helvetica", margin="0.2,0.12"];
      edge [color="#587089", fontcolor="#7f93a7", fontname="Helvetica", fontsize=10];
      M1 [label="Module 1\\nData & preprocessing"]; M2 [label="Module 2\\nRandom Forest"];
      PH [label="Module 3\\nPhysics rules", color="#ffb020"]; FU [label="Fusion\\nseverity matrix", color="#ffb020"];
      IN [label="Incidents\\nattack type · risk\\nresponse steps", color="#ffb020"]; UI [label="Dashboard\\n+ report", color="#ffb020"];
      M1 -> M2 [label="features"]; M1 -> PH [label="process readings"]; M2 -> FU [label="ai_score"];
      PH -> FU [label="rule hits"]; FU -> IN; IN -> UI; }'''
    st.graphviz_chart(dot)
    a, b = st.columns(2)
    with a:
        section("How an alert is decided")
        st.dataframe(fusion_table(), width="stretch")
        st.caption("Physics evidence decides the base severity. An AI flag raises it one step, and an AI flag alone is only a Low alert sent to review.")
    with b:
        section("Risk score (0-100)")
        st.markdown("- Base from severity: Low 20, Medium 40, High 65, Critical 85  \n- plus up to 10 for AI confidence  \n"
                    "- Incidents take the peak risk of their samples  \n- Attack type is inferred from which rules fired, and mapped to MITRE ATT&CK for ICS")
    section("Physics rules in use")
    rr = pd.DataFrame([{"rule": k, "severity": v[0], "threshold": THRESH_ACTIVE.get(k, ""), "what it checks": v[1],
                        "why": v[2] if len(v) > 2 else ""} for k, v in RULES_ACTIVE.items()])
    st.dataframe(rr, width="stretch", hide_index=True)
    section("Data lineage")
    lin = pd.DataFrame([{"": "✅" if p.exists() else "❌", "item": k, "path": str(p)} for k, p in itg.PATHS.items()])
    st.dataframe(lin, width="stretch", hide_index=True)

    if TANK:
        section("Self-test of the physics layer")
        st.caption("Injects the four stress-test attacks into Module 1's data, runs the full pipeline, and checks each one is detected and named.")
        if st.button("Run self-test"):
            d2, _ = td.load(split, True, seed)
            r2 = run_pipeline(d2, thr, dict(cfg), profile=tk)
            e2 = ia.enrich_incidents(build_incidents(r2), r2)
            seg, summ = evaluate_incidents(r2, e2)
            stress_seg = seg[seg["attack"].isin(td.STRESS_KEYS)]
            out = stress_seg.assign(result=np.where(stress_seg["detected"] & stress_seg["named_correctly"], "✅ detected + named",
                                                    np.where(stress_seg["detected"], "⚠️ detected, named differently", "❌ missed")))
            st.dataframe(out[["attack", "start", "end", "likely_attack", "result"]], width="stretch", hide_index=True)
            ok = bool(len(out)) and bool(stress_seg["detected"].all())
            (st.success if ok else st.error)("All stress-test attacks were detected." if ok else "At least one stress-test attack was missed.")
    section("Limitations")
    st.markdown("- This is an academic prototype on simulated/offline data, not a validated industrial safety product.  \n"
                "- Module 1's samples are independent random draws, so smooth-dynamics checks (rate of change) are not used.  \n"
                "- Stress-test attacks are authored by Module 3 to evaluate the physics layer; they are labelled and optional.  \n"
                "- Attack types are inferred from rule signatures and need analyst review.  \n"
                "- Do not connect experimental detection code to live control equipment without formal authorization.")
    section("Downloads")
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Incident report (.md)", ia.generate_report(res, ei, src, thr), "voltguard_incident_report.md", width="stretch")
    c2.download_button("⬇ Alerts (.csv)", res.to_csv(index=False), "alerts.csv", width="stretch")


pg = st.navigation([
    st.Page(page_overview, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(page_data, title="Module 1 · Data", icon="📊", url_path="data"),
    st.Page(page_model, title="Module 2 · Model", icon="🤖", url_path="model"),
    st.Page(page_alerts, title="Module 3 · Alerts", icon="🚨", url_path="alerts"),
    st.Page(page_about, title="About & Method", icon="📘", url_path="about"),
], position="top")
pg.run()
