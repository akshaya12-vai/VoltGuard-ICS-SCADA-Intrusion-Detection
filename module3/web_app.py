"""VoltGuard web dashboard (Modules 1-3).  Run from module3/:  streamlit run web_app.py"""
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

st.set_page_config(page_title="VoltGuard", page_icon="⚡", layout="wide")

try:
    from simulate_process import simulate
    from alert_engine import run_pipeline, build_incidents, evaluate, LEVELS
    from physics_rules import RULES, CONFIG
    import integration as itg
    import incident_analysis as ia
except ImportError as e:
    st.error(f"Missing module: {e}. Copy integration.py and incident_analysis.py into module3/.")
    st.stop()

ACCENT = "#22d3ee"
COLORS = {"Normal": "#22c55e", "Low": "#38bdf8", "Medium": "#eab308",
          "High": "#f97316", "Critical": "#ef4444"}
SRC_M2 = "Module 2 output (agreed 11 kV dataset)"
SRC_SIM = "Simulated (mock AI)"
SRC_UP = "Upload CSV"

st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1320px}
.vg-hero{background:linear-gradient(120deg,#0e7490 0%,#312e81 100%);border:1px solid #1f2a44;
 border-radius:18px;padding:26px 32px;margin-bottom:18px}
.vg-hero .tag{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;color:#a5f3fc}
.vg-hero h1{margin:4px 0 0;font-size:2rem;color:#fff}
.vg-hero p{margin:6px 0 0;color:#cbd5e1}
.vg-card{background:#111a2e;border:1px solid #1f2a44;border-radius:14px;padding:14px 16px;height:100%}
.vg-k{font-size:.72rem;letter-spacing:.09em;text-transform:uppercase;color:#94a3b8}
.vg-v{font-size:1.8rem;font-weight:700;line-height:1.25}
.vg-s{font-size:.8rem;color:#64748b}
.vg-banner{border-radius:14px;padding:14px 20px;margin-bottom:14px;font-weight:600;border:1px solid}
.vg-mod h4{margin:0 0 6px}.vg-mod p{color:#94a3b8;font-size:.88rem;margin:0 0 8px}
div[data-testid="stTabs"] button{font-weight:600}
</style>""", unsafe_allow_html=True)


# ---------- helpers ----------
def hero(title, sub, tag):
    st.markdown(f'<div class="vg-hero"><div class="tag">{tag}</div><h1>{title}</h1><p>{sub}</p></div>',
                unsafe_allow_html=True)


def cards(items):
    for col, (k, v, s, *c) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="vg-card"><div class="vg-k">{k}</div>'
                     f'<div class="vg-v" style="color:{c[0] if c else ACCENT}">{v}</div>'
                     f'<div class="vg-s">{s}</div></div>', unsafe_allow_html=True)


def style(fig, h=340):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=36, b=10), legend=dict(orientation="h", y=1.12))
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
def compute(src, thr, seed, split, cfg_items, up_bytes):
    info = {}
    try:
        if src == SRC_M2:
            df = pd.read_csv(itg.PATHS["m2out"], parse_dates=["timestamp"])
            check_schema(df)
        elif src == SRC_UP and up_bytes:
            df = pd.read_csv(io.BytesIO(up_bytes), parse_dates=["timestamp"])
            check_schema(df)
        else:
            df = simulate(seed=seed)
    except Exception as e:  # missing teammate files etc.
        df, info = simulate(seed=seed), {"fallback": str(e)}
    try:
        res = run_pipeline(df, thr, dict(cfg_items))
    except TypeError:
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
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=pr, recall=rc, f1=f1,
                accuracy=(tp + tn) / max(1, len(y)))


# ---------- sidebar / state ----------
with st.sidebar:
    st.markdown("### ⚡ VoltGuard")
    st.caption("Physics-aware ICS/SCADA intrusion detection")
    src = st.radio("Data source", [SRC_M2, SRC_SIM, SRC_UP], key="src")
    split, up_bytes = "test", None
    if src == SRC_UP:
        up = st.file_uploader("CSV with process columns + ai_score", type="csv")
        up_bytes = up.getvalue() if up else None
        if not up:
            st.info("No file yet, showing simulated data.")
    thr = st.slider("AI score threshold", 0.1, 0.95, 0.5, 0.05, key="thr")
    seed = int(st.number_input("Random seed", 0, 9999, 42, key="seed"))
    with st.expander("Physics thresholds (what-if)"):
        tol = st.slider("Voltage tolerance ±%", 1.0, 10.0, 5.0, 0.5)
        imax = st.slider("Current limit (A)", 250, 600, int(CONFIG["i_max"]), 10)
        tmax = st.slider("Temperature limit (°C)", 60, 110, int(CONFIG["temp_max"]))
        ptol = st.slider("Power mismatch tolerance (MW)", 0.1, 1.5, float(CONFIG["power_tol_mw"]), 0.05)
    cfg = dict(CONFIG)
    cfg.update(v_min=11 * (1 - tol / 100), v_max=11 * (1 + tol / 100), i_max=float(imax),
               temp_max=float(tmax), power_tol_mw=float(ptol))

res, ei, info = compute(src, thr, seed, split, tuple(sorted(cfg.items())), up_bytes)
if "fallback" in info:
    st.sidebar.warning(f"Module 1/2 files not usable, using simulation. ({info['fallback'][:90]})")


# ---------- pages ----------
def page_overview():
    hero("VoltGuard", "AI traffic detection + physical-rule validation for industrial control systems",
         "Overview")
    worst = max(res.tail(60)["final_severity"], key=LEVELS.index)
    txt, col = {"Normal": ("ALL CLEAR", "#22c55e"), "Low": ("ALL CLEAR", "#22c55e"),
                "Medium": ("WARNING", "#eab308"), "High": ("ALERT", "#f97316"),
                "Critical": ("CRITICAL ALERT", "#ef4444")}[worst]
    st.markdown(f'<div class="vg-banner" style="background:{col}22;border-color:{col};color:{col}">'
                f'● {txt} — highest severity in the latest 60 samples: {worst}</div>', unsafe_allow_html=True)
    phys_only = None
    if "true_attack" in res:
        phys_only = int(((res.true_attack == 1) & ~res.ai_flag & (res.physics_severity >= 2)).sum())
    cards([("Samples", f"{len(res):,}", "process readings"),
           ("Incidents (Medium+)", int(ei["max_severity"].isin(["Medium", "High", "Critical"]).sum()) if len(ei) else 0,
            "grouped alerts"),
           ("Critical samples", int((res.final_severity == "Critical").sum()), "need immediate review", "#ef4444"),
           ("Highest risk", int(ei["risk_score"].max()) if len(ei) else 0, "out of 100", "#f97316"),
           ("Caught only by physics", phys_only if phys_only is not None else "n/a",
            "attack samples the AI missed", "#a78bfa")])
    st.write("")
    ok1 = all(itg.PATHS[k].exists() for k in ("clean", "train", "test", "dictionary"))
    ok2 = itg.PATHS["model"].exists()
    mods = [("Module 1 · Data", "Dataset collection, cleaning, features, data dictionary.", ok1),
            ("Module 2 · AI detection", "Random Forest classifier: normal vs suspicious traffic.", ok2),
            ("Module 3 · Physics & alerts", "Physical rules, AI+physics fusion, incidents, dashboard.", True)]
    for col_, (t, d, ok) in zip(st.columns(3), mods):
        col_.markdown(f'<div class="vg-card vg-mod"><h4>{t}</h4><p>{d}</p>'
                      f'<span style="color:{"#22c55e" if ok else "#eab308"}">'
                      f'{"● files found" if ok else "● files missing (git pull)"}</span></div>',
                      unsafe_allow_html=True)
    st.write("")
    l, r = st.columns([1, 1.4])
    with l:
        st.subheader("Severity distribution")
        c = res["final_severity"].value_counts().reindex(LEVELS).dropna().reset_index()
        c.columns = ["severity", "samples"]
        st.plotly_chart(style(px.pie(c, names="severity", values="samples", hole=.55,
                                     color="severity", color_discrete_map=COLORS), 320), width="stretch")
    with r:
        st.subheader("Top incidents by risk")
        if len(ei):
            st.dataframe(ei.sort_values("risk_score", ascending=False).head(6)[
                ["incident", "start", "attack_type", "max_severity", "risk_score"]],
                width="stretch", hide_index=True,
                column_config={"risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d")})
        else:
            st.success("No incidents.")
    st.subheader("How an alert is decided")
    st.dataframe(fusion_table(), width="stretch")


def page_data():
    hero("Module 1 · Data collection & preprocessing", "Dataset quality, class balance, features and documentation", "Module 1")
    names = {"clean": "Cleaned dataset", "train": "Train split", "test": "Test split"}
    key = st.radio("Dataset", list(names), format_func=names.get, horizontal=True)
    df = load_df(key)
    if df is None:
        st.error(f"{itg.PATHS[key]} not found. Run git pull to get Module 1 files."); return
    lab = itg.find_label(df)
    cards([("Rows", f"{len(df):,}", names[key]), ("Columns", df.shape[1], f"label: {lab or 'not found'}"),
           ("Missing cells", int(df.isna().sum().sum()), "after cleaning", "#eab308"),
           ("Duplicate rows", int(df.duplicated().sum()), "should be 0", "#f97316"),
           ("Attack share", f"{itg.is_attack(df[lab]).mean():.1%}" if lab else "n/a", "of rows", "#ef4444")])
    t1, t2, t3, t4, t5 = st.tabs(["Class balance", "Features", "Correlation", "Dictionary & report", "Preview"])
    with t1:
        if lab:
            vc = df[lab].astype(str).value_counts().reset_index(); vc.columns = [lab, "count"]
            a, b = st.columns(2)
            a.plotly_chart(style(px.bar(vc, x=lab, y="count", color=lab), 330), width="stretch")
            b.plotly_chart(style(px.pie(vc, names=lab, values="count", hole=.5), 330), width="stretch")
        else:
            st.info("No label column detected.")
    with t2:
        num = [c for c in df.select_dtypes("number").columns if c != lab]
        if num:
            f = st.selectbox("Feature", num)
            s = df.sample(min(len(df), 20000), random_state=0)
            fig = px.histogram(s, x=f, color=s[lab].astype(str) if lab else None, nbins=50,
                               barmode="overlay", opacity=.65)
            st.plotly_chart(style(fig, 360), width="stretch")
            st.dataframe(df[num].describe().T, width="stretch")
    with t3:
        num = df.select_dtypes("number")
        if num.shape[1] > 1:
            cols = num.var().sort_values(ascending=False).head(12).index
            st.plotly_chart(style(px.imshow(num[cols].corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                            text_auto=".2f", aspect="auto"), 480), width="stretch")
    with t4:
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
    with t5:
        st.dataframe(df.head(300), width="stretch")


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
    hero("Module 2 · AI-based intrusion detection", "Random Forest performance on network traffic", "Module 2")
    sp = st.radio("Evaluate on", ["test", "train"], horizontal=True)
    try:
        E = eval_model(sp)
    except Exception as e:
        st.error(f"Could not load model / data: {e}"); return
    if E is None:
        st.error("Dataset not found. Run git pull."); return
    st.caption("  |  ".join(f"{k}: {v}" for k, v in E["meta"].items()))
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
    cards([("Accuracy", f"{m['accuracy']:.3f}", "all samples"), ("Precision", f"{m['precision']:.3f}", "alerts that were real"),
           ("Recall", f"{m['recall']:.3f}", "attacks caught", "#22c55e"), ("F1-score", f"{m['f1']:.3f}", "balance"),
           ("ROC AUC", f"{auc:.3f}" if auc else "n/a", "ranking quality", "#a78bfa")])
    st.write("")
    a, b = st.columns(2)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    a.plotly_chart(style(px.imshow(cm, x=["Pred normal", "Pred attack"], y=["True normal", "True attack"],
                                   text_auto=True, color_continuous_scale="Blues", title="Confusion matrix"), 340), width="stretch")
    if auc:
        fpr, tpr, _ = roc_curve(y, score)
        f = go.Figure(go.Scatter(x=fpr, y=tpr, name="ROC", line=dict(color=ACCENT)))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="chance", line=dict(dash="dash", color="#64748b")))
        f.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        b.plotly_chart(style(f, 340), width="stretch")
    c, d = st.columns(2)
    sw = pd.DataFrame([{"threshold": x, **{k: metrics_at(y, score, x)[k] for k in ("precision", "recall", "f1")}}
                       for x in np.round(np.arange(.05, .96, .05), 2)])
    c.plotly_chart(style(px.line(sw, x="threshold", y=["precision", "recall", "f1"], title="Threshold trade-off"), 320), width="stretch")
    h = pd.DataFrame({"score": score, "class": np.where(y == 1, "attack", "normal")})
    d.plotly_chart(style(px.histogram(h, x="score", color="class", nbins=40, barmode="overlay", opacity=.65,
                                      title="Score distribution", color_discrete_map={"attack": "#ef4444", "normal": "#22c55e"}), 320), width="stretch")
    if E["imp"]:
        fi = pd.DataFrame({"feature": E["feats"], "importance": E["imp"]}).sort_values("importance").tail(15)
        st.plotly_chart(style(px.bar(fi, x="importance", y="feature", orientation="h", title="Top feature importances"), 440), width="stretch")
    st.subheader("Try it: score a random traffic row")
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
    with st.expander("Evaluation report from Module 2"):
        st.code(itg.read_text("eval"), language="text")


def page_alerts():
    hero("Module 3 · Physics-aware analysis & alerts", "Rule violations, AI+physics fusion, incidents and response", "Module 3")
    sev_n = res["final_severity"].value_counts()
    cards([(l, int(sev_n.get(l, 0)), "samples", COLORS[l]) for l in LEVELS[::-1]])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Live timeline", "Incidents & response", "Physics rules", "Fusion & evaluation"])

    with t1:
        ts = res["timestamp"]
        lo, hi = st.slider("Time window", ts.min().to_pydatetime(), ts.max().to_pydatetime(),
                           (ts.min().to_pydatetime(), ts.max().to_pydatetime()))
        w = res[(ts >= lo) & (ts <= hi)]
        sig = [("current_a", "Current (A)"), ("voltage_kv", "Voltage (kV)"), ("power_mw", "Power (MW)"),
               ("transformer_temp_c", "Temp (°C)"), ("ai_score", "AI score")]
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=.025,
                            subplot_titles=[s[1] for s in sig])
        for i, (c, _) in enumerate(sig, 1):
            fig.add_trace(go.Scatter(x=w.timestamp, y=w[c], line=dict(color="#64748b", width=1), showlegend=False), row=i, col=1)
            for lvl in ["Low", "Medium", "High", "Critical"]:
                s = w[w.final_severity == lvl]
                if len(s) and (c != "ai_score" or lvl == "Low" or True):
                    fig.add_trace(go.Scatter(x=s.timestamp, y=s[c], mode="markers", name=lvl, legendgroup=lvl,
                                             showlegend=(i == 1), marker=dict(color=COLORS[lvl], size=4),
                                             text=s.alert_reason, hoverinfo="text+x+y"), row=i, col=1)
        fig.add_trace(go.Scatter(x=w.timestamp, y=w.power_calc_mw, line=dict(color=ACCENT, width=1, dash="dot"),
                                 name="V·I·pf (expected)"), row=3, col=1)
        fig.add_hline(y=thr, line_dash="dash", line_color="#64748b", row=5, col=1)
        for _, r in ei[ei.max_severity.isin(["Medium", "High", "Critical"])].iterrows():
            if r["end"] >= lo and r["start"] <= hi:
                fig.add_vrect(x0=r["start"], x1=r["end"], fillcolor=COLORS[r["max_severity"]], opacity=.13,
                              line_width=0, row="all", col=1)
        st.plotly_chart(style(fig, 820), width="stretch")
        st.caption("Shaded bands = Medium+ incidents. Dotted line on Power = value expected from voltage × current × power factor.")

    with t2:
        if ei.empty:
            st.success("No incidents.")
        else:
            show = st.multiselect("Severity", LEVELS[1:], default=["Medium", "High", "Critical"])
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
            st.download_button("⬇ Download incident report (.md)", ia.generate_report(res, ei, src, thr),
                               "voltguard_incident_report.md")
        with st.expander("Alert log"):
            sev = st.multiselect("Filter", LEVELS[1:], default=["Medium", "High", "Critical"], key="logsev")
            log = res[res.final_severity.isin(sev)][["timestamp", "final_severity", "ai_score", "physics_severity", "alert_reason"]]
            st.dataframe(log, width="stretch", hide_index=True)
            st.download_button("Download alerts CSV", res.to_csv(index=False), "alerts.csv")

    with t3:
        rows = []
        for rid, (sv, desc) in RULES.items():
            m_ = res["rules_triggered"].str.contains(rid)
            rows.append({"rule": rid, "severity (1-4)": sv, "description": desc, "violations": int(m_.sum()),
                         "first seen": res.loc[m_, "timestamp"].min() if m_.any() else None})
        rt = pd.DataFrame(rows)
        st.dataframe(rt, width="stretch", hide_index=True)
        st.plotly_chart(style(px.bar(rt, x="violations", y="rule", orientation="h", color="severity (1-4)",
                                     color_continuous_scale="OrRd"), 380), width="stretch")
        st.caption("Change limits in the sidebar (Physics thresholds) to see how the alerts react.")

    with t4:
        st.dataframe(fusion_table(), width="stretch")
        ev = evaluate(res)
        if ev is None:
            st.info("No ground-truth column (true_attack) in this data.")
        else:
            st.dataframe(ev, width="stretch", hide_index=True)
            m_ = ev.melt(id_vars="method", value_vars=["precision", "recall", "f1"])
            st.plotly_chart(style(px.bar(m_, x="method", y="value", color="variable", barmode="group"), 340), width="stretch")
            y = res["true_attack"].to_numpy()
            ph = (res["physics_severity"] >= 2).to_numpy()
            sw = []
            for x in np.round(np.arange(.1, .91, .1), 2):
                ai = (res["ai_score"] >= x).to_numpy()
                for name, p in (("AI only", ai), ("Physics only", ph), ("Combined (any alert)", ai | ph)):
                    m = metrics_at(y, p.astype(float), .5)
                    sw.append({"AI threshold": x, "method": name, "F1": m["f1"]})
            st.plotly_chart(style(px.line(pd.DataFrame(sw), x="AI threshold", y="F1", color="method",
                                          title="F1 vs AI threshold"), 340), width="stretch")
            st.caption("Combined (any alert) = physics Medium+ or AI flag; AI-only flags show up as Low alerts for review.")


pg = st.navigation([
    st.Page(page_overview, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(page_data, title="Module 1 · Data", icon="📊", url_path="data"),
    st.Page(page_model, title="Module 2 · Model", icon="🤖", url_path="model"),
    st.Page(page_alerts, title="Module 3 · Alerts", icon="🚨", url_path="alerts"),
], position="top")
pg.run()
