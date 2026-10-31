"""Fuses AI detection (Module 2) with physics-rule results (Module 3).

Fusion matrix (transparent & documented):
  physics Critical(4)               -> Critical
  physics High(3)  + AI flag        -> Critical
  physics High(3)  only             -> High
  physics Medium(2)+ AI flag        -> High
  physics Medium(2) only            -> Medium
  AI flag only (physics normal)     -> Low  (goes to the review queue)
  nothing                           -> Normal
"""
import re
import numpy as np
import pandas as pd
from physics_rules import apply_rules, validate_input, CONFIG
import incident_analysis as ia

LEVELS = ["Normal", "Low", "Medium", "High", "Critical"]
TRUE_NAME = {"false_data_injection": "False data injection", "breaker_manipulation": "Breaker manipulation",
             "replay_frozen_sensor": "Replay / frozen sensor", "overvoltage_injection": "Process parameter manipulation",
             "slow_power_drift": "Slow power drift",
             "module1_attack": "Overflow / over-pressure attack", "sustained_overflow": "Overflow / over-pressure attack",
             "valve_spoof": "Valve / flow spoofing", "inconsistent_sensors": "Sensor spoofing (inconsistent readings)"}
INC_COLS = ["incident", "start", "end", "duration_s", "alert_samples", "max_severity", "rules",
            "avg_ai_score", "likely_attack", "max_risk"]


def _fuse(phys, ai_flag):
    if phys >= 4: return "Critical"
    if phys == 3: return "Critical" if ai_flag else "High"
    if phys == 2: return "High" if ai_flag else "Medium"
    return "Low" if ai_flag else "Normal"


def join_module2(proc, m2, tolerance="1s"):
    """Attach Module 2's ai_score to process rows by nearest timestamp (unmatched -> 0)."""
    if "ai_score" not in m2.columns or "timestamp" not in m2.columns:
        raise ValueError("Module 2 file needs columns: timestamp, ai_score")
    p = proc.drop(columns=["ai_score"], errors="ignore").copy()
    m = m2[["timestamp", "ai_score"]].copy()
    p["timestamp"] = pd.to_datetime(p["timestamp"], errors="coerce")
    m["timestamp"] = pd.to_datetime(m["timestamp"], errors="coerce")
    p = p.dropna(subset=["timestamp"]).sort_values("timestamp")
    m = m.dropna().sort_values("timestamp")
    j = pd.merge_asof(p, m, on="timestamp", direction="nearest", tolerance=pd.Timedelta(tolerance))
    j["ai_score"] = j["ai_score"].fillna(0.0)
    return j.reset_index(drop=True)


def run_pipeline(df, ai_threshold=0.5, cfg=None, disabled=(), profile=None):
    """profile = a rules module (e.g. tank_rules) with validate_input/apply_rules/CONFIG; default = 11 kV feeder."""
    vi, ar, base_cfg = (validate_input, apply_rules, CONFIG) if profile is None else \
        (profile.validate_input, profile.apply_rules, profile.CONFIG)
    d = vi(df)
    dropped = d.attrs.get("dropped_rows", 0)
    d = ar(d, base_cfg if cfg is None else cfg, tuple(disabled or ()))
    if "ai_score" not in d:
        d["ai_score"] = 0.0
    d["ai_flag"] = d["ai_score"] >= ai_threshold
    d["final_severity"] = [_fuse(p, a) for p, a in zip(d["physics_severity"], d["ai_flag"])]

    def reason(r):
        parts = []
        if r["physics_reason"]: parts.append("PHYSICS: " + r["physics_reason"])
        if r["ai_flag"]: parts.append(f"AI: suspicious traffic (score {r['ai_score']:.2f})")
        return " || ".join(parts)
    d["alert_reason"] = d.apply(reason, axis=1)
    # per-sample risk 0-100: severity base + AI confidence bonus (max 10)
    d["risk_score"] = np.minimum(100, d["final_severity"].map(ia.BASE).fillna(0)
                                 + (10 * d["ai_score"]).round()).astype(int)
    d.attrs["dropped_rows"] = dropped
    return d


def build_incidents(d, min_sev="Medium", max_gap=5):
    """Group alert samples (>= min_sev) into incidents; gaps of <= max_gap quiet samples are merged."""
    on = d["final_severity"].map(LEVELS.index).to_numpy() >= LEVELS.index(min_sev)
    idx = np.flatnonzero(on)
    if len(idx) == 0:
        return pd.DataFrame(columns=INC_COLS)
    groups = np.split(idx, np.where(np.diff(idx) - 1 > max_gap)[0] + 1)
    dt = d["timestamp"].diff().dt.total_seconds().median()
    dt = 1.0 if pd.isna(dt) or dt <= 0 else float(dt)
    rows = []
    for n, g in enumerate(groups, 1):
        seg = d.iloc[g[0]:g[-1] + 1]
        rules = sorted({r for s in seg["rules_triggered"] for r in s.split(";") if r})
        sev = max(d.iloc[g]["final_severity"], key=LEVELS.index)
        avg_ai = round(float(seg["ai_score"].mean()), 2)
        rows.append({
            "incident": n, "start": seg["timestamp"].iloc[0], "end": seg["timestamp"].iloc[-1],
            "duration_s": int(round((seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds() + dt)),
            "alert_samples": len(g), "max_severity": sev, "rules": ", ".join(rules) or "AI only",
            "avg_ai_score": avg_ai,
            "likely_attack": ia.classify(set(rules)),
            "max_risk": int(seg["risk_score"].max())})
    return pd.DataFrame(rows, columns=INC_COLS)


def playbook_for(kind):
    text = ia.PLAYBOOK.get(kind, ia.PLAYBOOK["Anomalous process behaviour"])[1]
    return [s.strip().rstrip(".") + "." for s in re.split(r";\s*", text) if s.strip()]


def review_queue(d):
    """AI flagged it but physics looks normal: possible false positives for an analyst."""
    q = d[d["ai_flag"] & (d["physics_severity"] == 0)]
    return q[["timestamp", "ai_score", "alert_reason"]].reset_index(drop=True)


def _prf(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1, tp, fp, fn


def evaluate(d):
    """Per-sample AI-only vs physics-only vs combined against ground truth (None if no labels)."""
    if "true_attack" not in d:
        return None
    y = d["true_attack"] == 1
    methods = {"AI only": d["ai_flag"], "Physics only": d["physics_severity"] >= 2,
               "Combined (Medium+)": d["final_severity"].isin(["Medium", "High", "Critical"])}
    out = []
    for name, pred in methods.items():
        pr, rc, f1, tp, fp, fn = _prf(pred, y)
        out.append({"method": name, "precision": round(pr, 3), "recall": round(rc, 3),
                    "f1": round(f1, 3), "TP": tp, "FP": fp, "FN": fn})
    return pd.DataFrame(out)


def evaluate_by_attack(d):
    """Share of each attack type's samples detected by AI only, physics only and combined."""
    if "true_attack" not in d or "attack_type" not in d or not (d["true_attack"] == 1).any():
        return None
    med = d["final_severity"].isin(["Medium", "High", "Critical"])
    rows = []
    for k, g in d[d["true_attack"] == 1].groupby("attack_type"):
        rows.append({"attack_type": k, "samples": len(g), "AI only": round(float(g["ai_flag"].mean()), 2),
                     "Physics only": round(float((g["physics_severity"] >= 2).mean()), 2),
                     "Combined (Medium+)": round(float(med[g.index].mean()), 2)})
    return pd.DataFrame(rows)


def threshold_sweep(d, thresholds=None):
    """F1/precision/recall of each strategy as the AI threshold moves."""
    cols = ["threshold", "method", "precision", "recall", "f1"]
    if "true_attack" not in d:
        return pd.DataFrame(columns=cols)
    y = (d["true_attack"] == 1).to_numpy()
    ph = d["physics_severity"].to_numpy()
    ai_s = d["ai_score"].to_numpy()
    rows = []
    for t in (np.round(np.arange(.1, .91, .1), 2) if thresholds is None else thresholds):
        ai = ai_s >= t
        strat = {"AI only": ai, "Physics only": ph >= 2, "Any alert": (ph >= 2) | ai,
                 "High and above": (ph >= 3) | ((ph == 2) & ai)}
        for name, p in strat.items():
            pr, rc, f1, *_ = _prf(p, y)
            rows.append({"threshold": float(t), "method": name, "precision": pr, "recall": rc, "f1": f1})
    return pd.DataFrame(rows, columns=cols)


def evaluate_incidents(d, inc):
    """Per true attack: was an incident opened, how fast, and was it named correctly?"""
    cols = ["attack", "start", "end", "detected", "likely_attack", "named_correctly", "delay_s"]
    zero = {"attacks": 0, "detected": 0, "correctly_named": 0, "mean_delay_s": 0.0,
            "false_alarm_incidents": int(len(inc))}
    if "true_attack" not in d or "attack_type" not in d:
        return pd.DataFrame(columns=cols), zero
    y = d["true_attack"] == 1
    at = d["attack_type"]
    sid = (at != at.shift()).cumsum()
    rows = []
    for _, g in d[y].groupby(sid[y]):
        s, e, kind = g["timestamp"].iloc[0], g["timestamp"].iloc[-1], g["attack_type"].iloc[0]
        ov = inc[(inc["start"] <= e) & (inc["end"] >= s)] if len(inc) else inc
        det = len(ov) > 0
        rows.append({"attack": kind, "start": s, "end": e, "detected": det,
                     "likely_attack": ", ".join(sorted(set(ov["likely_attack"]))) if det else "-",
                     "named_correctly": bool(det and (ov["likely_attack"] == TRUE_NAME.get(kind, kind)).any()),
                     "delay_s": max(0.0, (ov["start"].min() - s).total_seconds()) if det else np.nan})
    seg = pd.DataFrame(rows, columns=cols)
    if len(inc):
        fa = sum(not ((seg["start"] <= r.end) & (seg["end"] >= r.start)).any() for r in inc.itertuples())
    else:
        fa = 0
    det_delays = seg.loc[seg["detected"], "delay_s"]
    return seg, {"attacks": len(seg), "detected": int(seg["detected"].sum()),
                 "correctly_named": int(seg["named_correctly"].sum()),
                 "mean_delay_s": float(det_delays.mean()) if len(det_delays) else 0.0,
                 "false_alarm_incidents": int(fa)}
