"""Fuses AI detection (Module 2) with physics-rule results (Module 3).

Fusion matrix (transparent & documented):
  physics Critical(4)               -> Critical
  physics High(3)  + AI flag        -> Critical
  physics High(3)  only             -> High
  physics Medium(2)+ AI flag        -> High
  physics Medium(2) only            -> Medium
  AI flag only (physics normal)     -> Low  (possible false positive, review)
  nothing                           -> Normal
"""
import pandas as pd
from physics_rules import apply_rules

LEVELS = ["Normal", "Low", "Medium", "High", "Critical"]


def _fuse(phys, ai_flag):
    if phys >= 4: return "Critical"
    if phys == 3: return "Critical" if ai_flag else "High"
    if phys == 2: return "High" if ai_flag else "Medium"
    return "Low" if ai_flag else "Normal"


def run_pipeline(df, ai_threshold=0.5, cfg=None):
    d = apply_rules(df) if cfg is None else apply_rules(df, cfg)
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
    return d


def build_incidents(d):
    """Group consecutive alerting samples into incidents."""
    a = d["final_severity"] != "Normal"
    grp = (a != a.shift()).cumsum()
    rows = []
    for n, (_, g) in enumerate(d[a].groupby(grp[a]), 1):
        rules = sorted({r for s in g["rules_triggered"] for r in s.split(";") if r})
        rows.append({
            "incident": n, "start": g["timestamp"].iloc[0], "end": g["timestamp"].iloc[-1],
            "samples": len(g),
            "max_severity": max(g["final_severity"], key=LEVELS.index),
            "rules": ", ".join(rules) or "AI only",
            "avg_ai_score": round(g["ai_score"].mean(), 2),
        })
    return pd.DataFrame(rows)


def evaluate(d):
    """Compare AI-only, physics-only, and combined against ground truth (if present)."""
    if "true_attack" not in d:
        return None
    y = d["true_attack"] == 1
    methods = {
        "AI only": d["ai_flag"],
        "Physics only": d["physics_severity"] >= 2,
        "Combined (Medium+)": d["final_severity"].isin(["Medium", "High", "Critical"]),
    }
    out = []
    for name, pred in methods.items():
        tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        out.append({"method": name, "precision": round(pr, 3), "recall": round(rc, 3),
                    "f1": round(f1, 3), "TP": int(tp), "FP": int(fp), "FN": int(fn)})
    return pd.DataFrame(out)
