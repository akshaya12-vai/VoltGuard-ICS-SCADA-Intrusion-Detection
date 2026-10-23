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
import numpy as np
import pandas as pd
from physics_rules import apply_rules, validate_input, CONFIG

LEVELS = ["Normal", "Low", "Medium", "High", "Critical"]

# Likely attack from the rules an incident triggered (first match wins).
SIGNATURES = [
    ("breaker_manipulation", "R7_BREAKER_OPEN_FLOW"),
    ("overvoltage_injection", "R1_VOLTAGE_RANGE"),
    ("replay_frozen_sensor", "R9_FROZEN_SENSOR"),
    ("false_data_injection", "R6_POWER_MISMATCH"),
    ("stealthy_power_drift", "R10_POWER_DRIFT"),
    ("overload_or_frequency_fault", "R3_OVERCURRENT"),
    ("overload_or_frequency_fault", "R2_FREQUENCY_RANGE"),
]
PHYS_BASE = {0: 0, 1: 20, 2: 40, 3: 65, 4: 85}


def likely_attack(rules):
    """rules: iterable of rule ids. Returns a readable guess, or 'unclassified anomaly'."""
    rules = set(rules)
    for name, rule in SIGNATURES:
        if rule in rules:
            return name
    return "unclassified anomaly" if rules else "AI-only anomaly"


def _fuse(phys, ai_flag):
    if phys >= 4: return "Critical"
    if phys == 3: return "Critical" if ai_flag else "High"
    if phys == 2: return "High" if ai_flag else "Medium"
    return "Low" if ai_flag else "Normal"


def join_module2(process_df, preds_df, on="timestamp", tolerance="1s"):
    """Attach Module 2 predictions (timestamp, ai_score) to process variables.
    Nearest-timestamp join within `tolerance`; unmatched rows get ai_score 0."""
    p = process_df.copy(); m = preds_df[[on, "ai_score"]].copy()
    p[on] = pd.to_datetime(p[on]); m[on] = pd.to_datetime(m[on])
    p = p.drop(columns=["ai_score"], errors="ignore").sort_values(on)
    out = pd.merge_asof(p, m.sort_values(on), on=on, direction="nearest",
                        tolerance=pd.Timedelta(tolerance))
    out["ai_score"] = out["ai_score"].fillna(0.0)
    return out


def run_pipeline(df, ai_threshold=0.5, cfg=None):
    d = apply_rules(validate_input(df), cfg or CONFIG)
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
    # Risk score 0-100 (for ranking only; final_severity stays the decision):
    #   physics base (0/20/40/65/85) + 5 per extra rule (max 10) + 15*ai_score if AI-flagged else 5*ai_score
    n_rules = d["rules_triggered"].map(lambda s: len([r for r in s.split(";") if r]))
    bonus = ((n_rules - 1).clip(lower=0) * 5).clip(upper=10)
    ai_part = np.where(d["ai_flag"], 15 * d["ai_score"], 5 * d["ai_score"])
    d["risk_score"] = (d["physics_severity"].map(PHYS_BASE) + bonus + ai_part).clip(upper=100).round(1)
    return d


def build_incidents(d, min_severity="Medium", max_gap=5):
    """Group alerting samples into incidents.
    Only samples >= min_severity count; gaps of <= max_gap samples are merged.
    Low (AI-only) alerts are reported separately via review_queue()."""
    a = d["final_severity"].map(LEVELS.index) >= LEVELS.index(min_severity)
    idx = list(d.index[a])
    groups, cur = [], []
    for k in idx:
        if cur and (d.index.get_loc(k) - d.index.get_loc(cur[-1])) > max_gap + 1:
            groups.append(cur); cur = []
        cur.append(k)
    if cur: groups.append(cur)
    rows = []
    for n, ks in enumerate(groups, 1):
        g = d.loc[ks[0]:ks[-1]]
        al = g[g["final_severity"].map(LEVELS.index) >= LEVELS.index(min_severity)]
        rules = sorted({r for s in al["rules_triggered"] for r in s.split(";") if r})
        rows.append({
            "incident": n, "start": g["timestamp"].iloc[0], "end": g["timestamp"].iloc[-1],
            "duration_s": int((g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds()) + 1,
            "alert_samples": len(al),
            "max_severity": max(al["final_severity"], key=LEVELS.index),
            "likely_attack": likely_attack(rules),
            "max_risk": float(round(al["risk_score"].max())),
            "rules": ", ".join(rules) or "AI only",
            "avg_ai_score": round(g["ai_score"].mean(), 2),
        })
    return pd.DataFrame(rows, columns=["incident", "start", "end", "duration_s", "alert_samples",
                                       "max_severity", "likely_attack", "max_risk", "rules", "avg_ai_score"])


def review_queue(d):
    """AI-only alerts (physics normal): candidate false positives for analyst review."""
    return d[d["final_severity"] == "Low"][["timestamp", "ai_score", "alert_reason"]]


def evaluate(d):
    """Compare AI-only, physics-only, and combined against ground truth (if present).
    NOTE: with the mock ai_score this demonstrates the design, not real performance."""
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


def evaluate_by_attack(d):
    """Per-attack-type recall for each method (class-wise results)."""
    if "attack_type" not in d:
        return None
    methods = {"AI only": d["ai_flag"], "Physics only": d["physics_severity"] >= 2,
               "Combined": d["final_severity"].isin(["Medium", "High", "Critical"])}
    rows = []
    for t in sorted(set(d["attack_type"]) - {"normal"}):
        m = d["attack_type"] == t
        rows.append({"attack_type": t, "samples": int(m.sum()),
                     **{k: round(float(v[m].mean()), 3) for k, v in methods.items()}})
    return pd.DataFrame(rows)


def evaluate_incidents(d, inc):
    """Incident-level scoring (needs true_attack): was each real attack detected, how fast,
    was it named correctly, and how many incidents match no real attack?
    Returns (per-attack DataFrame, summary dict) or (None, None)."""
    if "true_attack" not in d:
        return None, None
    y = (d["true_attack"] == 1).to_numpy()
    alert = (d["final_severity"].map(LEVELS.index) >= LEVELS.index("Medium")).to_numpy()
    step = (d["timestamp"].iloc[1] - d["timestamp"].iloc[0]).total_seconds() if len(d) > 1 else 1.0
    rows, spans, i = [], [], 0
    while i < len(d):
        if not y[i]:
            i += 1; continue
        j = i
        while j + 1 < len(d) and y[j + 1] and d["attack_type"].iloc[j + 1] == d["attack_type"].iloc[i]:
            j += 1
        t0, t1 = d["timestamp"].iloc[i], d["timestamp"].iloc[j]
        spans.append((t0, t1))
        hit = np.flatnonzero(alert[i:j + 1])
        m = inc[(inc["start"] <= t1) & (inc["end"] >= t0)] if len(inc) else inc
        named = m.iloc[0]["likely_attack"] if len(m) else "-"
        atk = d["attack_type"].iloc[i]
        rows.append({"attack_type": atk, "start": t0, "end": t1,
                     "detected": "Yes" if len(hit) else "No",
                     "delay_s": float(hit[0] * step) if len(hit) else np.nan,
                     "named_as": named,
                     "named_correctly": ("Yes" if named == atk else "No") if len(hit) else "-"})
        i = j + 1
    seg = pd.DataFrame(rows, columns=["attack_type", "start", "end", "detected", "delay_s",
                                      "named_as", "named_correctly"])
    false_alarms = sum(1 for r in inc.itertuples()
                       if not any(r.start <= t1 and r.end >= t0 for t0, t1 in spans)) if len(inc) else 0
    det = seg[seg["detected"] == "Yes"]
    summ = {"attacks": len(seg), "detected": len(det),
            "mean_delay_s": float(det["delay_s"].mean()) if len(det) else float("nan"),
            "correctly_named": int((seg["named_correctly"] == "Yes").sum()),
            "false_alarm_incidents": int(false_alarms)}
    return seg, summ