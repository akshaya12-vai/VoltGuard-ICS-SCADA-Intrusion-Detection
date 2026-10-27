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
from physics_rules import apply_rules, validate_input

LEVELS = ["Normal", "Low", "Medium", "High", "Critical"]


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


def run_pipeline(df, ai_threshold=0.5):
    d = apply_rules(validate_input(df))
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
            "rules": ", ".join(rules) or "AI only",
            "avg_ai_score": round(g["ai_score"].mean(), 2),
        })
    return pd.DataFrame(rows, columns=["incident", "start", "end", "duration_s", "alert_samples",
                                       "max_severity", "rules", "avg_ai_score"])


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
