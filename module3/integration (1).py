"""Connects Module 1 (data) and Module 2 (Random Forest) to Module 3 (physics + alerts).
Reads teammates' files directly from the repo - nothing is copied or modified."""
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from simulate_process import simulate

ROOT = Path(os.environ.get("VOLTGUARD_ROOT", Path(__file__).resolve().parent.parent))
PATHS = {  # the team layout: data/processed, models, reports
    "clean": ROOT / "data/processed/cleaned_dataset.csv",
    "train": ROOT / "data/processed/train_data.csv",
    "test": ROOT / "data/processed/test_data.csv",
    "model": ROOT / "models/random_forest.joblib",
    "dictionary": ROOT / "reports/data_dictionary.csv",
    "eval": ROOT / "reports/model_evaluation.txt",
    "prep": ROOT / "reports/preprocessing_report.txt",
    "m2out": ROOT / "data/processed/module2_output.csv",
}
LABEL_CANDIDATES = ["true_attack", "Attack_Label", "label", "Label", "class", "Class", "target", "Target", "attack",
                    "Attack", "attack_cat", "Attack_Type", "binary_label", "marker"]
NORMAL_NAMES = {"0", "0.0", "normal", "benign", "false", "no attack", "no_attack", "natural"}


def read_text(key):
    p = PATHS[key]
    return p.read_text(errors="replace") if p.exists() else f"(file not found: {p})"


def read_csv(key, nrows=None):
    p = PATHS[key]
    return pd.read_csv(p, nrows=nrows) if p.exists() else None


def find_label(df):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def is_attack(series):
    return ~series.astype(str).str.strip().str.lower().isin(NORMAL_NAMES)


def load_model():
    obj = joblib.load(PATHS["model"])
    if isinstance(obj, dict):  # some people save {"model": clf, ...}
        obj = next((v for v in obj.values() if hasattr(v, "predict")), obj)
    return obj


def model_features(model, df, label_col):
    feats = getattr(model, "feature_names_in_", None)
    if feats is None:
        feats = [c for c in df.select_dtypes("number").columns if c != label_col]
    return list(feats)


def score_traffic(model, df, label_col):
    """Return suspicion score 0-1 for each traffic row."""
    X = df[model_features(model, df, label_col)]
    if not hasattr(model, "predict_proba"):
        return np.asarray(model.predict(X), dtype=float)
    P = model.predict_proba(X)
    classes = list(model.classes_)
    normal = [i for i, c in enumerate(classes) if str(c).strip().lower() in NORMAL_NAMES]
    if normal:
        return 1 - P[:, normal[0]]
    return P[:, 1] if P.shape[1] == 2 else P.max(axis=1)


def build_integrated(n=3000, seed=42, split="test"):
    """Simulated process + REAL Random Forest scores.

    The network dataset has no timestamps/process variables, so we replay it alongside
    the simulation: during simulated attack windows we sample traffic rows labelled
    'attack'; otherwise 'normal'. Stealthy false-data-injection uses NORMAL traffic
    (no network signature) - this is the case physics rules are meant to catch."""
    sim = simulate(n, seed)
    traffic = read_csv(split)
    if traffic is None:
        raise FileNotFoundError(f"Missing {PATHS[split]} - run git pull")
    model = load_model()
    label = find_label(traffic)
    scores = score_traffic(model, traffic, label)
    rng = np.random.default_rng(seed)
    info = {"label_col": label, "traffic_rows": len(traffic), "split": split}

    if label is not None:
        att = is_attack(traffic[label]).to_numpy()
        pool_a = np.where(att)[0] if att.any() else np.arange(len(traffic))
        pool_n = np.where(~att)[0] if (~att).any() else np.arange(len(traffic))
        want_att = ((sim["true_attack"] == 1) &
                    (sim["attack_type"] != "false_data_injection")).to_numpy()
        idx = np.where(want_att, rng.choice(pool_a, n), rng.choice(pool_n, n))
        sim["traffic_label"] = traffic[label].to_numpy()[idx]
        info.update(attack_rows=int(att.sum()), normal_rows=int((~att).sum()))
    else:
        idx = np.arange(n) % len(traffic)
        info["note"] = "No label column found - traffic replayed sequentially."
    sim["ai_score"] = scores[idx]
    return sim, info


MOD3_COLS = ["timestamp", "voltage_kv", "current_a", "frequency_hz", "power_mw", "power_factor",
             "transformer_temp_c", "breaker_status"]


TANK_COLS = ["Timestamp", "Tank_Water_Level", "Pressure_Sensor", "Flow_Rate", "Valve_Status"]


def compatibility_report(tank=False):
    """Do Module 1 data, the Module 2 model and Module 3 agree on columns? -> [(check, ok, detail)]"""
    rows = []
    test = read_csv("test")
    need = TANK_COLS if tank else MOD3_COLS
    rows.append(("Module 1 test data found", test is not None, PATHS["test"].name))
    if test is not None:
        miss = [c for c in need if c not in test.columns]
        rows.append(("Module 1 data has the process columns Module 3 needs", not miss,
                     "missing: " + ", ".join(miss) if miss else "all present"))
        rows.append(("Module 1 data has a label column", find_label(test) is not None, str(find_label(test))))
    try:
        model = load_model()
        feats = model_features(model, test if test is not None else pd.DataFrame(), None)
        miss = [f for f in feats if test is None or f not in test.columns]
        rows.append(("Module 2 model features exist in Module 1 data", not miss,
                     f"{len(feats)} features: " + ", ".join(feats) if not miss else "model needs: " + ", ".join(miss)))
        if tank and test is not None and not miss:
            s = score_traffic(model, test.head(200), find_label(test))
            rows.append(("Module 2 model scores Module 1 data (ai_score)", bool(np.isfinite(s).all()),
                         f"{len(s)} rows scored, scores {s.min():.2f}-{s.max():.2f}"))
    except Exception as e:
        rows.append(("Module 2 model loads", False, str(e)[:80]))
    if not tank:
        out = read_csv("m2out", nrows=5)
        if out is None:
            rows.append(("Module 2 output (module2_output.csv) found", False, "run the Module 2 script"))
        else:
            miss = [c for c in MOD3_COLS + ["ai_score"] if c not in out.columns]
            rows.append(("Module 2 output has the columns Module 3 needs", not miss,
                         "missing: " + ", ".join(miss) if miss else "timestamp + process columns + ai_score"))
    return rows
