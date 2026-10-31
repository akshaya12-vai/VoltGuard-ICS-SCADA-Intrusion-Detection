"""Module 3's view of Module 1/2: reads Module 1's CSV files and Module 2's saved model (READ-ONLY, nothing
in Modules 1/2 is changed) and attaches the real ai_score.  Optional Module 3 stress tests add sustained,
labelled attacks that a single-sensor threshold model cannot see, so the physics layer can be evaluated."""
import numpy as np
import pandas as pd
import integration as itg

STRESS_KEYS = ["inconsistent_sensors", "valve_spoof", "replay_frozen_sensor", "sustained_overflow"]
SENSORS = ["Tank_Water_Level", "Pressure_Sensor", "Flow_Rate", "Valve_Status"]


def _windows(rng, n, reps):
    taken, wins = np.zeros(n, bool), []
    for key in STRESS_KEYS:
        for _ in range(reps):
            for _try in range(300):
                m = int(rng.integers(30, 61)); a = int(rng.integers(30, max(31, n - m - 10)))
                if not taken[max(0, a - 20):a + m + 20].any():
                    taken[a:a + m] = True; wins.append((a, a + m, key)); break
    return sorted(wins)


def inject_stress(df, seed=7):
    d = df.copy()
    rng = np.random.default_rng(seed)
    n = len(d)
    arr = {c: d[c].to_numpy(dtype=float).copy() for c in SENSORS}
    ta, at = d["true_attack"].to_numpy().copy(), d["attack_type"].to_numpy(dtype=object).copy()
    for a, b, key in _windows(rng, n, 1 if n <= 1500 else 2):
        m = b - a
        if key == "inconsistent_sensors":      # both readings look normal alone, but together are impossible
            arr["Tank_Water_Level"][a:b] = rng.normal(38, 1, m); arr["Pressure_Sensor"][a:b] = rng.normal(38, 1, m)
        elif key == "valve_spoof":             # valve reported closed while water keeps flowing
            arr["Valve_Status"][a:b] = 0
        elif key == "replay_frozen_sensor":    # readings replayed from one instant
            for c in SENSORS:
                arr[c][a:b] = arr[c][a]
        elif key == "sustained_overflow":      # Module 1's attack signature, but sustained for 30-60 s
            arr["Tank_Water_Level"][a:b] = rng.uniform(75, 95, m); arr["Pressure_Sensor"][a:b] = rng.uniform(45, 60, m)
        ta[a:b], at[a:b] = 1, key
    for c in SENSORS:
        d[c] = arr[c]
    d["Valve_Status"] = d["Valve_Status"].astype(int)
    d["true_attack"], d["attack_type"] = ta, at
    return d


def load(split="test", stress=False, seed=7):
    """Module 1 data (+ optional stress tests) with Module 2's real ai_score."""
    raw = itg.read_csv(split)
    if raw is None:
        raise FileNotFoundError(f"Module 1 file not found: {itg.PATHS[split]} (run Module 1's preprocessing.py)")
    df = raw.rename(columns={"Timestamp": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    label = "Attack_Label" if "Attack_Label" in df else None
    df["true_attack"] = df[label].astype(int) if label else 0
    df["attack_type"] = np.where(df["true_attack"] == 1, "module1_attack", "normal")
    if stress:
        df = inject_stress(df, seed)
    model = itg.load_model()                              # Module 2's saved Random Forest, unchanged
    df["ai_score"] = itg.score_traffic(model, df, label)  # uses the model's own feature names
    return df.reset_index(drop=True), {"rows": len(df), "split": split, "stress": stress}
