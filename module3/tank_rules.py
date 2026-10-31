"""Physics rules for the WATER-TANK process in Module 1's dataset (Tank_Water_Level, Pressure_Sensor,
Flow_Rate, Valve_Status). Same output columns as physics_rules.py, so the alert engine, incidents and
dashboards work unchanged.  Limits come from Module 1's data dictionary (level > 70 is a violation) and
from the normal-traffic statistics in Module 1's generator (level ~50+-5, pressure ~30+-2.1, flow ~12.5+-1.2)."""
import numpy as np
import pandas as pd

NAME = "Water tank (Module 1/2 dataset)"
REQUIRED_COLUMNS = ["timestamp", "Tank_Water_Level", "Pressure_Sensor", "Flow_Rate", "Valve_Status"]
SIGNALS = [("Tank_Water_Level", "Tank level"), ("Pressure_Sensor", "Pressure (PSI)"),
           ("Flow_Rate", "Flow rate"), ("Valve_Status", "Valve (1=open)")]
OVERLAY = ("Pressure_Sensor", "expected_pressure", "expected from level (0.6 x level)")

CONFIG = {
    "level_high": 70.0,          # alarm: Module 1 dictionary says level > 70 is a violation
    "level_warn": 65.0,          # ~3 sigma above the normal mean of 50
    "level_low": 30.0,           # ~4 sigma below normal
    "pressure_high": 40.0,       # ~5 sigma above the normal mean of 30
    "pressure_low": 20.0,
    "flow_min": 5.0, "flow_max": 20.0,
    "k_pressure_per_level": 0.6,  # hydrostatic: bottom pressure ~ proportional to level (30 / 50)
    "mismatch_tol": 12.0,        # PSI; ~3.3 sigma of the normal residual
    "valve_closed_run": 10,      # samples the valve must stay closed before flow is checked
    "flow_when_closed": 8.0,     # flow above this with a closed valve is impossible
    "frozen_window": 20,
}

# rule_id: (severity 1-4, description, rationale)
RULES = {
    "T1_LEVEL_HIGH": (3, "Tank level above the high alarm limit",
                      "Module 1 dictionary: level above 70 is a violation; the tank may overflow."),
    "T2_LEVEL_LOW": (3, "Tank level below the low alarm limit",
                     "Very low level risks pump dry-run or means the sensor has failed."),
    "T3_LEVEL_WARN": (2, "Tank level in the warning band",
                      "Unusually high but not yet at the alarm limit."),
    "T4_PRESSURE_HIGH": (3, "Pressure above the alarm limit",
                         "Normal pressure is about 30 PSI; much higher stresses pipes and the tank."),
    "T5_PRESSURE_LOW": (3, "Pressure below the low limit", "Loss of pressure: leak, pump failure or bad sensor."),
    "T6_FLOW_RANGE": (2, "Flow rate outside the plausible range", "Normal flow is about 12.5; far outside is implausible."),
    "T7_LEVEL_PRESSURE_MISMATCH": (2, "Pressure inconsistent with tank level",
                                   "Bottom pressure follows water level (hydrostatic); a gap means one sensor is wrong or spoofed."),
    "T8_VALVE_CLOSED_FLOW": (4, "Valve reported closed for 10+ s but flow continues",
                             "A closed valve cannot pass flow; valve status or flow reading is spoofed."),
    "T9_FROZEN_SENSORS": (2, "All sensor readings frozen",
                          "Real readings always carry noise; zero variance suggests a replay attack."),
}
RULE_THRESHOLDS = {
    "T1_LEVEL_HIGH": "> 70", "T2_LEVEL_LOW": "< 30", "T3_LEVEL_WARN": "65 - 70",
    "T4_PRESSURE_HIGH": "> 40 PSI", "T5_PRESSURE_LOW": "< 20 PSI", "T6_FLOW_RANGE": "outside 5 - 20",
    "T7_LEVEL_PRESSURE_MISMATCH": "|pressure - 0.6 x level| > 12 PSI",
    "T8_VALVE_CLOSED_FLOW": "valve = 0 for 10+ samples and flow > 8",
    "T9_FROZEN_SENSORS": "level and pressure unchanged for 20+ samples",
}
SEVERITY_NAMES = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
WHATIF = [("level_high", "Level alarm limit", 55, 90, 1), ("pressure_high", "Pressure alarm limit (PSI)", 35, 60, 1),
          ("mismatch_tol", "Level/pressure mismatch tolerance (PSI)", 4, 20, 1)]


def validate_input(df):
    d = df.rename(columns={"Timestamp": "timestamp"}).copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    for c in REQUIRED_COLUMNS[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    bad = d[REQUIRED_COLUMNS].isna().any(axis=1)
    if bad.all():
        raise ValueError("No valid rows after parsing timestamps/numbers.")
    d = d[~bad].sort_values("timestamp").reset_index(drop=True)
    if not set(d["Valve_Status"].unique()) <= {0, 1}:
        raise ValueError("Valve_Status must be 0 (closed) or 1 (open).")
    if "ai_score" in d:
        d["ai_score"] = pd.to_numeric(d["ai_score"], errors="coerce").fillna(0.0).clip(0, 1)
    d.attrs["dropped_rows"] = int(bad.sum())
    return d


def apply_rules(df, cfg=CONFIG, disabled=()):
    d = df.copy()
    L, P, F, V = d["Tank_Water_Level"], d["Pressure_Sensor"], d["Flow_Rate"], d["Valve_Status"]
    expected = cfg["k_pressure_per_level"] * L

    closed = V == 0                                        # sustained closed valve with flow still running
    run = (closed != closed.shift()).cumsum()
    t8 = closed & (closed.groupby(run).transform("size") >= cfg["valve_closed_run"]) \
        & (F.groupby(run).transform("mean") > cfg["flow_when_closed"])

    w = cfg["frozen_window"]                               # offline back-fill: flag the whole frozen run
    def flat(x):  # range (max - min) is exact for a constant series; rolling std leaves float noise
        return (x.rolling(w, min_periods=w).max() - x.rolling(w, min_periods=w).min()) < 1e-9
    fz = flat(L) & flat(P)
    fz = fz.fillna(False).astype(float)
    fz = fz[::-1].rolling(w, min_periods=1).max()[::-1].astype(bool)

    hit = pd.DataFrame({
        "T1_LEVEL_HIGH": L > cfg["level_high"],
        "T2_LEVEL_LOW": L < cfg["level_low"],
        "T3_LEVEL_WARN": (L > cfg["level_warn"]) & (L <= cfg["level_high"]),
        "T4_PRESSURE_HIGH": P > cfg["pressure_high"],
        "T5_PRESSURE_LOW": P < cfg["pressure_low"],
        "T6_FLOW_RANGE": (F < cfg["flow_min"]) | (F > cfg["flow_max"]),
        "T7_LEVEL_PRESSURE_MISMATCH": (P - expected).abs() > cfg["mismatch_tol"],
        "T8_VALVE_CLOSED_FLOW": t8,
        "T9_FROZEN_SENSORS": fz,
    }, index=d.index).fillna(False).astype(bool)
    for c in disabled:
        if c in hit.columns:
            hit[c] = False

    sev = np.array([RULES[c][0] for c in hit.columns])
    d["physics_severity"] = (hit.values * sev).max(axis=1).astype(int)
    d["rules_triggered"] = hit.apply(lambda r: ";".join(c for c in hit.columns if r[c]), axis=1)
    d["physics_reason"] = hit.apply(lambda r: " | ".join(RULES[c][1] for c in hit.columns if r[c]), axis=1)
    d["expected_pressure"] = expected
    return d


def rules_markdown():
    lines = ["| Rule | Severity | Threshold | Description | Rationale |", "|---|---|---|---|---|"]
    for k, (s, desc, why) in RULES.items():
        lines.append(f"| {k} | {s} ({SEVERITY_NAMES[s]}) | {RULE_THRESHOLDS[k]} | {desc} | {why} |")
    return "\n".join(lines)
