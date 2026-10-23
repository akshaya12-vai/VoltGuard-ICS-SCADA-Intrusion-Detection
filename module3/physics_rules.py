"""Physics-rule checker: documented operating constraints for an 11 kV feeder.

Every rule has an id, severity (1-4), description and rationale. RULES_DOC is
exported to RULES.md so the report always matches the code.
"""
import numpy as np
import pandas as pd

CONFIG = {
    "v_min": 10.45, "v_max": 11.55,        # +-5 % of 11 kV
    "f_min": 49.5, "f_max": 50.5,          # Hz (India grid)
    "i_max": 400.0,                        # A, overcurrent limit
    "temp_max": 85.0,                      # deg C, transformer
    "pf_min": 0.80, "pf_max": 1.00,
    "power_tol_mw": 0.40,                  # |P_reported - sqrt3*V*I*pf|
    "di_max": 80.0,                        # A per sample (rate of change)
    "frozen_window": 20,                   # samples with zero variance
    "breaker_open_current": 5.0,           # A flowing while breaker open
    "cusum_k": 0.10,                       # MW of residual tolerated per sample (drift allowance)
    "cusum_h": 1.5,                        # MW accumulated residual that raises the alarm
}

REQUIRED_COLUMNS = ["timestamp", "voltage_kv", "current_a", "frequency_hz", "power_mw",
                    "power_factor", "transformer_temp_c", "breaker_status"]

# rule_id: (severity 1-4, description, rationale)
RULES = {
    "R1_VOLTAGE_RANGE": (3, "Voltage outside +-5% of nominal",
                         "Distribution voltage is normally held within +-5% of 11 kV."),
    "R2_FREQUENCY_RANGE": (3, "Frequency outside 49.5-50.5 Hz",
                           "Grid frequency is tightly regulated around 50 Hz."),
    "R3_OVERCURRENT": (3, "Current above safe limit",
                       "Exceeds feeder/conductor rating (protection should trip)."),
    "R4_OVERTEMP": (3, "Transformer temperature too high",
                    "Above typical winding/oil limit; risks insulation damage."),
    "R5_PF_RANGE": (2, "Power factor implausible",
                    "Power factor must be within 0.80-1.00 for this feeder."),
    "R6_POWER_MISMATCH": (3, "Reported power inconsistent with V x I x pf",
                          "P = sqrt(3)*V*I*pf is a physical identity; a gap suggests data injection."),
    "R7_BREAKER_OPEN_FLOW": (4, "Breaker open but current still flowing (spoofed state/sensor)",
                             "An open breaker cannot carry load current."),
    "R8_RATE_OF_CHANGE": (2, "Current changed faster than physically plausible",
                          "Load cannot jump by more than di_max in one sample."),
    "R9_FROZEN_SENSOR": (2, "Sensor values frozen (possible replay attack)",
                         "Real measurements always carry noise; zero variance suggests replay."),
    "R10_POWER_DRIFT": (3, "Reported power slowly drifting from V x I x pf (stealthy injection)",
                        "Small biases stay under R6's limit but add up; a CUSUM of the residual catches them."),
}

RULE_THRESHOLDS = {
    "R1_VOLTAGE_RANGE": "10.45 - 11.55 kV", "R2_FREQUENCY_RANGE": "49.5 - 50.5 Hz",
    "R3_OVERCURRENT": "> 400 A", "R4_OVERTEMP": "> 85 C", "R5_PF_RANGE": "0.80 - 1.00",
    "R6_POWER_MISMATCH": "> 0.40 MW", "R7_BREAKER_OPEN_FLOW": "breaker=0 and I > 5 A",
    "R8_RATE_OF_CHANGE": "> 80 A/sample", "R9_FROZEN_SENSOR": "std < 1e-6 over 20 samples",
    "R10_POWER_DRIFT": "CUSUM of |P - calc| > 1.5 MW (k = 0.1)",
}
SEVERITY_NAMES = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}


def rules_markdown():
    lines = ["| Rule | Severity | Threshold | Description | Rationale |", "|---|---|---|---|---|"]
    for k, (s, d, why) in RULES.items():
        lines.append(f"| {k} | {s} ({SEVERITY_NAMES[s]}) | {RULE_THRESHOLDS[k]} | {d} | {why} |")
    return "\n".join(lines)


def validate_input(df):
    """Return a clean, time-sorted copy or raise ValueError listing problems."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    num = [c for c in REQUIRED_COLUMNS if c != "timestamp"]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    bad = d[REQUIRED_COLUMNS].isna().any(axis=1)
    if bad.all():
        raise ValueError("No valid rows after parsing timestamps/numbers.")
    d = d[~bad].sort_values("timestamp").reset_index(drop=True)
    if not set(d["breaker_status"].unique()) <= {0, 1}:
        raise ValueError("breaker_status must be 0 (open) or 1 (closed).")
    if "ai_score" in d:
        d["ai_score"] = pd.to_numeric(d["ai_score"], errors="coerce").fillna(0.0).clip(0, 1)
    d.attrs["dropped_rows"] = int(bad.sum())
    return d


def _cusum_alarm(resid, k, h):
    """One-sided CUSUM on |residual|. Grows when |r| > k, decays 3x faster when |r| < k,
    capped at 2h so the alarm clears soon after the drift ends."""
    s, cap, out = 0.0, 2 * h, []
    for x in resid.abs().to_numpy():
        s = s + x - k if x >= k else max(0.0, s - 3 * (k - x))
        s = min(s, cap)
        out.append(s > h)
    return pd.Series(out, index=resid.index)


def apply_rules(df, cfg=CONFIG):
    d = df.copy()
    v, i = d["voltage_kv"], d["current_a"]
    p_calc = np.sqrt(3) * v * i * d["power_factor"] / 1000
    w = cfg["frozen_window"]
    frozen = (i.rolling(w, min_periods=w).std() < 1e-6) & (v.rolling(w, min_periods=w).std() < 1e-6)
    # Offline back-fill: once a frozen run is confirmed, flag the whole run (incl. first w-1 samples)
    frozen = frozen.fillna(False).astype(float)
    frozen = frozen[::-1].rolling(w, min_periods=1).max()[::-1].astype(bool)

    hit = pd.DataFrame({
        "R1_VOLTAGE_RANGE": (v < cfg["v_min"]) | (v > cfg["v_max"]),
        "R2_FREQUENCY_RANGE": (d["frequency_hz"] < cfg["f_min"]) | (d["frequency_hz"] > cfg["f_max"]),
        "R3_OVERCURRENT": i > cfg["i_max"],
        "R4_OVERTEMP": d["transformer_temp_c"] > cfg["temp_max"],
        "R5_PF_RANGE": (d["power_factor"] < cfg["pf_min"]) | (d["power_factor"] > cfg["pf_max"]),
        "R6_POWER_MISMATCH": (d["power_mw"] - p_calc).abs() > cfg["power_tol_mw"],
        "R7_BREAKER_OPEN_FLOW": (d["breaker_status"] == 0) & (i > cfg["breaker_open_current"]),
        "R8_RATE_OF_CHANGE": i.diff().abs() > cfg["di_max"],
        "R9_FROZEN_SENSOR": frozen,
        "R10_POWER_DRIFT": _cusum_alarm(d["power_mw"] - p_calc, cfg["cusum_k"], cfg["cusum_h"]),
    }, index=d.index).fillna(False).astype(bool)

    sev = np.array([RULES[c][0] for c in hit.columns])
    d["physics_severity"] = (hit.values * sev).max(axis=1).astype(int)
    d["rules_triggered"] = hit.apply(lambda r: ";".join(c for c in hit.columns if r[c]), axis=1)
    d["physics_reason"] = hit.apply(
        lambda r: " | ".join(RULES[c][1] for c in hit.columns if r[c]), axis=1)
    d["power_calc_mw"] = p_calc
    return d