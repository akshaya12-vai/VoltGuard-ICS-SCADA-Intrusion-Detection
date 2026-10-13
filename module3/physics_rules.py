"""Physics-rule checker: documented operating constraints for an 11 kV feeder."""
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
}

# rule_id: (severity 1-4, description)
RULES = {
    "R1_VOLTAGE_RANGE":   (3, "Voltage outside +-5% of nominal"),
    "R2_FREQUENCY_RANGE": (3, "Frequency outside 49.5-50.5 Hz"),
    "R3_OVERCURRENT":     (3, "Current above safe limit"),
    "R4_OVERTEMP":        (3, "Transformer temperature too high"),
    "R5_PF_RANGE":        (2, "Power factor implausible"),
    "R6_POWER_MISMATCH":  (3, "Reported power inconsistent with V x I x pf"),
    "R7_BREAKER_OPEN_FLOW": (4, "Breaker open but current still flowing (spoofed state/sensor)"),
    "R8_RATE_OF_CHANGE":  (2, "Current changed faster than physically plausible"),
    "R9_FROZEN_SENSOR":   (2, "Sensor values frozen (possible replay attack)"),
}


def apply_rules(df, cfg=CONFIG):
    d = df.copy()
    v, i = d["voltage_kv"], d["current_a"]
    p_calc = np.sqrt(3) * v * i * d["power_factor"] / 1000
    w = cfg["frozen_window"]
    frozen = (i.rolling(w, min_periods=w).std() < 1e-6) & (v.rolling(w, min_periods=w).std() < 1e-6)

    hit = pd.DataFrame({
        "R1_VOLTAGE_RANGE": (v < cfg["v_min"]) | (v > cfg["v_max"]),
        "R2_FREQUENCY_RANGE": (d["frequency_hz"] < cfg["f_min"]) | (d["frequency_hz"] > cfg["f_max"]),
        "R3_OVERCURRENT": i > cfg["i_max"],
        "R4_OVERTEMP": d["transformer_temp_c"] > cfg["temp_max"],
        "R5_PF_RANGE": (d["power_factor"] < cfg["pf_min"]) | (d["power_factor"] > cfg["pf_max"]),
        "R6_POWER_MISMATCH": (d["power_mw"] - p_calc).abs() > cfg["power_tol_mw"],
        "R7_BREAKER_OPEN_FLOW": (d["breaker_status"] == 0) & (i > cfg["breaker_open_current"]),
        "R8_RATE_OF_CHANGE": i.diff().abs() > cfg["di_max"],
        "R9_FROZEN_SENSOR": frozen.fillna(False),
    }, index=d.index).fillna(False)

    sev = np.array([RULES[c][0] for c in hit.columns])
    d["physics_severity"] = (hit.values * sev).max(axis=1).astype(int)
    d["rules_triggered"] = hit.apply(lambda r: ";".join(c for c in hit.columns if r[c]), axis=1)
    d["physics_reason"] = hit.apply(
        lambda r: " | ".join(RULES[c][1] for c in hit.columns if r[c]), axis=1)
    d["power_calc_mw"] = p_calc
    return d
