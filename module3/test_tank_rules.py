"""Water-tank rule tests (no external files needed)."""
import numpy as np
import pandas as pd
import tank_rules as tk
from alert_engine import run_pipeline, build_incidents
import incident_analysis as ia
from tank_data import inject_stress


def normal(n=400, seed=0):
    r = np.random.default_rng(seed)   # same distributions as Module 1's generator
    return pd.DataFrame({"timestamp": pd.date_range("2026-10-13", periods=n, freq="s"),
                         "Tank_Water_Level": r.normal(50, 5, n), "Pressure_Sensor": r.normal(30, 2.1, n),
                         "Flow_Rate": r.normal(12.5, 1.2, n), "Valve_Status": r.choice([0, 1], n, p=[.1, .9]),
                         "ai_score": 0.0})


def hits(d):
    return set(";".join(tk.apply_rules(tk.validate_input(d))["rules_triggered"]).split(";")) - {""}


def test_normal_data_has_few_false_alarms():
    d = tk.apply_rules(tk.validate_input(normal(5000)))
    assert (d.physics_severity >= 2).mean() < 0.01


def test_module1_attack_signature():
    d = normal(); d.loc[10, ["Tank_Water_Level", "Pressure_Sensor"]] = [85, 52]
    assert {"T1_LEVEL_HIGH", "T4_PRESSURE_HIGH"} <= hits(d)


def test_level_pressure_mismatch():
    d = normal(); d.loc[10, ["Tank_Water_Level", "Pressure_Sensor"]] = [38, 38]
    assert "T7_LEVEL_PRESSURE_MISMATCH" in hits(d)


def test_valve_closed_needs_a_sustained_run():
    d = normal(); d["Valve_Status"] = 1
    d.loc[20:23, "Valve_Status"] = 0                    # 4 samples: sensor chatter, not an attack
    assert "T8_VALVE_CLOSED_FLOW" not in hits(d)
    d.loc[20:34, "Valve_Status"] = 0                    # 15 samples closed while flow ~12.5
    assert "T8_VALVE_CLOSED_FLOW" in hits(d)


def test_frozen_run_flagged():
    d = normal()
    for c in ["Tank_Water_Level", "Pressure_Sensor", "Flow_Rate", "Valve_Status"]:
        d.loc[50:79, c] = d.loc[50, c]
    r = tk.apply_rules(tk.validate_input(d))
    assert r.loc[50:79, "rules_triggered"].str.contains("T9").all()


def test_input_validation():
    for bad in (normal().drop(columns=["Flow_Rate"]), normal().assign(Valve_Status=2)):
        try: tk.validate_input(bad); assert False
        except ValueError: pass


def test_accepts_module1_column_name_Timestamp():
    assert "timestamp" in tk.validate_input(normal().rename(columns={"timestamp": "Timestamp"})).columns


def test_classification_and_stress_incidents():
    d = inject_stress(normal(1000).assign(true_attack=0, attack_type="normal"))
    res = run_pipeline(d, 0.5, profile=tk)
    types = set(ia.enrich_incidents(build_incidents(res), res)["attack_type"])
    assert {"Valve / flow spoofing", "Replay / frozen sensor", "Overflow / over-pressure attack",
            "Sensor spoofing (inconsistent readings)"} <= types
